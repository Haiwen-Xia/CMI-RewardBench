"""
Backbone components for Reward Model

Contains:
- Configuration dataclasses (DropoutConfig, NullEmbeddingConfig)
- Utility functions
- Base backbone functionality (init_mulan_modules, init_downsampler, etc.)
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
import logging
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass

from omegaconf import DictConfig, OmegaConf

from ..modules.transformer import Transformer, LayerNorm, TransformerDecoderLayer
from ..modules.utils import exists, frozen_params, default
from .audio import AudioSpectrogramTransformerPretrained
from .text import TextTransformerPretrained
from .downsample import build_downsampler

logger = logging.getLogger(__name__)


# ============ Configuration Dataclasses ============

@dataclass
class DropoutConfig:
    """Configuration for modality dropout during training"""
    dropout: float = 0.0
    length: int = 10  # Length of null embedding sequence


@dataclass
class NullEmbeddingConfig:
    """Configuration for null embeddings (used when modality is dropped out)"""
    text: DropoutConfig
    lyrics: DropoutConfig
    audio: DropoutConfig
    
    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "NullEmbeddingConfig":
        if d is None:
            return cls(
                text=DropoutConfig(dropout=0.2, length=10),
                lyrics=DropoutConfig(dropout=0.3, length=10),
                audio=DropoutConfig(dropout=0.5, length=10),
            )
        return cls(
            text=DropoutConfig(
                dropout=d.get('text', {}).get('dropout', 0.2),
                length=d.get('text', {}).get('length', 10)
            ),
            lyrics=DropoutConfig(
                dropout=d.get('lyrics', {}).get('dropout', 0.3),
                length=d.get('lyrics', {}).get('length', 10)
            ),
            audio=DropoutConfig(
                dropout=d.get('audio', {}).get('dropout', 0.5),
                length=d.get('audio', {}).get('length', 10)
            ),
        )


# ============ Utility Functions ============

def _mb(x): 
    return x / 1024 / 1024


def count_parameters(model):
    """Count number of parameters in PyTorch model"""
    n_all = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all_m = n_all / 1e6
    n_trainable_m = n_trainable / 1e6
    logger.info(f"Parameter Count: all {n_all_m:.3f}M; trainable {n_trainable_m:.3f}M")


def _get_nested(cfg, *keys):
    """Safely get nested value from config (dict or OmegaConf)"""
    for key in keys:
        if cfg is None:
            return None
        if isinstance(cfg, DictConfig):
            if key in cfg:
                cfg = cfg[key]
            else:
                return None
        elif isinstance(cfg, dict):
            cfg = cfg.get(key)
        else:
            return None
    return cfg


# ============ Backbone Mixin ============

class BackboneMixin:
    """Mixin class providing backbone initialization and utility methods
    
    To be used by RewardAttentionModel and other model classes.
    """
    
    # Head 参数前缀（用于分离存储/加载）
    HEAD_PARAM_PREFIXES = (
        'score_projector.',
        'single_score_projector.',
        'alignment_head.',
        'quality_head.',
    )
    
    def init_null_embeddings(self):
        """Initialize null embeddings for dropout during training"""
        # Parse null embedding config
        null_cfg = None
        if self.config is not None:
            null_cfg = _get_nested(self.config, 'model', 'null_embedding')
            if null_cfg is None:
                null_cfg = _get_nested(self.config, 'null_embedding')
            
            if isinstance(null_cfg, DictConfig):
                null_cfg = OmegaConf.to_container(null_cfg, resolve=True)
        
        self.null_embedding_config = NullEmbeddingConfig.from_dict(null_cfg)
        
        # Initialize learnable null embeddings
        text_len = self.null_embedding_config.text.length
        self.null_text_embedding = nn.Parameter(
            torch.randn(1, text_len, self.dim) * 0.02
        )
        
        lyrics_len = self.null_embedding_config.lyrics.length
        self.null_lyrics_embedding = nn.Parameter(
            torch.randn(1, lyrics_len, self.dim) * 0.02
        )
        
        audio_len = self.null_embedding_config.audio.length
        self.null_audio_embedding = nn.Parameter(
            torch.randn(1, audio_len, self.dim) * 0.02
        )
        
        # Category embeddings placeholder
        self.category_embeddings: Dict[str, nn.Parameter] = nn.ParameterDict()
        
        logger.info(f"Initialized null embeddings: text={text_len}, lyrics={lyrics_len}, audio={audio_len}")
        logger.info(f"Dropout rates: text={self.null_embedding_config.text.dropout}, "
                   f"lyrics={self.null_embedding_config.lyrics.dropout}, "
                   f"audio={self.null_embedding_config.audio.dropout}")
    
    def init_downsampler(self):
        """Initialize ref and eval downsamplers from config"""
        if self.config is None:
            self.ref_downsampler = build_downsampler("none", 0, 0)
            self.eval_downsampler = build_downsampler("none", 0, 0)
            self.text_downsampler = build_downsampler("none", 0, 0)
            self.lyrics_downsampler = build_downsampler("none", 0, 0)
            logger.info("No config provided, using identity downsamplers")
            return
        
        ds_cfg = _get_nested(self.config, 'model', 'downsample')
        if ds_cfg is not None:
            logger.info("config.model.downsample found")
        else:
            ds_cfg = _get_nested(self.config, 'downsample')
            if ds_cfg is not None:
                logger.info("config.downsample found")
        
        if ds_cfg is None:
            self.ref_downsampler = build_downsampler("none", 0, 0)
            self.eval_downsampler = build_downsampler("none", 0, 0)
            self.text_downsampler = build_downsampler("none", 0, 0)
            self.lyrics_downsampler = build_downsampler("none", 0, 0)
            logger.info("No downsample config found, using identity downsamplers")
            return
        
        if isinstance(ds_cfg, DictConfig):
            ds_cfg = OmegaConf.to_container(ds_cfg, resolve=True)
        
        all_configs = ds_cfg.get('configs', {})
        if not all_configs:
            logger.warning("No downsample configs defined, using identity")
            self.ref_downsampler = build_downsampler("none", 0, 0)
            self.eval_downsampler = build_downsampler("none", 0, 0)
            self.text_downsampler = build_downsampler("none", 0, 0)
            self.lyrics_downsampler = build_downsampler("none", 0, 0)
            return
        
        eval_name = ds_cfg.get('eval', 'none')
        if eval_name not in all_configs:
            raise ValueError(
                f"Downsample config '{eval_name}' not found. "
                f"Available configs: {list(all_configs.keys())}"
            )
        eval_cfg = all_configs[eval_name].copy()
        
        ref_name = ds_cfg.get('ref', None)
        if ref_name is None:
            ref_cfg = eval_cfg.copy()
            logger.info(f"Ref downsampler inherits from eval ('{eval_name}')")
        else:
            if ref_name not in all_configs:
                raise ValueError(
                    f"Downsample config '{ref_name}' not found. "
                    f"Available configs: {list(all_configs.keys())}"
                )
            ref_cfg = all_configs[ref_name].copy()
            logger.info(f"Ref downsampler: '{ref_name}', Eval downsampler: '{eval_name}'")
        
        text_name = ds_cfg.get('text', 'none')
        if text_name not in all_configs:
            raise ValueError(
                f"Downsample config '{text_name}' not found. "
                f"Available configs: {list(all_configs.keys())}"
            )
        text_cfg = all_configs[text_name].copy()
        
        eval_cfg['dim'] = self.dim
        ref_cfg['dim'] = self.dim
        text_cfg['dim'] = self.dim

        self.text_downsampler = build_downsampler(**text_cfg)
        self.lyrics_downsampler = build_downsampler(**text_cfg)
        self.eval_downsampler = build_downsampler(**eval_cfg)
        self.ref_downsampler = build_downsampler(**ref_cfg)
        
        logger.info(f"✓ Eval downsampler: {eval_name} -> {eval_cfg}")
        logger.info(f"✓ Ref downsampler: {ref_name or eval_name} -> {ref_cfg}")
        logger.info(f"✓ Text downsampler: {text_name} -> {text_cfg}")
    
    def init_mulan_modules(self, name):
        """Initialize MuQ-MuLan text and audio modules"""
        from muq import MuQMuLan
        mulan = MuQMuLan.from_pretrained(name)
        
        # Parse text_encoder_config
        text_tune = None
        if self.text_encoder_config is not None:
            if isinstance(self.text_encoder_config, dict):
                text_tune = self.text_encoder_config.get('tune', None)
            logger.info(f"Text encoder config: {self.text_encoder_config}")
        
        # Initialize text module with LoRA support if configured
        if self.text_lora_config is not None and self.text_lora_config.get('use_lora', False):
            logger.info(f"Initializing text module with LoRA: {self.text_lora_config}")
            original_text = mulan.mulan_module.text
            self.text_module = TextTransformerPretrained(
                model_name=original_text.model_name,
                dim=original_text.dim,
                model_dim=None,
                max_seq_len=original_text.max_seq_len,
                tf_depth=original_text.transformer.depth if hasattr(original_text.transformer, 'depth') else 12,
                frozen_pretrained=False,
                hf_hub_cache_dir=original_text.hf_hub_cache_dir,
                use_lora=True,
                lora_r=self.text_lora_config.get('lora_r', 8),
                lora_alpha=self.text_lora_config.get('lora_alpha', 16),
                lora_dropout=self.text_lora_config.get('lora_dropout', 0.1),
                lora_target_modules=self.text_lora_config.get('target_modules', None),
            )
            frozen_params(self.text_module.transformer)
            logger.info("Frozen text_module.transformer (currently not applying LoRA targets found)")
        else:
            self.text_module: TextTransformerPretrained = mulan.mulan_module.text 
            if self.freeze_text:
                frozen_params(self.text_module)
            
            if text_tune == 'transformer':
                if hasattr(self.text_module, 'transformer'):
                    for param in self.text_module.transformer.parameters():
                        param.requires_grad = True
                    n_params = sum(p.numel() for p in self.text_module.transformer.parameters())
                    logger.info(f"Unfroze text_module.transformer: {n_params/1e6:.2f}M params (text_encoder.tune='transformer')")
                else:
                    logger.warning("text_encoder.tune='transformer' but text_module has no 'transformer' attribute")
            elif text_tune == 'all':
                for param in self.text_module.parameters():
                    param.requires_grad = True
                n_params = sum(p.numel() for p in self.text_module.parameters())
                logger.info(f"Unfroze entire text_module: {n_params/1e6:.2f}M params (text_encoder.tune='all')")
            elif text_tune is not None:
                logger.warning(f"Unknown text_encoder.tune value: '{text_tune}'. Supported: 'transformer', 'all'")
        
        self.audio_module: AudioSpectrogramTransformerPretrained = mulan.mulan_module.audio
        if self.freeze_audio:
            frozen_params(self.audio_module)
            logger.info("Frozen audio_module (freeze_audio=True)")
        else:
            n_params = sum(p.numel() for p in self.audio_module.parameters())
            logger.info(f"Audio module trainable: {n_params/1e6:.2f}M params (freeze_audio=False)")
        
        if self.config is not None and self.config.get("train_muq_depth", 0) > 0:
            logger.info(len(self.audio_module.model.model.conformer.layers))
            for conv in self.audio_module.model.model.conformer.layers[-self.config.get("train_muq_depth", 0):]:
                for param in conv.parameters():
                    param.requires_grad = True
    
    def _apply_modality_dropout(
        self,
        embeds: torch.Tensor,
        mask: torch.Tensor,
        null_embedding: nn.Parameter,
        dropout_prob: float,
        is_null: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply dropout to a modality by replacing with null embedding"""
        batch_size = embeds.shape[0]
        device = embeds.device
        
        if self.training and dropout_prob > 0:
            dropout_mask = torch.rand(batch_size, device=device) < dropout_prob
        else:
            dropout_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        if is_null is not None:
            dropout_mask = dropout_mask | is_null.to(device)
        
        if not dropout_mask.any():
            return embeds, mask
        
        null_len = null_embedding.shape[1]
        null_embeds_batch = null_embedding.expand(batch_size, -1, -1).to(dtype=embeds.dtype, device=device)
        null_mask_batch = torch.ones(batch_size, null_len, dtype=mask.dtype, device=device)
        
        output_embeds = embeds.clone()
        output_mask = mask.clone()
        
        seq_len = embeds.shape[1]
        
        if null_len >= seq_len:
            output_embeds[dropout_mask] = null_embeds_batch[dropout_mask, :seq_len, :]
            output_mask[dropout_mask] = null_mask_batch[dropout_mask, :seq_len]
        else:
            output_embeds[dropout_mask, :null_len, :] = null_embeds_batch[dropout_mask]
            output_embeds[dropout_mask, null_len:, :] = 0
            output_mask[dropout_mask, :null_len] = 1
            output_mask[dropout_mask, null_len:] = 0
        
        return output_embeds, output_mask
    
    def _detect_null_content(
        self,
        embeds: torch.Tensor,
        mask: torch.Tensor,
        threshold: float = 0.1,
    ) -> torch.Tensor:
        """Detect which samples have effectively null/empty content"""
        valid_count = mask.sum(dim=1)
        total_count = mask.shape[1]
        is_null = valid_count < (total_count * threshold)
        return is_null
    
    def _apply_joint_transformer(self, prompt_embed, prompt_mask, eval_audio_embeds, eval_audio_mask):
        """Apply joint transformer to get eval audio embedding"""
        if self.attention_mode == 'SA':
            joint_embed = torch.cat([prompt_embed, eval_audio_embeds], dim=1)
            joint_mask = torch.cat([prompt_mask, eval_audio_mask], dim=1)
            joint_output = self.joint_transformer(joint_embed, mask=joint_mask)
            
            prompt_len = prompt_mask.shape[1]
            eval_audio_output = joint_output[:, prompt_len:, :]
            
            eval_audio_mask_expanded = eval_audio_mask.unsqueeze(-1)
            masked_sum = (eval_audio_output * eval_audio_mask_expanded).sum(dim=1)
            valid_count = eval_audio_mask.sum(dim=1, keepdim=True).clamp(min=1)
            eval_audio_embedding = masked_sum / valid_count

        else:  # attention_mode == 'CA'
            decoder_output = eval_audio_embeds
            for decoder_layer in self.joint_transformer:
                decoder_output = decoder_layer(
                    x=decoder_output,
                    encoder_hidden_states=prompt_embed,
                    self_attn_mask=eval_audio_mask,
                    cross_attn_mask_q=eval_audio_mask,
                    cross_attn_mask_kv=prompt_mask
                )
            
            eval_audio_mask_expanded = eval_audio_mask.unsqueeze(-1)
            masked_sum = (decoder_output * eval_audio_mask_expanded).sum(dim=1)
            valid_count = eval_audio_mask.sum(dim=1, keepdim=True).clamp(min=1)
            eval_audio_embedding = masked_sum / valid_count
        
        return eval_audio_embedding
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.gradient_checkpointing = True
        if hasattr(self.prompt_transformer, 'gradient_checkpointing_enable'):
            self.prompt_transformer.gradient_checkpointing_enable()
        if self.attention_mode == 'SA':
            if hasattr(self.joint_transformer, 'gradient_checkpointing_enable'):
                self.joint_transformer.gradient_checkpointing_enable()
        else:
            for layer in self.joint_transformer:
                if hasattr(layer, 'gradient_checkpointing_enable'):
                    layer.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        self.gradient_checkpointing = False
        if hasattr(self.prompt_transformer, 'gradient_checkpointing_disable'):
            self.prompt_transformer.gradient_checkpointing_disable()
        if self.attention_mode == 'SA':
            if hasattr(self.joint_transformer, 'gradient_checkpointing_disable'):
                self.joint_transformer.gradient_checkpointing_disable()
        else:
            for layer in self.joint_transformer:
                if hasattr(layer, 'gradient_checkpointing_disable'):
                    layer.gradient_checkpointing_disable()
    
    # ============ Backbone/Heads 分离存储与加载 ============
    
    def get_backbone_state_dict(self) -> dict:
        """获取 backbone 参数（不包含 heads）"""
        return {
            k: v for k, v in self.state_dict().items()
            if not any(k.startswith(prefix) for prefix in self.HEAD_PARAM_PREFIXES)
        }
    
    def get_heads_state_dict(self) -> dict:
        """获取 heads 参数"""
        return {
            k: v for k, v in self.state_dict().items()
            if any(k.startswith(prefix) for prefix in self.HEAD_PARAM_PREFIXES)
        }
    
    def load_backbone(
        self, 
        checkpoint_path: str, 
        strict: bool = False,
        trust_checkpoint: bool = False,
    ) -> tuple:
        """只加载 backbone 参数"""
        pkg = self._load_checkpoint(checkpoint_path, trust_checkpoint)
        
        if 'model' in pkg:
            full_state = pkg['model']
        elif 'backbone' in pkg:
            full_state = pkg['backbone']
        else:
            full_state = pkg
        
        backbone_state = {
            k: v for k, v in full_state.items()
            if not any(k.startswith(prefix) for prefix in self.HEAD_PARAM_PREFIXES)
        }
        
        missing, unexpected = self.load_state_dict(backbone_state, strict=False)
        logger.info(f"Loaded backbone from {checkpoint_path}: {len(backbone_state)} params")
        return missing, unexpected
    
    def load_heads(
        self,
        checkpoint_path: str,
        head_names: list = None,
        trust_checkpoint: bool = False,
    ) -> tuple:
        """只加载 heads 参数"""
        pkg = self._load_checkpoint(checkpoint_path, trust_checkpoint)
        
        if 'heads' in pkg:
            full_state = pkg['heads']
        elif 'model' in pkg:
            full_state = pkg['model']
        else:
            full_state = pkg
        
        if head_names is None:
            prefixes = self.HEAD_PARAM_PREFIXES
        else:
            prefixes = tuple(f"{name}." for name in head_names)
        
        heads_state = {
            k: v for k, v in full_state.items()
            if any(k.startswith(prefix) for prefix in prefixes)
        }
        
        if heads_state:
            missing, unexpected = self.load_state_dict(heads_state, strict=False)
            logger.info(f"Loaded heads from {checkpoint_path}: {len(heads_state)} params")
            return missing, unexpected
        return [], []
    
    def load_backbone_and_heads(
        self,
        backbone_path: str = None,
        heads_path: str = None,
        trust_checkpoint: bool = False,
    ) -> dict:
        """分别加载 backbone 和 heads"""
        result: dict = {'backbone_loaded': False, 'heads_loaded': False}
        
        if backbone_path:
            self.load_backbone(backbone_path, trust_checkpoint=trust_checkpoint)
            result['backbone_loaded'] = True
            result['backbone_path'] = backbone_path
        
        if heads_path:
            self.load_heads(heads_path, trust_checkpoint=trust_checkpoint)
            result['heads_loaded'] = True
            result['heads_path'] = heads_path
        
        return result
    
    @staticmethod
    def _load_checkpoint(path: str, trust_checkpoint: bool = False) -> dict:
        """统一 checkpoint 加载"""
        try:
            return torch.load(path, map_location='cpu', weights_only=True)
        except Exception:
            if trust_checkpoint:
                return torch.load(path, map_location='cpu', weights_only=False)
            raise
