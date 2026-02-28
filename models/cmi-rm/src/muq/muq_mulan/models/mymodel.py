import time
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
from ..modules.transformer import Transformer, LayerNorm, posemb_sincos_2d, GEGLU, FeedForward, TransformerDecoderLayer
from .audio import AudioSpectrogramTransformerPretrained
from .text import TextTransformerPretrained
from .text_t5 import T5TextEncoder
from .downsample import build_downsampler, MeanPoolMLPDownsampler, _DepthwiseConvDown
from ..modules.utils import exists, frozen_params, default
from ..utils.model_utils import DropoutConfig, NullEmbeddingConfig
import logging
logger = logging.getLogger(__name__)
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass


def _mb(x): 
    return x / 1024 / 1024


    
def count_parameters(model):
    global logger
    """Count number of parameters in PyTorch model,
    References: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7.
    """
    n_all = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # transform to (M)
    n_all_m = n_all / 1e6
    n_trainable_m = n_trainable / 1e6
    logger.info(f"Parameter Count: all {n_all_m:.3f}M; trainable {n_trainable_m:.3f}M")

# export HF_HOME = ...
    



class RewardAttentionModel(nn.Module):
    """Reward Model with multiple attention interaction modes
    
    Modes:
    - 'SA': Self-Attention - all tokens attend to all tokens
    - 'CA': Cross-Attention - eval_audio does self-attention + queries prompt embeddings (TransformerDecoder style)
    """
    def __init__(self,
                 model_name = 'OpenMuQ/MuQ-MuLan-large',
                 dim = 768,
                 mode = 'concat_text_late',
                 attention_mode = 'SA',  # 'SA' or 'CA'
                 mlp_dim = 512,
                 output_dim = 2,
                sr = 24000,
                prompt_tf_depth = 1,
                joint_tf_depth = 4,
                dim_head = 64,
                heads = 8,
                attn_dropout = 0.,
                ff_dropout = 0.,
                ff_mult = 4,
                use_layer_idx = -1,
                freeze_audio = True,
                freeze_text = True,
                # [DEPRECATED] train_muqmulan removed - use text_encoder.tune and train_muq_depth instead
                # LoRA config for text encoder
                text_lora_config = None,  # Dict with keys: use_lora, lora_r, lora_alpha, lora_dropout, target_modules
                # Text encoder config (new format): {name: 'muq_mulan', tune: 'transformer'}
                text_encoder_config = None, 
                cfg = None # full config, including the original hydra config; config.model to access information
                    ) -> None:
        super().__init__()
        self.dim = dim
        self.config = cfg
        self.text_lora_config = text_lora_config
        self.text_encoder_config = text_encoder_config

        
        self.prompt_transformer = Transformer(
            dim = dim,
            depth = prompt_tf_depth,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_mult = ff_mult,
            use_flash_attn = True  # Default to FlashAttention
        )
        self.attention_mode = attention_mode
        assert attention_mode in ['SA', 'CA'], f"attention_mode must be 'SA' or 'CA', got {attention_mode}"
        
        if attention_mode == 'SA':
            # Self-Attention: all tokens attend to all tokens
            self.joint_transformer = Transformer(
                dim = dim,
                depth = joint_tf_depth,
                dim_head = dim_head,
                heads = heads,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                ff_mult = ff_mult,
                use_flash_attn = True  # Default to FlashAttention
            )
        else:  # attention_mode == 'CA'
            # Cross-Attention: TransformerDecoder style (self-attn + cross-attn + FFN)
            self.joint_transformer = nn.ModuleList([
                TransformerDecoderLayer(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    ff_mult=ff_mult,
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    use_flash_attn=True  # Default to FlashAttention
                ) for _ in range(joint_tf_depth)
            ])
        
        self.score_projector = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, output_dim)
        )
        self.single_score_projector = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, 1)
        )
        
        # New separated heads for IF/MQ tasks
        # alignment_head: for Instruction Following (text-music alignment) preference
        self.alignment_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, 1)
        )
        # quality_head: for Music Quality preference
        self.quality_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, 1)
        )
        
        # ICL prompts configuration
        # Format: {task: [list of possible prompts]}
        # During training/inference, one prompt is randomly selected from the list
        self.icl_prompts = {
            'quality': [
                # Production & Recording Quality
                "Production-ready commercial music.",
                "Studio-quality recording with excellent mixing.",
                "Professional mastering with balanced dynamics.",
                "Crystal clear audio with pristine production.",
                # Musicality & Composition
                "Catchy melody with memorable hooks.",
                "Comfortable groove that makes you move.",
                "Tight and consistent rhythm throughout.",
                "Rich harmonic progression with depth.",
                "Well-structured arrangement with clear sections.",
                "Engaging musical narrative with a clear theme.",
                # Creativity & Innovation
                "Fresh and innovative sound elements.",
                "Creative use of textures and timbres.",
                "Unique sonic palette with interesting choices.",
                # Emotional & Artistic
                "Emotionally moving and expressive music.",
                "Strong artistic vision with coherent style.",
                "Compelling musical storytelling.",
                # Technical Excellence  
                "Excellent instrument balance and separation.",
                "Polished performance with precise execution.",
                "Seamless transitions between sections.",
                "Dynamic range that breathes naturally.",
            ],
            'alignment': [
                "This music perfectly matches the given description.",
                "The music aligns well with the text prompt.",
                "Faithful interpretation of the requested style.",
                "Accurately captures the intended mood and genre.",
            ],
        }
        
        self.mode = mode
        self.freeze_audio = freeze_audio
        self.freeze_text = freeze_text
        self.gradient_checkpointing = False # default off, will be changed via method calls
        self.model_dtype = None  # Will be lazily initialized on first forward pass
        self.init_mulan_modules(model_name)
        self.init_downsampler()
        self.init_null_embeddings()
    
    def init_null_embeddings(self):
        """Initialize null embeddings for dropout during training
        
        Null embeddings are learnable parameters used to replace actual embeddings
        when a modality is dropped out during training. This helps the model learn
        to handle missing modalities gracefully.
        """
        # Helper function to safely get nested config values
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
        
        # Parse null embedding config
        # Config can be at: config.model.null_embedding or config.null_embedding
        null_cfg = None
        if self.config is not None:
            # Try model.null_embedding first (Hydra config structure)
            null_cfg = _get_nested(self.config, 'model', 'null_embedding')
            if null_cfg is None:
                null_cfg = _get_nested(self.config, 'null_embedding')
            
            # Convert to dict if OmegaConf
            if isinstance(null_cfg, DictConfig):
                null_cfg = OmegaConf.to_container(null_cfg, resolve=True)
        
        self.null_embedding_config = NullEmbeddingConfig.from_dict(null_cfg)
        
        # Initialize learnable null embeddings
        # Text null embedding: [1, text_length, dim]
        text_len = self.null_embedding_config.text.length
        self.null_text_embedding = nn.Parameter(
            torch.zeros(1, text_len, self.dim) 
        )
        
        # Lyrics null embedding: [1, lyrics_length, dim]  
        lyrics_len = self.null_embedding_config.lyrics.length
        self.null_lyrics_embedding = nn.Parameter(
            torch.zeros(1, lyrics_len, self.dim)
        )
        
        # Audio null embedding: [1, audio_length, dim]
        audio_len = self.null_embedding_config.audio.length
        self.null_audio_embedding = nn.Parameter(
            torch.zeros(1, audio_len, self.dim)
        )
        
        # Category embeddings placeholder (for future use)
        # Dict: category_name -> embedding [N, D]
        self.category_embeddings: Dict[str, nn.Parameter] = nn.ParameterDict()
        
        logger.info(f"Initialized null embeddings: text={text_len}, lyrics={lyrics_len}, audio={audio_len}")
        logger.info(f"Dropout rates: text={self.null_embedding_config.text.dropout}, "
                   f"lyrics={self.null_embedding_config.lyrics.dropout}, "
                   f"audio={self.null_embedding_config.audio.dropout}")
        if self.null_embedding_config.skip_null:
            logger.info("skip_null=True: Dropped modalities will use empty sequences instead of null embeddings")
    
    def init_downsampler(self):
        """Initialize ref and eval downsamplers from config"""
        # Access downsample config from model.downsample or directly from config
        
        if self.config is None:
            # No config at all
            self.ref_downsampler = build_downsampler("none", 0, 0)
            self.eval_downsampler = build_downsampler("none", 0, 0)
            self.text_downsampler = build_downsampler("none", 0, 0)
            self.lyrics_downsampler = build_downsampler("none", 0, 0)
            logger.info("No config provided, using identity downsamplers")
            return
        
        # Helper function to safely get nested config values
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
        
        # Try to access downsample config (could be at config.model.downsample or config.downsample)
        ds_cfg = _get_nested(self.config, 'model', 'downsample')
        if ds_cfg is not None:
            logger.info("config.model.downsample found")
        else:
            ds_cfg = _get_nested(self.config, 'downsample')
            if ds_cfg is not None:
                logger.info("config.downsample found")
        
        if ds_cfg is None:
            # No downsample config found
            self.ref_downsampler = build_downsampler("none", 0, 0)
            self.eval_downsampler = build_downsampler("none", 0, 0)
            self.text_downsampler = build_downsampler("none", 0, 0)
            self.lyrics_downsampler = build_downsampler("none", 0, 0)
            logger.info("No downsample config found, using identity downsamplers")
            return
        
        # Convert to dict if OmegaConf
        if isinstance(ds_cfg, DictConfig):
            ds_cfg = OmegaConf.to_container(ds_cfg, resolve=True)
        
        # 1. 获取所有预定义配置
        all_configs = ds_cfg.get('configs', {})
        if not all_configs:
            logger.warning("No downsample configs defined, using identity")
            self.ref_downsampler = build_downsampler("none", 0, 0)
            self.eval_downsampler = build_downsampler("none", 0, 0)
            self.text_downsampler = build_downsampler("none", 0, 0)
            self.lyrics_downsampler = build_downsampler("none", 0, 0)
            return
        
        # 2. 获取 eval 配置名称
        eval_name = ds_cfg.get('eval', 'none')
        if eval_name not in all_configs:
            raise ValueError(
                f"Downsample config '{eval_name}' not found. "
                f"Available configs: {list(all_configs.keys())}"
            )
        eval_cfg = all_configs[eval_name].copy()
        
        # 3. 获取 ref 配置名称（默认继承 eval）
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
        # 4. 注入 dim（由模型提供）
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
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.gradient_checkpointing = True
        # Enable for prompt transformer
        if hasattr(self.prompt_transformer, 'gradient_checkpointing_enable'):
            self.prompt_transformer.gradient_checkpointing_enable()
        # Enable for joint transformer
        if self.attention_mode == 'SA':
            if hasattr(self.joint_transformer, 'gradient_checkpointing_enable'):
                self.joint_transformer.gradient_checkpointing_enable()
        else:  # CA mode
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
        
        
    def init_mulan_modules(self, name):
        from muq import MuQMuLan
        mulan = MuQMuLan.from_pretrained(name)
        
        # Parse text_encoder_config
        text_encoder_name = None
        text_tune = None
        text_encoder_kwargs = {}
        if self.text_encoder_config is not None:
            if isinstance(self.text_encoder_config, dict):
                text_encoder_name = self.text_encoder_config.get('name', 'muq_mulan')
                text_tune = self.text_encoder_config.get('tune', None)
                # Extract T5-specific config
                text_encoder_kwargs = {
                    'model_name': self.text_encoder_config.get('model_name', 'google/flan-t5-base'),
                    'max_seq_len': self.text_encoder_config.get('max_seq_len', 512),
                    'tune_last_n_layers': self.text_encoder_config.get('tune_last_n_layers', 0),
                    'use_lora': self.text_encoder_config.get('use_lora', False),
                    'lora_r': self.text_encoder_config.get('lora_r', 8),
                    'lora_alpha': self.text_encoder_config.get('lora_alpha', 16),
                    'lora_dropout': self.text_encoder_config.get('lora_dropout', 0.1),
                    'lora_target_modules': self.text_encoder_config.get('lora_target_modules', None),
                }
            logger.info(f"Text encoder config: {self.text_encoder_config}")
        
        # ============ Initialize Text Module ============
        # Support multiple text encoder backends: muq_mulan, t5
        
        if text_encoder_name == 't5':
            # Use T5TextEncoder (Flan-T5)
            logger.info(f"Initializing T5TextEncoder: {text_encoder_kwargs.get('model_name')}")
            
            # Determine freeze strategy
            freeze = self.freeze_text and text_tune is None
            tune_last_n = text_encoder_kwargs.get('tune_last_n_layers', 0)
            use_lora = text_encoder_kwargs.get('use_lora', False)
            
            # Handle text_tune overrides
            if text_tune == 'all':
                freeze = False
                tune_last_n = 0  # Will unfreeze all
            elif text_tune == 'transformer':
                # For T5, 'transformer' means tune last few layers
                freeze = True
                if tune_last_n == 0:
                    tune_last_n = 3  # Default: tune last 6 layers (half of flan-t5-base)
            
            self.text_module = T5TextEncoder(
                model_name=text_encoder_kwargs.get('model_name', 'google/flan-t5-base'),
                dim=self.dim,
                max_seq_len=text_encoder_kwargs.get('max_seq_len', 512),
                freeze=freeze,
                tune_last_n_layers=tune_last_n,
                use_lora=use_lora,
                lora_r=text_encoder_kwargs.get('lora_r', 8),
                lora_alpha=text_encoder_kwargs.get('lora_alpha', 16),
                lora_dropout=text_encoder_kwargs.get('lora_dropout', 0.1),
                lora_target_modules=text_encoder_kwargs.get('lora_target_modules', None),
            )
            self.text_module.print_layer_info()
            
        elif self.text_lora_config is not None and self.text_lora_config.get('use_lora', False):
            # Initialize MuQ-MuLan text module with LoRA support
            logger.info(f"Initializing text module with LoRA: {self.text_lora_config}")
            # Recreate text module with LoRA parameters
            from .text import TextTransformerPretrained
            original_text = mulan.mulan_module.text
            self.text_module = TextTransformerPretrained(
                model_name=original_text.model_name,
                dim=original_text.dim,
                model_dim=None,  # Will use default projection
                max_seq_len=original_text.max_seq_len,
                tf_depth=original_text.transformer.depth if hasattr(original_text.transformer, 'depth') else 12,
                frozen_pretrained=False,  # LoRA mode doesn't freeze pretrained model (managed by PEFT)
                hf_hub_cache_dir=original_text.hf_hub_cache_dir,
                use_lora=True,
                lora_r=self.text_lora_config.get('lora_r', 8),
                lora_alpha=self.text_lora_config.get('lora_alpha', 16),
                lora_dropout=self.text_lora_config.get('lora_dropout', 0.1),
                lora_target_modules=self.text_lora_config.get('target_modules', None),
            )
            
            # Fallback: just freeze transformer if structure is unexpected
            frozen_params(self.text_module.transformer)
            logger.info("Frozen text_module.transformer (currently not applying LoRA targets found)")
        else:
            # Use pretrained text module from MuQMuLan, backwards compatible
            self.text_module: TextTransformerPretrained = mulan.mulan_module.text 
            if self.freeze_text: # all freeze at first
                frozen_params(self.text_module)
            
            # Handle text_encoder_config.tune option
            if text_tune == 'transformer':
                # Unfreeze the transformer layers in text module
                if hasattr(self.text_module, 'transformer'):
                    for param in self.text_module.transformer.parameters():
                        param.requires_grad = True
                    n_params = sum(p.numel() for p in self.text_module.transformer.parameters())
                    logger.info(f"Unfroze text_module.transformer: {n_params/1e6:.2f}M params (text_encoder.tune='transformer')")
                else:
                    logger.warning("text_encoder.tune='transformer' but text_module has no 'transformer' attribute")
            elif text_tune == 'all':
                # Unfreeze all text module parameters
                for param in self.text_module.parameters():
                    param.requires_grad = True
                n_params = sum(p.numel() for p in self.text_module.parameters())
                logger.info(f"Unfroze entire text_module: {n_params/1e6:.2f}M params (text_encoder.tune='all')")
            elif text_tune is not None:
                logger.warning(f"Unknown text_encoder.tune value: '{text_tune}'. Supported: 'transformer', 'all'")
        
        # Audio module - freeze based on freeze_audio parameter
        self.audio_module: AudioSpectrogramTransformerPretrained = mulan.mulan_module.audio
        if self.freeze_audio: # freeze audio module first, if not, all trainable, therefore there is no need to make this false
            frozen_params(self.audio_module)
            logger.info("Frozen audio_module (freeze_audio=True)")
        else:
            # Keep audio module trainable
            n_params = sum(p.numel() for p in self.audio_module.parameters())
            logger.info(f"Audio module trainable: {n_params/1e6:.2f}M params (freeze_audio=False)")
        
        # train_muq_depth: Unfreeze last N conformer layers in MuQ backbone
        # This is independent of freeze_audio - allows fine-grained control
        if self.config is not None and self.config.get("train_muq_depth", 0) > 0:
            logger.info(len(self.audio_module.model.model.conformer.layers))
            for conv in self.audio_module.model.model.conformer.layers[-self.config.get("train_muq_depth", 0):]:
                for param in conv.parameters():
                    param.requires_grad = True
                    
    def get_text_embedding(self, texts, lyrics):
        """Extract and combine text embeddings from texts and lyrics
        
        Args:
            texts: List of text strings
            lyrics: List of lyrics strings
            
        Returns:
            Tuple of (combined_text_embeds, combined_text_mask)
        """
        if self.mode == 'concat_text_late':
            if self.freeze_text:
                with torch.no_grad():
                    text_embeds, texts_mask = self.text_module(
                        raw_texts=texts,
                        return_mean=False,
                        return_mask=True
                    )
                    lyrics_embeds, lyrics_mask = self.text_module(
                        raw_texts=lyrics,
                        return_mean=False,
                        return_mask=True
                    )
            else:
                text_embeds, texts_mask = self.text_module(
                    raw_texts=texts,
                    return_mean=False,
                    return_mask=True
                )
                lyrics_embeds, lyrics_mask = self.text_module(
                    raw_texts=lyrics,
                    return_mean=False,
                    return_mask=True
                )
            combined_text_embeds = torch.cat([text_embeds, lyrics_embeds], dim=1)
            combined_text_mask = torch.cat([texts_mask, lyrics_mask], dim=1)
        
        elif self.mode == 'concat_text_early':
            combined_texts = [t + " " + l for t, l in zip(texts, lyrics)]
            if self.freeze_text:
                with torch.no_grad():
                    combined_text_embeds, combined_text_mask = self.text_module(
                        raw_texts=combined_texts,
                        return_mean=False,
                        return_mask=True
                    )
            else:
                combined_text_embeds, combined_text_mask = self.text_module(
                    raw_texts=combined_texts,
                    return_mean=False,
                    return_mask=True
                )
        
        elif self.mode == 'text_only':
            if self.freeze_text:
                with torch.no_grad():
                    combined_text_embeds, combined_text_mask = self.text_module(
                        raw_texts=texts,
                        return_mean=False,
                        return_mask=True
                    )
            else:
                combined_text_embeds, combined_text_mask = self.text_module(
                    raw_texts=texts,
                    return_mean=False,
                    return_mask=True
                )
        
        return combined_text_embeds, combined_text_mask
    
    def get_text_embedding_separate(self, texts, lyrics) -> Dict[str, torch.Tensor]:
        """Extract text and lyrics embeddings separately for dropout support
        
        Args:
            texts: List of text strings
            lyrics: List of lyrics strings
            
        Returns:
            Dict with keys:
                - 'text_embeds': Text embeddings [batch, text_len, dim]
                - 'text_mask': Text mask [batch, text_len]
                - 'lyrics_embeds': Lyrics embeddings [batch, lyrics_len, dim]
                - 'lyrics_mask': Lyrics mask [batch, lyrics_len]
                - 'text_boundary': int - boundary index between text and lyrics when concatenated
        """
        result = {}
        
        if self.freeze_text:
            with torch.no_grad():
                text_embeds, text_mask = self.text_module(
                    raw_texts=texts,
                    return_mean=False,
                    return_mask=True
                )
                lyrics_embeds, lyrics_mask = self.text_module(
                    raw_texts=lyrics,
                    return_mean=False,
                    return_mask=True
                )
        else:
            text_embeds, text_mask = self.text_module(
                raw_texts=texts,
                return_mean=False,
                return_mask=True
            )
            lyrics_embeds, lyrics_mask = self.text_module(
                raw_texts=lyrics,
                return_mean=False,
                return_mask=True
            )
        
        result['text_embeds'] = text_embeds
        result['text_mask'] = text_mask
        result['lyrics_embeds'] = lyrics_embeds
        result['lyrics_mask'] = lyrics_mask
        result['text_boundary'] = text_embeds.shape[1]  # Where text ends, lyrics begins
        
        return result
    
    def forward_muq_mulan(
        self, 
        texts, 
        lyrics, 
        ref_audios=None, 
        ref_audio_masks=None, 
        eval_audios=None, 
        eval_audio_masks=None,
        # Dropout configuration (can override null_embedding_config)
        text_dropout: Optional[float] = None,
        lyrics_dropout: Optional[float] = None,
        ref_audio_dropout: Optional[float] = None,
    ):
        """Forward pass through MuQ-MuLan encoders with null embedding support
        
        Extract embeddings from text, reference audio, and evaluation audio using 
        pretrained MuQ-MuLan modules. When any modality is None, automatically uses
        the corresponding null embedding.
        
        Args:
            texts: List of text strings [batch]
            lyrics: List of lyrics strings [batch]
            ref_audios: Reference/prompt audio waveforms [batch, audio_len] or None
            ref_audio_masks: Reference audio masks [batch, seq_len] or None
            eval_audios: Evaluation audio waveforms [batch, audio_len] or None
            eval_audio_masks: Evaluation audio masks [batch, seq_len] or None
            text_dropout: Override dropout probability for text
            lyrics_dropout: Override dropout probability for lyrics
            ref_audio_dropout: Override dropout probability for reference audio
            
        Returns:
            dict with keys:
                - 'text_embeds': Text embeddings [batch, text_len, dim]
                - 'text_mask': Text mask [batch, text_len]
                - 'lyrics_embeds': Lyrics embeddings [batch, lyrics_len, dim]
                - 'lyrics_mask': Lyrics mask [batch, lyrics_len]
                - 'ref_audio_embeds': Reference audio embeddings [batch, ref_len, dim]
                - 'ref_audio_mask': Reference audio mask [batch, ref_len]
                - 'eval_audio_embeds': Evaluation audio embeddings [batch, eval_len, dim]
                - 'eval_audio_mask': Evaluation audio mask [batch, eval_len]
        """
        result = {}
        
        # Determine dropout probabilities
        text_p = text_dropout if text_dropout is not None else (
            self.null_embedding_config.text.dropout if self.training else 0.0
        )
        lyrics_p = lyrics_dropout if lyrics_dropout is not None else (
            self.null_embedding_config.lyrics.dropout if self.training else 0.0
        )
        ref_audio_p = ref_audio_dropout if ref_audio_dropout is not None else (
            self.null_embedding_config.audio.dropout if self.training else 0.0
        )
        
        # Get text embeddings (separate for text and lyrics)
        text_output = self.get_text_embedding_separate(texts, lyrics)
        text_embeds = text_output['text_embeds']
        text_mask = text_output['text_mask']
        lyrics_embeds = text_output['lyrics_embeds']
        lyrics_mask = text_output['lyrics_mask']
        
        device = text_embeds.device
        batch_size = text_embeds.shape[0]
        
        # Downsample text and lyrics
        text_embeds_ds, text_mask_ds = self.text_downsampler(text_embeds, text_mask)
        lyrics_embeds_ds, lyrics_mask_ds = self.lyrics_downsampler(lyrics_embeds, lyrics_mask)
        
        # Apply dropout to text
        text_embeds_ds, text_mask_ds = self._apply_modality_dropout(
            text_embeds_ds, text_mask_ds,
            self.null_text_embedding,
            dropout_prob=text_p,
        )
        
        # Apply dropout to lyrics
        lyrics_embeds_ds, lyrics_mask_ds = self._apply_modality_dropout(
            lyrics_embeds_ds, lyrics_mask_ds,
            self.null_lyrics_embedding,
            dropout_prob=lyrics_p,
        )
        
        result['text_embeds'] = text_embeds_ds
        result['text_mask'] = text_mask_ds
        result['lyrics_embeds'] = lyrics_embeds_ds
        result['lyrics_mask'] = lyrics_mask_ds
        
        # Get reference audio embeddings if provided, otherwise use null embedding
        if ref_audios is not None and ref_audio_masks is not None:
            if self.freeze_audio:
                with torch.no_grad():
                    ref_audio_embeds, new_ref_mask = self.audio_module(
                        ref_audios,
                        mask=ref_audio_masks,
                        return_mean=False,
                        return_mask=True
                    )
            else:
                ref_audio_embeds, new_ref_mask = self.audio_module(
                    ref_audios,
                    mask=ref_audio_masks,
                    return_mean=False,
                    return_mask=True
                )
            # Downsample
            ref_audio_embeds, new_ref_mask = self.ref_downsampler(ref_audio_embeds, new_ref_mask)
            # Apply dropout
            ref_audio_embeds, new_ref_mask = self._apply_modality_dropout(
                ref_audio_embeds, new_ref_mask,
                self.null_audio_embedding,
                dropout_prob=ref_audio_p,
            )
        else:
            # Use null embedding when ref_audios is None
            ref_audio_embeds = self.null_audio_embedding.expand(batch_size, -1, -1)
            new_ref_mask = torch.ones(batch_size, ref_audio_embeds.shape[1], dtype=torch.bool, device=device)
        
        result['ref_audio_embeds'] = ref_audio_embeds
        result['ref_audio_mask'] = new_ref_mask
        
        # Get evaluation audio embeddings if provided, otherwise use null embedding
        if eval_audios is not None and eval_audio_masks is not None:
            if self.freeze_audio:
                with torch.no_grad():
                    eval_audio_embeds, new_eval_mask = self.audio_module(
                        eval_audios,
                        mask=eval_audio_masks,
                        return_mean=False,
                        return_mask=True
                    )
            else:
                eval_audio_embeds, new_eval_mask = self.audio_module(
                    eval_audios,
                    mask=eval_audio_masks,
                    return_mean=False,
                    return_mask=True
                )
            # Downsample
            eval_audio_embeds, new_eval_mask = self.eval_downsampler(eval_audio_embeds, new_eval_mask)
        else:
            # Use null embedding when eval_audios is None
            eval_audio_embeds = self.null_audio_embedding.expand(batch_size, -1, -1)
            new_eval_mask = torch.ones(batch_size, eval_audio_embeds.shape[1], dtype=torch.bool, device=device)
        
        result['eval_audio_embeds'] = eval_audio_embeds
        result['eval_audio_mask'] = new_eval_mask
        
        return result
    
    def _apply_joint_transformer(self, prompt_embed, prompt_mask, eval_audio_embeds, eval_audio_mask):
        """Apply joint transformer to get eval audio embedding

        This method encapsulates the common logic for processing prompt and eval audio embeddings
        through the joint transformer, supporting both SA (Self-Attention) and CA (Cross-Attention) modes.

        Args:
            prompt_embed: Prompt embeddings after prompt_transformer [batch, prompt_len, dim]
            prompt_mask: Prompt mask [batch, prompt_len]
            eval_audio_embeds: Evaluation audio embeddings [batch, eval_len, dim]
            eval_audio_mask: Evaluation audio mask [batch, eval_len]
            
        Returns:
            eval_audio_embedding: Pooled eval audio embedding [batch, dim]
        """
        if self.attention_mode == 'SA':
            # Self-Attention: concatenate prompt and eval audio
            joint_embed = torch.cat([prompt_embed, eval_audio_embeds], dim=1)
            joint_mask = torch.cat([prompt_mask, eval_audio_mask], dim=1)
            joint_output = self.joint_transformer(joint_embed, mask=joint_mask)
            
            # Extract eval audio part
            prompt_len = prompt_mask.shape[1]
            eval_audio_output = joint_output[:, prompt_len:, :]
            
            # Masked mean pooling
            eval_audio_mask_expanded = eval_audio_mask.unsqueeze(-1)
            masked_sum = (eval_audio_output * eval_audio_mask_expanded).sum(dim=1)
            valid_count = eval_audio_mask.sum(dim=1, keepdim=True).clamp(min=1)
            eval_audio_embedding = masked_sum / valid_count

        else:  # attention_mode == 'CA'
            # Cross-Attention: eval audio attends to prompt
            decoder_output = eval_audio_embeds
            for decoder_layer in self.joint_transformer:
                decoder_output = decoder_layer(
                    x=decoder_output,
                    encoder_hidden_states=prompt_embed,
                    self_attn_mask=eval_audio_mask,
                    cross_attn_mask_q=eval_audio_mask,
                    cross_attn_mask_kv=prompt_mask
                )
            
            # Masked mean pooling
            eval_audio_mask_expanded = eval_audio_mask.unsqueeze(-1)
            masked_sum = (decoder_output * eval_audio_mask_expanded).sum(dim=1)
            valid_count = eval_audio_mask.sum(dim=1, keepdim=True).clamp(min=1)
            eval_audio_embedding = masked_sum / valid_count
        return eval_audio_embedding
    
    def forward(
        self, 
        texts, 
        lyrics, 
        audio_prompts, 
        audio_prompts_mask, 
        eval_audios, 
        eval_audios_mask, 
        return_embeddings=False,
        # Dropout configuration (can override null_embedding_config)
        text_dropout: Optional[float] = None,
        lyrics_dropout: Optional[float] = None,
        audio_dropout: Optional[float] = None,
    ):
        """Forward pass for reward model with modality dropout support
        
        Args:
            texts: List of text strings
            lyrics: List of lyrics strings
            audio_prompts: Prompt audio waveforms [batch, audio_len] or None
            audio_prompts_mask: Prompt audio mask [batch, seq_len] or None
            eval_audios: Evaluation audio waveforms [batch, audio_len]
            eval_audios_mask: Evaluation audio mask [batch, seq_len]
            return_embeddings: If True, return intermediate embeddings
            text_dropout: Override dropout probability for text
            lyrics_dropout: Override dropout probability for lyrics
            audio_dropout: Override dropout probability for reference audio
            
        Returns:
            scores: Reward scores [batch, output_dim] or dict if return_embeddings=True
        """
        # Get all embeddings through forward_muq_mulan (handles None + dropout automatically)
        mulan_output = self.forward_muq_mulan(
            texts=texts,
            lyrics=lyrics,
            ref_audios=audio_prompts,
            ref_audio_masks=audio_prompts_mask,
            eval_audios=eval_audios,
            eval_audio_masks=eval_audios_mask,
            text_dropout=text_dropout,
            lyrics_dropout=lyrics_dropout,
            ref_audio_dropout=audio_dropout,
        )
        
        text_embeds = mulan_output['text_embeds']
        text_mask = mulan_output['text_mask']
        lyrics_embeds = mulan_output['lyrics_embeds']
        lyrics_mask = mulan_output['lyrics_mask']
        ref_audio_embeds = mulan_output['ref_audio_embeds']
        ref_audio_mask = mulan_output['ref_audio_mask']
        eval_audio_embeds = mulan_output['eval_audio_embeds']
        eval_audio_mask = mulan_output['eval_audio_mask']
        
        # Combine text and lyrics based on mode
        if self.mode == 'concat_text_late':
            combined_text_embeds = torch.cat([text_embeds, lyrics_embeds], dim=1)
            combined_text_mask = torch.cat([text_mask, lyrics_mask], dim=1)
        elif self.mode == 'text_only':
            combined_text_embeds = text_embeds
            combined_text_mask = text_mask
        else:
            combined_text_embeds = torch.cat([text_embeds, lyrics_embeds], dim=1)
            combined_text_mask = torch.cat([text_mask, lyrics_mask], dim=1)
        
        # Combine text and reference audio for prompt embeddings
        combined_prompt_embeds = torch.cat([combined_text_embeds, ref_audio_embeds], dim=1)
        combined_prompt_mask = torch.cat([combined_text_mask, ref_audio_mask], dim=1)
        
        # Apply prompt transformer
        prompt_embed = self.prompt_transformer(combined_prompt_embeds, mask=combined_prompt_mask)
        
        # Apply joint transformer to get eval audio embedding
        eval_audio_embedding = self._apply_joint_transformer(
            prompt_embed, combined_prompt_mask, eval_audio_embeds, eval_audio_mask
        )
        
        # Compute scores
        scores = self.score_projector(eval_audio_embedding)
        single_score = self.single_score_projector(eval_audio_embedding)
        
        if return_embeddings:
            # Compute prompt embedding with masked pooling
            prompt_mask_expanded = combined_prompt_mask.unsqueeze(-1)
            prompt_masked_sum = (prompt_embed * prompt_mask_expanded).sum(dim=1)
            prompt_valid_count = combined_prompt_mask.sum(dim=1, keepdim=True).clamp(min=1)
            prompt_embedding = prompt_masked_sum / prompt_valid_count
            
            return {
                'scores': scores,
                'single_score': single_score,
                'prompt_embedding': prompt_embedding,
                'eval_audio_embedding': eval_audio_embedding,
                # Only return what's needed for hard negative resampling:
                'prompt_embed': prompt_embed,  # prompt transformer output
                'prompt_mask': combined_prompt_mask,  # combined mask
                'eval_audio_embeds_downsampled': eval_audio_embeds,  # downsampled eval audio
                'eval_audio_mask_downsampled': eval_audio_mask,  # eval audio mask
            }
        
        return scores
    
    def _apply_modality_dropout(
        self,
        embeds: torch.Tensor,
        mask: torch.Tensor,
        null_embedding: nn.Parameter,
        dropout_prob: float,
        is_null: Optional[torch.Tensor] = None,
        skip_null: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply dropout to a modality by replacing with null embedding or empty sequence
        
        Args:
            embeds: Input embeddings [batch, seq_len, dim]
            mask: Input mask [batch, seq_len]
            null_embedding: Null embedding to replace with [1, null_len, dim]
            dropout_prob: Probability of dropout (0.0 to 1.0)
            is_null: Optional boolean tensor [batch] indicating which samples have null content
                     If provided, those samples will always use null embedding
            skip_null: If True, dropout samples will have mask set to all zeros (empty sequence)
                      instead of being replaced with null embedding. If None, uses config default.
                     
        Returns:
            Tuple of (output_embeds, output_mask)
            - If dropped out and skip_null=True: mask all zeros (empty sequence)
            - If dropped out and skip_null=False: uses null_embedding
            - Otherwise: uses original embeds
        """
        batch_size = embeds.shape[0]
        device = embeds.device
        
        # Determine skip_null from config if not specified
        if skip_null is None:
            skip_null = self.null_embedding_config.skip_null
        
        # Determine which samples to dropout
        if self.training and dropout_prob > 0: #* already set
            # Random dropout during training
            dropout_mask = torch.rand(batch_size, device=device) < dropout_prob
        else:
            dropout_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Also dropout samples that have null content (if provided)
        if is_null is not None:
            dropout_mask = dropout_mask | is_null.to(device)
        
        if not dropout_mask.any():
            # No dropout needed
            return embeds, mask
        
        # Skip null mode: set mask to zeros for dropped samples (empty sequence)
        if skip_null:
            output_embeds = embeds.clone()
            output_mask = mask.clone()
            output_mask[dropout_mask] = 0  # All masked out -> empty sequence
            return output_embeds, output_mask
        
        # Standard mode: replace with null embedding
        # Expand null embedding to batch, ensuring dtype matches input embeddings
        null_len = null_embedding.shape[1]
        null_embeds_batch = null_embedding.expand(batch_size, -1, -1).to(dtype=embeds.dtype, device=device)  # [batch, null_len, dim]
        null_mask_batch = torch.ones(batch_size, null_len, dtype=mask.dtype, device=device)
        
        # Create output tensors
        # For samples that are dropped out, use null embedding
        # For samples that are not dropped out, use original embedding
        
        # Since embeddings may have different sequence lengths, we need to handle this carefully
        # Option 1: Pad to same length (not ideal for variable lengths)
        # Option 2: Replace entirely (cleaner)
        
        # We'll use option 2: return list-like behavior via masking
        # For simplicity, we return the null embedding for dropped samples
        # and original for non-dropped samples
        
        # Create output with same shape as input (pad or truncate null embedding if needed)
        output_embeds = embeds.clone()
        output_mask = mask.clone()
        
        seq_len = embeds.shape[1]
        
        if null_len >= seq_len:
            # Null embedding is longer or equal, truncate it
            output_embeds[dropout_mask] = null_embeds_batch[dropout_mask, :seq_len, :]
            output_mask[dropout_mask] = null_mask_batch[dropout_mask, :seq_len]
        else:
            # Null embedding is shorter, zero-pad
            output_embeds[dropout_mask, :null_len, :] = null_embeds_batch[dropout_mask]
            output_embeds[dropout_mask, null_len:, :] = 0
            output_mask[dropout_mask, :null_len] = 1
            output_mask[dropout_mask, null_len:] = 0
        
        return output_embeds, output_mask
    

    def forward_from_preextracted(
        self,
        prompt_text_embeds,
        prompt_text_mask,
        prompt_lyrics_embeds,
        prompt_lyrics_mask,
        prompt_audio_embeds,
        prompt_audio_mask,
        eval_audio_embeds,
        eval_audio_mask,
        return_embeddings=False,
        # Dropout configuration (can override null_embedding_config)
        text_dropout: Optional[float] = None,
        lyrics_dropout: Optional[float] = None,
        audio_dropout: Optional[float] = None,
        # Null indicators (for samples that are actually null)
        text_is_null: Optional[torch.Tensor] = None,
        lyrics_is_null: Optional[torch.Tensor] = None,
        audio_is_null: Optional[torch.Tensor] = None,
    ):
        """Forward pass using pre-extracted embeddings (skips MuQ-MuLan encoding)
        
        Args:
            prompt_text_embeds: [batch, text_len, dim]
            prompt_text_mask: [batch, text_len]
            prompt_lyrics_embeds: [batch, lyrics_len, dim]
            prompt_lyrics_mask: [batch, lyrics_len]
            prompt_audio_embeds: [batch, audio_len, dim]
            prompt_audio_mask: [batch, audio_len]
            eval_audio_embeds: [batch, eval_len, dim]
            eval_audio_mask: [batch, eval_len]
            return_embeddings: If True, return intermediate embeddings
            text_dropout: Override dropout probability for text (None = use config)
            lyrics_dropout: Override dropout probability for lyrics (None = use config)
            audio_dropout: Override dropout probability for audio (None = use config)
            text_is_null: Boolean tensor [batch] indicating which samples have null text
            lyrics_is_null: Boolean tensor [batch] indicating which samples have null lyrics
            audio_is_null: Boolean tensor [batch] indicating which samples have null audio
            
        Returns:
            dict with keys: scores, single_score, eval_audio_embedding, prompt_embedding
        """
        # Determine dropout probabilities
        text_p = text_dropout if text_dropout is not None else (
            self.null_embedding_config.text.dropout if self.training else 0.0
        )
        lyrics_p = lyrics_dropout if lyrics_dropout is not None else (
            self.null_embedding_config.lyrics.dropout if self.training else 0.0
        )
        audio_p = audio_dropout if audio_dropout is not None else (
            self.null_embedding_config.audio.dropout if self.training else 0.0
        )
        
        # ====== Auto dtype conversion for compatibility ======
        # Detect model dtype from a representative parameter (lazy initialization)
        if self.model_dtype is None:
            self.model_dtype = next(self.prompt_transformer.parameters()).dtype
        model_dtype = self.model_dtype
        
        # Get batch size and device from eval_audio_embeds (always present)
        batch_size = eval_audio_embeds.shape[0]
        device = eval_audio_embeds.device
        
        # Check skip_null mode for ablation study
        skip_null = self.null_embedding_config.skip_null
        
        # Convert all input embeddings to model dtype if needed
        if prompt_text_embeds is not None and prompt_text_embeds.dtype != model_dtype:
            prompt_text_embeds = prompt_text_embeds.to(model_dtype)
        if prompt_lyrics_embeds is not None and prompt_lyrics_embeds.dtype != model_dtype:
            prompt_lyrics_embeds = prompt_lyrics_embeds.to(model_dtype)
        if eval_audio_embeds.dtype != model_dtype:
            eval_audio_embeds = eval_audio_embeds.to(model_dtype)
        if prompt_audio_embeds is not None and prompt_audio_embeds.dtype != model_dtype:
            prompt_audio_embeds = prompt_audio_embeds.to(model_dtype)
        # ====== End dtype conversion ======
        
        # Handle no_condition mode: when prompt_text_embeds is None
        if prompt_text_embeds is None:
            text_embeds_ds = torch.zeros(batch_size, 0, self.dim, dtype=model_dtype, device=device)
            text_mask_ds = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
        else:
            # Apply downsamplers first (before dropout, so null embedding size matches)
            text_embeds_ds, text_mask_ds = self.text_downsampler(prompt_text_embeds, prompt_text_mask)
            # Apply modality dropout (after downsampling)
            text_embeds_ds, text_mask_ds = self._apply_modality_dropout(
                text_embeds_ds, text_mask_ds,
                self.null_text_embedding,
                dropout_prob=text_p,
                is_null=text_is_null,
            )
        
        # Handle no_condition mode: when prompt_lyrics_embeds is None
        if prompt_lyrics_embeds is None:
            lyrics_embeds_ds = torch.zeros(batch_size, 0, self.dim, dtype=model_dtype, device=device)
            lyrics_mask_ds = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
        else:
            lyrics_embeds_ds, lyrics_mask_ds = self.lyrics_downsampler(prompt_lyrics_embeds, prompt_lyrics_mask)
            lyrics_embeds_ds, lyrics_mask_ds = self._apply_modality_dropout(
                lyrics_embeds_ds, lyrics_mask_ds,
                self.null_lyrics_embedding,
                dropout_prob=lyrics_p,
                is_null=lyrics_is_null,
            )
        
        eval_audio_embeds_ds, eval_audio_mask_ds = self.eval_downsampler(eval_audio_embeds, eval_audio_mask)
        
        # Handle no_condition mode for prompt_audio_embeds
        if prompt_audio_embeds is None:
            ref_audio_embeds_ds = torch.zeros(batch_size, 0, self.dim, dtype=model_dtype, device=device)
            ref_audio_mask_ds = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
        else:
            ref_audio_embeds_ds, ref_audio_mask_ds = self.ref_downsampler(prompt_audio_embeds, prompt_audio_mask)
            ref_audio_embeds_ds, ref_audio_mask_ds = self._apply_modality_dropout(
                ref_audio_embeds_ds, ref_audio_mask_ds,
                self.null_audio_embedding,
                dropout_prob=audio_p,
                is_null=audio_is_null,
            )
        
        # Combine text embeddings based on mode
        if self.mode == 'concat_text_late':
            text_embeds = torch.cat([text_embeds_ds, lyrics_embeds_ds], dim=1)
            text_mask = torch.cat([text_mask_ds, lyrics_mask_ds], dim=1)
        elif self.mode == 'concat_text_early':
            raise ValueError("concat_text_early mode not supported in pre-extracted forward")
        elif self.mode == 'text_only':
            text_embeds = text_embeds_ds
            text_mask = text_mask_ds
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Combine text and reference audio for prompt embeddings
        if ref_audio_embeds_ds is not None and ref_audio_embeds_ds.shape[1] > 0:
            combined_prompt_embeds = torch.cat([text_embeds, ref_audio_embeds_ds], dim=1)
            combined_prompt_mask = torch.cat([text_mask, ref_audio_mask_ds], dim=1)
        else:
            combined_prompt_embeds = text_embeds
            combined_prompt_mask = text_mask
        
        # Apply prompt transformer
        prompt_embed = self.prompt_transformer(combined_prompt_embeds, mask=combined_prompt_mask)
        
        # Apply joint transformer to get eval audio embedding
        eval_audio_embedding = self._apply_joint_transformer(
            prompt_embed, combined_prompt_mask, eval_audio_embeds_ds, eval_audio_mask_ds
        )
        
        # Compute prompt embedding (mean pooling)
        prompt_mask_expanded = combined_prompt_mask.unsqueeze(-1)
        prompt_masked_sum = (prompt_embed * prompt_mask_expanded).sum(dim=1)
        prompt_valid_count = combined_prompt_mask.sum(dim=1, keepdim=True).clamp(min=1)
        prompt_embedding = prompt_masked_sum / prompt_valid_count
        
        scores = self.score_projector(eval_audio_embedding)
        single_score = self.single_score_projector(eval_audio_embedding)
        
        result = {
            'scores': scores,
            'single_score': single_score,
            'eval_audio_embedding': eval_audio_embedding,
            'prompt_embedding': prompt_embedding,
        }
        
        if return_embeddings:
            # Only return what's needed for hard negative resampling:
            result['prompt_embed'] = prompt_embed  # prompt transformer output
            result['prompt_mask'] = combined_prompt_mask  # combined mask
            result['eval_audio_embeds_downsampled'] = eval_audio_embeds_ds  # downsampled eval audio
            result['eval_audio_mask_downsampled'] = eval_audio_mask_ds  # eval audio mask
        
        return result
    
    def forward_raw_text(
        self,
        prompt_texts,
        prompt_lyrics,
        prompt_audio_embeds,
        prompt_audio_mask,
        eval_audio_embeds,
        eval_audio_mask,
        return_embeddings=False,
        # Dropout configuration (can override null_embedding_config)
        text_dropout: Optional[float] = None,
        lyrics_dropout: Optional[float] = None,
        audio_dropout: Optional[float] = None,
    ):
        """Forward pass with raw text + frozen audio embeddings (raw_text_frozen_audio mode)
        
        Text/lyrics are raw strings that need encoding, but audio is pre-extracted.
        
        Args:
            prompt_texts: List of text strings [batch]
            prompt_lyrics: List of lyrics strings [batch]
            prompt_audio_embeds: [batch, audio_len, dim] or None
            prompt_audio_mask: [batch, audio_len] or None
            eval_audio_embeds: [batch, eval_len, dim]
            eval_audio_mask: [batch, eval_len]
            return_embeddings: If True, return intermediate embeddings for reuse
            text_dropout: Override dropout probability for text
            lyrics_dropout: Override dropout probability for lyrics  
            audio_dropout: Override dropout probability for audio
        """
        # Determine dropout probabilities
        text_p = text_dropout if text_dropout is not None else (
            self.null_embedding_config.text.dropout if self.training else 0.0
        )
        lyrics_p = lyrics_dropout if lyrics_dropout is not None else (
            self.null_embedding_config.lyrics.dropout if self.training else 0.0
        )
        audio_p = audio_dropout if audio_dropout is not None else (
            self.null_embedding_config.audio.dropout if self.training else 0.0
        )
        
        # ====== Auto dtype conversion for compatibility ======
        # Detect model dtype from a representative parameter (lazy initialization)
        if self.model_dtype is None:
            self.model_dtype = next(self.prompt_transformer.parameters()).dtype
        model_dtype = self.model_dtype
        
        # Convert eval_audio_embeds to model dtype if needed
        if eval_audio_embeds.dtype != model_dtype:
            eval_audio_embeds = eval_audio_embeds.to(model_dtype)
        
        # Convert prompt_audio_embeds if provided
        if prompt_audio_embeds is not None and prompt_audio_embeds.dtype != model_dtype:
            prompt_audio_embeds = prompt_audio_embeds.to(model_dtype)
        # ====== End dtype conversion ======
        
        # Get batch size from eval_audio_embeds (always present)
        batch_size = eval_audio_embeds.shape[0]
        device = eval_audio_embeds.device
        
        # Check skip_null mode for ablation study
        skip_null = self.null_embedding_config.skip_null
        
        # Handle no_condition mode: when prompt_texts is None
        if prompt_texts is None:
            if skip_null:
                # Skip null mode: use empty sequence (L=0 with mask all zeros)
                text_embeds_ds = torch.zeros(batch_size, 0, self.dim, dtype=model_dtype, device=device)
                text_mask_ds = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
            else:
                # Use null text embedding directly (no text encoder call)
                null_text = self.null_text_embedding.to(device=device, dtype=model_dtype)
                text_embeds_ds = null_text.expand(batch_size, -1, -1)
                text_mask_ds = torch.ones(batch_size, text_embeds_ds.shape[1], dtype=torch.bool, device=device)
        else:
            # Get text embeddings using MuQ-MuLan text encoder
            text_output_text = self.get_text_embedding_separate(prompt_texts, prompt_texts)  # Use texts for both to get text only
            text_embeds = text_output_text['text_embeds']
            text_mask = text_output_text['text_mask']
            # Downsample text
            text_embeds_ds, text_mask_ds = self.text_downsampler(text_embeds, text_mask)
            # Apply dropout to text
            text_embeds_ds, text_mask_ds = self._apply_modality_dropout(
                text_embeds_ds, text_mask_ds,
                self.null_text_embedding,
                dropout_prob=text_p,
            )
        
        # Handle no_condition mode: when prompt_lyrics is None
        if prompt_lyrics is None:
            if skip_null:
                # Skip null mode: use empty sequence (L=0 with mask all zeros)
                lyrics_embeds_ds = torch.zeros(batch_size, 0, self.dim, dtype=model_dtype, device=device)
                lyrics_mask_ds = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
            else:
                # Use null lyrics embedding directly (no text encoder call)
                null_lyrics = self.null_lyrics_embedding.to(device=device, dtype=model_dtype)
                lyrics_embeds_ds = null_lyrics.expand(batch_size, -1, -1)
                lyrics_mask_ds = torch.ones(batch_size, lyrics_embeds_ds.shape[1], dtype=torch.bool, device=device)
        else:
            # Get lyrics embeddings using MuQ-MuLan text encoder
            lyrics_output = self.get_text_embedding_separate(prompt_lyrics, prompt_lyrics)  # Use lyrics for both to get lyrics only
            lyrics_embeds = lyrics_output['text_embeds']  # 'text_embeds' contains the first argument
            lyrics_mask = lyrics_output['text_mask']
            # Downsample lyrics
            lyrics_embeds_ds, lyrics_mask_ds = self.lyrics_downsampler(lyrics_embeds, lyrics_mask)
            # Apply dropout to lyrics
            lyrics_embeds_ds, lyrics_mask_ds = self._apply_modality_dropout(
                lyrics_embeds_ds, lyrics_mask_ds,
                self.null_lyrics_embedding,
                dropout_prob=lyrics_p,
            )
        
        # Combine text and lyrics
        if self.mode == 'concat_text_late':
            combined_text_embeds = torch.cat([text_embeds_ds, lyrics_embeds_ds], dim=1)
            combined_text_mask = torch.cat([text_mask_ds, lyrics_mask_ds], dim=1)
        elif self.mode == 'text_only':
            combined_text_embeds = text_embeds_ds
            combined_text_mask = text_mask_ds
        else:
            combined_text_embeds = torch.cat([text_embeds_ds, lyrics_embeds_ds], dim=1)
            combined_text_mask = torch.cat([text_mask_ds, lyrics_mask_ds], dim=1)
        
        # Downsample prompt audio if exists, otherwise use null embedding or empty
        if prompt_audio_embeds is not None:
            ref_audio_embeds, ref_audio_mask = self.ref_downsampler(prompt_audio_embeds, prompt_audio_mask)
            # Apply dropout to reference audio
            ref_audio_embeds, ref_audio_mask = self._apply_modality_dropout(
                ref_audio_embeds, ref_audio_mask,
                self.null_audio_embedding,
                dropout_prob=audio_p,
            )
        else:
            if skip_null:
                # Skip null mode: use empty sequence (L=0 with mask all zeros)
                ref_audio_embeds = torch.zeros(batch_size, 0, self.dim, dtype=model_dtype, device=device)
                ref_audio_mask = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
            else:
                # Use null embedding when prompt_audio_embeds is None
                null_audio = self.null_audio_embedding.to(device=device, dtype=model_dtype)
                ref_audio_embeds = null_audio.expand(batch_size, -1, -1)
                ref_audio_mask = torch.ones(batch_size, ref_audio_embeds.shape[1], dtype=torch.bool, device=device)

        
        # Downsample eval audio
        eval_audio_embeds_ds, eval_audio_mask_ds = self.eval_downsampler(eval_audio_embeds, eval_audio_mask)
        
        # Get prompt embedding
        prompt_input = torch.cat([combined_text_embeds, ref_audio_embeds], dim=1)
        prompt_mask = torch.cat([combined_text_mask, ref_audio_mask], dim=1)
        
        prompt_embed = self.prompt_transformer(prompt_input, mask=prompt_mask)
        
        # Compute prompt embedding with masked pooling
        prompt_mask_expanded = prompt_mask.unsqueeze(-1)
        prompt_masked_sum = (prompt_embed * prompt_mask_expanded).sum(dim=1)
        prompt_valid_count = prompt_mask.sum(dim=1, keepdim=True).clamp(min=1)
        prompt_embedding = prompt_masked_sum / prompt_valid_count
        
        # Apply joint transformer to get eval audio embedding
        eval_audio_embedding = self._apply_joint_transformer(
            prompt_embed, prompt_mask, eval_audio_embeds_ds, eval_audio_mask_ds
        )
        
        # Compute scores
        scores = self.score_projector(eval_audio_embedding)
        single_score = self.single_score_projector(eval_audio_embedding)
        
        result = {
            'scores': scores,
            'single_score': single_score,
            'prompt_embedding': prompt_embedding,
            'eval_audio_embedding': eval_audio_embedding,
        }
        
        # Return intermediate embeddings if requested (for hard negative resampling)
        if return_embeddings:
            # Only return what's needed for hard negative resampling:
            result['prompt_embed'] = prompt_embed  # prompt transformer output
            result['prompt_mask'] = prompt_mask  # combined mask
            result['eval_audio_embeds_downsampled'] = eval_audio_embeds_ds  # downsampled eval audio
            result['eval_audio_mask_downsampled'] = eval_audio_mask_ds  # eval audio mask
        
        return result
    
    # ============ Backbone + Heads 分离架构 ============
    
    # Head 参数前缀（用于分离存储/加载）
    HEAD_PARAM_PREFIXES = (
        'score_projector.',
        'single_score_projector.',
        'alignment_head.',
        'quality_head.',
    )
    
    def forward_backbone(
        self,
        prompt_texts,
        prompt_lyrics,
        prompt_audio_embeds,
        prompt_audio_mask,
        eval_audio_embeds,
        eval_audio_mask,
        icl_prompt: str = None,  # type: ignore
        text_dropout: Optional[float] = None,
        lyrics_dropout: Optional[float] = None,
        audio_dropout: Optional[float] = None,
    ) -> dict:
        """Backbone forward: 计算 joint embedding，不调用任何 head
        
        基于 forward_raw_text，但不调用 score_projector。
        支持 ICL prompt 注入（在 text 前 prepend）。
        
        Args:
            prompt_texts: List of text strings [batch]
            prompt_lyrics: List of lyrics strings [batch]
            prompt_audio_embeds: [batch, audio_len, dim] or None
            prompt_audio_mask: [batch, audio_len] or None
            eval_audio_embeds: [batch, eval_len, dim]
            eval_audio_mask: [batch, eval_len]
            icl_prompt: Optional ICL prompt to prepend to texts
                       Can be a key from self.icl_prompts(quality, alignment) or a custom string
            text_dropout: Override dropout probability for text
            lyrics_dropout: Override dropout probability for lyrics
            audio_dropout: Override dropout probability for audio
            
        Returns:
            dict with keys:
                - eval_audio_embedding: [B, dim] 用于 head forward
                - prompt_embedding: [B, dim] 聚合后的 prompt embedding
                - prompt_embed: [B, seq, dim] transformer 输出
                - prompt_mask: [B, seq] mask
        """
        # Handle ICL prompt injection
        if icl_prompt is not None:
            # If icl_prompt is a key (e.g., 'quality', 'alignment'), look up and sample
            if icl_prompt in self.icl_prompts:
                import random
                prompts_list = self.icl_prompts[icl_prompt]
                # Randomly select one prompt from the list
                icl_text = random.choice(prompts_list)
            else:
                # Assume it's a direct prompt string
                icl_text = icl_prompt
            # Prepend ICL prompt to each text
            prompt_texts = [f"{icl_text} {text}" for text in prompt_texts]
        
        # Determine dropout probabilities
        text_p = text_dropout if text_dropout is not None else (
            self.null_embedding_config.text.dropout if self.training else 0.0
        )
        lyrics_p = lyrics_dropout if lyrics_dropout is not None else (
            self.null_embedding_config.lyrics.dropout if self.training else 0.0
        )
        audio_p = audio_dropout if audio_dropout is not None else (
            self.null_embedding_config.audio.dropout if self.training else 0.0
        )
        
        # Get text embeddings using MuQ-MuLan text encoder
        text_output = self.get_text_embedding_separate(prompt_texts, prompt_lyrics)
        text_embeds = text_output['text_embeds']
        text_mask = text_output['text_mask']
        lyrics_embeds = text_output['lyrics_embeds']
        lyrics_mask = text_output['lyrics_mask']
        
        # Downsample text and lyrics separately
        text_embeds_ds, text_mask_ds = self.text_downsampler(text_embeds, text_mask)
        lyrics_embeds_ds, lyrics_mask_ds = self.lyrics_downsampler(lyrics_embeds, lyrics_mask)
        
        # Apply dropout to text and lyrics
        text_embeds_ds, text_mask_ds = self._apply_modality_dropout(
            text_embeds_ds, text_mask_ds,
            self.null_text_embedding,
            dropout_prob=text_p,
        )
        
        lyrics_embeds_ds, lyrics_mask_ds = self._apply_modality_dropout(
            lyrics_embeds_ds, lyrics_mask_ds,
            self.null_lyrics_embedding,
            dropout_prob=lyrics_p,
        )
        
        # Combine text and lyrics
        if self.mode == 'concat_text_late':
            combined_text_embeds = torch.cat([text_embeds_ds, lyrics_embeds_ds], dim=1)
            combined_text_mask = torch.cat([text_mask_ds, lyrics_mask_ds], dim=1)
        elif self.mode == 'text_only':
            combined_text_embeds = text_embeds_ds
            combined_text_mask = text_mask_ds
        else:
            combined_text_embeds = torch.cat([text_embeds_ds, lyrics_embeds_ds], dim=1)
            combined_text_mask = torch.cat([text_mask_ds, lyrics_mask_ds], dim=1)
        
        # Handle prompt audio
        if prompt_audio_embeds is not None:
            ref_audio_embeds, ref_audio_mask = self.ref_downsampler(prompt_audio_embeds, prompt_audio_mask)
            ref_audio_embeds, ref_audio_mask = self._apply_modality_dropout(
                ref_audio_embeds, ref_audio_mask,
                self.null_audio_embedding,
                dropout_prob=audio_p,
            )
        else:
            # Create empty ref audio
            batch_size = combined_text_embeds.shape[0]
            device = combined_text_embeds.device
            ref_audio_embeds = self.null_audio_embedding.expand(batch_size, -1, -1)
            ref_audio_mask = torch.ones(batch_size, ref_audio_embeds.shape[1], dtype=torch.bool, device=device)
        
        # ====== Auto dtype conversion for compatibility ======
        # Detect model dtype from a representative parameter (lazy initialization)
        if self.model_dtype is None:
            self.model_dtype = next(self.prompt_transformer.parameters()).dtype
        model_dtype = self.model_dtype
        
        # Convert all input embeddings to model dtype if needed
        if eval_audio_embeds.dtype != model_dtype:
            eval_audio_embeds = eval_audio_embeds.to(model_dtype)
        if prompt_audio_embeds is not None and prompt_audio_embeds.dtype != model_dtype:
            prompt_audio_embeds = prompt_audio_embeds.to(model_dtype)
        # ====== End dtype conversion ======
        
        # Downsample eval audio
        eval_audio_embeds_ds, eval_audio_mask_ds = self.eval_downsampler(eval_audio_embeds, eval_audio_mask)
        
        # Combine prompt embeddings
        prompt_input = torch.cat([combined_text_embeds, ref_audio_embeds], dim=1)
        prompt_mask = torch.cat([combined_text_mask, ref_audio_mask], dim=1)
        
        # Apply prompt transformer
        prompt_embed = self.prompt_transformer(prompt_input, mask=prompt_mask)
        
        # Compute prompt embedding with masked pooling
        prompt_mask_expanded = prompt_mask.unsqueeze(-1)
        prompt_masked_sum = (prompt_embed * prompt_mask_expanded).sum(dim=1)
        prompt_valid_count = prompt_mask.sum(dim=1, keepdim=True).clamp(min=1)
        prompt_embedding = prompt_masked_sum / prompt_valid_count
        
        # Apply joint transformer to get eval audio embedding
        eval_audio_embedding = self._apply_joint_transformer(
            prompt_embed, prompt_mask, eval_audio_embeds_ds, eval_audio_mask_ds
        )
        
        return {
            'eval_audio_embedding': eval_audio_embedding,
            'prompt_embedding': prompt_embedding,
            'prompt_embed': prompt_embed,
            'prompt_mask': prompt_mask,
            'eval_audio_embeds_downsampled': eval_audio_embeds_ds,
            'eval_audio_mask_downsampled': eval_audio_mask_ds,
        }
    
    def forward_alignment(
        self,
        prompt_texts,
        prompt_lyrics,
        prompt_audio_embeds,
        prompt_audio_mask,
        eval_audio_embeds,
        eval_audio_mask,
        use_icl: bool = False,
        icl_prompt: str = None,  # type: ignore
        **kwargs,
    ) -> dict:
        """Forward for alignment/Instruction Following task
        
        Args:
            use_icl: If True, use ICL mode with alignment prompt
            icl_prompt: Override ICL prompt key (default: 'alignment') or direct prompt string
            **kwargs: Passed to forward_backbone
            
        Returns:
            dict with 'score' [B, 1] and backbone outputs
        """
        # Determine ICL prompt
        if use_icl:
            icl = icl_prompt or 'alignment'  # Use 'alignment' task prompts by default
        else:
            icl = None
        
        # Get backbone output
        backbone_out = self.forward_backbone(
            prompt_texts=prompt_texts,
            prompt_lyrics=prompt_lyrics,
            prompt_audio_embeds=prompt_audio_embeds,
            prompt_audio_mask=prompt_audio_mask,
            eval_audio_embeds=eval_audio_embeds,
            eval_audio_mask=eval_audio_mask,
            icl_prompt=icl,
            **kwargs,
        )
        
        # Apply alignment head
        embedding = backbone_out['eval_audio_embedding']
        score = self.alignment_head(embedding)
        
        return {
            'score': score,
            'eval_audio_embedding': embedding,
            'prompt_embedding': backbone_out['prompt_embedding'],
            'task': 'alignment',
        }
    
    def forward_quality(
        self,
        prompt_texts,
        prompt_lyrics,
        prompt_audio_embeds,
        prompt_audio_mask,
        eval_audio_embeds,
        eval_audio_mask,
        use_icl: bool = False,
        icl_prompt: str = None,  # type: ignore
        use_alignment_head: bool = False,
        **kwargs,
    ) -> dict:
        """Forward for music quality task
        
        Args:
            use_icl: If True, use ICL mode with quality prompt
            icl_prompt: Override ICL prompt (default: 'quality_high')
            use_alignment_head: If True (with ICL), use alignment_head instead of quality_head
            **kwargs: Passed to forward_backbone
            
        Returns:
            dict with 'score' [B, 1] and backbone outputs
        """
        # Determine ICL prompt
        if use_icl:
            icl = icl_prompt or 'quality'  # Use 'quality' task prompts by default
        else:
            icl = None
        
        # Get backbone output
        backbone_out = self.forward_backbone(
            prompt_texts=prompt_texts,
            prompt_lyrics=prompt_lyrics,
            prompt_audio_embeds=prompt_audio_embeds,
            prompt_audio_mask=prompt_audio_mask,
            eval_audio_embeds=eval_audio_embeds,
            eval_audio_mask=eval_audio_mask,
            icl_prompt=icl,
            **kwargs,
        )
        
        # Apply head
        embedding = backbone_out['eval_audio_embedding']
        
        # ICL mode can optionally use alignment_head
        if use_icl and use_alignment_head:
            score = self.alignment_head(embedding)
            head_used = 'alignment_head (ICL)'
        else:
            score = self.quality_head(embedding)
            head_used = 'quality_head'
        
        return {
            'score': score,
            'eval_audio_embedding': embedding,
            'prompt_embedding': backbone_out['prompt_embedding'],
            'task': 'quality',
            'head_used': head_used,
        }
    
    def forward_for_task(
        self,
        task: str,
        prompt_texts,
        prompt_lyrics,
        prompt_audio_embeds,
        prompt_audio_mask,
        eval_audio_embeds,
        eval_audio_mask,
        **kwargs,
    ) -> dict:
        """Unified task-specific forward
        
        Args:
            task: 'IF' | 'MQ' | 'alignment' | 'quality'
            **kwargs: Passed to task-specific forward
            
        Returns:
            dict with 'score' and other outputs
        """
        task_map = {
            'IF': self.forward_alignment,
            'alignment': self.forward_alignment,
            'MQ': self.forward_quality,
            'quality': self.forward_quality,
        }
        
        forward_fn = task_map.get(task)
        if forward_fn is None:
            raise ValueError(f"Unknown task: {task}. Available: {list(task_map.keys())}")
        
        return forward_fn(
            prompt_texts=prompt_texts,
            prompt_lyrics=prompt_lyrics,
            prompt_audio_embeds=prompt_audio_embeds,
            prompt_audio_mask=prompt_audio_mask,
            eval_audio_embeds=eval_audio_embeds,
            eval_audio_mask=eval_audio_mask,
            **kwargs,
        )
    
    # ============ Backbone/Heads 分离存储与加载 ============
    
    def get_backbone_state_dict(self) -> dict:
        """获取 backbone 参数（不包含 heads）"""
        return {
            k: v for k, v in self.state_dict().items()
            if not any(k.startswith(prefix) for prefix in self.HEAD_PARAM_PREFIXES)
        }
    

    def forward_comparison(
        self, 
        texts, 
        lyrics, 
        audio_prompts, 
        audio_prompts_mask, 
        audio_as, 
        audio_a_masks, 
        audio_bs, 
        audio_b_masks
    ):
        """Forward pass for pairwise comparison
        
        Computes preference logits for two audio samples (a and b) given the same prompt.
        Used for preference prediction training with win/loss labels.
        
        Args:
            texts: List of text strings [batch]
            lyrics: List of lyrics strings [batch]
            audio_prompts: Prompt audio waveforms [batch, audio_len]
            audio_prompts_mask: Prompt audio mask [batch, seq_len]
            audio_as: Audio A waveforms [batch, audio_len]
            audio_a_masks: Audio A mask [batch, seq_len]
            audio_bs: Audio B waveforms [batch, audio_len]
            audio_b_masks: Audio B mask [batch, seq_len]
            
        Returns:
            logits: Preference logits [batch, 2, 2]
                   - dim 1: (instruction_following, music_quality)
                   - dim 2: (audio_a_win, audio_b_win)
        """
        # Get scores for audio_a
        scores_a = self.forward(
            texts=texts,
            lyrics=lyrics,
            audio_prompts=audio_prompts,
            audio_prompts_mask=audio_prompts_mask,
            eval_audios=audio_as,
            eval_audios_mask=audio_a_masks
        )  # [batch, output_dim=2]
        
        # Get scores for audio_b
        scores_b = self.forward(
            texts=texts,
            lyrics=lyrics,
            audio_prompts=audio_prompts,
            audio_prompts_mask=audio_prompts_mask,
            eval_audios=audio_bs,
            eval_audios_mask=audio_b_masks
        )  # [batch, output_dim=2]
        
        # Stack to create logits: [batch, 2, 2]
        # For each dimension (instruction_following, music_quality):
        #   logits[:, dim, 0] = score_a (audio_a wins)
        #   logits[:, dim, 1] = score_b (audio_b wins)
        logits = torch.stack([scores_a, scores_b], dim=2)  # [batch, 2, 2]
        
        return logits
    
    def forward_comparison_from_preextracted(
        self,
        prompt_text_embeds,
        prompt_text_mask,
        prompt_lyrics_embeds,
        prompt_lyrics_mask,
        prompt_audio_embeds,
        prompt_audio_mask,
        gen_a_audio_embeds,
        gen_a_audio_mask,
        gen_b_audio_embeds,
        gen_b_audio_mask
    ):
        """Forward pass for pairwise comparison using pre-extracted embeddings
        
        Args:
            prompt_text_embeds: [batch, text_len, dim]
            prompt_text_mask: [batch, text_len]
            prompt_lyrics_embeds: [batch, lyrics_len, dim]
            prompt_lyrics_mask: [batch, lyrics_len]
            prompt_audio_embeds: [batch, audio_len, dim]
            prompt_audio_mask: [batch, audio_len]
            gen_a_audio_embeds: [batch, gen_a_len, dim]
            gen_a_audio_mask: [batch, gen_a_len]
            gen_b_audio_embeds: [batch, gen_b_len, dim]
            gen_b_audio_mask: [batch, gen_b_len]
            
        Returns:
            logits: Preference logits [batch, 2, 2]
        """
        # Get scores for generation A
        scores_a = self.forward_from_preextracted(
            prompt_text_embeds=prompt_text_embeds,
            prompt_text_mask=prompt_text_mask,
            prompt_lyrics_embeds=prompt_lyrics_embeds,
            prompt_lyrics_mask=prompt_lyrics_mask,
            prompt_audio_embeds=prompt_audio_embeds,
            prompt_audio_mask=prompt_audio_mask,
            eval_audio_embeds=gen_a_audio_embeds,
            eval_audio_mask=gen_a_audio_mask
        ).get('scores')  # [batch, output_dim=2]
        
        # Get scores for generation B
        scores_b = self.forward_from_preextracted(
            prompt_text_embeds=prompt_text_embeds,
            prompt_text_mask=prompt_text_mask,
            prompt_lyrics_embeds=prompt_lyrics_embeds,
            prompt_lyrics_mask=prompt_lyrics_mask,
            prompt_audio_embeds=prompt_audio_embeds,
            prompt_audio_mask=prompt_audio_mask,
            eval_audio_embeds=gen_b_audio_embeds,
            eval_audio_mask=gen_b_audio_mask
        ).get('scores')  # [batch, output_dim=2]
        
        # Stack to create logits: [batch, 2, 2]
        logits = torch.stack([scores_a, scores_b], dim=2)
        
        return logits


class SanityCheckModel(nn.Module):
    """Sanity check model with minimal learnable parameters
    
    Uses simple mean pooling + linear layers for instruction following and quality scoring.
    Keeps the same interface as RewardAttentionModel for drop-in replacement.
    """
    def __init__(
        self,
        model_name='OpenMuQ/MuQ-MuLan-large',
        dim=768,
        mlp_dim=512,
        output_dim=2,
        sr=24000,
        freeze_audio=True,
        freeze_text=True,
        cfg=None,
        # Transformer config for eval audio self-attention
        eval_tf_depth=2,
        dim_head=64,
        heads=8,
        attn_dropout=0.,
        ff_dropout=0.,
        ff_mult=4,
        **kwargs  # Accept and ignore other args for compatibility
    ):
        super().__init__()
        self.dim = dim
        self.mlp_dim = mlp_dim
        self.output_dim = output_dim
        self.config = cfg
        self.freeze_audio = freeze_audio
        self.freeze_text = freeze_text
        self.gradient_checkpointing = False
        
        # Learnable layers
        self.linearLayer = nn.Linear(2 * self.dim, self.dim)
        self.scorelayer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, 1)
        )
        
        # Transformer for eval audio self-attention (used in forward_raw_text_frozen_audio)
        self.eval_tf_depth = eval_tf_depth
        if eval_tf_depth > 0:
            self.eval_transformer = Transformer(
                dim=dim,
                depth=eval_tf_depth,
                dim_head=dim_head,
                heads=heads,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                ff_mult=ff_mult,
                use_flash_attn=True
            )
        else:
            self.eval_transformer = None
        
        # Initialize MuQ-MuLan modules (same as RewardAttentionModel)
        self.init_mulan_modules(model_name)
        
        logger.info(f"Created SanityCheckModel with dim={dim}, mlp_dim={mlp_dim}, eval_tf_depth={eval_tf_depth}")
    
    def init_mulan_modules(self, name):
        """Initialize frozen MuQ-MuLan encoders"""
        from muq import MuQMuLan
        mulan = MuQMuLan.from_pretrained(name)
        self.text_module = mulan.mulan_module.text 
        self.audio_module = mulan.mulan_module.audio
        
        if self.freeze_text:
            frozen_params(self.text_module)
        frozen_params(self.audio_module)
        
        logger.info("Initialized frozen MuQ-MuLan modules")
    
    def gradient_checkpointing_enable(self):
        """Compatibility method - does nothing for this simple model"""
        self.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Compatibility method - does nothing for this simple model"""
        self.gradient_checkpointing = False
    
    def mean_pooling(self, embeds, mask):
        """Mean pooling with masking"""
        mask_expanded = mask.unsqueeze(-1)  # [B, seq_len, 1]
        masked_sum = (embeds * mask_expanded).sum(dim=1)  # [B, dim]
        valid_count = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        return masked_sum / valid_count  # [B, dim]
    
    # def forward_from_preextracted(
    #     self,
    #     prompt_text_embeds,
    #     prompt_text_mask,
    #     prompt_lyrics_embeds,
    #     prompt_lyrics_mask,
    #     prompt_audio_embeds,
    #     prompt_audio_mask,
    #     eval_audio_embeds,
    #     eval_audio_mask
    # ):
    #     """Forward pass using pre-extracted embeddings
        
    #     Returns same format as RewardAttentionModel for compatibility
    #     """
    #     # Mean pool eval audio
    #     eval_audio_pooled = self.mean_pooling(eval_audio_embeds, eval_audio_mask)  # [B, dim]
        
    #     # Mean pool prompt text
    #     text_pooled = self.mean_pooling(prompt_text_embeds, prompt_text_mask)  # [B, dim]
        
    #     # Mean pool lyrics (or use zeros if None)
    #     if prompt_lyrics_embeds is not None:
    #         lyrics_pooled = self.mean_pooling(prompt_lyrics_embeds, prompt_lyrics_mask)
    #     else:
    #         lyrics_pooled = torch.zeros_like(text_pooled)
        
    #     # Combine text + lyrics and project to dim
    #     prompt_embeds = torch.cat([text_pooled, lyrics_pooled], dim=-1)  # [B, 2*dim]
    #     prompt_embeds = self.linearLayer(prompt_embeds)  # [B, dim]
        
    #     # Instruction following score (cosine similarity with epsilon to avoid NaN)
    #     eps = 1e-8
    #     instruction_score = (prompt_embeds * eval_audio_pooled).sum(dim=-1, keepdim=True) / \
    #                        ((prompt_embeds.norm(dim=-1, keepdim=True) + eps) * (eval_audio_pooled.norm(dim=-1, keepdim=True) + eps))  # [B, 1]
        
    #     # Music quality score (MLP on eval audio)
    #     quality_score = self.scorelayer(eval_audio_pooled)  # [B, 1]
        
    #     # Combine scores
    #     scores = torch.cat([instruction_score, quality_score], dim=-1)  # [B, 2]
        
    #     return {
    #         'scores': scores,
    #         'single_score': quality_score,
    #         'eval_audio_embedding': eval_audio_pooled,
    #         'prompt_embedding': prompt_embeds
    #     }
    
    # def forward_comparison_from_preextracted(
    #     self,
    #     prompt_text_embeds,
    #     prompt_text_mask,
    #     prompt_lyrics_embeds,
    #     prompt_lyrics_mask,
    #     prompt_audio_embeds,
    #     prompt_audio_mask,
    #     gen_a_audio_embeds,
    #     gen_a_audio_mask,
    #     gen_b_audio_embeds,
    #     gen_b_audio_mask
    # ):
    #     """Forward pass for pairwise comparison using pre-extracted embeddings"""
    #     # Get scores for generation A
    #     scores_a = self.forward_from_preextracted(
    #         prompt_text_embeds=prompt_text_embeds,
    #         prompt_text_mask=prompt_text_mask,
    #         prompt_lyrics_embeds=prompt_lyrics_embeds,
    #         prompt_lyrics_mask=prompt_lyrics_mask,
    #         prompt_audio_embeds=prompt_audio_embeds,
    #         prompt_audio_mask=prompt_audio_mask,
    #         eval_audio_embeds=gen_a_audio_embeds,
    #         eval_audio_mask=gen_a_audio_mask
    #     )['scores']  # [batch, 2]
        
    #     # Get scores for generation B
    #     scores_b = self.forward_from_preextracted(
    #         prompt_text_embeds=prompt_text_embeds,
    #         prompt_text_mask=prompt_text_mask,
    #         prompt_lyrics_embeds=prompt_lyrics_embeds,
    #         prompt_lyrics_mask=prompt_lyrics_mask,
    #         prompt_audio_embeds=prompt_audio_embeds,
    #         prompt_audio_mask=prompt_audio_mask,
    #         eval_audio_embeds=gen_b_audio_embeds,
    #         eval_audio_mask=gen_b_audio_mask
    #     )['scores']  # [batch, 2]
        
    #     # Stack to create logits: [batch, 2, 2]
    #     logits = torch.stack([scores_a, scores_b], dim=2)
        
    #     return logits
    
    def forward(self, texts, lyrics, audio_prompts, audio_prompts_mask, eval_audios, eval_audios_mask):
        """Forward pass with raw audio (for compatibility - not optimized)"""
        raise NotImplementedError(
            "SanityCheckModel only supports forward_from_preextracted. "
            "Please use pre-extracted embeddings (use_preextracted=True in config)."
        )
    
    def forward_comparison(self, texts, lyrics, audio_prompts, audio_prompts_mask, audio_as, audio_a_masks, audio_bs, audio_b_masks):
        """Forward comparison with raw audio (for compatibility - not optimized)"""
        raise NotImplementedError(
            "SanityCheckModel only supports forward_comparison_from_preextracted. "
            "Please use pre-extracted embeddings (use_preextracted=True in config)."
        )
    
    def forward_raw_text(
        self,
        prompt_texts,
        prompt_lyrics,
        prompt_audio_embeds,
        prompt_audio_mask,
        eval_audio_embeds,
        eval_audio_mask,
        return_embeddings=False,
    ):
        """Forward pass for raw text + frozen audio embeddings mode
        
        This method IGNORES all prompt information (text, lyrics, audio prompt)
        and only processes eval_audio_embeds through self-attention transformer.
        
        Args:
            prompt_texts: List of text strings [batch] - IGNORED
            prompt_lyrics: List of lyrics strings [batch] - IGNORED
            prompt_audio_embeds: [batch, audio_len, dim] or None - IGNORED
            prompt_audio_mask: [batch, audio_len] or None - IGNORED
            eval_audio_embeds: [batch, eval_len, dim]
            eval_audio_mask: [batch, eval_len]
            return_embeddings: Whether to return intermediate embeddings
            
        Returns:
            dict with keys:
                - 'scores': [batch, 2] (instruction_following=0, quality_score)
                - 'single_score': [batch, 1] quality score
                - 'eval_audio_embedding': [batch, dim] pooled eval audio
        """
        # Apply self-attention transformer to eval audio if configured
        if self.eval_transformer is not None:
            eval_embeds = self.eval_transformer(eval_audio_embeds, mask=eval_audio_mask)
        else:
            eval_embeds = eval_audio_embeds
        
        # Mean pooling
        eval_audio_pooled = self.mean_pooling(eval_embeds, eval_audio_mask)  # [B, dim]
        
        # Quality score (MLP on eval audio)
        quality_score = self.scorelayer(eval_audio_pooled)  # [B, 1]
        
        # Instruction following score = 0 (no prompt info used)
        instruction_score = torch.zeros_like(quality_score)  # [B, 1]
        
        # Combine scores
        scores = torch.cat([instruction_score, quality_score], dim=-1)  # [B, 2]
        
        result = {
            'scores': scores,
            'single_score': quality_score,
            'eval_audio_embedding': eval_audio_pooled,
        }
        
        return result
    
    def forward_comparison_raw_text(
        self,
        prompt_texts,
        prompt_lyrics,
        prompt_audio_embeds,
        prompt_audio_mask,
        gen_a_audio_embeds,
        gen_a_audio_mask,
        gen_b_audio_embeds,
        gen_b_audio_mask,
    ):
        """Forward comparison for raw text + frozen audio embeddings mode
        
        This method IGNORES all prompt information and compares two audio generations
        using only their eval audio embeddings processed through self-attention.
        
        Args:
            prompt_texts: List of text strings [batch] - IGNORED
            prompt_lyrics: List of lyrics strings [batch] - IGNORED
            prompt_audio_embeds: [batch, audio_len, dim] or None - IGNORED
            prompt_audio_mask: [batch, audio_len] or None - IGNORED
            gen_a_audio_embeds: [batch, gen_a_len, dim]
            gen_a_audio_mask: [batch, gen_a_len]
            gen_b_audio_embeds: [batch, gen_b_len, dim]
            gen_b_audio_mask: [batch, gen_b_len]
            
        Returns:
            logits: [batch, 2, 2] preference logits
        """
        # Get scores for generation A
        scores_a = self.forward_raw_text_frozen_audio(
            prompt_texts=prompt_texts,
            prompt_lyrics=prompt_lyrics,
            prompt_audio_embeds=prompt_audio_embeds,
            prompt_audio_mask=prompt_audio_mask,
            eval_audio_embeds=gen_a_audio_embeds,
            eval_audio_mask=gen_a_audio_mask,
        )['scores']  # [batch, 2]
        
        # Get scores for generation B
        scores_b = self.forward_raw_text_frozen_audio(
            prompt_texts=prompt_texts,
            prompt_lyrics=prompt_lyrics,
            prompt_audio_embeds=prompt_audio_embeds,
            prompt_audio_mask=prompt_audio_mask,
            eval_audio_embeds=gen_b_audio_embeds,
            eval_audio_mask=gen_b_audio_mask,
        )['scores']  # [batch, 2]
        
        # Stack to create logits: [batch, 2, 2]
        logits = torch.stack([scores_a, scores_b], dim=2)
        
        return logits
