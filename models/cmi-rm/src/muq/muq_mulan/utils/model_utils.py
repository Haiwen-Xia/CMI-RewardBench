"""Model Configuration and Loading Utilities

This module provides:
1. ModelConfig: Configuration class for model initialization
2. load_config: Load configuration from YAML file
3. load_model: Load model weights with support for frozen/non-frozen parts

Loading Strategy:
- If checkpoint is None: Load all from pretrained (from_pretrained)
- If checkpoint has 'frozen_modules' key: Load frozen from pretrained, trainable from checkpoint
- Otherwise: Load all from checkpoint (for fully saved models)
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union

import torch
import yaml
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@dataclass
class DropoutConfig:
    """Configuration for modality dropout during training"""
    dropout: float = 0.0
    length: int = 10  # Length of null embedding sequence
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DropoutConfig":
        return cls(
            dropout=d.get('dropout', 0.0),
            length=d.get('length', 10)
        )


@dataclass
class NullEmbeddingConfig:
    """Configuration for null embeddings (used when modality is dropped out)
    
    When skip_null=True (ablation mode), dropped modalities produce empty sequences
    instead of null embeddings. This requires the model to handle L=0 gracefully.
    """
    text: DropoutConfig = field(default_factory=lambda: DropoutConfig(dropout=0.2, length=10))
    lyrics: DropoutConfig = field(default_factory=lambda: DropoutConfig(dropout=0.3, length=10))
    audio: DropoutConfig = field(default_factory=lambda: DropoutConfig(dropout=0.5, length=10))
    skip_null: bool = False  # Ablation: skip null embeddings, use empty sequence instead
    
    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "NullEmbeddingConfig":
        if d is None:
            return cls()
        return cls(
            text=DropoutConfig.from_dict(d.get('text', {})),
            lyrics=DropoutConfig.from_dict(d.get('lyrics', {})),
            audio=DropoutConfig.from_dict(d.get('audio', {})),
            skip_null=d.get('skip_null', False),
        )


@dataclass
class LoadConfig:
    """Configuration for loading model weights
    
    Attributes:
        checkpoint_path: Path to checkpoint file (.pt)
        frozen_from_pretrained: Whether to load frozen modules from pretrained
        pretrained_name: Name of pretrained model (e.g., 'OpenMuQ/MuQ-MuLan-large')
    """
    checkpoint_path: Optional[str] = None
    frozen_from_pretrained: bool = True  # Load frozen parts from from_pretrained
    pretrained_name: str = "OpenMuQ/MuQ-MuLan-large"
    strict: bool = False  # Whether to use strict mode when loading state_dict
    
    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "LoadConfig":
        if d is None:
            return cls()
        return cls(
            checkpoint_path=d.get('checkpoint_path', None),
            frozen_from_pretrained=d.get('frozen_from_pretrained', True),
            pretrained_name=d.get('pretrained_name', 'OpenMuQ/MuQ-MuLan-large'),
            strict=d.get('strict', False)
        )


@dataclass
class TextEncoderConfig:
    """Configuration for text encoder
    
    Attributes:
        name: Encoder name ('muq_mulan' or 't5')
        tune: What to tune (None = frozen, 'transformer' = tune transformer layers)
    """
    name: str = "muq_mulan"
    tune: Optional[str] = None  # None, 'transformer', 'all'
    
    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> Optional["TextEncoderConfig"]:
        if d is None:
            return None
        # Handle both old string format and new dict format
        if isinstance(d, str):
            return cls(name=d, tune=None)
        return cls(
            name=d.get('name', 'muq_mulan'),
            tune=d.get('tune', None)
        )


@dataclass
class ModelConfig:
    """Configuration for RewardAttentionModel
    
    Reads from YAML config and provides default values for all model parameters.
    """
    # Model architecture
    name: str = "reward"  # 'sanity' or 'reward'
    model_name: str = "OpenMuQ/MuQ-MuLan-large"
    dim: int = 768
    mode: str = "concat_text_late"  # 'concat_text_late', 'concat_text_early', 'text_only'
    attention_mode: str = "SA"  # 'SA' or 'CA'
    mlp_dim: int = 512
    output_dim: int = 2
    sr: int = 24000
    
    # Transformer depths
    prompt_tf_depth: int = 1
    joint_tf_depth: int = 4
    
    # Attention parameters
    dim_head: int = 64
    heads: int = 8
    attn_dropout: float = 0.0
    ff_dropout: float = 0.0
    ff_mult: int = 4
    
    # Layer configuration
    use_layer_idx: int = -1
    
    # Freezing configuration
    freeze_audio: bool = True
    freeze_text: bool = True
    # train_muqmulan: DEPRECATED - removed, use text_encoder_config.tune and train_muq_depth instead
    train_muq_depth: int = 0
    
    # Text encoder configuration (new format)
    # Can be: string 'muqmulan' (old) or dict {name: 'muq_mulan', tune: 'transformer'} (new)
    text_encoder: str = "muqmulan"  # Deprecated: use text_encoder_config
    text_encoder_config: Optional[Dict[str, Any]] = None  # {name: 'muq_mulan', tune: 'transformer'}
    
    # LoRA configuration
    text_lora_config: Optional[Dict[str, Any]] = None
    
    # Loading configuration
    load_config: LoadConfig = field(default_factory=LoadConfig)
    
    # Null embedding / dropout configuration
    null_embedding_config: NullEmbeddingConfig = field(default_factory=NullEmbeddingConfig)
    
    # Category embedding configuration (for future use)
    category_embeddings: Optional[Dict[str, Dict[str, Any]]] = None
    
    # Downsample configuration (kept as dict for flexibility)
    downsample: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        """Create ModelConfig from dictionary"""
        # Handle nested configs
        load_config = LoadConfig.from_dict(d.get('load_config', None))
        null_embedding_config = NullEmbeddingConfig.from_dict(d.get('null_embedding', None))
        
        # Handle text_encoder config (can be string or dict)
        text_encoder_raw = d.get('text_encoder', 'muqmulan')
        if isinstance(text_encoder_raw, dict):
            text_encoder_config = text_encoder_raw
            text_encoder = text_encoder_raw.get('name', 'muqmulan')
        else:
            text_encoder_config = None
            text_encoder = text_encoder_raw
        
        return cls(
            name=d.get('name', 'reward'),
            model_name=d.get('model_name', 'OpenMuQ/MuQ-MuLan-large'),
            dim=d.get('dim', 768),
            mode=d.get('mode', 'concat_text_late'),
            attention_mode=d.get('attention_mode', 'SA'),
            mlp_dim=d.get('mlp_dim', 512),
            output_dim=d.get('output_dim', 2),
            sr=d.get('sr', 24000),
            prompt_tf_depth=d.get('prompt_tf_depth', 1),
            joint_tf_depth=d.get('joint_tf_depth', 4),
            dim_head=d.get('dim_head', 64),
            heads=d.get('heads', 8),
            attn_dropout=d.get('attn_dropout', 0.0),
            ff_dropout=d.get('ff_dropout', 0.0),
            ff_mult=d.get('ff_mult', 4),
            use_layer_idx=d.get('use_layer_idx', -1),
            freeze_audio=d.get('freeze_audio', True),
            freeze_text=d.get('freeze_text', True),
            # train_muqmulan removed - ignored for backward compatibility
            train_muq_depth=d.get('train_muq_depth', 0),
            text_encoder=text_encoder,
            text_encoder_config=text_encoder_config,
            text_lora_config=d.get('text_lora_config', None),
            load_config=load_config,
            null_embedding_config=null_embedding_config,
            category_embeddings=d.get('category_embeddings', None),
            downsample=d.get('downsample', None),
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ModelConfig":
        """Create ModelConfig from YAML file"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Handle both flat config and nested 'model' key
        if 'model' in config:
            return cls.from_dict(config['model'])
        return cls.from_dict(config)
    
    @classmethod
    def from_omegaconf(cls, cfg: DictConfig) -> "ModelConfig":
        """Create ModelConfig from OmegaConf DictConfig"""
        d = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(d, dict):
            if 'model' in d:
                return cls.from_dict(d['model'])
            return cls.from_dict(d)
        raise ValueError(f"Expected dict, got {type(d)}")
    
    @classmethod
    def from_checkpoint_path(cls, checkpoint_path: str) -> "ModelConfig":
        """Create ModelConfig from checkpoint file path.
        
        The config.yaml is expected at: checkpoint_path.parent.parent / 'config.yaml'
        e.g., /exp/20240101_1200/ckpt/model.pt -> /exp/20240101_1200/config.yaml
        
        Args:
            checkpoint_path: Path to checkpoint .pt file
            
        Returns:
            ModelConfig instance (only model-related config)
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # config.yaml is at ckpt_path.parent.parent / 'config.yaml'
        config_path = ckpt_path.parent.parent / 'config.yaml'
        if not config_path.exists():
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            if 'config' in checkpoint:
                # Save config to temp file and load from there
                with tempfile.NamedTemporaryFile(mode='w+', suffix='.yaml', delete=False) as tmpf:
                    yaml.dump(checkpoint['config'], tmpf)
                    tmpf_path = tmpf.name
                logger.info(f"Loading model config from checkpoint 'config' key")
                model_config = cls.from_yaml(tmpf_path)
                os.remove(tmpf_path)
                return model_config
            # Fallback: try ckpt_path.parent / 'config.yaml'
            config_path = ckpt_path.parent / 'config.yaml'
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config not found. Searched:\n"
                f"  - {ckpt_path.parent.parent / 'config.yaml'}\n"
                f"  - {ckpt_path.parent / 'config.yaml'}"
            )
        
        logger.info(f"Loading model config from: {config_path}")
        return cls.from_yaml(str(config_path))
    
    @classmethod
    def from_experiment_dir(cls, experiment_dir: str) -> "ModelConfig":
        """Create ModelConfig from experiment directory.
        
        The config.yaml is expected at: experiment_dir / 'config.yaml'
        
        Args:
            experiment_dir: Path to experiment directory
            
        Returns:
            ModelConfig instance (only model-related config)
        """
        exp_dir = Path(experiment_dir)
        config_path = exp_dir / 'config.yaml'
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        logger.info(f"Loading model config from: {config_path}")
        return cls.from_yaml(str(config_path))
    
    def compare_with(self, other: "ModelConfig", warn_on_diff: bool = True) -> Dict[str, tuple]:
        """Compare this config with another config.
        
        Args:
            other: Another ModelConfig to compare with
            warn_on_diff: If True, log warnings for differences
            
        Returns:
            Dict of differing fields: {field_name: (self_value, other_value)}
        """
        diffs = {}
        self_dict = self.to_dict()
        other_dict = other.to_dict()
        
        # Compare all fields
        all_keys = set(self_dict.keys()) | set(other_dict.keys())
        for key in all_keys:
            self_val = self_dict.get(key)
            other_val = other_dict.get(key)
            if self_val != other_val:
                diffs[key] = (self_val, other_val)
                if warn_on_diff:
                    logger.warning(
                        f"ModelConfig mismatch for '{key}': "
                        f"current={self_val}, checkpoint={other_val}"
                    )
        
        return diffs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'model_name': self.model_name,
            'dim': self.dim,
            'mode': self.mode,
            'attention_mode': self.attention_mode,
            'mlp_dim': self.mlp_dim,
            'output_dim': self.output_dim,
            'sr': self.sr,
            'prompt_tf_depth': self.prompt_tf_depth,
            'joint_tf_depth': self.joint_tf_depth,
            'dim_head': self.dim_head,
            'heads': self.heads,
            'attn_dropout': self.attn_dropout,
            'ff_dropout': self.ff_dropout,
            'ff_mult': self.ff_mult,
            'use_layer_idx': self.use_layer_idx,
            'freeze_audio': self.freeze_audio,
            'freeze_text': self.freeze_text,
            # train_muqmulan removed
            'train_muq_depth': self.train_muq_depth,
            'text_encoder': self.text_encoder,
            'text_lora_config': self.text_lora_config,
            'load_config': {
                'checkpoint_path': self.load_config.checkpoint_path,
                'frozen_from_pretrained': self.load_config.frozen_from_pretrained,
                'pretrained_name': self.load_config.pretrained_name,
                'strict': self.load_config.strict,
            },
            'null_embedding': {
                'text': {'dropout': self.null_embedding_config.text.dropout, 'length': self.null_embedding_config.text.length},
                'lyrics': {'dropout': self.null_embedding_config.lyrics.dropout, 'length': self.null_embedding_config.lyrics.length},
                'audio': {'dropout': self.null_embedding_config.audio.dropout, 'length': self.null_embedding_config.audio.length},
            },
            'category_embeddings': self.category_embeddings,
            'downsample': self.downsample,
        }


def load_config(config_path: str) -> ModelConfig:
    """Load model configuration from YAML file
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        ModelConfig instance
    """
    return ModelConfig.from_yaml(config_path)


def load_model_weights(
    model: torch.nn.Module,
    checkpoint_path: Optional[str] = None,
    frozen_from_pretrained: bool = True,
    strict: bool = False,
    device: str = 'cpu',
) -> torch.nn.Module:
    """Load model weights with support for mixed loading strategies
    
    Loading strategy:
    1. If checkpoint_path is None: Model already initialized from pretrained
    2. If frozen_from_pretrained=True: Load only trainable params from checkpoint
    3. If frozen_from_pretrained=False: Load all params from checkpoint
    
    Args:
        model: Model to load weights into (should be initialized)
        checkpoint_path: Path to checkpoint file (.pt)
        frozen_from_pretrained: If True, skip frozen module weights from checkpoint
        strict: Whether to use strict mode for state_dict loading
        device: Device to load weights to
        
    Returns:
        Model with loaded weights
    """
    if checkpoint_path is None:
        logger.info("No checkpoint specified, using pretrained weights")
        return model
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    if frozen_from_pretrained:
        # Filter out frozen modules (text_module, audio_module)
        # These will be loaded from pretrained
        filtered_state_dict = {}
        frozen_prefixes = ['text_module.', 'audio_module.']
        
        for key, value in state_dict.items():
            is_frozen = any(key.startswith(prefix) for prefix in frozen_prefixes)
            if not is_frozen:
                filtered_state_dict[key] = value
            else:
                logger.debug(f"Skipping frozen weight: {key}")
        
        logger.info(f"Loading {len(filtered_state_dict)}/{len(state_dict)} weights "
                   f"(frozen modules loaded from pretrained)")
        state_dict = filtered_state_dict
    
    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    
    if missing_keys:
        logger.info(f"Missing keys (expected for frozen modules): {len(missing_keys)}")
        if len(missing_keys) <= 10:
            for key in missing_keys:
                logger.debug(f"  - {key}")
    
    if unexpected_keys:
        logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
    
    return model


def create_model_from_config(config: Union[ModelConfig, DictConfig, Dict[str, Any]]) -> torch.nn.Module:
    """Create model from configuration
    
    Args:
        config: ModelConfig, OmegaConf DictConfig, or dict
        
    Returns:
        Initialized model (with pretrained weights for frozen modules)
    """
    # Convert to ModelConfig if needed
    if isinstance(config, DictConfig):
        model_config = ModelConfig.from_omegaconf(config)
    elif isinstance(config, dict):
        model_config = ModelConfig.from_dict(config)
    elif isinstance(config, ModelConfig):
        model_config = config
    else:
        raise TypeError(f"Expected ModelConfig, DictConfig, or dict, got {type(config)}")
    
    # Import model classes (avoid circular import)
    from ..models.mymodel import RewardAttentionModel, SanityCheckModel
    
    if model_config.name == 'sanity':
        model = SanityCheckModel(
            model_name=model_config.model_name,
            dim=model_config.dim,
            mlp_dim=model_config.mlp_dim,
            output_dim=model_config.output_dim,
            sr=model_config.sr,
            freeze_audio=model_config.freeze_audio,
            freeze_text=model_config.freeze_text,
            cfg=model_config.to_dict(),
        )
        logger.info("Created SanityCheckModel")
    else:
        model = RewardAttentionModel(
            model_name=model_config.model_name,
            dim=model_config.dim,
            mode=model_config.mode,
            attention_mode=model_config.attention_mode,
            mlp_dim=model_config.mlp_dim,
            output_dim=model_config.output_dim,
            sr=model_config.sr,
            prompt_tf_depth=model_config.prompt_tf_depth,
            joint_tf_depth=model_config.joint_tf_depth,
            dim_head=model_config.dim_head,
            heads=model_config.heads,
            attn_dropout=model_config.attn_dropout,
            ff_dropout=model_config.ff_dropout,
            ff_mult=model_config.ff_mult,
            use_layer_idx=model_config.use_layer_idx,
            freeze_audio=model_config.freeze_audio,
            freeze_text=model_config.freeze_text,
            text_lora_config=model_config.text_lora_config,
            text_encoder_config=model_config.text_encoder_config,
            cfg=model_config.to_dict(),
        )
        logger.info(f"Created RewardAttentionModel with attention_mode={model_config.attention_mode}")
    
    # Load checkpoint if specified
    if model_config.load_config.checkpoint_path:
        model = load_model_weights(
            model=model,
            checkpoint_path=model_config.load_config.checkpoint_path,
            frozen_from_pretrained=model_config.load_config.frozen_from_pretrained,
            strict=model_config.load_config.strict,
        )
    
    return model


# Alias for backward compatibility
def load_model(
    config: Union[str, ModelConfig, DictConfig, Dict[str, Any]],
    checkpoint_path: Optional[str] = None,
    device: str = 'cpu',
) -> torch.nn.Module:
    """Load model from configuration
    
    Args:
        config: Path to YAML file, or ModelConfig/DictConfig/dict
        checkpoint_path: Optional override for checkpoint path
        device: Device to load model to
        
    Returns:
        Loaded model
    """
    # Load config if path
    if isinstance(config, str):
        model_config = ModelConfig.from_yaml(config)
    elif isinstance(config, DictConfig):
        model_config = ModelConfig.from_omegaconf(config)
    elif isinstance(config, dict):
        model_config = ModelConfig.from_dict(config)
    elif isinstance(config, ModelConfig):
        model_config = config
    else:
        raise TypeError(f"Expected str, ModelConfig, DictConfig, or dict, got {type(config)}")
    
    # Override checkpoint path if provided
    if checkpoint_path is not None:
        model_config.load_config.checkpoint_path = checkpoint_path
    
    # Create and load model
    model = create_model_from_config(model_config)
    model = model.to(device)
    
    return model


if __name__ == '__main__':
    # Simple test
    import tempfile
    
    # Test ModelConfig creation
    config = ModelConfig()
    print(f"Default ModelConfig: {config.name}, dim={config.dim}")
    
    # Test from_dict
    config_dict = {
        'name': 'reward',
        'dim': 768,
        'null_embedding': {
            'text': {'dropout': 0.2, 'length': 10},
            'lyrics': {'dropout': 0.3, 'length': 10},
            'audio': {'dropout': 0.5, 'length': 10},
        }
    }
    config = ModelConfig.from_dict(config_dict)
    print(f"From dict: text_dropout={config.null_embedding_config.text.dropout}")
    
    # Test to_dict round-trip
    config_rt = ModelConfig.from_dict(config.to_dict())
    print(f"Round-trip: text_dropout={config_rt.null_embedding_config.text.dropout}")
    
    print("✓ model_utils tests passed")
