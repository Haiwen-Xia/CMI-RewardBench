"""
Reward Model Components

Main exports:
- RewardAttentionModel: Full reward model with attention interaction
- SanityCheckModel: Simple baseline model
- DownstreamModel: Wrapper for downstream task predictions
- BackboneMixin: Shared initialization methods for models
"""

from .mymodel import (
    RewardAttentionModel,
    SanityCheckModel,
)



__all__ = [
    # Main models
    'RewardAttentionModel',
    'SanityCheckModel',
    
]
