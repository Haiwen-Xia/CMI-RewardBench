"""
Flan-T5 Text Encoder Module

A text encoder implementation using Flan-T5-Base for the reward model.
Supports:
- Fine-tuning the last M layers
- LoRA fine-tuning
- Forward pass with raw text input

Usage:
    encoder = T5TextEncoder(
        model_name='google/flan-t5-base',
        tune_last_n_layers=2,  # Fine-tune last 2 encoder layers
    )
    
    # Or with LoRA
    encoder = T5TextEncoder(
        model_name='google/flan-t5-base',
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
    )
"""

from typing import Optional, List, Union
import logging

import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5Config, AutoTokenizer

logger = logging.getLogger(__name__)


def frozen_params(module: nn.Module):
    """Freeze all parameters in a module"""
    for param in module.parameters():
        param.requires_grad = False


class T5TextEncoder(nn.Module):
    """Text encoder based on Flan-T5
    
    Uses T5 encoder (not the full T5 with decoder) to extract text embeddings.
    
    Args:
        model_name: HuggingFace model name (e.g., 'google/flan-t5-base', 'google/flan-t5-large')
        dim: Output dimension (if different from model dim, a projection is applied)
        max_seq_len: Maximum sequence length for tokenization
        freeze: Whether to freeze the pretrained model
        tune_last_n_layers: Number of encoder layers to fine-tune from the end (0 means all frozen)
        use_lora: Whether to use LoRA for parameter-efficient fine-tuning
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: LoRA dropout rate
        lora_target_modules: Target modules for LoRA (default: q, k, v, o projections)
        cache_dir: HuggingFace cache directory
    """
    
    def __init__(
        self,
        model_name: str = 'google/flan-t5-base',
        dim: int = 768,
        max_seq_len: int = 512,
        freeze: bool = True,
        tune_last_n_layers: int = 0,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.use_lora = use_lora
        self.tune_last_n_layers = tune_last_n_layers
        self.cache_dir = cache_dir
        
        # Load T5 encoder (not full T5 with decoder)
        self.model = T5EncoderModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        
        # Get model dimension
        self.model_dim = self.model.config.d_model
        self.num_layers = self.model.config.num_layers
        
        logger.info(f"Loaded T5 encoder: {model_name}")
        logger.info(f"  - Model dim: {self.model_dim}")
        logger.info(f"  - Num layers: {self.num_layers}")
        logger.info(f"  - Vocab size: {self.model.config.vocab_size}")
        
        # Projection layer if output dim differs from model dim
        if self.model_dim != dim:
            self.proj = nn.Linear(self.model_dim, dim)
            logger.info(f"  - Added projection: {self.model_dim} -> {dim}")
        else:
            self.proj = nn.Identity()
        
        # Tokenizer (lazy initialization)
        self._tokenizer = None
        
        # Apply freezing / fine-tuning strategy
        if use_lora:
            self._apply_lora(lora_r, lora_alpha, lora_dropout, lora_target_modules)
        elif freeze:
            frozen_params(self.model)
            if tune_last_n_layers > 0:
                self._unfreeze_last_n_layers(tune_last_n_layers)
        
        # Log parameter stats
        self._log_param_stats()
    
    def _apply_lora(
        self, 
        lora_r: int, 
        lora_alpha: int, 
        lora_dropout: float, 
        target_modules: Optional[List[str]]
    ):
        """Apply LoRA to the T5 encoder"""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError("peft library is required for LoRA. Install with: pip install peft")
        
        # Default target modules for T5
        if target_modules is None:
            # T5 uses SelfAttention with q, k, v, o naming
            target_modules = ["q", "k", "v", "o"]
        
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            inference_mode=False,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        logger.info(f"Applied LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        logger.info(f"  - Target modules: {target_modules}")
    
    def _unfreeze_last_n_layers(self, n: int):
        """Unfreeze the last n encoder layers"""
        if n <= 0:
            return
        
        if n > self.num_layers:
            logger.warning(f"tune_last_n_layers ({n}) > num_layers ({self.num_layers}), unfreezing all layers")
            n = self.num_layers
        
        # T5EncoderModel structure: model.encoder.block[i]
        encoder_blocks = self.model.encoder.block
        
        for i in range(self.num_layers - n, self.num_layers):
            for param in encoder_blocks[i].parameters():
                param.requires_grad = True
            logger.info(f"  - Unfroze encoder layer {i}")
        
        # Also unfreeze final layer norm
        if hasattr(self.model.encoder, 'final_layer_norm'):
            for param in self.model.encoder.final_layer_norm.parameters():
                param.requires_grad = True
            logger.info("  - Unfroze final_layer_norm")
    
    def _log_param_stats(self):
        """Log parameter statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params_count = total_params - trainable_params
        
        logger.info(f"Parameter stats:")
        logger.info(f"  - Total: {total_params / 1e6:.2f}M")
        logger.info(f"  - Trainable: {trainable_params / 1e6:.2f}M ({100 * trainable_params / total_params:.2f}%)")
        logger.info(f"  - Frozen: {frozen_params_count / 1e6:.2f}M")
    
    @property
    def tokenizer(self):
        """Lazy load tokenizer"""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
        return self._tokenizer
    
    @property
    def device(self):
        return next(self.model.parameters()).device
    
    @property
    def dtype(self):
        return next(self.model.parameters()).dtype
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        raw_texts: Optional[List[str]] = None,
        return_mean: bool = True,
        return_mask: bool = False,
        return_all_layers: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """Forward pass
        
        Args:
            input_ids: Tokenized input IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            raw_texts: List of raw text strings (will be tokenized)
            return_mean: If True, return mean-pooled output; else return sequence output
            return_mask: If True, also return attention mask
            return_all_layers: If True, also return all hidden states
            
        Returns:
            embeddings: Text embeddings [batch, dim] or [batch, seq_len, dim]
            (optional) attention_mask: [batch, seq_len]
            (optional) all_hidden_states: Tuple of hidden states from all layers
        """
        # Tokenize if raw texts provided
        if raw_texts is not None:
            inputs = self.tokenizer(
                raw_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_seq_len,
            )
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
        
        assert input_ids is not None, "Either input_ids or raw_texts must be provided"
        
        # Forward through T5 encoder
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=return_all_layers,
            return_dict=True,
        )
        
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, model_dim]
        
        # Project if needed
        hidden_states = self.proj(hidden_states)  # [batch, seq_len, dim]
        
        # Mean pooling with attention mask
        if return_mean:
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
                masked_sum = (hidden_states * mask_expanded).sum(dim=1)  # [batch, dim]
                mask_sum = mask_expanded.sum(dim=1).clamp(min=1e-9)  # [batch, 1]
                pooled = masked_sum / mask_sum  # [batch, dim]
            else:
                pooled = hidden_states.mean(dim=1)  # [batch, dim]
            
            result = pooled
        else:
            result = hidden_states
        
        # Prepare return values
        if return_all_layers and return_mask:
            all_hidden = outputs.hidden_states if return_all_layers else None
            return result, all_hidden, attention_mask
        elif return_all_layers:
            all_hidden = outputs.hidden_states
            return result, all_hidden
        elif return_mask:
            return result, attention_mask
        
        return result
    
    def get_num_layers(self) -> int:
        """Get the number of encoder layers"""
        return self.num_layers
    
    def print_layer_info(self):
        """Print detailed layer information"""
        print(f"\n{'='*60}")
        print(f"T5 Text Encoder: {self.model_name}")
        print(f"{'='*60}")
        print(f"Number of encoder layers: {self.num_layers}")
        print(f"Model dimension: {self.model_dim}")
        print(f"Output dimension: {self.dim}")
        print(f"Max sequence length: {self.max_seq_len}")
        print(f"\nEncoder blocks:")
        for i, block in enumerate(self.model.encoder.block):
            trainable = any(p.requires_grad for p in block.parameters())
            status = "✓ Trainable" if trainable else "✗ Frozen"
            print(f"  Layer {i}: {status}")
        print(f"{'='*60}\n")


def main():
    """Main function for testing the T5TextEncoder"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test T5 Text Encoder")
    parser.add_argument("--model", type=str, default="google/flan-t5-base", help="Model name")
    parser.add_argument("--tune_last_n", type=int, default=0, help="Number of layers to fine-tune")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("Testing T5TextEncoder")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Tune last N layers: {args.tune_last_n}")
    print(f"Use LoRA: {args.use_lora}")
    
    # Initialize encoder
    if args.use_lora:
        encoder = T5TextEncoder(
            model_name=args.model,
            use_lora=True,
            lora_r=args.lora_r,
            freeze=False,
        )
    else:
        encoder = T5TextEncoder(
            model_name=args.model,
            freeze=True,
            tune_last_n_layers=args.tune_last_n,
        )
    
    encoder = encoder.to(args.device)
    
    # Print layer info
    encoder.print_layer_info()
    
    # Test forward pass with raw text
    test_texts = [
        "A beautiful piano melody with soft strings in the background.",
        "Energetic electronic dance music with heavy bass drops.",
        "Calm acoustic guitar playing a folk song.",
    ]
    
    print(f"\nTest forward pass with {len(test_texts)} texts:")
    for i, text in enumerate(test_texts):
        print(f"  [{i}] {text[:50]}...")
    
    # Forward pass
    with torch.no_grad() if not (args.use_lora or args.tune_last_n > 0) else torch.enable_grad():
        # Mean pooled output
        output_mean = encoder(raw_texts=test_texts, return_mean=True)
        print(f"\nMean pooled output shape: {output_mean.shape}")
        print(f"Output dtype: {output_mean.dtype}")
        
        # Sequence output with mask
        output_seq, mask = encoder(raw_texts=test_texts, return_mean=False, return_mask=True)
        print(f"Sequence output shape: {output_seq.shape}")
        print(f"Attention mask shape: {mask.shape}")
        
        # With all hidden states
        output_mean, all_hidden = encoder(raw_texts=test_texts, return_mean=True, return_all_layers=True)
        print(f"Number of hidden states: {len(all_hidden)}")
        print(f"Hidden state shapes: {[h.shape for h in all_hidden[:3]]}...")
    
    # Test gradient flow if fine-tuning
    if args.use_lora or args.tune_last_n > 0:
        print(f"\nTesting gradient flow...")
        output = encoder(raw_texts=test_texts, return_mean=True)
        loss = output.sum()
        loss.backward()
        
        # Check which parameters have gradients
        has_grad = []
        no_grad = []
        for name, param in encoder.named_parameters():
            if param.grad is not None:
                has_grad.append(name)
            elif param.requires_grad:
                no_grad.append(name)
        
        print(f"  Parameters with gradients: {len(has_grad)}")
        if has_grad:
            print(f"    Examples: {has_grad[:3]}...")
    
    print(f"\n{'='*60}")
    print("Test completed successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
