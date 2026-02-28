from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from x_clip.tokenizer import tokenizer
from einops import rearrange, repeat, reduce, pack, unpack
from transformers import AutoTokenizer,XLMRobertaModel,AutoModelForCausalLM

from ..modules.utils import *
from ..modules.transformer import Transformer, LayerNorm
from ..modules.utils import frozen_params


# text transformer

class TextTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_tokens = tokenizer.vocab_size,
        max_seq_len = 256,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4,
        pad_id = 0
    ):
        super().__init__()
        self.dim = dim

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.depth = depth
        self.max_seq_len = max_seq_len

        self.cls_token = nn.Parameter(torch.randn(dim))

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_mult = ff_mult
        )

        self.pad_id = pad_id
        self.norm = LayerNorm(dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        x = None,
        raw_texts: Optional[List[str]] = None,
        mask = None,
        return_all_layers = False
    ):
        assert exists(x) ^ exists(raw_texts)

        if exists(raw_texts):
            x = tokenizer.tokenize(raw_texts).to(self.device)

        if not exists(mask):
            mask = x != self.pad_id

        b, n, device = *x.shape, x.device

        # token embedding + positional embedding

        x = self.token_emb(x)

        assert n <= self.max_seq_len, f'text sequence length {n} must be less than {self.max_seq_len}'

        x = x + self.pos_emb(torch.arange(n, device = device))

        # cls tokens, as in bert

        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)
        x, ps = pack([cls_tokens, x], 'b * d')

        # account for attending to cls token with self attention mask

        mask = F.pad(mask, (1, 0), value = True)

        # attention

        x, all_layers = self.transformer(x, mask = mask, return_all_layers = True)

        # unpack the cls tokens

        cls_tokens, _ = unpack(x, ps, 'b * d')

        out = self.norm(cls_tokens)

        if not return_all_layers:
            return out

        return out, all_layers

class TextPretrainedModelType:
    Roberta = 'roberta'
    Qwen = 'qwen'

class TextTransformerPretrained(nn.Module):
    def __init__(
        self,
        model_name = 'xlm-roberta-base',
        dim = 768,
        model_dim = None,
        max_seq_len = 256,
        tf_depth = 12,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4,
        frozen_pretrained = True,
        hf_hub_cache_dir = None,
        # LoRA parameters
        use_lora = False,
        lora_r = 8,
        lora_alpha = 16,
        lora_dropout = 0.1,
        lora_target_modules = None,
    ):
        super().__init__()
        self.dim = dim 

        self.model_name = model_name

        self.hf_hub_cache_dir = hf_hub_cache_dir

        self.pretrained_model_type = self._get_pretrained_model_type(model_name)

        self.model = self._init_pretrained_model()

        self._tokenizer = None

        self.max_seq_len = max_seq_len

        self.transformer = Transformer(
            dim = dim,
            depth = tf_depth,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_mult = ff_mult
        ) # if tf_depth > 0 else torch.nn.Identity()

        is_proj = exists(model_dim) and model_dim != dim

        self.proj = nn.Linear(model_dim, dim) if is_proj else torch.nn.Identity()
        
        # Apply LoRA or freeze pretrained model
        self.use_lora = use_lora
        if use_lora:
            self._apply_lora(lora_r, lora_alpha, lora_dropout, lora_target_modules)
        elif frozen_pretrained:
            frozen_params(self.model)
            
        self.frozen_pretrained = frozen_pretrained and not use_lora
    
    @staticmethod
    def _get_pretrained_model_type(model_name):
        if 'xlm-roberta' in model_name:
            return TextPretrainedModelType.Roberta
        elif 'Qwen' in model_name:
            return TextPretrainedModelType.Qwen
        else:
            raise ValueError(f"Unknown pretrained model named: {model_name}")
    
    def _init_pretrained_model(self):
        if self.pretrained_model_type == TextPretrainedModelType.Roberta:
            model = XLMRobertaModel.from_pretrained(self.model_name, trust_remote_code=True, cache_dir=self.hf_hub_cache_dir)
        elif self.pretrained_model_type == TextPretrainedModelType.Qwen:
            model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True, fp16=True, cache_dir=self.hf_hub_cache_dir)
        else:
            raise ValueError(f"Failed to init pretrained model type: {self.pretrained_model_type}")
        return model
    
    def _apply_lora(self, lora_r, lora_alpha, lora_dropout, target_modules):
        """Apply LoRA to pretrained model"""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError("peft library is required for LoRA. Install with: pip install peft")
        
        import logging
        logger = logging.getLogger(__name__)
        
        # Set default target_modules based on model type
        if target_modules is None:
            if self.pretrained_model_type == TextPretrainedModelType.Roberta:
                # XLM-RoBERTa: query, key, value, dense (output projection)
                target_modules = ["query", "key", "value", "dense"]
            elif self.pretrained_model_type == TextPretrainedModelType.Qwen:
                # Qwen: attention q_proj, k_proj, v_proj, o_proj
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            else:
                raise ValueError(f"Unknown model type for LoRA: {self.pretrained_model_type}")
        
        # Create LoRA config
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            inference_mode=False
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters info
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"LoRA applied to {self.model_name}: "
            f"trainable params: {trainable_params:,} / {all_params:,} "
            f"({100 * trainable_params / all_params:.2f}%)"
        )
        logger.info(f"LoRA config: r={lora_r}, alpha={lora_alpha}, "
                   f"dropout={lora_dropout}, targets={target_modules}")
    
    def _init_tokenizer(self):
        if self.pretrained_model_type == TextPretrainedModelType.Roberta:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, cache_dir=self.hf_hub_cache_dir)
        elif self.pretrained_model_type == TextPretrainedModelType.Qwen:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, cache_dir=self.hf_hub_cache_dir)
            tokenizer.pad_token = '<|im_end|>'
        else:
            raise ValueError(f"Failed to init tokenizer of pretrained model type: {self.pretrained_model_type}")
        return tokenizer

    @property
    def tokenizer(self):
        if not exists(self._tokenizer):
            self._tokenizer = self._init_tokenizer()
        return self._tokenizer
    
    @property
    def device(self):
        return next(self.model.parameters()).device
    
    @property
    def dtype(self):
        return next(self.transformer.parameters()).dtype
    
    
    def pred_pretrained_model_hidden(self, **kw):
        if self.pretrained_model_type == TextPretrainedModelType.Roberta:
            outputs = self.model(**kw)
            outputs = outputs.last_hidden_state
        elif self.pretrained_model_type == TextPretrainedModelType.Qwen:
            last_hidden_state = self.model(**kw, output_hidden_states=True)['hidden_states'][-1]
            outputs = last_hidden_state.to(dtype = self.dtype)
        else:
            raise ValueError(f"Unknown pretrained model type: {self.pretrained_model_type}")
        return outputs

    def forward(
        self,
        x = None,
        raw_texts: Optional[List[str]] = None,
        mask = None,
        return_all_layers = False,
        return_mean = True,
        return_mask = False
    ):
        assert exists(x) ^ exists(raw_texts)
        attention_mask = None
        
        # Use torch.no_grad() only when not using LoRA
        if self.use_lora:
            # LoRA mode: allow gradients
            if exists(raw_texts):
                model_max_len = self.model.config.max_position_embeddings
                effective_max_len = min(self.max_seq_len, 512)
                inputs = self.tokenizer(
                    raw_texts, 
                    return_tensors='pt', 
                    padding=True,
                    truncation=True,
                    max_length=effective_max_len,
                    return_token_type_ids=False
                )                
                inputs = inputs.to(self.device)
                attention_mask = inputs['attention_mask']

            if exists(mask):
                mask_long = mask.long() if mask.dtype != torch.long else mask
                inputs['attention_mask'] = mask_long
                attention_mask = mask_long
            
            outputs = self.pred_pretrained_model_hidden(**inputs)
        else:
            # Frozen mode: no gradients needed
            with torch.no_grad():
                if exists(raw_texts):
                    model_max_len = self.model.config.max_position_embeddings
                    effective_max_len = min(self.max_seq_len, 512)
                    inputs = self.tokenizer(
                        raw_texts, 
                        return_tensors='pt', 
                        padding=True,
                        truncation=True,
                        max_length=effective_max_len,
                        return_token_type_ids=False
                    )                
                    inputs = inputs.to(self.device)
                    attention_mask = inputs['attention_mask']

                if exists(mask):
                    mask_long = mask.long() if mask.dtype != torch.long else mask
                    inputs['attention_mask'] = mask_long
                    attention_mask = mask_long
                
                outputs = self.pred_pretrained_model_hidden(**inputs)
        
        # Convert attention mask to bool if needed
        if exists(attention_mask) and attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()
        outputs = self.proj(outputs)

        # Pass mask to transformer for attention
        outputs, layer_results = self.transformer(outputs, mask=attention_mask, return_all_layers=True)
        
        if return_mean:
            # Masked mean pooling: only average over non-padded tokens
            if exists(attention_mask):
                # Expand mask to match hidden dimension: [batch, seq_len] -> [batch, seq_len, 1]
                mask_expanded = attention_mask.unsqueeze(-1).float()
                # Weighted sum divided by number of valid tokens
                masked_sum = (outputs * mask_expanded).sum(dim=-2)
                mask_sum = mask_expanded.sum(dim=-2).clamp(min=1e-9)
                outputs = masked_sum / mask_sum
                
                # Log padding statistics
                import logging
                logger = logging.getLogger(__name__)
                seq_lengths = attention_mask.sum(dim=-1)
                max_len = attention_mask.shape[-1]
                logger.debug(f"Text sequence lengths: {seq_lengths.tolist()}, max_len: {max_len}, padding ratios: {[(max_len - l.item()) / max_len for l in seq_lengths]}")
            else:
                outputs = outputs.mean(dim = -2)
        
        if return_mask:
            if return_all_layers:
                return outputs, layer_results, attention_mask
            return outputs, attention_mask
        
        if return_all_layers:
            return outputs, layer_results
        return outputs
