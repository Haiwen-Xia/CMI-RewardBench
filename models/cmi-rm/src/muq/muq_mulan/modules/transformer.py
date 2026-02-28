# attention
from regex import P
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange
from .utils import l2norm, default, exists

# PyTorch native Flash Attention support (>= 2.0)
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    PYTORCH_FLASH_AVAILABLE = hasattr(F, 'scaled_dot_product_attention')
except ImportError:
    PYTORCH_FLASH_AVAILABLE = False
    SDPBackend = None
    sdpa_kernel = None

# 2d sinusoidal positional embedding
# simple vit paper shows it is good enough compared to learned

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'

    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    pe = pe.type(dtype)

    return rearrange(pe, '(h w) d -> h w d', h = h, w = w)

# biasless layernorm

class LayerNorm(nn.Module):
    def __init__(self, dim, scale = True):
        super().__init__()
        self.learned_gamma = nn.Parameter(torch.ones(dim)) if scale else None

        self.register_buffer('gamma', torch.ones(dim), persistent = False)
        self.register_buffer('beta', torch.zeros(dim), persistent = False)

    def forward(self, x):
        # Ensure gamma and beta match input dtype (for mixed precision training)
        gamma = default(self.learned_gamma, self.gamma)
        beta = self.beta
        if gamma.dtype != x.dtype:
            gamma = gamma.to(x.dtype)
        if beta.dtype != x.dtype:
            beta = beta.to(x.dtype)
        return F.layer_norm(x, x.shape[-1:], gamma, beta)

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

def FeedForward(dim, mult = 4, dropout = 0.):
    dim_hidden = int(dim * mult * 2 / 3)

    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, dim_hidden * 2, bias = False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_hidden, dim, bias = False)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        scale = 8,
        use_flash_attn = True
    ):
        super().__init__()
        self.heads = heads
        self.scale = scale
        self.causal = causal
        self.dim_head = dim_head
        self.dropout = dropout
        inner_dim = dim_head * heads

        # Determine if we can use FlashAttention
        self.use_flash_attn = use_flash_attn and PYTORCH_FLASH_AVAILABLE
        if use_flash_attn and not PYTORCH_FLASH_AVAILABLE:
            print("Warning: FlashAttention requested but not available. Falling back to standard attention.")

        self.norm = LayerNorm(dim)

        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        rel_pos_bias = None,
        mask = None
    ):
        b, n, _, device = *x.shape, x.device

        # prenorm
        x = self.norm(x)

        # project for queries, keys, values
        q, k, v = self.to_q(x), *self.to_kv(x).chunk(2, dim = -1)

        # Rearrange to (b, h, n, d) for multi-head attention
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))
        
        # Apply L2 normalization and scaling (QK-Norm style)
        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # Use Flash Attention (SDPA) if available and conditions are met
        use_sdpa = (
            self.use_flash_attn and PYTORCH_FLASH_AVAILABLE and
            not self.causal and not exists(rel_pos_bias)
        )
        
        if use_sdpa:
            # Convert boolean mask to attention mask for SDPA
            # SDPA expects: True = attend, or float mask where -inf = don't attend
            if exists(mask):
                # mask: [b, n] boolean, True = valid token
                # Convert to [b, 1, 1, n] float mask for broadcasting
                attn_mask = (~mask).to(q.dtype) * -1e9   # [b, n]
                attn_mask = attn_mask[:, None, None, :]  # [b, 1, 1, n]
            else:
                attn_mask = None
            
            # Use scaled_dot_product_attention (supports Flash Attention backend)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
                scale=self.scale
            )
        else:
            # Standard attention path (for causal or rel_pos_bias cases)
            sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

            if exists(rel_pos_bias):
                rel_pos_bias = rel_pos_bias.to(compute_dtype)
                sim = sim + rel_pos_bias

            if exists(mask):
                mask_expanded = rearrange(mask, 'b j -> b 1 1 j')
                sim = sim.masked_fill(~mask_expanded, -torch.finfo(sim.dtype).max)

            if self.causal:
                i, j = sim.shape[-2:]
                causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
                sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

            attn = sim.softmax(dim = -1)
            attn = self.attn_dropout(attn)
            out = einsum('b h i j, b h j d -> b h i d', attn, v)
        
        # merge heads
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        use_flash_attn = True
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.gradient_checkpointing = False
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, use_flash_attn = use_flash_attn),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout),
            ]))
    
    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def forward(
        self,
        x,
        rel_pos_bias = None,
        mask = None,
        return_all_layers = False
    ):
        # Handle empty sequence (L=0) gracefully - return as-is without processing
        if x.shape[1] == 0:
            if return_all_layers:
                return x, None
            return x
        
        if return_all_layers:
            layers = []

        for attn, ff in self.layers:
            if self.gradient_checkpointing and self.training:
                # Use checkpoint for memory efficiency during training
                x = torch.utils.checkpoint.checkpoint(attn, x, rel_pos_bias, mask, use_reentrant=False) + x
                x = torch.utils.checkpoint.checkpoint(ff, x, use_reentrant=False) + x
            else:
                x = attn(x, rel_pos_bias = rel_pos_bias, mask = mask) + x
                x = ff(x) + x
            if return_all_layers:
                layers.append(x)

        if not return_all_layers:
            return x
        
        return x, torch.stack(layers[:-1]) if len(self.layers)>1 else None
    
class CrossAttentionLayer(nn.Module):
    """Cross-attention layer reusing transformer.Attention
    Query: from decoder (eval_audio)
    Key/Value: from encoder (prompt)
    """
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        use_flash_attn = True
    ):
        super().__init__()
        from ..modules.transformer import Attention, PYTORCH_FLASH_AVAILABLE
        
        self.norm_q = LayerNorm(dim)
        self.norm_kv = LayerNorm(dim)
        
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * heads * 2, bias=False)
        
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.dropout = dropout
        
        # Determine if we can use PyTorch Flash Attention
        self.use_flash_attn = use_flash_attn and PYTORCH_FLASH_AVAILABLE
        if use_flash_attn and not PYTORCH_FLASH_AVAILABLE:
            print("Warning: FlashAttention requested but PyTorch SDPA not available. Falling back to standard attention.")
        
        self.attn_dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(dim_head * heads, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x, encoder_hidden_states, mask_q=None, mask_kv=None):
        """
        Args:
            x: Query [batch, len_q, dim]
            encoder_hidden_states: Key/Value [batch, len_kv, dim]
            mask_q: Query mask [batch, len_q]
            mask_kv: Key/Value mask [batch, len_kv]
        """
        b, len_q, _ = x.shape
        len_kv = encoder_hidden_states.shape[1]
        device = x.device
        
        # Normalize
        x_norm = self.norm_q(x)
        enc_norm = self.norm_kv(encoder_hidden_states)
        
        # Project to multi-head
        q = self.to_q(x_norm)
        k, v = self.to_kv(enc_norm).chunk(2, dim=-1)
        
        # Rearrange to (b, h, n, d) for multi-head attention
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        
        # Use Flash Attention (SDPA) if available
        use_sdpa = self.use_flash_attn and PYTORCH_FLASH_AVAILABLE
        
        if use_sdpa:
            # Build attention mask for cross-attention: [b, h, len_q, len_kv]
            # Only mask_kv matters for cross-attention (which keys to attend to)
            if exists(mask_kv):
                # mask_kv: [b, len_kv] boolean, True = valid token
                # Expand to [b, 1, 1, len_kv] then broadcast to [b, h, len_q, len_kv]
                attn_mask = torch.zeros(b, 1, 1, len_kv, dtype=q.dtype, device=device)
                attn_mask = attn_mask.masked_fill(~mask_kv.unsqueeze(1).unsqueeze(2), float('-inf'))
            else:
                attn_mask = None
            
            # Use scaled_dot_product_attention
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
                scale=self.scale
            )
        else:
            # Standard attention path
            sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
            
            if exists(mask_kv):
                mask_kv_expanded = rearrange(mask_kv, 'b j -> b 1 1 j')
                sim = sim.masked_fill(~mask_kv_expanded, -torch.finfo(sim.dtype).max)
            
            attn = sim.softmax(dim=-1)
            attn = self.attn_dropout(attn)
            out = einsum('b h i j, b h j d -> b h i d', attn, v)
        
        # Merge heads
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerDecoderLayer(nn.Module):
    """TransformerDecoder layer: Self-Attention → Cross-Attention → FFN
    with LayerNorm pre-normalization and residual connections
    """
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        use_flash_attn = True
    ):
        super().__init__()
        self.gradient_checkpointing = False
        # 1. Self-Attention (reuse Attention from transformer.py)
        from ..modules.transformer import Attention
        self.self_attn = Attention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            dropout=attn_dropout,
            use_flash_attn=use_flash_attn
        )
        
        # 2. Cross-Attention
        self.cross_attn = CrossAttentionLayer(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            dropout=attn_dropout,
            use_flash_attn=use_flash_attn
        )
        
        # 3. FFN
        self.ffn = FeedForward(
            dim=dim,
            mult=ff_mult,
            dropout=ff_dropout
        )
    
    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
    def forward(self, x, encoder_hidden_states, self_attn_mask=None, cross_attn_mask_q=None, cross_attn_mask_kv=None):
        """
        Args:
            x: [batch, len, dim] - decoder input
            encoder_hidden_states: [batch, enc_len, dim] - encoder output
            self_attn_mask: [batch, len] - self-attention mask
            cross_attn_mask_q: [batch, len] - cross-attention query mask
            cross_attn_mask_kv: [batch, enc_len] - cross-attention key/value mask
        """
        if self.gradient_checkpointing and self.training:
            # Use checkpoint for memory efficiency during training
            x = torch.utils.checkpoint.checkpoint(
                self.self_attn, x, None, self_attn_mask, use_reentrant=False
            ) + x
            
            def cross_attn_wrapper(x, enc, mask_q, mask_kv):
                return self.cross_attn(x, enc, mask_q=mask_q, mask_kv=mask_kv)
            
            cross_out = torch.utils.checkpoint.checkpoint(
                cross_attn_wrapper, x, encoder_hidden_states, cross_attn_mask_q, cross_attn_mask_kv,
                use_reentrant=False
            )
            x = cross_out + x
            
            x = torch.utils.checkpoint.checkpoint(self.ffn, x, use_reentrant=False) + x
        else:
            # 1. Self-Attention + Residual
            x = self.self_attn(x, mask=self_attn_mask) + x
            
            # 2. Cross-Attention + Residual
            cross_out = self.cross_attn(x, encoder_hidden_states, mask_q=cross_attn_mask_q, mask_kv=cross_attn_mask_kv)
            x = cross_out + x
            
            # 3. FFN + Residual
            x = self.ffn(x) + x
        
        return x
