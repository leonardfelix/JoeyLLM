"""
Custom GPT-2 Implementation (Matching Hugging Face)
 
This script implements GPT-2 architecture identical to Hugging Face's version.
Contains extensive comments for educational purposes.
 
Key Features:
- Full HF compatibility for weight loading
- Layer names match HF's parameter names
- Detailed documentation for each component
- Clean, modular architecture
"""
 
import torch
import torch.nn as nn
import math
 
class GPT2Config:
    """
    Configuration class for GPT-2 model with preset sizes
    """
    model_configs = {
        "tiny":    {"vocab_size": 50257, "max_seq_len": 1024, "hidden_dim": 512,  "num_heads": 8,  "num_layers": 2,  "dropout": 0.1},
        "small":   {"vocab_size": 50257, "max_seq_len": 1024, "hidden_dim": 768,  "num_heads": 12, "num_layers": 12, "dropout": 0.1},
        "medium":  {"vocab_size": 50257, "max_seq_len": 1024, "hidden_dim": 1024, "num_heads": 16, "num_layers": 24, "dropout": 0.1},
        "large":   {"vocab_size": 50257, "max_seq_len": 1024, "hidden_dim": 1280, "num_heads": 20, "num_layers": 36, "dropout": 0.1},
        "xl":      {"vocab_size": 50257, "max_seq_len": 1024, "hidden_dim": 1600, "num_heads": 25, "num_layers": 48, "dropout": 0.1}
    }

    def __init__(self, model_size=None, **kwargs):
        """
        Initialize configuration with either:
        - A predefined model size (tiny/small/medium/large/xl)
        - Custom parameters via keyword arguments
        
        Example custom config:
        GPT2Config(hidden_dim=512, num_heads=8, num_layers=4)
        """
        if model_size:
            assert model_size in self.model_configs, f"Invalid model size: {model_size}"
            self.__dict__.update(self.model_configs[model_size])
        else:
            # Default values
            defaults = {
                "vocab_size": 50257,
                "max_seq_len": 1024,
                "hidden_dim": 768,
                "num_heads": 12,
                "num_layers": 12,
                "dropout": 0.1
            }
            # Update with provided kwargs
            defaults.update(kwargs)
            self.__dict__.update(defaults)
            
        self.model_size = model_size  # Track if using predefined size

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_dim // self.num_heads
        
        self.c_attn = nn.Linear(self.hidden_dim, 3 * self.hidden_dim)
        self.c_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # Register causal mask as buffer
        self.register_buffer("causal_mask", None)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Create fresh causal mask for current sequence length
        device = x.device
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1
        )
        
        # Generate causal mask if not exists
        if self.causal_mask is None or self.causal_mask.size(0) != seq_len:
            self.causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
                diagonal=1
            )

        # Project QKV
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.hidden_dim, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores with causal masking
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        # attn_scores = attn_scores.masked_fill(self.causal_mask[:seq_len, :seq_len], float('-inf'))
        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        context = attn_probs @ v
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.c_proj(context)
 
class TransformerBlock(nn.Module):
    """
    GPT-2 Transformer Block (identical to HF implementation)
    
    Contains:
    1. LayerNorm -> MultiHeadAttention -> Residual Connection
    2. LayerNorm -> FeedForward -> Residual Connection
    """
    def __init__(self, config):
        super().__init__()
        # Attention components
        self.ln_1 = nn.LayerNorm(config.hidden_dim)
        self.attn = MultiHeadAttention(config)
        
        # Feed-forward components
        self.ln_2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),  # HF's 'c_fc'
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim),  # HF's 'c_proj'
            nn.Dropout(config.dropout)
        )
 
    def forward(self, x):
        # Attention phase
        x = x + self.attn(self.ln_1(x))
        # Feed-forward phase
        x = x + self.mlp(self.ln_2(x))
        return x
 
class GPT2Model(nn.Module):
    """
    Full GPT-2 Model Architecture
    
    Structure matches Hugging Face's implementation exactly:
    - Token and positional embeddings
    - Stack of transformer blocks
    - Final layer normalization
    - Language modeling head
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
 
        # Embedding layers (names match HF)
        self.wte = nn.Embedding(config.vocab_size, config.hidden_dim)  # Token embeddings
        self.wpe = nn.Embedding(config.max_seq_len, config.hidden_dim)  # Position embeddings
        
        # Transformer blocks (named 'h' like HF implementation)
        self.h = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        
        # Final layer norm (named 'ln_f' like HF)
        self.ln_f = nn.LayerNorm(config.hidden_dim)
        
        # Language modeling head (no bias like HF)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
 
    def forward(self, input_ids):
        """
        Forward pass through GPT-2
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            
        Returns:
            logits: Prediction scores [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.size()
        
        # Create position IDs [0, 1, 2, ..., seq_len-1]
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        
        # Get embeddings
        token_embeds = self.wte(input_ids)
        pos_embeds = self.wpe(pos_ids)
        x = token_embeds + pos_embeds
 
        # Process through transformer blocks
        for block in self.h:
            x = block(x)
 
        # Final layer norm
        x = self.ln_f(x)
        
        # Language model head
        return self.lm_head(x)
 
# --------------------------
# Helper Function for Weight Loading
# --------------------------
 
def load_hf_weights(custom_model, hf_model):
    """
    Load weights from Hugging Face model into custom GPT-2
    with proper weight transposition for linear layers
    """
    hf_weights = hf_model.state_dict()
    custom_weights = custom_model.state_dict()

    # Direct name mappings (most parameters match)
    name_mapping = {
        'transformer.wte.weight': 'wte.weight',
        'transformer.wpe.weight': 'wpe.weight',
        'transformer.ln_f.weight': 'ln_f.weight',
        'transformer.ln_f.bias': 'ln_f.bias',
        'lm_head.weight': 'lm_head.weight'
    }

    # Copy directly matching parameters
    for hf_name, custom_name in name_mapping.items():
        if hf_name in hf_weights:
            custom_weights[custom_name] = hf_weights[hf_name]

    # Copy transformer blocks with transposition
    for layer_idx in range(custom_model.config.num_layers):
        # LayerNorm parameters (no transposition needed)
        custom_weights[f'h.{layer_idx}.ln_1.weight'] = hf_weights[f'transformer.h.{layer_idx}.ln_1.weight']
        custom_weights[f'h.{layer_idx}.ln_1.bias'] = hf_weights[f'transformer.h.{layer_idx}.ln_1.bias']
        custom_weights[f'h.{layer_idx}.ln_2.weight'] = hf_weights[f'transformer.h.{layer_idx}.ln_2.weight']
        custom_weights[f'h.{layer_idx}.ln_2.bias'] = hf_weights[f'transformer.h.{layer_idx}.ln_2.bias']

        # Attention projections (transpose weights)
        # c_attn: [768, 2304] -> [2304, 768]
        custom_weights[f'h.{layer_idx}.attn.c_attn.weight'] = hf_weights[f'transformer.h.{layer_idx}.attn.c_attn.weight'].T
        custom_weights[f'h.{layer_idx}.attn.c_attn.bias'] = hf_weights[f'transformer.h.{layer_idx}.attn.c_attn.bias']
        
        # c_proj: [768, 768] -> [768, 768] (no transpose needed for square matrix)
        custom_weights[f'h.{layer_idx}.attn.c_proj.weight'] = hf_weights[f'transformer.h.{layer_idx}.attn.c_proj.weight'].T
        custom_weights[f'h.{layer_idx}.attn.c_proj.bias'] = hf_weights[f'transformer.h.{layer_idx}.attn.c_proj.bias']

        # MLP parameters (transpose weights)
        # c_fc: [768, 3072] -> [3072, 768]
        custom_weights[f'h.{layer_idx}.mlp.0.weight'] = hf_weights[f'transformer.h.{layer_idx}.mlp.c_fc.weight'].T
        custom_weights[f'h.{layer_idx}.mlp.0.bias'] = hf_weights[f'transformer.h.{layer_idx}.mlp.c_fc.bias']
        
        # c_proj: [3072, 768] -> [768, 3072]
        custom_weights[f'h.{layer_idx}.mlp.2.weight'] = hf_weights[f'transformer.h.{layer_idx}.mlp.c_proj.weight'].T
        custom_weights[f'h.{layer_idx}.mlp.2.bias'] = hf_weights[f'transformer.h.{layer_idx}.mlp.c_proj.bias']

    # Load weights into custom model
    custom_model.load_state_dict(custom_weights)
    return custom_model