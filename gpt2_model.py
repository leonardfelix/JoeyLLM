import torch
import torch.nn as nn
import math

class GPT2Config:
    """
    Configuration for the GPT-2 model, allowing easy switching between different sizes.

    Args:
        model_size (str): Choose from 'small', 'medium', 'large', 'xl'.
    """
    model_configs = {
        "small":  {"vocab_size": 50257, "max_seq_len": 1024, "hidden_dim": 768,  "num_heads": 12, "num_layers": 12, "dropout": 0.1},
        "medium": {"vocab_size": 50257, "max_seq_len": 1024, "hidden_dim": 1024, "num_heads": 16, "num_layers": 24, "dropout": 0.1},
        "large":  {"vocab_size": 50257, "max_seq_len": 1024, "hidden_dim": 1280, "num_heads": 20, "num_layers": 36, "dropout": 0.1},
        "xl":     {"vocab_size": 50257, "max_seq_len": 1024, "hidden_dim": 1600, "num_heads": 25, "num_layers": 48, "dropout": 0.1}
    }

    def __init__(self, model_size="small"):
        assert model_size in self.model_configs, f"Invalid model size: {model_size}. Choose from {list(self.model_configs.keys())}"
        self.__dict__.update(self.model_configs[model_size])  # Load the configuration into instance variables
        self.model_size = model_size

class MultiHeadSelfAttention(nn.Module):
    """ Implements Multi-Head Self-Attention. """
    
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        assert hidden_dim % num_heads == 0, "Hidden size must be divisible by number of heads."
        
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)

        return self.out_proj(attn_output)

class TransformerBlock(nn.Module):
    """ Implements a single Transformer block (Attention + Feedforward). """

    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config.hidden_dim, config.num_heads, config.dropout)
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        x = self.norm1(x + self.attention(x))
        return self.norm2(x + self.ffn(x))

class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.hidden_dim)
        # Disable bias to match HF's lm_head
        self.output_layer = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        token_embeddings = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        position_embeddings = self.position_embedding(positions)
        x = token_embeddings + position_embeddings

        for block in self.blocks:
            x = block(x)

        return self.output_layer(self.norm(x))
