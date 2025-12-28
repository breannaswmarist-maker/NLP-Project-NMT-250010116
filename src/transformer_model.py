import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. 基础组件: Normalization & Embedding
# ==========================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    这是 LayerNorm 的一种变体，常用于现代 LLM (如 LLaMA)，计算更高效。
    """
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.g = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # RMSNorm 不减去均值，只除以均方根
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / (norm + self.eps) * self.g

class PositionalEncoding(nn.Module):
    """
    绝对位置编码 (Sinusoidal Absolute Position Embedding)
    """
    def __init__(self, emb_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为 buffer，不是参数，不参与更新
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [batch, seq_len, emb_dim]
        return x + self.pe[:, :x.size(1), :]

# ==========================================
# 2. 核心 Transformer 架构
# ==========================================

class TransformerNMT(nn.Module):
    def __init__(self, 
                 src_vocab_size, 
                 trg_vocab_size, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 emb_dim=512, 
                 n_heads=8, 
                 n_layers=3, 
                 ffn_dim=2048, 
                 dropout=0.1, 
                 norm_type='layernorm',  # 'layernorm' vs 'rmsnorm'
                 pos_type='absolute',    # 'absolute' (sinusoidal) vs 'learnable'
                 max_len=1000):
        super().__init__()
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.emb_dim = emb_dim
        
        # 1. Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, emb_dim)
        self.trg_embedding = nn.Embedding(trg_vocab_size, emb_dim)
        
        # 2. Position Embedding Strategy (Ablation)
        if pos_type == 'absolute':
            self.pos_encoder = PositionalEncoding(emb_dim, max_len)
            self.pos_decoder = PositionalEncoding(emb_dim, max_len)
        elif pos_type == 'learnable':
            self.pos_encoder = nn.Embedding(max_len, emb_dim)
            self.pos_decoder = nn.Embedding(max_len, emb_dim)
        else:
            raise ValueError("pos_type must be 'absolute' or 'learnable'")
            
        self.pos_type = pos_type
            
        # 3. Normalization Strategy (Ablation)
        # 我们自定义 LayerNorm 工厂函数，传给 PyTorch 的 Transformer 组件
        norm_layer = nn.LayerNorm if norm_type == 'layernorm' else RMSNorm
        
        # 4. Transformer Core
        # 使用 PyTorch 官方高度优化的组件，但注入自定义的 Norm
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, 
                                                   dim_feedforward=ffn_dim, dropout=dropout, 
                                                   batch_first=True, norm_first=True) # Pre-Norm 更加稳定
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, norm=norm_layer(emb_dim))
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=n_heads, 
                                                   dim_feedforward=ffn_dim, dropout=dropout, 
                                                   batch_first=True, norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers, norm=norm_layer(emb_dim))
        
        self.fc_out = nn.Linear(emb_dim, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def create_mask(self, src, trg):
        # src: [batch, src_len]
        # trg: [batch, trg_len]
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        trg_len = trg.shape[1]
        
        # src_mask: 屏蔽 padding [batch, 1, 1, src_len] -> 扩展给 multi-head attention
        # PyTorch Transformer 需要的 mask 格式：
        # key_padding_mask: (True 被屏蔽) [batch, seq_len]
        src_key_padding_mask = (src == self.src_pad_idx)
        trg_key_padding_mask = (trg == self.trg_pad_idx)
        
        # trg_mask: 屏蔽 padding + 屏蔽未来时刻 (Casual Mask)
        # PyTorch 这里的 mask 是 additive mask: 0 是保留, -inf 是屏蔽
        trg_mask = nn.Transformer.generate_square_subsequent_mask(trg_len).to(src.device)
        
        return src_key_padding_mask, trg_key_padding_mask, trg_mask

    def forward(self, src, trg):
        # src: [batch, src_len]
        # trg: [batch, trg_len]
        
        src_key_padding_mask, trg_key_padding_mask, trg_mask = self.create_mask(src, trg)
        
        # Embedding
        src_emb = self.src_embedding(src) * math.sqrt(self.emb_dim)
        trg_emb = self.trg_embedding(trg) * math.sqrt(self.emb_dim)
        
        # Add Positional Encoding
        if self.pos_type == 'learnable':
            # Create simple 0...len indices
            src_positions = torch.arange(0, src.shape[1]).unsqueeze(0).repeat(src.shape[0], 1).to(src.device)
            trg_positions = torch.arange(0, trg.shape[1]).unsqueeze(0).repeat(trg.shape[0], 1).to(trg.device)
            src_emb = src_emb + self.pos_encoder(src_positions)
            trg_emb = trg_emb + self.pos_decoder(trg_positions)
        else:
            # Sinusoidal
            src_emb = self.pos_encoder(src_emb)
            trg_emb = self.pos_decoder(trg_emb)
            
        src_emb = self.dropout(src_emb)
        trg_emb = self.dropout(trg_emb)
        
        # Transformer Process
        memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        
        output = self.transformer_decoder(trg_emb, memory, 
                                          tgt_mask=trg_mask,
                                          tgt_key_padding_mask=trg_key_padding_mask,
                                          memory_key_padding_mask=src_key_padding_mask)
        
        return self.fc_out(output)