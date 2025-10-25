import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# Positional Encoding
# ----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # shape (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

# ----------------------
# ProbSparse Attention
# ----------------------
class ProbAttention(nn.Module):
    """Simplified version of ProbSparse Attention"""
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        super().__init__()
        self.mask_flag = mask_flag
        self.factor = factor
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        # x: (B, L, D)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (queries.size(-1) ** 0.5)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, values)
        return out, attn

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.attention = attention
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, attn_mask=None):
        B, L, D = x.shape
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        out, _ = self.attention(q, k, v, attn_mask)
        out = self.out(out)
        return out

# ----------------------
# Encoder Layer
# ----------------------
class EncoderLayer(nn.Module):
    def __init__(self, attention_layer, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super().__init__()
        self.attn = attention_layer
        self.linear1 = nn.Linear(d_model, d_ff or d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff or d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, attn_mask=None):
        new_x = self.attn(x, attn_mask)
        x = x + new_x
        x = self.norm1(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + x2
        x = self.norm2(x)
        return x

class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x = layer(x, attn_mask)
        if self.norm:
            x = self.norm(x)
        return x, None

# ----------------------
# Informer Model
# ----------------------
class Informer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pred_len = cfg['prediction_horizon']
        self.n_vars = cfg['feature_dimensionality']

        input_dim = cfg['input_dim'] if isinstance(cfg['input_dim'], int) else cfg['input_dim'].get('value', 16)
        hidden_dim = cfg['hidden_dim'] if isinstance(cfg['hidden_dim'], int) else cfg['hidden_dim'].get('value', 64)

        self.input_proj = nn.Linear(input_dim * self.n_vars, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)

        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(ProbAttention(mask_flag=False, factor=5), hidden_dim, cfg['nhead']),
                d_model=hidden_dim,
                d_ff=hidden_dim*4,
                dropout=cfg['dropout'],
                activation='relu'
            ) for _ in range(cfg['num_layers'])
        ], norm_layer=nn.LayerNorm(hidden_dim))

        self.head = nn.Linear(hidden_dim, self.pred_len)

    def forward(self, x):
        # x: (B, T, N, F)
        B, T, N, F = x.shape
        x_flat = x.reshape(B, T, N * F)
        h = self.input_proj(x_flat)
        h = self.pos_encoding(h)
        enc_out, _ = self.encoder(h)
        h_last = enc_out[:, -1, :]
        preds = self.head(h_last)
        return preds


class SeriesDecomposition(nn.Module):
    """简单移动平均分解"""
    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AvgPool1d(kernel_size, stride=1, padding=kernel_size//2, count_include_pad=False)

    def forward(self, x):
        # x: (B, T, D)
        trend = self.avg_pool(x.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend

class Autoformer(nn.Module):
    """Autoformer for multi-step forecasting."""

    def __init__(self, cfg):
        super().__init__()
        self.pred_len = cfg['prediction_horizon']
        self.n_vars = cfg['feature_dimensionality']

        # 处理可能为字典的配置项
        input_dim = cfg['input_dim'] if isinstance(cfg['input_dim'], int) else cfg['input_dim'].get('value', 16)
        hidden_dim = cfg['hidden_dim'] if isinstance(cfg['hidden_dim'], int) else cfg['hidden_dim'].get('value', 64)

        self.input_proj = nn.Linear(input_dim * self.n_vars, hidden_dim)

        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_dim)

        # Series decomposition
        self.decomp = SeriesDecomposition(kernel_size=25)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=cfg['nhead'],
            dim_feedforward=hidden_dim * 4,
            dropout=cfg['dropout'],
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg['num_layers'])

        # Prediction head
        self.head = nn.Linear(hidden_dim, self.pred_len)

    def forward(self, x):
        # x: (B, T, N, F)
        B, T, N, F = x.shape
        x_flat = x.reshape(B, T, N * F)
        h = self.input_proj(x_flat)

        # add positional encoding
        h = h + self.pos_encoding(h)

        # series decomposition
        seasonal, trend = self.decomp(h)

        # encoder
        h_enc = self.encoder(seasonal)

        # add trend back
        h_out = h_enc + trend

        # take last time step
        h_last = h_out[:, -1, :]
        preds = self.head(h_last)
        return preds


