import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def sinusoidal_pos_encoding(seq_len, d_model, device):
    pe = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, T, d)


class VariableWiseEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.conv1d = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, T, N, F) -> (B*N, T, F)
        B, T, N, F = x.shape
        x_flat = x.permute(0, 2, 1, 3).contiguous().view(B * N, T, F)

        # (B*N, T, F) -> (B*N, F, T) -> (B*N, d, T)
        x_flat = x_flat.transpose(1, 2)
        u = self.conv1d(x_flat)
        u = self.bn(u)
        u = self.relu(u)

        # (B*N, d, T) -> (B*N, T, d) -> (B, T, N, d)
        u = u.transpose(1, 2)  # 64*8,200,64)
        u = u.reshape(B, N, T, -1).transpose(1, 2)  # 64,200,8,64

        return u


class TemporalEncoder(nn.Module):

    def __init__(self, d_model, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, u):
        # u: (B, T, N, d)
        B, T, N, d = u.shape
        u_flat = u.permute(0, 2, 1, 3).contiguous().view(B * N, T, d)
        pe = sinusoidal_pos_encoding(T, d, device=u.device)
        u_flat = u_flat + pe  #

        # Apply shared transformer to each sequence
        h_flat = self.transformer(u_flat)

        # h_flat (b*n, t, d) Reshape back to (B, T, N, d)
        h = h_flat.reshape(B, N, T, -1).transpose(1, 2)  # 64,200,8,64

        return h

class EdgeScoring(nn.Module):
    def __init__(self, d_model, use_bilinear=True):
        super().__init__()
        self.use_bilinear = use_bilinear
        if use_bilinear:
            self.W_e = nn.Parameter(torch.randn(d_model, d_model))
            nn.init.xavier_uniform_(self.W_e)
        else:

            pass

    def forward(self, h):
        # h: (B, T, N, d)
        B, T, N, d = h.shape

        # Reshape to (B*T, N, d)
        h_bt = h.reshape(B * T, N, d)

        if self.use_bilinear:
            # (B*T, N, d) @ (d, d) @ (B*T, d, N) -> (B*T, N, N)
            e = torch.bmm(h_bt, self.W_e.unsqueeze(0).expand(B * T, -1, -1))
            e = torch.bmm(e, h_bt.transpose(1, 2))
        else:
            e = torch.bmm(h_bt, h_bt.transpose(1, 2))
        # print(e)
        # Apply sigmoid to get edge weights
        a = torch.sigmoid(e)  # [0,1]
        a = a.reshape(B, T, N, N)

        return a


class DynamicGNN(nn.Module):

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.W_msg = nn.Linear(d_model, d_model)
        self.W_res = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, a):
        # h: (B, T, N, d), a: (B, T, N, N)
        B, T, N, d = h.shape

        # Reshape to (B*T, N, d) and (B*T, N, N)
        h_bt = h.reshape(B * T, N, d)
        a_bt = a.reshape(B * T, N, N)

        # Message passing: H' = σ(A @ (H W_msg) + H W_res)
        h_msg = self.W_msg(h_bt)  # (B*T, N, d)
        h_res = self.W_res(h_bt)  # (B*T, N, d)

        # Graph convolution
        h_updated = torch.bmm(a_bt, h_msg) + h_res
        h_updated = F.relu(h_updated)
        h_updated = self.dropout(h_updated)

        # Reshape back to (B, T, N, d)
        h_out = h_updated.view(B, T, N, d)

        return h_out

class MainEffectHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, h):
        # h: (B, T, N, d)
        B, T, N, d = h.shape

        # Apply linear transformation to each node at each time
        m = self.linear(h)  # (B, T, N, 1)
        m = m.squeeze(-1)  # (B, T, N)

        return m

class EdgeMask(nn.Module):
    def __init__(self, d_model, hidden_dim=32, use_gumbel=False, tau=0.5):
        super().__init__()
        self.use_gumbel = use_gumbel
        self.tau = tau

        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h, I_full):
        # h: (B,T,N,d), I_full: (B,T,N,N)
        B,T,N,d = h.shape
        h_i = h.unsqueeze(3).expand(B,T,N,N,d)  # (B,T,N,N,d)
        h_j = h.unsqueeze(2).expand(B,T,N,N,d)  # (B,T,N,N,d)
        pair_feat = torch.cat([h_i,h_j], dim=-1)  # (B,T,N,N,2d)

        logits = self.mlp(pair_feat).squeeze(-1)  # (B,T,N,N)

        if self.use_gumbel:
            # Gumbel-Softmax trick
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)+1e-9)+1e-9)
            M = torch.sigmoid((logits + gumbel_noise) / self.tau)
        else:
            M = torch.sigmoid(logits)
        I_sparse = I_full * M
        return I_sparse, M




class InteractionHeadGated(nn.Module):

    def __init__(self, d_model, r_factor=16, att_dim=16):
        super().__init__()
        self.r = r_factor
        self.P = nn.Linear(d_model, r_factor, bias=False)
        self.Q = nn.Linear(r_factor, att_dim, bias=False)
        self.K = nn.Linear(r_factor, att_dim, bias=False)

    def forward(self, h):
        B, T, N, d = h.shape
        z = self.P(h)  # (B,T,N,r)
        z_bt = z.reshape(B * T, N, self.r)  # (B*T,N,r)
        base = torch.bmm(z_bt, z_bt.transpose(1, 2))  # (B*T,N,N)
        q = self.Q(z).reshape(B * T, N, -1)  # (B*T,N,a)
        k = self.K(z).reshape(B * T, N, -1)  # (B*T,N,a)
        att = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])  # (B*T,N,N)
        gate = torch.sigmoid(att)  # [0,1]

        s = base * gate
        return s.view(B, T, N, N)
        # return i


class TemporalAggregator(nn.Module):

    def __init__(self, prediction_horizon):
        super().__init__()
        H = prediction_horizon
        self.beta = nn.Parameter(torch.zeros(H))
        self.bias = nn.Parameter(torch.zeros(H))

    def forward(self, y_t):
        B, T = y_t.shape
        t_idx = torch.arange(T, device=y_t.device, dtype=y_t.dtype)  # 0..T-1
        logits = self.beta.view(-1, 1) * t_idx.view(1, -1) + self.bias.view(-1, 1)  # (H,T)
        W = torch.softmax(logits, dim=-1)  # (H,T)
        preds = y_t @ W.t()  # (B,H)
        return preds, W


class DynamicInteractionModelCoupled(nn.Module):

    def __init__(self, params):
        super().__init__()
        self.prediction_horizon = params['prediction_horizon']
        self.N = params['feature_dimensionality']
        self.T=params['T']

        # 1) 编码与动态图
        self.variable_encoder = VariableWiseEncoder(params['input_dim'], params['hidden_dim']).cuda()
        self.temporal_encoder = TemporalEncoder(
            params['hidden_dim'], params['nhead'], params['num_layers'],
            params['dropout']).cuda()
        self.edge_scoring = EdgeScoring(params['hidden_dim'], params['use_bilinear']).cuda()
        self.dynamic_gnn = DynamicGNN(params['hidden_dim'], params['dropout']).cuda()
        self.edge_mask=EdgeMask(params['hidden_dim'],hidden_dim=32,use_gumbel=True,
    tau=1.0   ).cuda()
        self.main_effect_head = MainEffectHead(params['hidden_dim']).cuda()
        self.interaction_head = InteractionHeadGated(params['hidden_dim'],
                                                        r_factor=params['r_factor'],
                                                        att_dim=min(32, params['r_factor'])).cuda()

        P = (self.N * (self.N - 1)) // 2
        self.w_main = nn.Parameter(
            torch.randn(self.N) * 0.5)
        self.w_int = nn.Parameter(torch.randn(P) * 2)
        nn.init.normal_(self.w_main, std=0.02)
        nn.init.normal_(self.w_int, std=0.02)

        self.time_agg = TemporalAggregator(self.prediction_horizon).cuda()
        self.scale_main = nn.Parameter(torch.tensor(1.0))
        self.scale_int = nn.Parameter(torch.tensor(1.0))

    def upper_flatten(self, I):
        B, T, N, _ = I.shape
        mask = torch.triu(torch.ones(N, N, device=I.device, dtype=torch.bool), diagonal=1)
        return I[..., mask]  # (B,T,P)

    def forward(self, x):
        # x: (B,T,N,F)
        B, T, N, F = x.shape
        u = self.variable_encoder(x)  # (B,T,N,d)
        h = self.temporal_encoder(u)  # (B,T,N,d)
        a = self.edge_scoring(h)  # (B,T,N,N)
        h_upd = self.dynamic_gnn(h, a)  # (B,T,N,d)

        # 主效应 & 交互（门控）
        m = self.main_effect_head(h_upd)  # (B,T,N)
        I_full = self.interaction_head(h_upd)  # (B,T,N,N)
        I_flat_full = self.upper_flatten(I_full)  # (B,T,P)
        I_sparse, M = self.edge_mask(h_upd, I_full)
        I_flat = self.upper_flatten(I_sparse)      # (B,T,P)
        y_main_t = torch.einsum('btn,n->bt', m,
                                self.w_main)
        y_int_t = torch.einsum('btp,p->bt', I_flat, self.w_int)
        y_int_t_full = torch.einsum('btp,p->bt', I_flat_full, self.w_int)

        y_t = y_main_t + y_int_t  # (B,T)
        y_t_orig = y_main_t + y_int_t_full  # (B,T) original (no mask)

        # 多步预测：时间聚合
        preds, W_time = self.time_agg(y_t)
        preds_orig, W_time = self.time_agg(y_t_orig)  # (B,H), (H,T)
        sparsity_loss = torch.mean(M)

        W_broadcast = W_time.unsqueeze(0)  # (1,H,T)
        contrib_main_htn = (W_broadcast.unsqueeze(-1) *
                            self.w_main.view(1, 1, 1, -1) * m.unsqueeze(1))  # (B,H,T,N)
        contrib_main_rank = torch.softmax(contrib_main_htn, dim=-1)
        # （24,48,16）
        contrib_int_htp = (W_broadcast.unsqueeze(-1) *
                           self.w_int.view(1, 1, 1, -1) * I_flat.unsqueeze(1))  # (B,H,T,P)
        contrib_int_rank = torch.softmax(contrib_int_htp, dim=-1)

        return {
            'prediction': preds,  # (B,H)
            'preds_orig': preds_orig,
            'main_effects': m,  # (B,T,N)
            'interactions': I_full,  # (B,T,N,N)
            'interactions_flat': I_flat,  # (B,T,P)
            # 'edge_weights': a,                 # (B,T,N,N)
            'node_embeddings': h_upd,  # (B,T,N,d)
            'y_t': y_t,  # (B,T) 逐时刻原子输出
            'y_main_t': y_main_t,  # (B,T)
            'y_int_t': y_int_t,  # (B,T)
            'time_weights': W_time,  # (H,T)
            'contrib_main_htn': contrib_main_htn,
            'contrib_main_rank':contrib_main_rank,# (B,H,T,N)
            'contrib_int_htp': contrib_int_htp,
            'contrib_int_rank':contrib_int_rank,# (B,H,T,P)
            # 'aux_preds': preds_aux             # (B,T-1)
            'sparsity_loss': sparsity_loss
        }


# ---------- 小工具：正弦位置编码 ----------
# -------------------------
# Sinusoidal Positional Encoding
# -------------------------

# =========================
# 1) Transformer Baseline
# =========================
class Transformer(nn.Module):
    """Simple Transformer for multi-step forecasting."""

    def __init__(self, params):
        super().__init__()
        self.prediction_horizon = params['prediction_horizon']
        self.n_vars = params['feature_dimensionality']
        self.input_dim = params['input_dim']
        in_size = self.n_vars * self.input_dim

        self.input_proj = nn.Linear(in_size, params['hidden_dim'])
        enc_layer = nn.TransformerEncoderLayer(
            d_model=params['hidden_dim'],
            nhead=params['nhead'],
            dim_feedforward=params['hidden_dim'] * 4,
            dropout=params['dropout'],
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=params['num_layers'])
        self.head = nn.Linear(params['hidden_dim'], self.prediction_horizon)
        self.norm = nn.LayerNorm(params['hidden_dim'])
        self.dropout = nn.Dropout(params['dropout'])

    def forward(self, x):
        # x: (B, T, N, F)
        B, T, N, F = x.shape
        assert N == self.n_vars and F == self.input_dim

        x_flat = x.reshape(B, T, N * F)  # (B, T, N*F)
        h = self.input_proj(x_flat)  # (B, T, hidden)
        h = self.dropout(self.norm(h))

        # Add positional encoding
        pe = sinusoidal_pos_encoding(T, h.size(-1), x.device)
        h = h + pe

        # Transformer encoding
        h_enc = self.encoder(h)  # (B, T, hidden)
        h_last = h_enc[:, -1, :]  # (B, hidden)
        preds = self.head(h_last)  # (B, H)

        return preds  # 直接返回预测张量

# =====================
# 2) LSTM Baseline
# =====================
class LSTM(nn.Module):
    """Simple LSTM for multi-step forecasting."""

    def __init__(self, params):
        super().__init__()
        self.prediction_horizon = params['prediction_horizon']
        self.n_vars = params['feature_dimensionality']
        self.input_dim = params['input_dim']

        in_size = self.n_vars * self.input_dim
        self.lstm = nn.LSTM(
            input_size=in_size,
            hidden_size=params['hidden_dim'],
            num_layers=params['num_layers'],
            dropout=params['dropout'] if params['num_layers'] > 1 else 0.0,
            batch_first=True
        )
        self.head = nn.Linear(params['hidden_dim'], self.prediction_horizon)

    def forward(self, x):
        B, T, N, F = x.shape
        assert N == self.n_vars and F == self.input_dim

        x_flat = x.reshape(B, T, N * F)
        h_seq, (h_n, c_n) = self.lstm(x_flat)
        h_last = h_seq[:, -1, :]
        preds = self.head(h_last)
        return preds


# =====================
# 3) GRU Baseline
# =====================
class GRU(nn.Module):
    """Simple GRU for multi-step forecasting."""

    def __init__(self, params):
        super().__init__()
        self.prediction_horizon = params['prediction_horizon']
        self.n_vars = params['feature_dimensionality']
        self.input_dim = params['input_dim']

        in_size = self.n_vars * self.input_dim
        self.gru = nn.GRU(
            input_size=in_size,
            hidden_size=params['hidden_dim'],
            num_layers=params['num_layers'],
            dropout=params['dropout'] if params['num_layers'] > 1 else 0.0,
            batch_first=True
        )
        self.head = nn.Linear(params['hidden_dim'], self.prediction_horizon)

    def forward(self, x):
        B, T, N, F = x.shape
        assert N == self.n_vars and F == self.input_dim

        x_flat = x.reshape(B, T, N * F)
        h_seq, h_n = self.gru(x_flat)
        h_last = h_seq[:, -1, :]
        preds = self.head(h_last)
        return preds
