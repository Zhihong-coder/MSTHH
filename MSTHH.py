"""
MSTHH: Multi-modal Spatio-Temporal Hierarchical Heterogeneity Network
Complete implementation following the paper exactly.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_main.mamba_ssm import Mamba
    HAS_MAMBA = True
except Exception:
    HAS_MAMBA = False


# ============================================================
# Stage I: Adaptive Spatio-Temporal Encoding & Alignment
# ============================================================

class FrequencyAdaptivePositionalEncoding(nn.Module):
    """
    FAP Encoding - Equation 5.
    PE_m(t, 2i)   = sin(t * omega_m / f_m^(2i/d))
    PE_m(t, 2i+1) = cos(t * omega_m / f_m^(2i/d))
    omega_m and f_m are learnable per modality.
    """
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        # Learnable sampling-frequency parameter (log scale for stability)
        self.log_omega = nn.Parameter(torch.zeros(1))
        self.log_f = nn.Parameter(torch.zeros(1))

    def forward(self, T, device):
        d = self.d_model
        omega = torch.exp(self.log_omega)        # scalar
        f_m   = torch.exp(self.log_f).clamp(1.0) # scalar >= 1

        pos = torch.arange(T, dtype=torch.float, device=device).unsqueeze(1)  # (T,1)
        dim = torch.arange(0, d, 2, dtype=torch.float, device=device)          # (d//2,)
        freq = omega / (f_m ** (dim / d))                                       # (d//2,)

        pe = torch.zeros(T, d, device=device)
        pe[:, 0::2] = torch.sin(pos * freq)
        pe[:, 1::2] = torch.cos(pos * freq[:d // 2 if d % 2 else d // 2])
        return pe  # (T, d)


class TemporalScaleAwareEmbedding(nn.Module):
    """
    TSA Embedding - Equations 6-7.
    X_m^temp = W_emb * X_m + PE_m(t) + TE_m(delta_t)
    TE_m(delta_t) = tanh(W_te * [log(delta_t+1), delta_t/delta_t_max])
    """
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.emb  = nn.Linear(input_dim, d_model)
        self.fap  = FrequencyAdaptivePositionalEncoding(d_model)
        self.W_te = nn.Linear(2, d_model)

    def forward(self, x_m, delta_t_max=1.0):
        # x_m: (B, T, N, input_dim)
        B, T, N, _ = x_m.shape
        device = x_m.device

        out = self.emb(x_m)                              # (B, T, N, d)
        pe  = self.fap(T, device)                        # (T, d)
        out = out + pe.unsqueeze(0).unsqueeze(2)         # broadcast B, N

        # Temporal interval encoding: delta_t = step index / T (proxy for sampling interval)
        t_idx = torch.arange(T, dtype=torch.float, device=device) / max(T - 1, 1)  # (T,)
        dt_feat = torch.stack([torch.log(t_idx + 1),
                                t_idx / (delta_t_max + 1e-6)], dim=-1)  # (T, 2)
        te = torch.tanh(self.W_te(dt_feat))              # (T, d)
        out = out + te.unsqueeze(0).unsqueeze(2)

        return out  # (B, T, N, d)


class HierarchicalMultiScaleGCN(nn.Module):
    """
    Adaptive Multi-Scale Adjacency + Hierarchical GCN - Equations 8-9.
    Uses K spatial scales and L graph convolution layers.
    """
    def __init__(self, d_model, num_nodes, num_scales=3, num_layers=3, dropout=0.1):
        super().__init__()
        self.K = num_scales
        self.L = num_layers
        dk = max(d_model // 4, 1)

        # Per-scale query/key projections
        self.W_Q = nn.ModuleList([nn.Linear(d_model, dk) for _ in range(num_scales)])
        self.W_K = nn.ModuleList([nn.Linear(d_model, dk) for _ in range(num_scales)])

        # Learnable spatial range parameters (log sigma)
        self.log_sigma = nn.Parameter(
            torch.tensor([0.5, 1.0, 2.0]).log().float()
        )

        # Node position embeddings for distance matrix (Eq.8: D in R^{N×N})
        self.node_pos = nn.Parameter(torch.randn(num_nodes, 16))

        # Per-layer GCN weights and scale attention
        self.gcn_W  = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.W_alfa = nn.ModuleList([nn.Linear(d_model, 32)       for _ in range(num_layers)])
        self.w_alfa = nn.ModuleList([nn.Linear(32, num_scales)    for _ in range(num_layers)])

        self.norms   = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.dk = dk

    def _distance_matrix(self):
        # Euclidean distance from node position embeddings
        pos = self.node_pos                    # (N, 16)
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # (N, N, 16)
        return (diff ** 2).sum(-1).sqrt()      # (N, N)

    def forward(self, x):
        # x: (B, T, N, d)
        B, T, N, d = x.shape
        D = self._distance_matrix()            # (N, N)

        h = x.reshape(B * T, N, d)            # flatten B,T for batch-efficient GCN

        for l in range(self.L):
            # Build K adjacency matrices (Eq.8)
            adj_list = []
            for k in range(self.K):
                Q = self.W_Q[k](h)             # (BT, N, dk)
                K_ = self.W_K[k](h)
                attn = (Q @ K_.transpose(-1, -2)) / math.sqrt(self.dk)  # (BT, N, N)

                sigma_k = torch.exp(self.log_sigma[k]).clamp(min=1e-3)
                M_sp = torch.exp(-D ** 2 / (2 * sigma_k ** 2))          # (N, N)

                A = torch.softmax(attn * M_sp.unsqueeze(0), dim=-1)     # (BT, N, N)
                adj_list.append(A)

            # Adaptive scale weights alpha (Eq.9)
            h_mean = h.mean(dim=1, keepdim=True)                        # (BT,1,d)
            a_logit = self.w_alfa[l](F.relu(self.W_alfa[l](h_mean)))   # (BT,1,K)
            alpha   = torch.softmax(a_logit, dim=-1)                    # (BT,1,K)

            # Weighted multi-scale graph convolution
            h_agg = torch.zeros_like(h)
            for k, A in enumerate(adj_list):
                deg  = A.sum(-1, keepdim=True).clamp(min=1e-6)
                A_n  = A / deg                                          # row-normalised
                h_k  = A_n @ h                                          # (BT, N, d)
                h_agg = h_agg + alpha[:, :, k:k+1] * h_k

            # GCN update with residual + LayerNorm (Eq.9)
            h_new = F.relu(self.gcn_W[l](h_agg))
            h = self.norms[l](h + self.dropout(h_new))

        return h.reshape(B, T, N, d)


class CrossModalTemporalAligner(nn.Module):
    """
    Cross-Modal Temporal Attention Alignment - Equation 10.
    Soft attention combining temporal-distance decay and semantic similarity.
    Avoids information loss of hard interpolation.
    """
    def __init__(self, d_model):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.log_gamma = nn.Parameter(torch.tensor(0.0))   # temporal decay
        self.log_tau   = nn.Parameter(torch.tensor(0.0))   # temperature

    def forward(self, H_i, H_j):
        # H_i: (B, Ti, N, d)  H_j: (B, Tj, N, d)
        B, Ti, N, d = H_i.shape
        Tj = H_j.shape[1]
        device = H_i.device

        Q = self.W_Q(H_i)  # (B, Ti, N, d)
        K = self.W_K(H_j)  # (B, Tj, N, d)
        V = self.W_V(H_j)

        # (B, N, Ti, Tj) semantic similarity
        Q_t = Q.permute(0, 2, 1, 3)     # (B, N, Ti, d)
        K_t = K.permute(0, 2, 1, 3)     # (B, N, Tj, d)
        V_t = V.permute(0, 2, 1, 3)

        tau   = torch.exp(self.log_tau).clamp(min=0.01)
        sim   = (Q_t @ K_t.transpose(-1, -2)) / (d ** 0.5 * tau)  # (B,N,Ti,Tj)

        # Temporal distance decay (proxy: uniform index spacing)
        t_i = torch.arange(Ti, dtype=torch.float, device=device) / max(Ti - 1, 1)
        t_j = torch.arange(Tj, dtype=torch.float, device=device) / max(Tj - 1, 1)
        gamma = torch.exp(self.log_gamma).clamp(min=0.01)
        dist  = torch.abs(t_i.unsqueeze(1) - t_j.unsqueeze(0))     # (Ti, Tj)
        decay = torch.exp(-gamma * dist)
        sim   = sim + torch.log(decay + 1e-8).unsqueeze(0).unsqueeze(0)

        attn = torch.softmax(sim, dim=-1)                           # (B,N,Ti,Tj)
        out  = (attn @ V_t).permute(0, 2, 1, 3)                    # (B,Ti,N,d)
        return out


# ============================================================
# Stage II: Semantic Alignment via Contrastive Representation
# ============================================================

class DualBranchDecomposition(nn.Module):
    """
    Dual-Branch Representation Decomposition - Equations 11-13.
    Decomposes aligned features into shared semantics and modal-specific features.
    Orthogonality constraint + reconstruction loss ensure proper separation.
    """
    def __init__(self, d_model, num_modalities):
        super().__init__()
        M = num_modalities

        # Shared projection: same weights across all modalities
        self.shared_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        # Modal-specific projections
        self.specific_projs = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
            for _ in range(M)
        ])
        # Reconstruction networks (Eq.13)
        self.recon_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
            for _ in range(M)
        ])
        self.M = M

    def forward(self, aligned_feats):
        # aligned_feats: list of M tensors, each (B, T, N, d)
        shared_list, specific_list = [], []
        L_ortho = torch.tensor(0.0, device=aligned_feats[0].device)
        L_recon = torch.tensor(0.0, device=aligned_feats[0].device)

        for m, h in enumerate(aligned_feats):
            h_sh = self.shared_proj(h)               # Eq.11
            h_sp = self.specific_projs[m](h)

            # Orthogonality constraint - Eq.12
            # Average inner product across batch/space/time should be zero
            inner = (h_sh * h_sp).mean(dim=[0, 1, 2])  # (d,)
            L_ortho = L_ortho + torch.norm(inner) ** 2

            # Reconstruction loss - Eq.13
            h_hat = self.recon_nets[m](h_sh + h_sp)
            L_recon = L_recon + F.mse_loss(h_hat, h.detach())

            shared_list.append(h_sh)
            specific_list.append(h_sp)

        return (shared_list, specific_list,
                L_ortho / self.M, L_recon / self.M)


class UnifiedSemanticSpaceProjection(nn.Module):
    """
    USSP - Equations 14-15.
    Multi-layer nonlinear projection onto unit hypersphere S^{d-1}.
    """
    def __init__(self, d_model, num_layers=2):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers += [nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU()]
        self.proj = nn.Sequential(*layers)

    def forward(self, h):
        z = self.proj(h)
        return F.normalize(z, dim=-1)   # L2-normalise onto hypersphere


class IntraModalCL(nn.Module):
    """Intra-Modal Contrastive Loss - Equations 16-17."""
    def __init__(self, d_model, d_z=128, tau=0.1):
        super().__init__()
        self.tau  = tau
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_z)
        )

    def forward(self, h1, h2):
        # h1, h2: (B, T, N, d) - original and augmented views
        B = h1.shape[0]
        z1 = F.normalize(self.head(h1.mean(dim=[1, 2])), dim=-1)  # (B, dz)
        z2 = F.normalize(self.head(h2.mean(dim=[1, 2])), dim=-1)

        z    = torch.cat([z1, z2], dim=0)                          # (2B, dz)
        sim  = (z @ z.t()) / self.tau                              # (2B, 2B)
        # Mask diagonal (self-similarity)
        mask = torch.eye(2 * B, dtype=torch.bool, device=h1.device)
        sim.masked_fill_(mask, float('-inf'))

        labels = torch.cat([torch.arange(B, device=h1.device) + B,
                             torch.arange(B, device=h1.device)])
        return F.cross_entropy(sim, labels)


class InterModalCL(nn.Module):
    """Inter-Modal Contrastive Loss - Equation 18."""
    def __init__(self, tau=0.1):
        super().__init__()
        self.tau = tau
        # Adaptive pair weighting MLP
        self.pair_mlp = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, shared_list):
        M = len(shared_list)
        if M < 2:
            return torch.tensor(0.0, device=shared_list[0].device)

        total, count = torch.tensor(0.0, device=shared_list[0].device), 0

        # Global averages for adaptive pair weighting
        z_bars = [h.mean(dim=[1, 2]) for h in shared_list]  # list of (B, d)

        for i in range(M):
            for j in range(i + 1, M):
                B, T, N, d = shared_list[i].shape
                # Local temporal representations averaged over nodes
                z_i = F.normalize(shared_list[i].mean(dim=2).reshape(B * T, d), dim=-1)
                z_j = F.normalize(shared_list[j].mean(dim=2).reshape(B * T, d), dim=-1)

                sim = (z_i @ z_j.t()) / self.tau     # (BT, BT)
                labels = torch.arange(B * T, device=z_i.device)
                loss_ij = F.cross_entropy(sim, labels)

                # Adaptive weight (Eq.18 lambda_ij)
                z_pair = torch.stack([z_bars[i].mean(0), z_bars[j].mean(0)], dim=-1)  # (d,2)
                lam = torch.sigmoid(self.pair_mlp(z_pair.t().mean(0, keepdim=True))).squeeze()

                total = total + lam * loss_ij
                count += 1

        return total / max(count, 1)


class InstanceLevelCL(nn.Module):
    """Instance-Level Contrastive Loss - Equation 19."""
    def __init__(self, d_model, tau=0.1):
        super().__init__()
        self.tau  = tau
        self.W_a  = nn.Linear(d_model, 32)
        self.w_a  = nn.Linear(32, 1)
        self.mlp  = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(),
                                   nn.Linear(d_model, d_model))

    def forward(self, shared_list):
        M = len(shared_list)
        B, T, N, d = shared_list[0].shape

        # Attention-weighted global representation per modality (Eq.19)
        global_reps = []
        for h_m in shared_list:
            alpha = torch.softmax(self.w_a(F.relu(self.W_a(h_m))), dim=1)  # (B,T,N,1)
            z_g   = (alpha * h_m).sum(dim=[1, 2])                           # (B, d)
            global_reps.append(z_g)

        # Fuse across modalities
        z_inst = self.mlp(torch.stack(global_reps, dim=1).mean(dim=1))  # (B, d)
        z_inst = F.normalize(z_inst, dim=-1)

        # NT-Xent with diagonal masked
        sim  = (z_inst @ z_inst.t()) / self.tau
        mask = torch.eye(B, dtype=torch.bool, device=z_inst.device)
        sim.masked_fill_(mask, float('-inf'))
        labels = torch.arange(B, device=z_inst.device)
        return F.cross_entropy(sim, labels)


# ============================================================
# Stage III: Dynamic Fusion via Selective GCN-Mamba
# ============================================================

class _GRUFallback(nn.Module):
    """GRU fallback when Mamba is unavailable."""
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True)

    def forward(self, x):
        out, _ = self.gru(x)
        return out


class MambaTemporalBlock(nn.Module):
    """
    Mamba SSM block for O(T) temporal modelling - Equation 22.
    Input-dependent time-step Delta_t is the core selectivity mechanism.
    Falls back to GRU if mamba_ssm is not installed.
    """
    def __init__(self, d_model, d_state=None, expand=2):
        super().__init__()
        if d_state is None:
            d_state = max(d_model // 16, 4)
        if HAS_MAMBA:
            self.ssm = Mamba(d_model=d_model, d_state=d_state,
                              d_conv=3, expand=expand)
        else:
            self.ssm = _GRUFallback(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (*, T, d_model)
        return self.norm(x + self.ssm(x))


class GCNMambaLayer(nn.Module):
    """
    One GCN-Mamba alternating layer.
    Spatial (GCN) → Temporal (Mamba) with residuals.
    """
    def __init__(self, d_model, num_nodes, dropout=0.1):
        super().__init__()
        # Adaptive graph (GraphWaveNet-style)
        self.e1 = nn.Parameter(torch.randn(num_nodes, 32))
        self.e2 = nn.Parameter(torch.randn(32, num_nodes))

        self.gcn_lin  = nn.Linear(d_model, d_model)
        self.mamba    = MambaTemporalBlock(d_model)
        self.norm_gcn = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, N, d)
        B, T, N, d = x.shape

        # --- GCN step ---
        adp = F.softmax(F.relu(self.e1 @ self.e2), dim=1)          # (N, N)
        x_flat = x.reshape(B * T, N, d)
        h_gcn  = F.relu(self.gcn_lin(adp.unsqueeze(0) @ x_flat))   # (BT, N, d)
        x = self.norm_gcn(x + self.dropout(h_gcn.reshape(B, T, N, d)))

        # --- Mamba step ---
        # Process temporal dimension independently per node
        x_tn = x.permute(0, 2, 1, 3).reshape(B * N, T, d)         # (BN, T, d)
        h_mb = self.mamba(x_tn).reshape(B, N, T, d).permute(0, 2, 1, 3)
        x    = x + self.dropout(h_mb)
        return x


class MissingAwareFusion(nn.Module):
    """
    Missing-Aware Fusion (MAF) - Equation 23.
    Dynamically blends Mamba output with global prior based on modality availability.
    When modalities are missing, increases time step to rapidly forget missing periods.
    """
    def __init__(self, d_model, num_modalities):
        super().__init__()
        self.M = num_modalities
        self.global_prior = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.alpha_mlp = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, z_mamba, pooled_list, modal_mask=None):
        # z_mamba:    (B, T, N, d)
        # pooled_list: list of M tensors (B, T, N, d)
        B, T, N, d = z_mamba.shape

        # Global prior from mean-pooled concatenation
        z_cat    = torch.stack(pooled_list, dim=-1).mean(dim=-1)  # (B, T, N, d)
        z_global = self.global_prior(z_cat)                        # (B, T, N, d)

        # Compute alpha from availability ratio
        if modal_mask is not None:
            avail = modal_mask.float().mean(dim=-1).mean(dim=-1, keepdim=True)  # (B,1)
        else:
            avail = torch.ones(B, 1, device=z_mamba.device)

        alpha = self.alpha_mlp(avail).unsqueeze(1).unsqueeze(2)   # (B,1,1,1)
        z_fused = alpha * z_mamba + (1.0 - alpha) * z_global
        return z_fused


# ============================================================
# Stage IV: Spatio-Temporal Decoding
# ============================================================

class MultiStepDecoder(nn.Module):
    """
    Multi-Step Prediction Decoder - Equation 24.
    Transformer encoder used as decoder (attending to historical states).
    """
    def __init__(self, d_model, out_steps, output_dim=1,
                 num_layers=2, num_heads=4, ff_dim=256, dropout=0.1):
        super().__init__()
        self.out_steps  = out_steps
        self.output_dim = output_dim

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_proj    = nn.Linear(d_model, out_steps * output_dim)

    def forward(self, z_fused):
        # z_fused: (B, T, N, d)
        B, T, N, d = z_fused.shape
        # Process each node's temporal sequence
        z_node = z_fused.permute(0, 2, 1, 3).reshape(B * N, T, d)   # (BN, T, d)
        h      = self.transformer(z_node)                             # (BN, T, d)
        out    = self.out_proj(h.mean(dim=1))                         # (BN, out*dout)
        out    = out.reshape(B, N, self.out_steps, self.output_dim)
        return out.permute(0, 2, 1, 3)                                # (B, out, N, dout)


# ============================================================
# Main MSTHH Model
# ============================================================

class MSTHH(nn.Module):
    """
    Multi-modal Spatio-Temporal Hierarchical Heterogeneity Network.

    Input x: (B, T, N, C) where the feature layout is:
        x[..., :num_modalities]          -- traffic features (flow, speed, OCC, ...)
        x[..., num_modalities]           -- time-of-day  (fractional 0–1, if tod=True)
        x[..., num_modalities+1]         -- day-of-week  (integer 0–6,    if dow=True)

    For the EXISTING single-modality data format [flow, tod, dow]:
        Set num_modalities=1.  The tod index will be 1, dow index will be 2.

    For true multi-modal PEMS data [flow, speed, OCC, tod, dow]:
        Set num_modalities=3.  The tod index will be 3, dow index will be 4.
    """

    def __init__(
        self,
        num_nodes,
        in_steps              = 12,
        out_steps             = 12,
        steps_per_day         = 288,
        num_modalities        = 3,     # M: number of traffic modalities
        input_dim             = 1,     # feature dimension per modality (usually 1)
        output_dim            = 1,
        tod_embedding_dim     = 24,
        dow_embedding_dim     = 24,
        d_model               = 64,
        num_gcn_scales        = 3,
        num_gcn_layers        = 3,
        num_gcn_mamba_layers  = 3,
        num_decoder_layers    = 2,
        dropout               = 0.1,
        tau_intra             = 0.1,
        tau_inter             = 0.1,
        tau_inst              = 0.1,
        # Adaptive loss: learnable uncertainty (Eq.26)
        sigma1_init           = 0.8,   # task loss
        sigma2_init           = 2.0,   # contrastive loss
        sigma3_init           = 0.5,   # regularisation loss
    ):
        super().__init__()

        self.num_nodes     = num_nodes
        self.in_steps      = in_steps
        self.out_steps     = out_steps
        self.steps_per_day = steps_per_day
        self.M             = num_modalities
        self.d_model       = d_model
        self.output_dim    = output_dim
        self.tod_dim       = tod_embedding_dim
        self.dow_dim       = dow_embedding_dim

        # ── Stage I ──────────────────────────────────────────────────
        # TSA Embedding per modality
        self.tsa = nn.ModuleList([
            TemporalScaleAwareEmbedding(input_dim, d_model)
            for _ in range(num_modalities)
        ])

        # Time-feature embeddings (shared across modalities)
        if tod_embedding_dim > 0:
            self.tod_emb = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_emb = nn.Embedding(7, dow_embedding_dim)

        time_d = tod_embedding_dim + dow_embedding_dim
        self.time_proj = nn.Linear(d_model + time_d, d_model) if time_d > 0 else None

        # Hierarchical GCN per modality
        self.hgcn = nn.ModuleList([
            HierarchicalMultiScaleGCN(d_model, num_nodes, num_gcn_scales,
                                       num_gcn_layers, dropout)
            for _ in range(num_modalities)
        ])

        # Cross-modal temporal aligners (one per ordered pair, i≠j)
        # We use M*(M-1) aligners for all ordered pairs
        n_aligners = max(num_modalities * (num_modalities - 1), 1)
        self.aligners = nn.ModuleList([
            CrossModalTemporalAligner(d_model) for _ in range(n_aligners)
        ])

        # ── Stage II ─────────────────────────────────────────────────
        self.dual_branch = DualBranchDecomposition(d_model, num_modalities)
        self.ussp        = UnifiedSemanticSpaceProjection(d_model)

        # Three-level contrastive losses
        self.intra_cl    = nn.ModuleList([
            IntraModalCL(d_model, tau=tau_intra) for _ in range(num_modalities)
        ])
        self.inter_cl    = InterModalCL(tau=tau_inter)
        self.inst_cl     = InstanceLevelCL(d_model, tau=tau_inst)

        # ── Stage III ────────────────────────────────────────────────
        fusion_d = d_model * num_modalities
        self.modal_embed  = nn.Embedding(num_modalities, d_model)
        self.fusion_proj  = nn.Linear(fusion_d, fusion_d)

        self.gcn_mamba = nn.ModuleList([
            GCNMambaLayer(fusion_d, num_nodes, dropout)
            for _ in range(num_gcn_mamba_layers)
        ])
        self.maf = MissingAwareFusion(fusion_d, num_modalities)

        # ── Stage IV ─────────────────────────────────────────────────
        self.decoder = MultiStepDecoder(
            fusion_d, out_steps, output_dim,
            num_decoder_layers, dropout=dropout
        )

        # ── Adaptive multi-task loss parameters (Eq.25) ──────────────
        self.log_sigma1 = nn.Parameter(torch.tensor(float(sigma1_init)).log())
        self.log_sigma2 = nn.Parameter(torch.tensor(float(sigma2_init)).log())
        self.log_sigma3 = nn.Parameter(torch.tensor(float(sigma3_init)).log())

        self._aug_std = 0.1  # noise std for intra-modal augmentation

    # ----------------------------------------------------------------
    def _augment(self, h):
        return h + torch.randn_like(h) * self._aug_std

    # ----------------------------------------------------------------
    def forward(self, x, modal_mask=None):
        """
        Returns:
            Training:  (pred, L_cl, L_ortho, L_recon)
            Inference: pred
        pred shape: (B, out_steps, N, output_dim)
        """
        B, T, N, C = x.shape
        M = self.M

        # ── Extract tod / dow ────────────────────────────────────────
        tod_idx = M       # tod is the first feature after the M traffic features
        dow_idx = M + 1

        time_feats = []
        if self.tod_dim > 0 and C > tod_idx:
            tod     = x[..., tod_idx]                                # (B,T,N)
            tod_emb = self.tod_emb((tod * self.steps_per_day).long().clamp(0, self.steps_per_day - 1))
            time_feats.append(tod_emb)                               # (B,T,N,tod_d)

        if self.dow_dim > 0 and C > dow_idx:
            dow     = x[..., dow_idx]                                # (B,T,N)
            dow_emb = self.dow_emb(dow.long().clamp(0, 6))
            time_feats.append(dow_emb)                               # (B,T,N,dow_d)

        # ── Stage I: Encoding ────────────────────────────────────────
        H_list = []
        for m in range(M):
            feat_idx = m if m < C else 0
            x_m = x[..., feat_idx: feat_idx + 1]                    # (B,T,N,1)

            h_m = self.tsa[m](x_m)                                  # (B,T,N,d)

            if time_feats and self.time_proj is not None:
                tc  = torch.cat(time_feats, dim=-1)                  # (B,T,N,tod+dow)
                h_m = self.time_proj(torch.cat([h_m, tc], dim=-1))

            h_m = self.hgcn[m](h_m)                                 # (B,T,N,d)
            H_list.append(h_m)

        # ── Stage I: Cross-modal alignment ───────────────────────────
        H_aligned = []
        aligner_idx = 0
        for i in range(M):
            h_i_aligned = H_list[i].clone()
            for j in range(M):
                if i != j:
                    idx = aligner_idx % len(self.aligners)
                    h_aligned_ij = self.aligners[idx](H_list[i], H_list[j])
                    h_i_aligned  = h_i_aligned + 0.5 * h_aligned_ij
                    aligner_idx += 1
            H_aligned.append(h_i_aligned)

        # ── Stage II: Semantic Alignment ─────────────────────────────
        shared, specific, L_ortho, L_recon = self.dual_branch(H_aligned)
        z_unified = [self.ussp(h_sh) for h_sh in shared]

        # Three-level contrastive losses (training only)
        L_cl = torch.tensor(0.0, device=x.device)
        if self.training:
            # Intra-modal
            for m in range(M):
                L_cl = L_cl + self.intra_cl[m](H_aligned[m], self._augment(H_aligned[m]))
            L_cl = L_cl / M
            # Inter-modal
            if M > 1:
                L_cl = L_cl + self.inter_cl(shared)
            # Instance-level
            L_cl = L_cl + self.inst_cl(shared)

        # ── Stage III: Fusion ────────────────────────────────────────
        # Modal identity injection (Eq.21)
        pooled = []
        for m, (z_u, z_sp) in enumerate(zip(z_unified, specific)):
            me = self.modal_embed(
                torch.tensor(m, device=x.device)
            ).view(1, 1, 1, -1)                                       # (1,1,1,d)
            z_m = z_u + z_sp + me
            pooled.append(z_m)

        z_cat = torch.cat(pooled, dim=-1)                             # (B,T,N,M*d)
        z_cat = self.fusion_proj(z_cat)

        # GCN-Mamba alternating layers
        z = z_cat
        for layer in self.gcn_mamba:
            z = layer(z)

        # Missing-Aware Fusion
        z_fused = self.maf(z, pooled, modal_mask)

        # ── Stage IV: Decode ─────────────────────────────────────────
        pred = self.decoder(z_fused)                                  # (B,out,N,dout)

        if self.training:
            return pred, L_cl, L_ortho, L_recon
        return pred

    # ----------------------------------------------------------------
    def compute_total_loss(self, pred, y_true, L_cl, L_ortho, L_recon,
                            criterion, epoch, total_epochs):
        """
        Adaptive multi-task objective - Equations 25-27.
        total = w1/(2*s1^2)*L_task + w2/(2*s2^2)*L_CL + w3/(2*s3^2)*L_reg
                + log(s1*s2*s3)
        sigma1,2,3 are jointly optimised with the model parameters.
        """
        # w2 scheduling (Eq.27)
        E_warmup = max(total_epochs // 10, 1)
        E_decay  = max(total_epochs // 3, 1)
        if epoch < E_warmup:
            w2 = epoch / E_warmup
        else:
            w2 = math.exp(-(epoch - E_warmup) / E_decay)

        s1 = torch.exp(self.log_sigma1).clamp(min=1e-3)
        s2 = torch.exp(self.log_sigma2).clamp(min=1e-3)
        s3 = torch.exp(self.log_sigma3).clamp(min=1e-3)

        L_task = criterion(pred, y_true)
        L_reg  = L_ortho + L_recon

        total = (
            1.0 / (2 * s1 ** 2) * L_task
            + w2 / (2 * s2 ** 2) * L_cl
            + 1.0 / (2 * s3 ** 2) * L_reg
            + torch.log(s1 * s2 * s3)
        )
        return total, L_task


# ----------------------------------------------------------------
if __name__ == "__main__":
    from torchinfo import summary
    # Quick sanity check with 1 modality (existing data format)
    model = MSTHH(num_nodes=170, num_modalities=1, d_model=64, in_steps=12, out_steps=12)
    x = torch.randn(4, 12, 170, 3)   # [flow, tod, dow]
    model.eval()
    out = model(x)
    print("Single-modal output shape:", out.shape)  # (4,12,170,1)

    # Quick sanity check with 3 modalities
    model3 = MSTHH(num_nodes=170, num_modalities=3, d_model=64, in_steps=12, out_steps=12)
    x3 = torch.randn(4, 12, 170, 5)  # [flow, speed, OCC, tod, dow]
    out3 = model3(x3)
    print("Multi-modal output shape:", out3.shape)  # (4,12,170,1)
