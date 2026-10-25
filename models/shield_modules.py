import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'ContextClusteringLayer',
    'MoDDecoderLayer',
    'ShieldFeedForward',
]


class ShieldFeedForward(nn.Module):
    """Two-layer feed-forward network with ReLU activation."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w_2(F.relu(self.w_1(x)))


class ContextClusteringLayer(nn.Module):
    """
    Context-based soft clustering layer with task prompt (SHIELD, Goh et al., ICML 2025).
    
    Clusters customer node features into Nc latent cluster centers based on problem constraints (task prompt).
    """
    def __init__(self, d_model=128, n_clusters=5, n_iters=5, task_dim=5, 
                 use_task_prompt=True, softmax_dim="cluster", 
                 cluster_use_layernorm_state=False, cluster_include_depot=False):
        super().__init__()
        self.d_model = d_model
        self.n_clusters = n_clusters
        self.n_iters = n_iters
        self.task_dim = task_dim
        self.use_task_prompt = use_task_prompt
        self.softmax_dim = softmax_dim
        self.cluster_use_layernorm_state = cluster_use_layernorm_state
        self.cluster_include_depot = cluster_include_depot

        # Trainable initial cluster centers: C0 [Nc, d]
        self.C0 = nn.Parameter(torch.randn(n_clusters, d_model) * (1.0 / math.sqrt(d_model)))

        self.W_H = nn.Linear(d_model, d_model)
        self.W_C = nn.Linear(2 * d_model if use_task_prompt else d_model, d_model)
        
        if use_task_prompt:
            self.W_prompt = nn.Linear(task_dim, d_model, bias=False)
        else:
            self.W_prompt = None

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, H, gamma=None):
        """
        Args:
            H: Node embeddings from encoder [B, N, d], where index 0 is depot.
            gamma: Task / constraint multi-hot vector [B, task_dim] or [1, task_dim].
        Returns:
            C_init: Initial cluster representation for context [B, Nc, d]
            psi_full: Soft assignment weights for all nodes [B, N, Nc] (padded with 0 at depot if excluded)
        """
        B, N, d = H.shape

        # Select customer nodes for clustering
        if not self.cluster_include_depot:
            H_nodes = H[:, 1:, :]  # [B, N_cust, d]
        else:
            H_nodes = H

        N_clustered = H_nodes.size(1)

        # Compute task prompt alpha
        if self.use_task_prompt and gamma is not None:
            if gamma.dim() == 1:
                gamma = gamma.unsqueeze(0)
            if gamma.size(0) == 1 and B > 1:
                gamma = gamma.expand(B, -1)
            alpha = self.W_prompt(gamma)  # [B, d]
        else:
            alpha = None

        C = self.C0.unsqueeze(0).expand(B, self.n_clusters, d)  # [B, Nc, d]
        psi = None
        C_sum = None

        for _ in range(self.n_iters):
            H_hat = self.W_H(H_nodes)  # [B, N_clustered, d]

            if alpha is not None:
                alpha_expanded = alpha.unsqueeze(1).expand(-1, self.n_clusters, -1)  # [B, Nc, d]
                C_in = torch.cat([C, alpha_expanded], dim=-1)  # [B, Nc, 2d]
            else:
                C_in = C

            C_hat = self.W_C(C_in)  # [B, Nc, d]

            logits = torch.matmul(H_hat, C_hat.transpose(1, 2)) / math.sqrt(d)  # [B, N_clustered, Nc]

            if self.softmax_dim == "cluster":
                psi = F.softmax(logits, dim=-1)  # normalize over clusters for each node
            elif self.softmax_dim == "node":
                psi = F.softmax(logits, dim=1)   # normalize over nodes for each cluster
            else:
                raise ValueError(f"Unknown softmax_dim: {self.softmax_dim}")

            C_sum = torch.einsum('bnj,bnd->bjd', psi, H_nodes)  # [B, Nc, d]
            C = self.layer_norm(C_hat + C_sum)  # residual + LayerNorm

        # Pad psi back to full [B, N, Nc] so depot has 0 weight
        if not self.cluster_include_depot:
            psi_depot = torch.zeros(B, 1, self.n_clusters, device=H.device, dtype=H.dtype)
            psi_full = torch.cat([psi_depot, psi], dim=1)  # [B, N, Nc]
        else:
            psi_full = psi

        C_init = C if self.cluster_use_layernorm_state else C_sum
        return C_init, psi_full


class MoDDecoderLayer(nn.Module):
    """
    Decoder layer with Mixture-of-Depths (MoD) routing (SHIELD, Goh et al., ICML 2025).
    
    Tokens correspond to POMO agents (P = pomo_size).
    Intra-instance routing selects top-k agents (k = ceil(beta * P)) to execute
    Cross-Attention and Feed-Forward; unselected agents bypass via residual connection.
    """
    def __init__(self, d_model=128, n_heads=8, qkv_dim=16, ff_dim=512, 
                 beta=0.10, use_mod=True, router_activation="none"):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.qkv_dim = qkv_dim
        self.ff_dim = ff_dim
        self.beta = beta
        self.use_mod = use_mod
        self.router_activation = router_activation

        # Router for selecting top-k agents
        self.router = nn.Linear(d_model, 1)

        # Cross-Attention projections
        self.Wq = nn.Linear(d_model, n_heads * qkv_dim, bias=False)
        self.Wk = nn.Linear(d_model, n_heads * qkv_dim, bias=False)
        self.Wv = nn.Linear(d_model, n_heads * qkv_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * qkv_dim, d_model)

        self.feed_forward = ShieldFeedForward(d_model, ff_dim)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def compute_kv(self, H):
        """
        Precompute Key and Value projections once per instance on encoder output H.
        Args:
            H: [B, N, d]
        Returns:
            K: [B, n_heads, N, qkv_dim]
            V: [B, n_heads, N, qkv_dim]
        """
        B, N, _ = H.shape
        K = self.Wk(H).view(B, N, self.n_heads, self.qkv_dim).transpose(1, 2)
        V = self.Wv(H).view(B, N, self.n_heads, self.qkv_dim).transpose(1, 2)
        return K, V

    def _block(self, q_tokens, K, V, mask_tokens=None):
        """
        Apply Cross-Attention and Feed-Forward on the given query tokens.
        Args:
            q_tokens: [B, k, d]
            K, V: [B, n_heads, N, qkv_dim]
            mask_tokens: [B, k, N]
        Returns:
            f: [B, k, d] (transformation without outer residual)
        """
        B, k, d = q_tokens.shape
        q_norm = self.norm1(q_tokens)
        
        # Multi-Head Cross Attention
        Q = self.Wq(q_norm).view(B, k, self.n_heads, self.qkv_dim).transpose(1, 2)  # [B, n_heads, k, qkv_dim]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.qkv_dim)  # [B, n_heads, k, N]

        if mask_tokens is not None:
            # mask_tokens has -inf for illegal nodes
            scores = scores + mask_tokens.unsqueeze(1)  # broadcast over heads

        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V)  # [B, n_heads, k, qkv_dim]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, k, self.n_heads * self.qkv_dim)
        attn_out = self.out_proj(attn_out)  # [B, k, d]

        # Sublayer 1 residual & norm
        x = q_norm + attn_out
        x_norm = self.norm2(x)
        ff_out = self.feed_forward(x_norm)  # [B, k, d]
        
        # Block output transformation f
        f = attn_out + ff_out
        return f

    def forward(self, h, K, V, mask=None, agent_done=None):
        """
        Args:
            h: Query embeddings for all agents [B, P, d]
            K, V: Precomputed keys/values [B, n_heads, N, qkv_dim]
            mask: Infeasibility mask [B, P, N] (0 for valid, -inf for invalid)
            agent_done: Boolean tensor [B, P] indicating finished agents
        Returns:
            h_out: Updated agent embeddings [B, P, d]
        """
        B, P, d = h.shape

        if not self.use_mod or self.beta >= 1.0:
            # Dense execution: all agents are updated
            f = self._block(h, K, V, mask)
            return h + f

        # Mixture-of-Depths (MoD) routing
        r_raw = self.router(h).squeeze(-1)  # [B, P]
        if self.router_activation == "sigmoid":
            r = torch.sigmoid(r_raw)
        else:
            r = r_raw

        # Exclude finished agents from top-k competition
        if agent_done is not None:
            r_for_topk = r.masked_fill(agent_done, -1e9)
        else:
            r_for_topk = r

        k = max(1, math.ceil(self.beta * P))
        top_idx = r_for_topk.topk(k, dim=1).indices  # [B, k]

        # Gather selected tokens and masks
        top_idx_expanded_d = top_idx.unsqueeze(-1).expand(-1, -1, d)
        q_sel = h.gather(1, top_idx_expanded_d)  # [B, k, d]
        r_sel = r.gather(1, top_idx)  # [B, k]

        if mask is not None:
            N = mask.size(-1)
            top_idx_expanded_m = top_idx.unsqueeze(-1).expand(-1, -1, N)
            m_sel = mask.gather(1, top_idx_expanded_m)  # [B, k, N]
        else:
            m_sel = None

        # Execute block on selected top-k tokens only
        f_sel = self._block(q_sel, K, V, m_sel)  # [B, k, d]

        # MoD formulation (Eq. 4): out = q + r_sel * f
        out_sel = q_sel + r_sel.unsqueeze(-1) * f_sel  # [B, k, d]

        # Scatter back: unselected agents retain their input h
        h_out = h.scatter(1, top_idx_expanded_d, out_sel)
        return h_out
