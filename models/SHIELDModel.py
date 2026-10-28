import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .MOEModel import MTL_Encoder
from .shield_modules import ContextClusteringLayer, MoDDecoderLayer

__all__ = ['SHIELDModel']


def _get_encoding(encoded_nodes, node_index_to_pick):
    """
    Gather node embeddings for selected indices across pomo agents.
    encoded_nodes: [B, N, d]
    node_index_to_pick: [B, pomo]
    returns: [B, pomo, d]
    """
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    return picked_nodes


class SHIELDModel(nn.Module):
    """
    SHIELD: Multi-task Multi-distribution Vehicle Routing Solver with Sparsity and Hierarchy
    (Goh et al., ICML 2025).
    
    Architecture:
    1. MoE Encoder (6 layers, top-2 MoE routing)
    2. Context-based Soft Clustering Layer (with task prompt gamma, Nc clusters, 5 iterations)
    3. Step-by-step Decoder:
       - Context Query h(c) (Eq. 7): W_combine([h_last, cluster_state]) + h_first
       - Deep Decoder with Mixture-of-Depths (MoD) (Eq. 4, 3 layers, top-beta routing across agents)
       - Pointer decision (single-head dot-product attention + tanh 10 clip + mask + softmax)
       - Per-agent cluster state update (Eq. 8): c_j <- c_j - psi[i, j] * H[i]
    """
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = model_params['embedding_dim']
        self.sqrt_embedding_dim = model_params.get('sqrt_embedding_dim', math.sqrt(self.embedding_dim))
        self.logit_clipping = model_params.get('logit_clipping', 10.0)
        self.eval_type = model_params['eval_type']
        self.problem = model_params.get('problem', None)
        self.aux_loss = 0

        # SHIELD Hyperparameters
        self.decoder_layers_num = model_params.get('decoder_layers', 3)
        self.use_mod = model_params.get('use_mod', True)
        self.mod_beta = model_params.get('mod_beta', 0.10)
        self.use_cluster = model_params.get('use_cluster', True)
        self.n_clusters = model_params.get('n_clusters', 5)
        self.cluster_iters = model_params.get('cluster_iters', 5)
        self.use_task_prompt = model_params.get('use_task_prompt', True)
        self.task_dim = model_params.get('task_dim', 5)
        self.softmax_dim = model_params.get('softmax_dim', 'cluster')
        self.cluster_use_layernorm_state = model_params.get('cluster_use_layernorm_state', False)
        self.cluster_include_depot = model_params.get('cluster_include_depot', False)
        self.router_activation = model_params.get('router_activation', 'none')

        # 1. MoE Encoder (preserves original MVMoE architecture)
        self.encoder = MTL_Encoder(**model_params)

        # 2. Context-based Soft Clustering Layer
        if self.use_cluster:
            self.clustering_layer = ContextClusteringLayer(
                d_model=self.embedding_dim,
                n_clusters=self.n_clusters,
                n_iters=self.cluster_iters,
                task_dim=self.task_dim,
                use_task_prompt=self.use_task_prompt,
                softmax_dim=self.softmax_dim,
                cluster_use_layernorm_state=self.cluster_use_layernorm_state,
                cluster_include_depot=self.cluster_include_depot
            )
        else:
            self.clustering_layer = None

        # 3. Context Query Projections (Eq. 7)
        # h_last combines last node embedding + 4 dynamic features (load, time, length, open)
        self.W_last = nn.Linear(self.embedding_dim + 4, self.embedding_dim)
        # h_first projects first node embedding
        self.W_first = nn.Linear(self.embedding_dim, self.embedding_dim)

        if self.use_cluster:
            # W_combine maps concat([h_last, flatten(cluster_state)]) -> d
            self.W_combine = nn.Linear(self.embedding_dim + self.n_clusters * self.embedding_dim, self.embedding_dim)
        else:
            self.W_combine = None

        # 4. Deep Decoder Layers (MoD)
        head_num = model_params['head_num']
        qkv_dim = model_params['qkv_dim']
        ff_dim = model_params['ff_hidden_dim']

        self.decoder_layers = nn.ModuleList([
            MoDDecoderLayer(
                d_model=self.embedding_dim,
                n_heads=head_num,
                qkv_dim=qkv_dim,
                ff_dim=ff_dim,
                beta=self.mod_beta,
                use_mod=self.use_mod,
                router_activation=self.router_activation
            ) for _ in range(self.decoder_layers_num)
        ])

        # State storage across decoding steps
        self.encoded_nodes = None       # [B, N, d]
        self.single_head_key = None     # [B, d, N]
        self.cached_kv = None           # List of (K, V) per decoder layer
        self.cluster_state = None       # [B, P, Nc, d]
        self.psi = None                 # [B, N, Nc]
        self.start_node = None          # [B, P]
        
        self.device = torch.device('cuda', torch.cuda.current_device()) if 'device' not in model_params.keys() else model_params['device']

    def pre_forward(self, reset_state):
        """
        Executed once per problem instance before rollout.
        Computes node embeddings, initial clustering, and caches K/V projections.
        """
        self.device = next(self.parameters()).device

        depot_xy = reset_state.depot_xy.to(self.device)
        node_xy = reset_state.node_xy.to(self.device)
        node_demand = reset_state.node_demand.to(self.device)
        node_tw_start = reset_state.node_tw_start.to(self.device)
        node_tw_end = reset_state.node_tw_end.to(self.device)

        node_xy_demand_tw = torch.cat((
            node_xy,
            node_demand[:, :, None],
            node_tw_start[:, :, None],
            node_tw_end[:, :, None]
        ), dim=2)

        # 1. MoE Encoder forward
        self.encoded_nodes, moe_loss = self.encoder(depot_xy, node_xy_demand_tw)
        self.aux_loss = moe_loss  # Load-balancing loss for Trainer

        B, N, d = self.encoded_nodes.shape
        self.single_head_key = self.encoded_nodes.transpose(1, 2)  # [B, d, N]

        # 2. Precompute & cache K, V for all decoder layers
        self.cached_kv = [layer.compute_kv(self.encoded_nodes) for layer in self.decoder_layers]

        # 3. Context-based Soft Clustering
        if self.use_cluster and self.clustering_layer is not None:
            gamma = getattr(reset_state, 'prob_emb', None)
            if gamma is None:
                # Fallback multi-hot vector [C, O, B, L, TW]
                gamma = torch.FloatTensor([[1, 0, 0, 0, 0]])
            gamma = gamma.to(self.device)
            C_init, self.psi = self.clustering_layer(self.encoded_nodes, gamma=gamma)
            
            # Per-agent cluster state will be instantiated when pomo_size is known in forward()
            self.C_init = C_init  # [B, Nc, d]
            self.cluster_state = None
        else:
            self.C_init = None
            self.cluster_state = None
            self.psi = None

        self.start_node = None

    def set_eval_type(self, eval_type):
        self.eval_type = eval_type

    def _update_cluster_state(self, selected):
        """
        Vectorized update of cluster states for all agents according to Eq. 8:
        C_state[b, p, j] -= psi[b, selected[b, p], j] * H[b, selected[b, p]]
        Depot (node 0) has psi = 0, so visiting depot causes no change.
        """
        if not self.use_cluster or self.cluster_state is None or self.psi is None:
            return

        selected = selected.to(self.device)
        B, P = selected.shape
        # Gather psi for selected nodes: [B, P, Nc]
        idx_psi = selected.unsqueeze(-1).expand(-1, -1, self.n_clusters)
        psi_sel = self.psi.gather(1, idx_psi)  # [B, P, Nc]

        # Gather H for selected nodes: [B, P, d]
        idx_H = selected.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
        H_sel = self.encoded_nodes.gather(1, idx_H)  # [B, P, d]

        # Outer product: [B, P, Nc, d]
        delta = psi_sel.unsqueeze(-1) * H_sel.unsqueeze(2)
        self.cluster_state = self.cluster_state - delta

    def forward(self, state, selected=None):
        """
        POMO Rollout step.
        """
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        orig_device = state.BATCH_IDX.device

        # Step 0: Initial position at depot
        if state.selected_count == 0:
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long, device=self.device)
            prob = torch.ones(size=(batch_size, pomo_size), device=self.device)
            
            # Initialize cluster state per agent [B, P, Nc, d]
            if self.use_cluster and self.C_init is not None:
                self.cluster_state = self.C_init.unsqueeze(1).expand(-1, pomo_size, -1, -1).clone()

        # Step 1: Multi-start node assignment
        elif state.selected_count == 1:
            selected = state.START_NODE.to(self.device)
            self.start_node = selected  # Cache first node for h_first (Eq. 7)
            prob = torch.ones(size=(batch_size, pomo_size), device=self.device)

            # Subtract START_NODE from cluster state as it visits the first customer
            if self.use_cluster:
                self._update_cluster_state(selected)

        # Step >= 2: Autoregressive decoding with MoD and Cluster guidance
        else:
            # 1. Query components
            cur_node = state.current_node.to(self.device)
            encoded_last_node = _get_encoding(self.encoded_nodes, cur_node)  # [B, P, d]
            attr = torch.cat((
                state.load.to(self.device)[:, :, None],
                state.current_time.to(self.device)[:, :, None],
                state.length.to(self.device)[:, :, None],
                state.open.to(self.device)[:, :, None]
            ), dim=2)  # [B, P, 4]

            input_last = torch.cat((encoded_last_node, attr), dim=2)  # [B, P, d+4]
            h_last = self.W_last(input_last)  # [B, P, d]

            first_node = self.start_node if self.start_node is not None else torch.zeros_like(cur_node)
            encoded_first_node = _get_encoding(self.encoded_nodes, first_node)  # [B, P, d]
            h_first = self.W_first(encoded_first_node)  # [B, P, d]

            # 2. Context Query h(c) (Eq. 7)
            if self.use_cluster and self.cluster_state is not None:
                cluster_flat = self.cluster_state.flatten(start_dim=2)  # [B, P, Nc*d]
                combine_in = torch.cat([h_last, cluster_flat], dim=-1)  # [B, P, d + Nc*d]
                h = self.W_combine(combine_in) + h_first                # [B, P, d]
            else:
                h = h_last + h_first                                     # [B, P, d]

            # 3. Deep Decoder Layers with MoD (Eq. 4)
            agent_done = getattr(state, 'finished', None)
            if agent_done is not None:
                agent_done = agent_done.to(self.device)
            ninf_mask = state.ninf_mask.to(self.device)

            for i, layer in enumerate(self.decoder_layers):
                K_i, V_i = self.cached_kv[i]
                h = layer(h, K_i, V_i, mask=ninf_mask, agent_done=agent_done)

            # 4. Final Pointer Attention
            # score: [B, P, N]
            score = torch.matmul(h, self.single_head_key) / self.sqrt_embedding_dim
            score_clipped = self.logit_clipping * torch.tanh(score)
            score_masked = score_clipped + ninf_mask

            probs = F.softmax(score_masked, dim=-1)  # [B, P, N]

            # 5. Selection (Greedy or Softmax Sampling)
            batch_idx = state.BATCH_IDX.to(self.device)
            pomo_idx = state.POMO_IDX.to(self.device)

            if selected is None:
                while True:
                    if self.training or self.eval_type == 'softmax':
                        try:
                            selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                                .squeeze(dim=1).reshape(batch_size, pomo_size)
                        except Exception as exception:
                            print(f">> Catch Exception: {exception}, on instances of {state.PROBLEM}")
                            exit(0)
                    else:
                        selected = probs.argmax(dim=-1)

                    prob = probs[batch_idx, pomo_idx, selected].reshape(batch_size, pomo_size)
                    if (prob != 0).all():
                        break
            else:
                selected = selected.to(self.device)
                prob = probs[batch_idx, pomo_idx, selected].reshape(batch_size, pomo_size)

            # 6. Update cluster state with chosen nodes (Eq. 8)
            if self.use_cluster:
                self._update_cluster_state(selected)

        return selected.to(orig_device), prob.to(orig_device)
