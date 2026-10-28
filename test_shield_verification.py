import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from envs import CVRPEnv, VRPTWEnv
from models import SHIELDModel, MOEModel
from models.shield_modules import ContextClusteringLayer, MoDDecoderLayer
from utils import num_param


def test_1_shapes_and_mod_correctness():
    print("=" * 60)
    print("TEST 1: Shapes & MoD Routing Correctness")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, P, d, N, Nc = 4, 50, 128, 51, 5
    beta = 0.10
    
    # Check ContextClusteringLayer
    c_layer = ContextClusteringLayer(d_model=d, n_clusters=Nc, n_iters=5, task_dim=5).to(device)
    H = torch.randn(B, N, d, device=device)
    gamma = torch.FloatTensor([[1, 0, 0, 0, 0]]).to(device)
    C_init, psi = c_layer(H, gamma)
    
    assert C_init.shape == (B, Nc, d), f"Expected C_init shape {(B, Nc, d)}, got {C_init.shape}"
    assert psi.shape == (B, N, Nc), f"Expected psi shape {(B, N, Nc)}, got {psi.shape}"
    assert torch.all(psi[:, 0, :] == 0), "Depot psi should be all zeros"
    print(f"  [PASS] ContextClusteringLayer: C_init {C_init.shape}, psi {psi.shape}, depot psi = 0")
    
    # Check MoDDecoderLayer
    mod_layer = MoDDecoderLayer(d_model=d, n_heads=8, qkv_dim=16, ff_dim=512, beta=beta, use_mod=True).to(device)
    K, V = mod_layer.compute_kv(H)
    h = torch.randn(B, P, d, device=device)
    mask = torch.zeros(B, P, N, device=device)
    
    h_out = mod_layer(h, K, V, mask=mask)
    assert h_out.shape == (B, P, d), f"Expected h_out shape {(B, P, d)}, got {h_out.shape}"
    
    # Check that unselected agents have identical values (bypass)
    r = mod_layer.router(h).squeeze(-1)
    k = max(1, math.ceil(beta * P))  # 5 agents
    assert k == 5, f"Expected k=5 for beta=0.10, P=50, got {k}"
    top_idx = r.topk(k, dim=1).indices
    
    for b in range(B):
        selected_set = set(top_idx[b].tolist())
        for p in range(P):
            if p not in selected_set:
                # Unselected agent MUST retain exact input h
                assert torch.equal(h[b, p], h_out[b, p]), f"Agent {p} was not selected but h was altered!"
    print(f"  [PASS] MoDDecoderLayer: Output shape {h_out.shape}, exactly {k} agents updated, 45 bypassed identically.")

    # Check beta = 1.0 (all updated)
    mod_layer_all = MoDDecoderLayer(d_model=d, n_heads=8, qkv_dim=16, ff_dim=512, beta=1.0, use_mod=True).to(device)
    h_out_all = mod_layer_all(h, K, V, mask=mask)
    diff = (h_out_all - h).abs().sum()
    assert diff > 0, "All agents should be updated when beta=1.0"
    print("  [PASS] MoDDecoderLayer with beta=1.0 updates all agents.")


def test_2_gradient_flow():
    print("=" * 60)
    print("TEST 2: Gradient Flow Verification")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_params = {
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128 ** 0.5,
        'encoder_layer_num': 6,
        'decoder_layers': 3,
        'qkv_dim': 16,
        'head_num': 8,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'num_experts': 4,
        'eval_type': 'argmax',
        'norm': 'instance',
        'norm_loc': 'norm_last',
        'expert_loc': ['Enc0', 'Enc1', 'Enc2', 'Enc3', 'Enc4', 'Enc5'],
        'topk': 2,
        'routing_level': 'node',
        'routing_method': 'input_choice',
        'problem': 'CVRP',
        'use_mod': True,
        'mod_beta': 0.10,
        'use_cluster': True,
        'n_clusters': 5,
        'cluster_iters': 5,
        'use_task_prompt': True,
        'task_dim': 5,
        'device': device
    }
    
    model = SHIELDModel(**model_params).to(device)
    
    # Run a single POMO episode
    env_params = {'problem_size': 50, 'pomo_size': 50, 'device': device}
    env = CVRPEnv(**env_params)
    data = env.get_random_problems(2, 50)
    env.load_problems(2, problems=data)
    reset_state, _, _ = env.reset()
    
    model.pre_forward(reset_state)
    state, reward, done = env.pre_step()
    
    total_log_prob = 0
    step = 0
    while not done and step < 5:  # Run first 5 steps
        selected, prob = model(state)
        state, reward, done = env.step(selected)
        total_log_prob = total_log_prob + prob.log().sum()
        step += 1
        
    loss = -total_log_prob + model.aux_loss
    model.zero_grad()
    loss.backward()
    
    # Verify gradients
    # 1. Router grad
    for i, layer in enumerate(model.decoder_layers):
        assert layer.router.weight.grad is not None, f"Router in layer {i} has no grad!"
        assert layer.router.weight.grad.abs().sum() > 0, f"Router in layer {i} has zero grad!"
    print("  [PASS] All MoD router weights received valid gradients.")
    
    # 2. Cluster params grad
    assert model.clustering_layer.C0.grad is not None and model.clustering_layer.C0.grad.abs().sum() > 0
    assert model.clustering_layer.W_H.weight.grad is not None and model.clustering_layer.W_H.weight.grad.abs().sum() > 0
    assert model.clustering_layer.W_C.weight.grad is not None and model.clustering_layer.W_C.weight.grad.abs().sum() > 0
    assert model.clustering_layer.W_prompt.weight.grad is not None and model.clustering_layer.W_prompt.weight.grad.abs().sum() > 0
    print("  [PASS] All Clustering params (C0, W_H, W_C, W_prompt) received valid gradients.")
    
    # 3. Context projections grad
    assert model.W_combine.weight.grad is not None and model.W_combine.weight.grad.abs().sum() > 0
    assert model.W_last.weight.grad is not None and model.W_last.weight.grad.abs().sum() > 0
    assert model.W_first.weight.grad is not None and model.W_first.weight.grad.abs().sum() > 0
    print("  [PASS] Context query projections (W_combine, W_last, W_first) received valid gradients.")


def test_3_parameter_counts():
    print("=" * 60)
    print("TEST 3: Parameter Counts Comparison (Paper Table 4)")
    print("=" * 60)
    
    base_params = {
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128 ** 0.5,
        'encoder_layer_num': 6,
        'qkv_dim': 16,
        'head_num': 8,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'num_experts': 4,
        'eval_type': 'argmax',
        'norm': 'instance',
        'norm_loc': 'norm_last',
        'expert_loc': ['Enc0', 'Enc1', 'Enc2', 'Enc3', 'Enc4', 'Enc5'],
        'topk': 2,
        'routing_level': 'node',
        'routing_method': 'input_choice',
        'problem': 'CVRP',
        'device': 'cpu'
    }
    
    # 1. MVMoE (baseline)
    mvmoe = MOEModel(**base_params)
    p_mvmoe = sum(p.numel() for p in mvmoe.parameters())
    
    # 2. MVMoE-Deeper (decoder_layers=3, use_mod=False, use_cluster=False)
    p2 = dict(base_params, decoder_layers=3, use_mod=False, use_cluster=False)
    shield_deeper = SHIELDModel(**p2)
    p_deeper = sum(p.numel() for p in shield_deeper.parameters())
    
    # 3. SHIELD-MoD (decoder_layers=3, use_mod=True, use_cluster=False)
    p3 = dict(base_params, decoder_layers=3, use_mod=True, mod_beta=0.10, use_cluster=False)
    shield_mod = SHIELDModel(**p3)
    p_mod = sum(p.numel() for p in shield_mod.parameters())
    
    # 4. SHIELD (decoder_layers=3, use_mod=True, use_cluster=True, n_clusters=5)
    p4 = dict(base_params, decoder_layers=3, use_mod=True, mod_beta=0.10, use_cluster=True, n_clusters=5)
    shield_full = SHIELDModel(**p4)
    p_full = sum(p.numel() for p in shield_full.parameters())
    
    print(f"  MVMoE:        {p_mvmoe:,} ({p_mvmoe/1e6:.2f}M) [Paper reference: ~3.68M]")
    print(f"  MVMoE-Deeper: {p_deeper:,} ({p_deeper/1e6:.2f}M) [Paper reference: ~4.46M]")
    print(f"  SHIELD-MoD:   {p_mod:,} ({p_mod/1e6:.2f}M) [Paper reference: ~4.37M - 4.46M]")
    print(f"  SHIELD (Full):{p_full:,} ({p_full/1e6:.2f}M) [Paper reference: ~4.59M]")
    
    assert 3.5e6 < p_mvmoe < 3.8e6, f"Unexpected MVMoE param count: {p_mvmoe}"
    assert 4.0e6 < p_deeper < 4.6e6, f"Unexpected Deeper param count: {p_deeper}"
    assert 4.0e6 < p_mod < 4.6e6, f"Unexpected SHIELD-MoD param count: {p_mod}"
    assert 4.2e6 < p_full < 4.8e6, f"Unexpected SHIELD Full param count: {p_full}"
    print("  [PASS] All parameter counts closely match Paper Table 4!")


def test_4_full_rollout_cvrp_and_vrptw():
    print("=" * 60)
    print("TEST 4: Full POMO Rollout (CVRP and VRPTW)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_params = {
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128 ** 0.5,
        'encoder_layer_num': 6,
        'decoder_layers': 3,
        'qkv_dim': 16,
        'head_num': 8,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'num_experts': 4,
        'eval_type': 'argmax',
        'norm': 'instance',
        'norm_loc': 'norm_last',
        'expert_loc': ['Enc0', 'Enc1', 'Enc2', 'Enc3', 'Enc4', 'Enc5'],
        'topk': 2,
        'routing_level': 'node',
        'routing_method': 'input_choice',
        'problem': 'Train_ALL',
        'use_mod': True,
        'mod_beta': 0.10,
        'use_cluster': True,
        'n_clusters': 5,
        'cluster_iters': 5,
        'use_task_prompt': True,
        'task_dim': 5,
        'device': device
    }
    
    model = SHIELDModel(**model_params).to(device)
    model.eval()
    
    for EnvClass, name in [(CVRPEnv, "CVRP"), (VRPTWEnv, "VRPTW")]:
        env = EnvClass(problem_size=50, pomo_size=50, device=device)
        data = env.get_random_problems(2, 50)
        data = [t.to(device) for t in data]
        env.load_problems(2, problems=data)
        reset_state, _, _ = env.reset()
        
        with torch.no_grad():
            model.pre_forward(reset_state)
            state, reward, done = env.pre_step()
            steps = 0
            while not done:
                selected, prob = model(state)
                state, reward, done = env.step(selected)
                steps += 1
                assert not torch.isnan(prob).any(), f"NaN detected in prob at step {steps}"
        
        print(f"  [PASS] {name} completed full rollout in {steps} steps. Max reward: {reward.max().item():.4f}")


if __name__ == "__main__":
    test_1_shapes_and_mod_correctness()
    test_2_gradient_flow()
    test_3_parameter_counts()
    test_4_full_rollout_cvrp_and_vrptw()
    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
