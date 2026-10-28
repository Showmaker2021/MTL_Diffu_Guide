"""
denoiser_probe.py -- two diagnostic probes for the SlotDiffModel latent denoiser.

Probe 1  (--probe mse):  Denoising MSE vs clean-signal fraction alpha.
    Corrupt z_0 = cat(xy, demand, tw_start, tw_end) via the model's own linear
    schedule  z_t = sqrt(alpha)*z_0 + sqrt(1-alpha)*eps  and predict eps under
    four slot conditions:  matched | shuffled | mean | unconditional.
    Reports per-feature normalized x0-space MSE and the "instance-specific
    information recovered fraction"
        frac(alpha,f) = 1 - L_matched(f) / L_shuffled(f)
    0  =>  matched slots add nothing over a different instance's slots;
    ~1 =>  the denoiser recovers the per-instance feature fully.

Probe 2  (--probe robust):  Slot-assignment consistency under coordinate noise.
    Nudge coordinates by N(0, sigma^2), sigma in [0.005, 0.05], keep demand/tw,
    and measure the fraction of customer->slot assignments that survive, on the
    diffusion-trained checkpoint vs the RL-only checkpoint. Direct test of
    "training with the diffusion objective yields more coordinate-robust slots".

Conventions inherited from the model (SlotDiffModel.py):
    * The denoiser is a NOISE predictor on a LINEAR schedule alpha_t = 1 - t/T.
    * z_0 is the 5-dim node row: xy(2), demand(1), tw_start(1), tw_end(1).
    * Slot init samples eps ~ N(0,1) FRESH on every forward (:81-82). Probe 2
      thus injects the SAME eps for a clean/noisy pair and averages over several
      eps draws; otherwise the number is dominated by init randomness.
    * Probe 1 keeps slots from a CLEAN encode fixed and perturbs only z_t. It
      does NOT re-run slot attention on noised features (that would silently
      switch to the noisy_slots training distribution and corrupt the semantics).
    * The "no diff" network is a separately-trained RL-only checkpoint (weights
      differ), so probe 2 is a training-objective ablation, not unplugging the
      denoiser at inference.

Timestamps are zero for these random problems (envs/VRPLEnv.py:230-231), so the
two tw channels have zero variance; their per-feature curves are flagged rather
than interpreted. Random-input envs only: --problem must be a single-base variant
(VRPL, CVRP, ...) whose get_random_problems returns (depot, xy, demand, route).
"""
import os
import sys
import argparse
import csv

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running from anywhere; utils.py + models/ live at the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils import get_env, get_model, seed_everything

FEAT_NAMES = ["xy", "xy", "demand", "tw_start", "tw_end"]
CONDS = ["matched", "shuffled", "mean", "unconditional"]


# =============================================================================
# Feature construction + slot attention (mirrors SlotDiffModel.pre_forward)
# =============================================================================
def build_z0(env, data, device):
    """Return (depot_xy [B,1,2], z0 [B,N,5]) from a random-problems tuple."""
    depot_xy, node_xy, node_demand, _ = data
    depot_xy = torch.Tensor(depot_xy).float().to(device)
    node_xy = torch.Tensor(node_xy).float().to(device)
    node_demand = torch.Tensor(node_demand).float().to(device)
    B, N, _ = node_xy.shape
    tw_start = torch.zeros(B, N, device=device)
    tw_end = torch.zeros(B, N, device=device)
    z0 = torch.cat((node_xy, node_demand[:, :, None], tw_start[:, :, None], tw_end[:, :, None]), dim=2)
    return depot_xy, z0


def slot_attention(sa, H, eps):
    """SlotAttentionModule.forward with a FIXED init eps (deterministic pairing).

    Same math as SlotDiffModel.py:74-99 but takes `eps` explicitly.
    Returns slots [B,K,D] and the customer-aware attention map [B,K,N+1].
    """
    B, N, D = H.shape
    mu = sa.slots_mu.expand(B, sa.slot_num, D)
    sigma = torch.exp(sa.slots_log_sigma.expand(B, sa.slot_num, D))
    slots = (mu + sigma * eps).reshape(B * sa.slot_num, D)
    attn_map = None
    for i in range(sa.iter_num):
        if i == sa.iter_num - 1:
            slots, attn_map = sa.iterative_attention(slots, H, return_attention=True)
        else:
            slots = sa.iterative_attention(slots, H, return_attention=False)
    return slots.reshape(B, sa.slot_num, D), attn_map


def encode(model, env, data, eps, device):
    """Clean encode -> (slots, attention, z0). Customers are attn[:, :, 1:]. """
    depot_xy, z0 = build_z0(env, data, device)
    H = torch.cat((model.encoder.embedding_depot(depot_xy),
                   model.encoder.embedding_node(z0)), dim=1)
    slots, attn = slot_attention(model.encoder.slot_attention_module, H, eps)
    return slots, attn, z0


def encode_noisy_coord(model, env, data, eps, sigma, device):
    """Encode a copy whose coordinates only are perturbed by N(0, sigma); eps fixed."""
    depot_xy, z0 = build_z0(env, data, device)
    z0_n = z0.clone()
    z0_n[:, :, :2] = (z0_n[:, :, :2] + sigma * torch.randn_like(z0_n[:, :, :2])).clamp(0, 1)
    H = torch.cat((model.encoder.embedding_depot(depot_xy),
                   model.encoder.embedding_node(z0_n)), dim=1)
    slots, attn = slot_attention(model.encoder.slot_attention_module, H, eps)
    return slots, attn, z0


# =============================================================================
# PROBE 1 -- denoising MSE vs alpha
# =============================================================================
def _deranged_perm(B, device):
    """Permutation with no fixed point; requires B >= 2, else identity."""
    if B < 2:
        return torch.arange(B, device=device)
    while True:
        p = torch.randperm(B, device=device)
        if not (p == torch.arange(B, device=device)).all():
            return p


@torch.no_grad()
def probe_mse(model, env, device, cfg, out_dir):
    denoiser = model.denoiser
    denoiser.eval()

    # Keep clear of alpha=1 (t=0: target eps is independent of z0 -> MSE=1 by
    # construction, uninformative) and alpha~0 (z0 fully erased, degenerate).
    alphas = np.linspace(0.05, 0.95, cfg.grid_alpha)
    t_for_alpha = [int((1.0 - a) * (model.max_timesteps - 1)) for a in alphas]

    num = len(alphas)
    feat = 5
    sum_x0 = {c: np.zeros((num, feat)) for c in CONDS}
    feat_var = np.zeros(feat)
    counts = 0

    for b in range(cfg.batches):
        data = env.get_random_problems(cfg.batch_size, cfg.problem_size)
        slots, _attn, z0 = encode(model, env, data, cfg.eps, device)   # CLEAN slots
        slots = slots.reshape(cfg.batch_size, cfg.slot_num, -1)
        B = cfg.batch_size

        feat_var_n = z0.var(dim=(0, 1)).cpu().numpy()
        assert B >= 2, "probe mse needs batch_size >= 2 for a non-identity shuffle"
        perm = _deranged_perm(B, device)
        mean_slots = slots.mean(dim=1, keepdim=True)        # [B,1,D] prototype
        zero_slots = torch.zeros_like(slots)

        for ai, (a, t) in enumerate(zip(alphas, t_for_alpha)):
            eps_n = torch.randn_like(z0)
            alpha = torch.full((B, 1, 1), a, device=device)
            z_t = torch.sqrt(alpha) * z0 + torch.sqrt(1 - alpha) * eps_n
            tvec = torch.full((B,), t, device=device)

            variants = {
                "matched": slots,
                "shuffled": slots[perm],
                "mean": mean_slots,
                "unconditional": zero_slots,
            }
            for c in CONDS:
                eps_hat = denoiser(z_t, tvec, variants[c])
                x0_hat = (z_t - torch.sqrt(1 - alpha) * eps_hat) / torch.sqrt(alpha)
                mse_x0 = (x0_hat - z0) ** 2
                sum_x0[c][ai] += mse_x0.mean(dim=(0, 1)).cpu().numpy()

        feat_var += feat_var_n
        counts += 1

    feat_var = np.maximum(feat_var / counts, 1e-6)
    avg_x0 = {c: s / counts for c, s in sum_x0.items()}
    norm_x0 = {c: avg_x0[c] / feat_var for c in CONDS}

    # recovered fraction in x0 space; clamp to [0,1]
    frac = np.clip(1.0 - avg_x0["matched"] / (avg_x0["shuffled"] + 1e-12), 0.0, 1.0)

    _save_mse(alphas, avg_x0, norm_x0, frac, feat_var, out_dir)
    print(f"   saved probe_mse.csv / probe_mse_x0.png / probe_mse_recovered_fraction.png  "
          f"[per-feature var = {np.round(feat_var, 4)}]")


def _save_mse(alphas, avg_x0, norm_x0, frac, feat_var, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for ai, a in enumerate(alphas):
        for c in CONDS:
            for f in range(5):
                rows.append({
                    "alpha": round(float(a), 4), "cond": c, "feature": FEAT_NAMES[f],
                    "mse_x0": float(avg_x0[c][ai][f]),
                    "mse_x0_norm": float(norm_x0[c][ai][f]),
                })
        for f in range(5):
            rows.append({
                "alpha": round(float(a), 4), "cond": "recovered_frac", "feature": FEAT_NAMES[f],
                "mse_x0": float(frac[ai][f]), "mse_x0_norm": float("nan"),
            })
    with open(os.path.join(out_dir, "probe_mse.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["alpha", "cond", "feature", "mse_x0", "mse_x0_norm"])
        w.writeheader(); w.writerows(rows)

    # per-feature normalized x0 MSE vs alpha (5 panels)
    fig, axes = plt.subplots(1, 5, figsize=(19, 3.4), sharey=True)
    for f in range(5):
        ax = axes[f]
        for c in CONDS:
            ax.plot(alphas, norm_x0[c][:, f], marker="o", ms=3, label=c)
        ax.axhline(1.0, color="grey", ls="--", lw=0.8)
        ax.set_title(f"feat: {FEAT_NAMES[f]} (var={feat_var[f]:.4f})")
        ax.set_xlabel("alpha (clean-signal fraction)")
    axes[0].set_ylabel("normalized x0 MSE")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "probe_mse_x0.png"), dpi=150); plt.close(fig)

    # recovered fraction vs alpha (per feature)
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))
    for f in range(5):
        ax2.plot(alphas, frac[:, f], marker="o", ms=4, label=f"feat {FEAT_NAMES[f]}")
    ax2.set_xlabel("alpha (clean-signal fraction)")
    ax2.set_ylabel("instance-info recovered fraction (x0)")
    ax2.set_ylim(0, 1.05); ax2.legend(fontsize=8, frameon=False); ax2.grid(alpha=0.3)
    fig2.tight_layout(); fig2.savefig(os.path.join(out_dir, "probe_mse_recovered_fraction.png"), dpi=150)
    plt.close(fig2)


# =============================================================================
# PROBE 2 -- assignment consistency under coordinate noise
# =============================================================================
def _pair_overlap(a_clean, a_noisy):
    """Fraction of customer nodes assigned to the same slot, slot order aligned.

    Hungarian-aligns *noisy* slot labels to *clean* slot labels by maximizing
    co-occurrence, then reports what fraction of customer->slot assignments
    survive. A pure relabeling (same partition) scores 1.0.
    """
    K = max(a_clean.max().item(), a_noisy.max().item()) + 1
    cnt = torch.zeros(K, K)
    for si in range(K):
        for ti in range(K):
            cnt[si, ti] = ((a_clean == si) & (a_noisy == ti)).sum()
    r, c = linear_sum_assignment(-cnt.numpy())   # clean-slot r -> noisy-slot c[r]
    # map noisy label -> matched clean label (inverse of c over the bijection)
    perm = torch.full((K,), -1, dtype=torch.long, device=a_clean.device)
    perm[c] = torch.tensor(r, dtype=torch.long, device=a_clean.device)
    mapped = perm[a_noisy]                     # [N]
    aligned = torch.where(mapped >= 0, mapped, a_noisy)
    return (a_clean == aligned).float().mean().item()


@torch.no_grad()
def probe_robust(model, env, device, cfg, out_dir, label):
    sigmas = [0.0] + list(np.linspace(0.005, 0.05, cfg.grid_sigma))
    rows = []

    for sigma in sigmas:
        ovl = 0.0
        total = 0
        for _e in range(cfg.num_eps):
            eps = torch.randn(cfg.batch_size, cfg.slot_num,
                              model.encoder.slot_attention_module.slots_mu.size(-1), device=device)
            for _b in range(cfg.batches):
                data = env.get_random_problems(cfg.batch_size, cfg.problem_size)
                _s, attn_c, _ = encode(model, env, data, eps, device)
                if sigma == 0.0:
                    attn_n = attn_c
                else:
                    _s, attn_n, _ = encode_noisy_coord(model, env, data, eps, sigma, device)
                a_c = attn_c[:, :, 1:].argmax(dim=1)   # [B,N]
                a_n = attn_n[:, :, 1:].argmax(dim=1)
                ovl += sum(_pair_overlap(a_c[b], a_n[b]) for b in range(cfg.batch_size))
                total += cfg.batch_size
        rows.append({"model": label, "sigma": float(sigma), "overlap": ovl / total})
        print(f"      sigma={sigma:.3f}  overlap={ovl / total:.4f}")

    fpath = os.path.join(out_dir, "probe_robust.csv")
    new = not os.path.exists(fpath)
    with open(fpath, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "sigma", "overlap"])
        if new:
            w.writeheader()
        w.writerows(rows)
    return rows


def plot_robust(out_dir):
    fpath = os.path.join(out_dir, "probe_robust.csv")
    if not os.path.exists(fpath):
        return
    with open(fpath) as fh:
        rows = list(csv.DictReader(fh))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for m in {r["model"] for r in rows}:
        mr = sorted([r for r in rows if r["model"] == m], key=lambda r: float(r["sigma"]))
        ax.plot([float(r["sigma"]) for r in mr], [float(r["overlap"]) for r in mr], marker="o", ms=4, label=m)
    ax.set_xlabel("coordinate noise sigma")
    ax.set_ylabel("assignment consistency (1 = identical)")
    ax.set_ylim(0, 1.05); ax.legend(frameon=False); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "probe_robust.png"), dpi=150); plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
def make_model(model_params, model_type, checkpoint, device, enable_diffusion):
    params = dict(model_params)
    params["enable_slot_diffusion"] = enable_diffusion
    params["device"] = device
    model = get_model(model_type)(**params).to(device)
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser(description="Denoiser slot probes")
    p.add_argument("--probe", choices=["mse", "robust", "both"], default="both")
    p.add_argument("--checkpoint", default="epoch-5000.pt", help="diffusion-trained checkpoint")
    p.add_argument("--checkpoint_nodiff", default="epoch-5000_no_diff.pt", help="RL-only checkpoint")
    p.add_argument("--model_type", default="SlotDiffMOEModel",
                   help="architecture of the checkpoint (SlotDiffModel or SlotDiffMOEModel)")
    p.add_argument("--problem", default="VRPL",
                   choices=["CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW"])
    p.add_argument("--problem_size", type=int, default=50)
    p.add_argument("--embedding_dim", type=int, default=128)
    p.add_argument("--slot_num", type=int, default=32)
    p.add_argument("--slot_iter_num", type=int, default=3)
    p.add_argument("--denoiser_heads", type=int, default=4)
    p.add_argument("--denoiser_layers", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--batches", type=int, default=4)
    p.add_argument("--num_eps", type=int, default=3, help="slot-init eps draws for probe 2")
    p.add_argument("--grid_alpha", type=int, default=9)
    p.add_argument("--grid_sigma", type=int, default=6)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--out_dir", default="results/denoiser_probe")
    args = p.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(">> device", device, "| checkpoint", args.checkpoint)

    env_cls = get_env(args.problem)[0]
    env = env_cls(problem_size=args.problem_size, pomo_size=args.problem_size, device=device)

    model_params = {
        "embedding_dim": args.embedding_dim,
        "sqrt_embedding_dim": args.embedding_dim ** 0.5,
        "encoder_layer_num": 6, "decoder_layer_num": 1,
        "qkv_dim": 16, "head_num": 8, "logit_clipping": 10,
        "ff_hidden_dim": 512, "num_experts": 4, "eval_type": "argmax",
        "norm": "instance", "norm_loc": "norm_last",
        "expert_loc": ["Enc0", "Enc1", "Enc2", "Enc3", "Enc4", "Enc5", "Dec"],
        "problem": args.problem,
        "topk": 2, "routing_level": "node", "routing_method": "input_choice",
        "slot_num": args.slot_num, "slot_iter_num": args.slot_iter_num,
        "max_timesteps": 1000,
        "denoiser_heads": args.denoiser_heads, "denoiser_layers": args.denoiser_layers,
        "diffusion_slot_mode": "noisy_slots",
    }

    # cfg object carries scalar knobs into probes; eps for probe1 is fixed seed-derived
    class _C: pass
    cfg = _C()
    cfg.batch_size = args.batch_size; cfg.batches = args.batches
    cfg.slot_num = args.slot_num; cfg.problem_size = args.problem_size
    cfg.grid_alpha = args.grid_alpha; cfg.grid_sigma = args.grid_sigma
    cfg.num_eps = args.num_eps
    cfg.eps = torch.randn(args.batch_size, args.slot_num, args.embedding_dim, device=device)

    if args.probe in ("mse", "both"):
        model = make_model(model_params, args.model_type, args.checkpoint, device,
                           enable_diffusion=True)
        if not hasattr(model, "denoiser"):
            print("!! checkpoint has no denoiser; skipping probe mse")
        else:
            probe_mse(model, env, device, cfg, args.out_dir)

    if args.probe in ("robust", "both"):
        model = make_model(model_params, args.model_type, args.checkpoint, device,
                           enable_diffusion=True)
        probe_robust(model, env, device, cfg, args.out_dir, label="SOLID (diff)")
        if args.checkpoint_nodiff and os.path.exists(args.checkpoint_nodiff):
            model_nd = make_model(model_params, args.model_type, args.checkpoint_nodiff, device,
                                  enable_diffusion=False)
            probe_robust(model_nd, env, device, cfg, args.out_dir, label="SOLID-no-diff (RL)")
        plot_robust(args.out_dir)
        print("   saved probe_robust.csv / probe_robust.png")


if __name__ == "__main__":
    main()
