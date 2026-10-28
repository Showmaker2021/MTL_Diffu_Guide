import os, random, math, time
import json
import pytz
import argparse
import pprint as pp
from datetime import datetime
from pathlib import Path

from Trainer import Trainer
from utils import *


def args2dict(args):
    env_params = {
        "problem_size": args.problem_size, 
        "pomo_size": args.pomo_size
    }
    
    model_params = {
        # Architecture
        "embedding_dim": args.embedding_dim, 
        "sqrt_embedding_dim": args.sqrt_embedding_dim,
        "encoder_layer_num": args.encoder_layer_num, 
        "decoder_layer_num": args.decoder_layer_num,
        "qkv_dim": args.qkv_dim, 
        "head_num": args.head_num, 
        "logit_clipping": args.logit_clipping,
        "ff_hidden_dim": args.ff_hidden_dim, 
        "num_experts": args.num_experts, 
        "eval_type": args.eval_type,
        "norm": args.norm, 
        "norm_loc": args.norm_loc, 
        "expert_loc": args.expert_loc, 
        "problem": args.problem,
        "topk": args.topk, 
        "routing_level": args.routing_level, 
        "routing_method": args.routing_method,
        
        # Slot Attention
        "slot_num": args.slot_num, 
        "slot_iter_num": args.slot_iter_num,
        
        # ========================================
        # DUAL-TASK: Slot Diffusion (NEW!)
        # ========================================
        "enable_slot_diffusion": args.enable_slot_diffusion,
        "max_timesteps": args.max_timesteps,
        "denoiser_heads": args.denoiser_heads,
        "denoiser_layers": args.denoiser_layers,
        "diffusion_slot_mode": args.diffusion_slot_mode,
        
        # Legacy Reconstruction (optional)
        "enable_slot_reconstruction": args.enable_slot_reconstruction,
        "demand_max1": not args.no_demand_max1,
        # ========================================
        # SHIELD Baseline Params (ICML 2025)
        # ========================================
        "decoder_layers": args.decoder_layers,
        "use_mod": args.use_mod,
        "mod_beta": args.mod_beta,
        "use_cluster": args.use_cluster,
        "n_clusters": args.n_clusters,
        "cluster_iters": args.cluster_iters,
        "use_task_prompt": args.use_task_prompt,
        "task_dim": args.task_dim,
        "softmax_dim": args.softmax_dim,
        "cluster_use_layernorm_state": args.cluster_use_layernorm_state,
        "cluster_include_depot": args.cluster_include_depot,
        "router_activation": args.router_activation,
    }
    
    optimizer_params = {
        "optimizer": {
            "lr": args.lr, 
            "weight_decay": args.weight_decay
        },
        "scheduler": {
            "milestones": args.milestones, 
            "gamma": args.gamma
        }
    }
    
    trainer_params = {
        "epochs": args.epochs, 
        "train_episodes": args.train_episodes,
        "train_batch_size": args.train_batch_size, 
        "log_interval": args.log_interval,
        "validation_interval": args.validation_interval,
        "model_save_interval": args.model_save_interval, 
        "checkpoint": args.checkpoint,
        
        # ========================================
        # DUAL-TASK: Loss Weights (NEW!)
        # ========================================
        "lambda_diffusion": args.lambda_diffusion,
        "lambda_recon": args.lambda_recon,
        "lambda_contrastive": args.lambda_contrastive,
    }

    return env_params, model_params, optimizer_params, trainer_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts + Slot Diffusion")
    
    # =========================================================================
    # ENV PARAMS
    # =========================================================================
    parser.add_argument('--problem', type=str, default="Train_ALL", 
                        choices=["Train_ALL", "CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW",
                                "OVRPB", "OVRPL", "VRPBL", "VRPBTW", "VRPLTW",
                                "OVRPBL", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"])
    parser.add_argument('--problem_size', type=int, default=50)
    parser.add_argument('--pomo_size', type=int, default=50, 
                        help="the number of start node, should <= problem size")

    # =========================================================================
    # MODEL PARAMS
    # =========================================================================
    parser.add_argument('--model_type', type=str, default="SlotDiffModel", choices=["MOE_LIGHT_Mixed", "MOE_Mixed", "MTL_Mixed","SINGLE", "MTL", "MOE", "MOE_LIGHT", "SlotDiffModel", "SlotModel", "SlotDiffMOEModel", "SHIELD", "MTLKD", "URS"])    
    # Basic Architecture
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--sqrt_embedding_dim', type=float, default=128**(1/2))
    parser.add_argument('--encoder_layer_num', type=int, default=None,
                        help="encoder layers (default: MTLKD 1, URS 12, other models 6)")
    parser.add_argument('--decoder_layer_num', type=int, default=None,
                        help="decoder layers (default: 6 for MTLKD, 1 for other models)")
    parser.add_argument('--qkv_dim', type=int, default=16)
    parser.add_argument('--head_num', type=int, default=8)
    parser.add_argument('--logit_clipping', type=float, default=None,
                        help='logit clipping (default: URS 50, other models 10)')
    parser.add_argument('--no_demand_max1', action='store_true',
                        help='disable URS node-demand rescaling by each instance maximum')
    parser.add_argument('--ff_hidden_dim', type=int, default=512)
    
    # MoE Settings
    parser.add_argument('--num_experts', type=int, default=4, 
                        help="the number of FFN in a MOE layer")
    parser.add_argument('--topk', type=int, default=2, 
                        help="how many experts (on average) to route for each input")
    parser.add_argument('--expert_loc', type=str, nargs='+', 
                        default=['Enc0', 'Enc1', 'Enc2', 'Enc3', 'Enc4', 'Enc5', 'Dec'], 
                        help="where to use MOE")
    parser.add_argument('--routing_level', type=str, default="node", 
                        choices=["node", "instance", "problem"], 
                        help="routing level for MOE")
    parser.add_argument('--routing_method', type=str, default="input_choice", 
                        choices=["input_choice", "expert_choice", "soft_moe", "random"], 
                        help="only for token-level and instance-level routing")
    
    # Evaluation & Normalization
    parser.add_argument('--eval_type', type=str, default="argmax", 
                        choices=["argmax", "softmax"])
    parser.add_argument('--norm', type=str, default=None,
                        choices=["batch", "batch_no_track", "instance", "layer", "rezero", "none"])
    parser.add_argument('--norm_loc', type=str, default="norm_last", 
                        choices=["norm_first", "norm_last"], 
                        help="whether conduct normalization before MHA/FFN/MOE")
    
    # Slot Attention
    parser.add_argument('--slot_num', type=int, default=16,
                        help="number of slots for slot attention")
    parser.add_argument('--slot_iter_num', type=int, default=3,
                        help="number of iterations for slot attention")
    
    # =========================================================================
    # DUAL-TASK: SLOT DIFFUSION (NEW!)
    # =========================================================================
    parser.add_argument('--enable_slot_diffusion', action='store_true', default=False,
                        help="Enable slot diffusion for representation learning")
    parser.add_argument('--max_timesteps', type=int, default=1000,
                        help="Maximum timesteps for diffusion process")
    parser.add_argument('--denoiser_heads', type=int, default=4,
                        help="Number of attention heads in denoiser")
    parser.add_argument('--denoiser_layers', type=int, default=2,
                        help="Number of transformer layers in denoiser")
    parser.add_argument('--diffusion_slot_mode', type=str, default='normal',
                        choices=['normal', 'shuffled', 'stopgrad', 'noisy_slots'],
                        help="How slots are fed to denoiser for ablation: normal, shuffled (batch derangement), stopgrad (S.detach()), noisy_slots (S_tilde = SlotAttention(Embed(Xt)))")
    
    # Legacy Reconstruction (optional, usually disabled)
    parser.add_argument('--enable_slot_reconstruction', action='store_true', default=False,
                        help="Enable legacy slot reconstruction loss")

    # =========================================================================
    # SHIELD BASELINE PARAMS (ICML 2025)
    # =========================================================================
    parser.add_argument('--decoder_layers', type=int, default=3,
                        help="number of decoder layers for SHIELD / deeper baseline")
    parser.add_argument('--use_mod', action=argparse.BooleanOptionalAction, default=True,
                        help="whether to use Mixture-of-Depths in SHIELD decoder")
    parser.add_argument('--mod_beta', type=float, default=0.10,
                        help="MoD routing capacity beta (top-beta agents per step)")
    parser.add_argument('--use_cluster', action=argparse.BooleanOptionalAction, default=True,
                        help="whether to use context-based clustering in SHIELD")
    parser.add_argument('--n_clusters', type=int, default=5,
                        help="number of clusters Nc in SHIELD")
    parser.add_argument('--cluster_iters', type=int, default=5,
                        help="number of iterations B in SHIELD clustering")
    parser.add_argument('--use_task_prompt', action=argparse.BooleanOptionalAction, default=True,
                        help="whether to prompt clusters with task gamma")
    parser.add_argument('--task_dim', type=int, default=5,
                        help="dimension of task/constraint vector gamma")
    parser.add_argument('--softmax_dim', type=str, default="cluster", choices=["cluster", "node"],
                        help="softmax axis for clustering psi")
    parser.add_argument('--cluster_use_layernorm_state', action=argparse.BooleanOptionalAction, default=False,
                        help="whether cluster state uses LayerNorm")
    parser.add_argument('--cluster_include_depot', action=argparse.BooleanOptionalAction, default=False,
                        help="whether depot is included in clustering")
    parser.add_argument('--router_activation', type=str, default="none", choices=["none", "sigmoid"],
                        help="activation for MoD router score")
    # =========================================================================
    # OPTIMIZER PARAMS
    # =========================================================================
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optimizer_type', choices=['Adam', 'AdamW'], default=None,
                        help='optimizer (default: URS AdamW, other methods Adam)')
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--milestones', type=int, nargs='+', default=None,
                        help='LR decay milestones (default: URS 451, other methods 4501)')
    parser.add_argument('--gamma', type=float, default=0.1, 
                        help='new_lr = lr * gamma')

    # =========================================================================
    # TRAINER PARAMS
    # =========================================================================
    parser.add_argument('--epochs', type=int, default=None,
                        help="final training epoch (default: 900 for MTLKD, 500 for URS, 5000 otherwise)")
    parser.add_argument('--train_episodes', type=int, default=10000 * 2, 
                        help="the num. of training instances per epoch")
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--log_interval', type=int, default=0,
                        help='print batch progress every N batches (0: epoch logs only)')
    parser.add_argument('--strict_aux_losses', action='store_true',
                        help='stop on auxiliary-loss errors or non-finite total loss')
    parser.add_argument('--validation_interval', type=int, default=5000)
    parser.add_argument('--model_save_interval', type=int, default=100)
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help="resume training")
    parser.add_argument('--teacher_dir', type=Path,
                        default=Path(__file__).resolve().parent / 'pretrained' / 'mtlkd')
    parser.add_argument('--teacher_size', type=int, default=100)
    parser.add_argument('--no_download_teachers', action='store_true',
                        help='require cached MTLKD teachers instead of downloading missing ones')
    parser.add_argument('--pomo_chunk_size', type=int, default=8,
                        help='MTLKD trajectories per chunk; gradients accumulate across chunks')
    
    # =========================================================================
    # DUAL-TASK: LOSS WEIGHTS (NEW!)
    # =========================================================================
    parser.add_argument('--lambda_diffusion', type=float, default=0.1,
                        help="Weight for slot diffusion loss (Nhánh 2: Representation Learning)")
    parser.add_argument('--lambda_recon', type=float, default=0.0,
                        help="Weight for legacy reconstruction loss (usually 0)")
    parser.add_argument('--lambda_contrastive', type=float, default=0.01,
                        help="Weight for slot contrastive loss (ensure slot diversity)")

    # =========================================================================
    # SETTINGS (GPU, SEED, etc.)
    # =========================================================================
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--log_dir', type=str, default="./results")
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--occ_gpu', type=float, default=0., 
                        help="preallocate a fraction of GPU memory; use sparingly")

    # =========================================================================
    # PARSE & VALIDATE
    # =========================================================================
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = {'MTLKD': 900, 'URS': 500}.get(args.model_type, 5000)
    if args.encoder_layer_num is None:
        args.encoder_layer_num = {'MTLKD': 1, 'URS': 12}.get(args.model_type, 6)
    if args.logit_clipping is None:
        args.logit_clipping = 50 if args.model_type == 'URS' else 10
    if args.optimizer_type is None:
        args.optimizer_type = 'AdamW' if args.model_type == 'URS' else 'Adam'
    if args.milestones is None:
        args.milestones = [451] if args.model_type == 'URS' else [4501]
    if args.norm is None:
        args.norm = 'none' if args.model_type == 'MTLKD' else 'instance'
    if args.decoder_layer_num is None:
        args.decoder_layer_num = 6 if args.model_type == 'MTLKD' else 1
    if args.model_type == 'URS':
        if args.enable_slot_diffusion or args.enable_slot_reconstruction:
            parser.error('Slot auxiliary objectives do not apply to URS')
        if args.norm != 'instance' or args.norm_loc != 'norm_last' or args.decoder_layer_num != 1:
            parser.error('URS uses instance normalization after residuals and a single adaptive decoder')
        args.lambda_diffusion = args.lambda_recon = args.lambda_contrastive = 0.0
        if Path(args.log_dir) == Path('results'):
            args.log_dir = str(Path(args.log_dir) / 'URS')
    if args.model_type == 'MTLKD':
        from baselines.mtlkd.training import main as train_mtlkd
        from baselines.mtlkd.protocol import TRAIN_PROBLEMS
        if args.problem != 'Train_ALL' and args.problem not in TRAIN_PROBLEMS:
            parser.error('MTLKD training requires Train_ALL or one of its six teacher tasks')
        if args.enable_slot_diffusion or args.enable_slot_reconstruction:
            parser.error('Slot auxiliary objectives do not apply to MTLKD')
        if args.encoder_layer_num != 1 or args.norm != 'none':
            parser.error('The upstream MTLKD architecture uses one encoder layer and --norm none')
        if args.optimizer_type != 'Adam':
            parser.error('The integrated MTLKD protocol uses Adam')
        if args.epochs < 1 or args.train_episodes < 1 or args.train_batch_size < 1 or args.pomo_chunk_size < 1:
            parser.error('epochs, episodes, batch size, and chunk size must be positive')
        args.log_dir = Path(args.log_dir)
        if args.log_dir == Path('results'):
            args.log_dir = args.log_dir / 'MTLKD'
        args.log_interval = args.log_interval or 10
        train_mtlkd(args=args)
        raise SystemExit(0)
    
    # Print configuration
    print("=" * 80)
    print("CONFIGURATION:")
    print("=" * 80)
    pp.pprint(vars(args))
    print("=" * 80)
    
    # Convert args to dict
    env_params, model_params, optimizer_params, trainer_params = args2dict(args)
    
    # Seed everything
    seed_everything(args.seed)

    # Validation
    if args.problem == "ALL" and args.model_type == "Single":
        assert False, "Cannot solve multiple problems with Single model, please use MTL/MOE/MOE_LIGHT instead."
    
    if args.enable_slot_diffusion:
        print("\n>> DUAL-TASK MODE ENABLED:")
        print(f"   - Nhánh 1 (Policy): VRP Solver")
        print(f"   - Nhánh 2 (Representation): Slot Diffusion")
        print(f"   - Diffusion Timesteps: {args.max_timesteps}")
        print(f"   - Denoiser: {args.denoiser_layers} layers, {args.denoiser_heads} heads")
        print(f"   - Loss Weights: λ_diffusion={args.lambda_diffusion}, λ_contrastive={args.lambda_contrastive}")
        print()

    # =========================================================================
    # SETUP LOGGING & GPU
    # =========================================================================
    torch.set_printoptions(threshold=1000000)
    process_start_time = datetime.now(pytz.timezone("Asia/Singapore"))
    args.log_path = os.path.join(args.log_dir, process_start_time.strftime("%Y%m%d_%H%M%S"))
    print(">> Log Path: {}".format(args.log_path))
    
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    
    if not args.no_cuda and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if not (0 <= args.gpu_id < gpu_count):
            print(f"ERROR: --gpu_id {args.gpu_id} out of range. This machine has {gpu_count} GPU(s): 0..{gpu_count-1}.")
            print(f"Available: {[torch.cuda.get_device_name(i) for i in range(gpu_count)]}")
            raise SystemExit(1)
        occumpy_mem(args) if args.occ_gpu != 0. else print(">> No occupation needed")
        args.device = torch.device('cuda', args.gpu_id)
        torch.cuda.set_device(args.gpu_id)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        args.device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')
    
    print(">> USE_CUDA: {}, CUDA_DEVICE_NUM: {}".format(not args.no_cuda, args.gpu_id))
    env_params['device'] = args.device
    model_params['device'] = args.device

    # =========================================================================
    # START TRAINING
    # =========================================================================
    print("\n" + "=" * 80)
    print(">> Start {} Training using {} Model ...".format(args.problem, args.model_type))
    print("=" * 80 + "\n")
    
    with open(os.path.join(args.log_path, 'config.json'), 'w') as config_file:
        json.dump(vars(args), config_file, indent=2, default=str)
        config_file.write('\n')

    trainer = Trainer(
        args=args, 
        env_params=env_params, 
        model_params=model_params, 
        optimizer_params=optimizer_params, 
        trainer_params=trainer_params
    )
    trainer.run()
    
    print("\n" + "=" * 80)
    print(">> Finish {} Training using {} Model ...".format(args.problem, args.model_type))
    print("=" * 80)
