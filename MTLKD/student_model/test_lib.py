import os
import torch
import sys

from Tester import Tester

from utils import *

# 环境配置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROBLEM = "CVRP"#VRPTW

def get_model_params():
    """返回模型参数配置"""
    return {
        "model_params": {
            "embedding_dim": 128,
            "sqrt_embedding_dim": 128 ** (1 / 2),
            "encoder_layer_num": 1,
            "decoder_layer_num": 6,
            "qkv_dim": 16,
            "head_num": 8,
            "logit_clipping": 10,
            "ff_hidden_dim": 512,
            "eval_type": "greedy",
            "norm": None,
            "test": False
        },
        "model_type": "MTLKD_model"
    }

def get_env_params(problem_size):
    """返回环境参数配置"""
    return {"problem_size": problem_size, "pomo_size": 1}

def get_test_params():
    """返回测试参数配置"""
    return {
        'compute_gap': True,
        'sample_size': 8,
        'aug_factor': 8,
        "log_dir": "./result",
        "checkpoint": "./result/epoch-850.pt",
        "R3C": False,
        "R3C_iter": 20,
    }


def setup_device(gpu_id=0):
    """设置计算设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda', gpu_id)
        torch.cuda.set_device(gpu_id)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')
    print(f">> Using device: {device}")
    return device

def run_testing(device):
    """执行CVRP测试"""
    test_params = get_test_params()
    model_params = get_model_params()

    print(f"\n>> Start Testing {PROBLEM}")
    current_params = test_params.copy()
    current_params.update({
        "problem": PROBLEM,
        "test_set_path": f"../data/CVRP-LIB",#cvrplib
        # "test_set_path": f"./data/Vrp-Set-Solomon",#vrptw
        "baseline":None,
        "episodes": 100,
        "batch_size": 100
    })
    tester = Tester(
        device=device,
        env_params=get_env_params(100),
        model_params=model_params,
        tester_params=current_params
    )
    tester.run()

if __name__ == "__main__":
    seed_everything(2025)
    device = setup_device(gpu_id=0)
    run_testing(device)
    print(">> CVRP Test Completed!")