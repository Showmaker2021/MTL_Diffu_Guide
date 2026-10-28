import pytz
import argparse
import pprint as pp
from datetime import datetime
import os
import torch
import sys
from Tester import Tester
from utils import seed_everything

# 环境配置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 添加项目根目录
sys.path.insert(0, ".")
sys.path.insert(0, "..")
sys.path.insert(0, "../..")
sys.path.insert(0, "../../..")

from utils import *

# 全局配置
PROBLEM_TYPES = [
    "CVRP", "VRPTW", "VRPB", "VRPL", "OVRP",
    "OVRPB", "OVRPL", "VRPBL", "VRPBTW", "VRPLTW",
    "OVRPBL", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"
]
PROBLEM_SIZES = [100, 200, 500, 1000]
EPISODE_CONFIG = {100: 1000, 200: 128, 500: 128, 1000: 128}
BATCH_SIZE_CONFIG={100: 1000, 200: 128, 500: 128, 1000: 128}
BASELINE_SOLVERS = {
    "CVRP": "pyvrp",
    "VRPTW": "pyvrp",
    "VRPB": "or_tools",
    "VRPL": "pyvrp",
    "OVRP": "pyvrp",
    "OVRPTW": "pyvrp",
    "OVRPB": "or_tools",
    "OVRPL": "pyvrp",
    "VRPBL": "or_tools",
    "VRPBTW": "or_tools",
    "VRPLTW": "pyvrp",
    "OVRPBL": "or_tools",
    "OVRPBTW": "or_tools",
    "OVRPLTW": "pyvrp",
    "VRPBLTW": "or_tools",
    "OVRPBLTW": "or_tools"
}


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
        "model_type": "HD_model"
    }


def get_env_params(problem_size):
    """返回环境参数配置"""
    return {"problem_size": problem_size, "pomo_size": 1}


def get_test_params():
    """返回测试参数配置"""
    return {
        'compute_gap': True,
        'sample_size': 8,
        'aug_factor': 8,  # 8-fold VRP augmentation for the AUGMENTATION score
        "log_dir": "./result",
        "checkpoint": "./result/epoch-850.pt",
        "R3C": False, #
        "R3C_iter": 200,
    }


def generate_filenames(problem, size):
    """自动生成数据集和baseline文件名"""
    dataset = f"{problem.lower()}{size}_{EPISODE_CONFIG[size]}uni.pkl"
    baseline = f"{BASELINE_SOLVERS[problem]}_{dataset}"
    return dataset, baseline


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


def run_testing(device, test_problems, test_sizes):
    """执行测试流程"""
    test_params = get_test_params()
    model_params = get_model_params()

    for problem in test_problems:
        print(f"\n>> Start Testing {problem}")
        for size in test_sizes:
            print(f"\n=== Problem Size: {size} ===")

            # 生成配置
            dataset, baseline = generate_filenames(problem, size)
            env_params = get_env_params(size)

            # 更新测试参数
            current_params = test_params.copy()
            current_params.update({
                "problem": problem,
                "test_set_path": f"../data/{problem}/{dataset}",
                "baseline": f"../data/{problem}/{baseline}",
                "episodes": EPISODE_CONFIG[size],
                "batch_size": BATCH_SIZE_CONFIG[size]
            })

            # 运行测试
            tester = Tester(
                device=device,
                env_params=env_params,
                model_params=model_params,
                tester_params=current_params
            )
            tester.run()


if __name__ == "__main__":
    # 初始化配置
    seed = 2025
    gpu_id =0
    seed_everything(seed)

    test_problems = ["CVRP", "VRPTW", "VRPB", "VRPL", "OVRP", "OVRPB", "OVRPL", "VRPBL",
                     "VRPBTW", "VRPLTW", "OVRPBL", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"]

    test_sizes = [100,500,1000]

    # Default inference technique: greedy
    # If you want to use the proposed R3C as inference technique, please modify the parameter "R3C" into True, and customize the
    # budget by changing "R3C_iter".

    device = setup_device(gpu_id)

    # 运行测试
    run_testing(device, test_problems, test_sizes)
    print(">> All Tests Completed!")