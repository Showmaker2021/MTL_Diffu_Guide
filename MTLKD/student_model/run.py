import pytz
import argparse
import pprint as pp
from datetime import datetime
import os

import torch

from student_model.utils import seed_everything

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"  # 建议初始值
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # PSDNCO for utils
from Trainer import Trainer
from utils import *

def args2dict():
    env_params = {"problem_size":100, "pomo_size": 1,"train_problems":["CVRP","VRPTW","OVRP","VRPL","VRPB","OVRPTW"]}
    model_params={"model_params":  {"embedding_dim": 128, "sqrt_embedding_dim": 128**(1/2),
                    "encoder_layer_num": 1, "decoder_layer_num": 6,
                    "qkv_dim": 16, "head_num": 8, "logit_clipping": 10,
                    "ff_hidden_dim": 512,  "eval_type": "softmax",
                    "norm": None,   "use_attention_scoring":True
                    },
           "model_type":"MTLKD_model"
    }
    optimizer_params = {"optimizer": {"lr": 1e-4, "weight_decay": 1e-6},
                        "scheduler": {"milestones": [500,600,700,800,850], "gamma":0.5}}
    trainer_params = {
        "log_dir":"./result",
        "epochs": 900, "train_episodes":6*250*100,
        "train_per_batch_size": 250, "use_val":True,"validation_interval": 50,"val_com_gap":True,
        "val_data_dir":"../data",
        "model_save_interval": 10, "checkpoint": None,"val_problem":["CVRP","VRPTW","OVRP","VRPL","VRPB","OVRPTW"],
                      # "rl_loss":False,"emb_loss":False,"emb_att":False,'probs':True,'decoder_att':False,
                      }
    teacher_params = [
        {'problem': "CVRP",
         'eval_type': 'argmax',
         'norm_loc': "norm_last",
         'norm': "instance",
         'model_type': "SINGLE",
         'embedding_dim': 128,
         'sqrt_embedding_dim': 128 ** (1 / 2),
         'encoder_layer_num': 6,
         'decoder_layer_num': 1,
         'qkv_dim': 16,
         'head_num': 8,
         'logit_clipping': 10,
         'ff_hidden_dim': 512,
         'model_path': "../pretrained/cvrp100.pt"},

        {'problem': "VRPTW",
         'eval_type': 'argmax',
         'norm': "instance",
         'norm_loc': "norm_last",
         'model_type': "SINGLE",
         'embedding_dim': 128,
         'sqrt_embedding_dim': 128 ** (1 / 2),
         'encoder_layer_num': 6,
         'decoder_layer_num': 1,
         'qkv_dim': 16,
         'head_num': 8,
         'logit_clipping': 10,
         'ff_hidden_dim': 512,
         'model_path': "../pretrained/vrptw100.pt"
         },
        {'problem': "OVRP",
         'eval_type': 'argmax',
         'norm': "instance",
         'norm_loc': "norm_last",
         'model_type': "SINGLE",
         'embedding_dim': 128,
         'sqrt_embedding_dim': 128 ** (1 / 2),
         'encoder_layer_num': 6,
         'decoder_layer_num': 1,
         'qkv_dim': 16,
         'head_num': 8,
         'logit_clipping': 10,
         'ff_hidden_dim': 512,
         'model_path': "../pretrained/ovrp100.pt"
         },
        {'problem': "VRPL",
         'eval_type': 'argmax',
         'norm': "instance",
         'norm_loc': "norm_last",
         'model_type': "SINGLE",
         'embedding_dim': 128,
         'sqrt_embedding_dim': 128 ** (1 / 2),
         'encoder_layer_num': 6,
         'decoder_layer_num': 1,
         'qkv_dim': 16,
         'head_num': 8,
         'logit_clipping': 10,
         'ff_hidden_dim': 512,
         'model_path': "../pretrained/vrpl100.pt"
         },
        {'problem': "VRPB",
         'eval_type': 'argmax',
         'norm': "instance",
         'norm_loc': "norm_last",
         'model_type': "SINGLE",
         'embedding_dim': 128,
         'sqrt_embedding_dim': 128 ** (1 / 2),
         'encoder_layer_num': 6,
         'decoder_layer_num': 1,
         'qkv_dim': 16,
         'head_num': 8,
         'logit_clipping': 10,
         'ff_hidden_dim': 512,
         'model_path': "../pretrained/vrpb100.pt"
         },

        {'problem': "OVRPTW",
         'eval_type': 'argmax',
         'norm': "instance",
         'norm_loc': "norm_last",
         'model_type': "SINGLE",
         'embedding_dim': 128,
         'sqrt_embedding_dim': 128 ** (1 / 2),
         'encoder_layer_num': 6,
         'decoder_layer_num': 1,
         'qkv_dim': 16,
         'head_num': 8,
         'logit_clipping': 10,
         'ff_hidden_dim': 512,
         'model_path': "../pretrained/ovrptw100.pt"
         },

    ]
    return env_params, model_params, optimizer_params, trainer_params,teacher_params


if __name__ == "__main__":
    seed=2025
    gpu_id=1
    env_params, model_params, optimizer_params, trainer_params ,teacher_params= args2dict()
    seed_everything(seed)

    print("env_params",env_params)
    print("model_params",model_params)
    print("optimizer_params",optimizer_params)
    print("trainer_params",trainer_params )
    # set log & gpu
    torch.set_printoptions(threshold=1000000)
    process_start_time = datetime.now(pytz.timezone("Asia/Singapore"))
    log_dir=trainer_params["log_dir"]
    log_path = os.path.join(log_dir, process_start_time.strftime("%Y%m%d_%H%M%S"))
    print(">> Log Path: {}".format(log_path))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if torch.cuda.is_available():
        device = torch.device('cuda',gpu_id)
        torch.cuda.set_device(gpu_id)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')
    print(">> USE_CUDA_DEVICE_NUM: {}".format( gpu_id))

    # start training
    print(">> Start {} Training using {} Model ...".format(env_params["train_problems"], model_params["model_type"]))
    trainer = Trainer(env_params=env_params, model_params=model_params, optimizer_params=optimizer_params, trainer_params=trainer_params,teacher_params=teacher_params,log_path=log_path,device=device)
    trainer.run()
    print(">> Finish {} Training using {} Model ...")
