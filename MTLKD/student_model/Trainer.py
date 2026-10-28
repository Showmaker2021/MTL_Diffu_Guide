import os
from random import random
import numpy as np
import torch
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from torch.utils.data import DataLoader
from data_loader import VRPDataset
from model.model import MTLKD_model
from utils import num_param, TimeEstimator, AverageMeter, show, get_opt_sol_path, load_dataset
from teacher_model.SINGLEModel import SINGLEModel
from env import VRP_Env, get_per_reset_state, get_per_state

def load_optimizer_with_loose_match(optimizer, ckpt_path, device='cpu'):
    """
    宽松加载 optimizer state_dict：跳过不匹配的 param_groups。
    返回 True 表示完全匹配；False 表示部分加载。
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    opt_state = ckpt.get('optimizer_state_dict', {})

    # 1. 先检查分组数量是否一致
    if len(opt_state['param_groups']) != len(optimizer.param_groups):
        print(f'[WARN] param_groups 数量不一致：{len(opt_state["param_groups"])} vs {len(optimizer.param_groups)}')
        return False

    # 2. 逐一比对张量形状
    matched_state = {'state': {}, 'param_groups': optimizer.state_dict()['param_groups']}
    for gid, (ckpt_group, curr_group) in enumerate(zip(opt_state['param_groups'], optimizer.param_groups)):
        if ckpt_group['params'] != curr_group['params']:
            print(f'[WARN] 第 {gid} 组参数 id 列表不匹配，跳过该组')
            continue
        matched_state['state'].update({pid: opt_state['state'][pid] for pid in ckpt_group['params'] if pid in opt_state['state']})
    optimizer.load_state_dict(matched_state)
    return True
def load_teacher_model(teacher_params,device):
    models=[]
    for param in teacher_params:
        if param['model_type'] == "SINGLE":
            model=SINGLEModel
        else:
            return NotImplementedError
        tem_model=model(**param)
        checkpoint = torch.load(param['model_path'], map_location=device)
        tem_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        tem_model.eval()
        models.append(tem_model)
    return models

class Trainer:
    def __init__(self, env_params, model_params, optimizer_params, trainer_params,teacher_params,log_path,device):
        # save arguments
        self.env_params = env_params
        self.trainProblemNum=len(self.env_params["train_problems"])
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        # self.set_seed(2025)
        self.device = device
        self.log_path = log_path
        self.result_log = {"val_score": [], "val_gap": []}

        self.env = VRP_Env(**self.env_params) # a list of envs classes (different problems), remember to initialize it!
        self.student_model = MTLKD_model(**self.model_params["model_params"])
        self.teacher_models = load_teacher_model(teacher_params,self.device)
        self.optimizer = Optimizer(self.student_model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])
        num_param(self.student_model)
        # Restore
        self.start_epoch = 1
        self.load_checkpoint()
        # utility
        self.time_estimator = TimeEstimator()
    def load_checkpoint(self):
        if self.trainer_params["checkpoint"] is not None:
            checkpoint_fullname = self.trainer_params["checkpoint"]
            checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
            self.student_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.start_epoch = 1 + checkpoint['epoch']
            load_optimizer_with_loose_match(self.optimizer, self.trainer_params["checkpoint"], device=self.device)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = checkpoint['epoch'] - 1
            print(">> Checkpoint (Epoch: {}) Loaded!".format(checkpoint['epoch']))
    def set_seed(self,seed=2025):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            print('=================================================================')
            self._train_one_epoch(epoch)
            self.scheduler.step()
            # Logs & Checkpoint
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            print("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['model_save_interval']
            val=self.trainer_params["use_val"]
            val_com_gap=self.trainer_params["val_com_gap"]
            if val==True:
                self._val(epoch,val_com_gap)
            if all_done or (epoch % model_save_interval == 0):
                print("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.student_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log
                }
                torch.save(checkpoint_dict, '{}/epoch-{}.pt'.format(self.log_path, epoch))
    def _val(self,epoch,val_com_gap):
        # MTL Validation & save latest images
        validation_interval = self.trainer_params['validation_interval']

        if epoch == 1 or (epoch % validation_interval == 0):
            val_problems = self.trainer_params['val_problem']
            val_episodes, problem_size = 1000, self.env_params['problem_size']
            dir = [os.path.join("../data", prob) for prob in val_problems]
            paths = ["{}{}_1000uni.pkl".format(prob.lower(), problem_size) for prob in val_problems]
            for i, path in enumerate(paths):
                # if no optimal solution provided, set compute_gap to False
                score, gap = self._val_and_stat(dir[i], path, self.env, problem=val_problems[i], batch_size=1000,
                                                val_episodes=val_episodes, compute_gap=val_com_gap)
                self.result_log["val_score"].append(score)
                self.result_log["val_gap"].append(gap)

            score_image_prefix = '{}/latest_val_score'.format(self.log_path)
            gap_image_prefix = '{}/latest_val_gap'.format(self.log_path)
            x, y1, y2, label = [], [], [], []
            for i, path in enumerate(paths):
                y1.append([r for j, r in enumerate(self.result_log["val_score"]) if j % len(paths) == i])
                y2.append([r for j, r in enumerate(self.result_log["val_gap"]) if j % len(paths) == i])
                x.append([j * validation_interval for j in range(len(y1[-1]))])
                label.append(val_problems[i])
            show(x, y1, label, title="Validation", xdes="Epoch", ydes="Score", path="{}.pdf".format(score_image_prefix))
            show(x, y2, label, title="Validation", xdes="Epoch", ydes="Opt. Gap (%)",
                 path="{}.pdf".format(gap_image_prefix))

    def _train_one_epoch(self, epoch):
        episode = 0
        score_AM, loss_AM = AverageMeter(), AverageMeter()
        train_num_episode = self.trainer_params['train_episodes']
        dataset=VRPDataset(self.trainer_params['train_episodes'],train_env=self.env.train_problems,
                           problem_size=self.env.problem_size,
                           per_batch=self.trainer_params["train_per_batch_size"],device=self.device)
        train_loader = DataLoader(dataset, batch_size=self.trainer_params["train_per_batch_size"]*self.trainProblemNum, shuffle=False, )
        for batch in train_loader:
            batch_size = batch[0].size(0)
            agv_loss,agv_score=self._train_one_batch(batch)
            score_AM.update(agv_score,1)
            loss_AM.update(agv_loss,1)
            episode += batch_size
        # Log Once, for each epoch
        print('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'.
              format(epoch, 100. * episode / train_num_episode, score_AM.avg, loss_AM.avg))



    def _compute_loss(self, selected, all_unselected_indices, stu_probs, state):
        # stu_probs的是以0，未被选择的节点，补0组成的节点的概率，那么我们需要从教师模型中获取对应的节点的probs,但是，对于补0的后面的概率，概率都应该为0
        tea_probss = []#教师模型给出的答案
        train_per_batch_size=self.trainer_params["train_per_batch_size"]
        for i in range(self.trainProblemNum):
            with torch.no_grad():
                per_state = get_per_state(state, i * train_per_batch_size, (i + 1) * train_per_batch_size)
                _, _, tea_probs = self.teacher_models[i](per_state, selected=selected[i * train_per_batch_size:(
                                                                                                                       i + 1) * train_per_batch_size],
                                                         returnProbs=True, return_att=False)
                tea_probs = tea_probs.float()
                unvistited_probs = torch.gather(tea_probs, 2, all_unselected_indices[i * train_per_batch_size:(
                                                                                                                  i + 1) * train_per_batch_size])

                # 对于后面补的0，替换概率为0，第一个0不替换
                unvistited_probs[:, :, 1:] = torch.where(
                    all_unselected_indices[i * train_per_batch_size:(i + 1) * train_per_batch_size, :, 1:] == 0,
                    torch.tensor(1e-9, dtype=torch.float32, device=tea_probs.device),
                    unvistited_probs[:, :, 1:]
                )
                tea_probss.append(unvistited_probs)

        tea_probss = torch.cat(tea_probss, dim=0)
        tea_probss = tea_probss + 1e-9
        stu_probs = stu_probs + 1e-9
        # loss_probs=10*nn.MSELoss()(tea_probss.detach(),stu_probs)
        loss_probs = (tea_probss * (tea_probss.log() - stu_probs.log())).sum(dim=2).mean()

        return loss_probs


    def _train_one_batch(self, batch):
        # ===== 初始化 =====
        self.student_model.train()
        self.student_model.set_eval_type("softmax")
        batch_size = batch[0].size(0)
        self.env.load_problems(batch_size, problems=batch, aug_factor=1)
        reset_state, _, _ = self.env.reset()
        state, reward, done = self.env.pre_step()
        train_per_batch_size=self.trainer_params["train_per_batch_size"]
        for i in range(self.trainProblemNum):
            with torch.no_grad():
                #初始状态划分，并为教师模型pre_forward
                tmp_reset_state=get_per_reset_state(reset_state,i * train_per_batch_size,(i + 1) * train_per_batch_size)
                self.teacher_models[i].pre_forward(tmp_reset_state)
        loss_list=[]
        while not done:
            torch.cuda.empty_cache()
            self.optimizer.zero_grad()
            selected,all_unselected_indices, prob, stu_probs = self.student_model(reset_state,state, returnProbs=True)
            if state.selected_count >2:
                loss = self._compute_loss(selected, all_unselected_indices, stu_probs, state)
                # 梯度更新
                loss.backward()
                # print(decoder_loss)
                loss_list.append(loss)
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                self.optimizer.step()
            # 状态更新
            state, reward, done = self.env.step(selected)

        reward1=reward.reshape(self.trainProblemNum, -1, self.env_params["pomo_size"])
        max_pomo_reward, _ = reward1.max(dim=2)  # get best results from pomo
        per_problem_reward= -max_pomo_reward.mean(dim=1)  # get best results from augmentation
        # 打印每个问题的奖励（动态适应 train_problems 长度）
        reward_info = ", ".join(
            f"{problem}: {per_problem_reward[i]:.2f}"
            for i, problem in enumerate(self.env_params['train_problems'])
        )
        # 安全计算 decoder loss
        avg_loss = sum(loss_list) / len(loss_list) if loss_list else 0.0
        # 最终打印
        print(f"{reward_info}, loss_decoder: {avg_loss:.4f}")
        return avg_loss,sum(per_problem_reward) / len(per_problem_reward)

    def _val_one_batch(self, data, env, aug_factor=1, eval_type="argmax"):
        self.student_model.eval()
        self.student_model.set_eval_type(eval_type)
        batch_size = data.size(0) if isinstance(data, torch.Tensor) else data[0].size(0)
        with torch.no_grad():
            env.load_problems(batch_size, problems=data, aug_factor=aug_factor)
            reset_state, _, _ = self.env.reset()
            state, reward, done = self.env.pre_step()
            while not done:
                selected, unselected_indices, prob, stu_probs = self.student_model(reset_state, state, returnProbs=True)
                state, reward, done = self.env.step(selected)
        # Return
        aug_reward = reward.reshape(aug_factor, batch_size, env.pomo_size)
        # shape: (augmentation, batch, pomo)
        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        no_aug_score = -max_pomo_reward[0, :].float()  # negative sign to make positive value
        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        aug_score = -max_aug_pomo_reward.float()  # negative sign to make positive value

        return no_aug_score, aug_score

    def _val_and_stat(self,dir, val_path, env,problem, batch_size=500, val_episodes=1000, compute_gap=False):
        no_aug_score_list, aug_score_list, no_aug_gap_list, aug_gap_list = [], [], [], []
        episode, no_aug_score, aug_score = 0, torch.zeros(0).to(self.device), torch.zeros(0).to(self.device)

        while episode < val_episodes:
            # torch.cuda.empty_cache()
            remaining = val_episodes - episode
            bs = min(batch_size, remaining)
            data = env.load_dataset(os.path.join(dir, val_path),problem=problem, offset=episode, num_samples=bs)
            no_aug, aug = self._val_one_batch(data, env, aug_factor=1, eval_type="argmax")

            # print("my:",no_aug0.mean(),"orl",no_aug.mean())
            no_aug_score = torch.cat((no_aug_score, no_aug), dim=0)
            aug_score = torch.cat((aug_score, aug), dim=0)
            episode += bs
            # print(aug_score.mean().item())

        no_aug_score_list.append(round(no_aug_score.mean().item(), 4))
        aug_score_list.append(round(aug_score.mean().item(), 4))

        if compute_gap:
            opt_sol = load_dataset(get_opt_sol_path(dir, problem, data[1].size(1)), disable_print=True)[: val_episodes]
            opt_sol = [i[0] for i in opt_sol]
            gap = [(no_aug_score[j].item() - opt_sol[j]) / opt_sol[j] * 100 for j in range(val_episodes)]
            no_aug_gap_list.append(round(sum(gap) / len(gap), 4))
            gap = [(aug_score[j].item() - opt_sol[j]) / opt_sol[j] * 100 for j in range(val_episodes)]
            aug_gap_list.append(round(sum(gap) / len(gap), 4))
            print(">> Val Score on {}: NO_AUG_Score: {}, NO_AUG_Gap: {}% --> AUG_Score: {}, AUG_Gap: {}% ".format(val_path, no_aug_score_list, no_aug_gap_list, aug_score_list, aug_gap_list))
            return aug_score_list[0], aug_gap_list[0]
        else:
            print(">> Val Score on {}: NO_AUG_Score: {}, --> AUG_Score: {}".format(val_path, no_aug_score_list, aug_score_list))
            return aug_score_list[0], 0