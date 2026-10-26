import numpy as np
import torch
from torch.utils.data import Dataset
import random


class VRPDataset(Dataset):
    def __init__(self, num_samples, train_env, problem_size, per_batch, device, seed=2025):
        # 设置所有相关的随机种子
        self.seed = seed
        # self._set_seeds()

        self.num_samples = num_samples
        self.problem_size = problem_size
        self.backhaul_ratio = 0.2
        self.train_problems = train_env
        self.num_pro = len(self.train_problems)
        self.per_batch = per_batch
        self.batch = self.per_batch * self.num_pro
        self.speed = 1.0
        self.depot_start, self.depot_end = 0., 3.  # tw for depot [0, 3]
        self.device = device

    def _set_seeds(self):
        """设置所有相关的随机种子以确保可重复性"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 调用您的数据生成函数（需稍作修改以支持单样本生成）

        current_pro=self.train_problems[int(idx%self.batch/self.per_batch)]
        depot_xy, node_xy, node_demand, route_limit, service_time, tw_start, tw_end, problems_type = self.vrpbltw_data_to_others(
            self.get_data(1,self.problem_size,normalized=True), current_pro)

        return (depot_xy.squeeze(0),
            node_xy.squeeze(0),
            node_demand.squeeze(0),
             route_limit.squeeze(0),
            service_time.squeeze(0),
           tw_start.squeeze(0),
            tw_end.squeeze(0),
             problems_type # 取单个元素
                )
        # 返回字典形式的数据（可根据模型输入需求调整）
        # return {
        #     "depot_xy": depot_xy.squeeze(0),  # 去除batch维度
        #     "node_xy": node_xy.squeeze(0),
        #     "node_demand": node_demand.squeeze(0),
        #     "route_limit": route_limit.squeeze(0),
        #     "service_time": service_time.squeeze(0),
        #     "tw_start": tw_start.squeeze(0),
        #     "tw_end": tw_end.squeeze(0),
        #     "problem_type": problems_type # 取单个元素
        # }

    def vrpbltw_data_to_others(self, vrpbltw_data, target_problem):
        depot_xy, node_xy, node_demand, route_limit, service_time, tw_start, tw_end = vrpbltw_data

        if "B" not in target_problem:
            node_demand = abs(node_demand)
        if "L" not in target_problem:
            route_limit *= 0.
        if "TW" not in target_problem:
            service_time *= 0.
            tw_start *= 0.
            tw_end *= 0.
        return depot_xy, node_xy, node_demand, route_limit, service_time, tw_start, tw_end, target_problem

    def get_data(self, batch_size, problem_size, normalized=True):
        depot_xy = torch.rand(size=(batch_size, 1, 2))  # (batch, 1, 2)
        node_xy = torch.rand(size=(batch_size, problem_size, 2))  # (batch, problem, 2)

        if problem_size == 20:
            demand_scaler = 30
        elif problem_size == 50:
            demand_scaler = 40
        elif problem_size == 100:
            demand_scaler = 50
        elif problem_size == 200:
            demand_scaler = 70
        else:
            raise NotImplementedError

        route_limit = torch.ones(batch_size)*3.0

        # time windows (vehicle speed = 1.):
        #   1. The setting of "MTL for Routing Problem with Zero-Shot Generalization".
        """
        self.depot_start, self.depot_end = 0., 4.6.
        a, b, c = 0.15, 0.18, 0.2
        service_time = a + (b - a) * torch.rand(batch_size, problem_size)
        tw_length = b + (c - b) * torch.rand(batch_size, problem_size)
        c = (node_xy - depot_xy).norm(p=2, dim=-1)
        h_max = (self.depot_end - service_time - tw_length) / c * self.speed - 1
        tw_start = (1 + (h_max - 1) * torch.rand(batch_size, problem_size)) * c / self.speed
        tw_end = tw_start + tw_length
        """
        #   2. See "Learning to Delegate for Large-scale Vehicle Routing" in NeurIPS 2021.
        #   Note: this setting follows a similar procedure as in Solomon, and therefore is more realistic and harder.
        service_time = torch.ones(batch_size, problem_size) * 0.2
        travel_time = (node_xy - depot_xy).norm(p=2, dim=-1) / self.speed
        a, b = self.depot_start + travel_time, self.depot_end - travel_time - service_time
        time_centers = (a - b) * torch.rand(batch_size, problem_size) + b
        time_half_width = (service_time / 2 - self.depot_end / 3) * torch.rand(batch_size,
                                                                               problem_size) + self.depot_end / 3
        tw_start = torch.clamp(time_centers - time_half_width, min=self.depot_start, max=self.depot_end)
        tw_end = torch.clamp(time_centers + time_half_width, min=self.depot_start, max=self.depot_end)
        # shape: (batch, problem)

        # check tw constraint: feasible solution must exist (i.e., depot -> a random node -> depot must be valid).
        instance_invalid, round_error_epsilon = False, 0.00001
        total_time = torch.max(0 + (depot_xy - node_xy).norm(p=2, dim=-1) / self.speed, tw_start) + service_time + (
                node_xy - depot_xy).norm(p=2, dim=-1) / self.speed > self.depot_end + round_error_epsilon
        # (batch, problem)
        instance_invalid = total_time.any()

        if instance_invalid:
            print(">> Invalid instances, Re-generating ...")
            return self.get_data(batch_size, problem_size, normalized=normalized)
        elif normalized:
            node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(
                demand_scaler)  # (batch, problem)
            backhauls_index = torch.randperm(problem_size)[:int(problem_size * self.backhaul_ratio)]
            node_demand[:, backhauls_index] = -1 * node_demand[:, backhauls_index]
            return depot_xy, node_xy, node_demand, route_limit, service_time, tw_start, tw_end
        else:
            node_demand = torch.Tensor(
                np.random.randint(1, 10, size=(batch_size, problem_size)))  # (unnormalized) shape: (batch, problem)

            capacity = torch.Tensor(np.full(batch_size, demand_scaler))
            return depot_xy, node_xy, node_demand, capacity, route_limit, service_time, tw_start, tw_end
