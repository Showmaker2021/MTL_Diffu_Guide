import os, re, time

import torch
from torch.optim import Adam as Optimizer

from env import VRP_Env, random_rearrange_subtours, parallel_split_solutions
from model.model import MTLKD_model
from utils import *

from torch.utils.data import DataLoader, TensorDataset

class Tester:
    def __init__(self, device, env_params, model_params, tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # ENV, MODEL, & Load checkpoint
        self.env = VRP_Env(**env_params)  # Env Class
        self.check_env=VRP_Env(**env_params)
        self.device = device


        self.model = MTLKD_model(**self.model_params["model_params"])
        # num_param(self.model)

        checkpoint_fullname = tester_params["checkpoint"]
        checkpoint = torch.load(checkpoint_fullname, map_location=self.device,weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # print(">> Checkpoint (Epoch: {}) Loaded!".format(checkpoint['epoch']))
        if tester_params['test_set_path'] is None or tester_params['test_set_path'].endswith(".pkl"):
            self.data_dir = "./data"
        else:
            self.path_list = [os.path.join(tester_params['test_set_path'], f) for f in sorted(os.listdir(tester_params['test_set_path']))] \
                if os.path.isdir(tester_params['test_set_path']) else [tester_params['test_set_path']]
            assert self.path_list[-1].endswith(".vrp") or self.path_list[-1].endswith(".txt"), "Unsupported file types."
        self.time_estimator = TimeEstimator()

    def run(self):
        if self.tester_params['test_set_path'] is None or self.tester_params['test_set_path'].endswith(".pkl"):
            start_time = time.time()
            self._test(compute_gap=self.tester_params["compute_gap"])
            print(">> Evaluation finished within {:.2f}s\n".format(time.time() - start_time))
        else:
            for path in self.path_list:
                if self.tester_params['problem']=='CVRP':
                    self._solve_cvrplib(self.model, path, self.env)
                elif self.tester_params['problem']=='VRPTW':
                    self._solve_cvrptwlib(self.model, path, self.env)
                else:
                    raise NotImplementedError



    def _test(self, compute_gap=False):
        self.time_estimator.reset()
        score_AM, gap_AM = AverageMeter(), AverageMeter()
        aug_score_AM, aug_gap_AM = AverageMeter(), AverageMeter()
        scores, aug_scores = torch.zeros(0).to(self.device), torch.zeros(0).to(self.device)
        if self.tester_params["R3C"]:
            r3c_score_AM, r3c_gap_am = AverageMeter(), AverageMeter()
            r3c_aug_score_AM, r3c_aug_gap_AM = AverageMeter(), AverageMeter()
            r3c_scores, r3c_aug_scores = torch.zeros(0).to(self.device), torch.zeros(0).to(self.device)
        episode, test_num_episode = 0, self.tester_params['episodes']
        data_path = self.tester_params['test_set_path']
        # data0 = self.env.lehd_txt_data(data_path, test_num_episode)
        opt_sols=[]
        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['batch_size'], remaining)
            data = self.env.load_dataset(data_path, problem=self.tester_params['problem'], offset=episode, num_samples=batch_size)
            # batch = (data0[0][episode:end_idx],data0[1][episode:end_idx],data0[2][episode:end_idx],data0[3][episode:end_idx],data0[4][episode:end_idx],
            #          data0[5][episode:end_idx],data0[6][episode:end_idx],data0[7][episode:end_idx])
            if self.tester_params["R3C"]:
                score, aug_score, all_score, all_aug_score,rrc_score, rrc_aug_score, rrc_all_score, rrc_all_aug_score = self._test_one_batch(self.model, data, self.env)
                r3c_score_AM.update(rrc_score, batch_size)
                r3c_aug_score_AM.update(rrc_aug_score, batch_size)
                r3c_scores = torch.cat((r3c_scores, rrc_all_score), dim=0)
                r3c_aug_scores = torch.cat((r3c_aug_scores, rrc_all_aug_score), dim=0)
            else:
                score, aug_score, all_score, all_aug_score = self._test_one_batch(self.model, data, self.env)
            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)
            scores = torch.cat((scores, all_score), dim=0)
            aug_scores = torch.cat((aug_scores, all_aug_score), dim=0)
            episode += batch_size
        mean_opt=0
        if compute_gap:


            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)

            all_done = (episode == test_num_episode)

            if all_done:
                opt_sol = load_dataset(self.tester_params["baseline"], disable_print=True)[
                          0:test_num_episode]
                opt_sol = [abs(i[0]) for i in opt_sol]
                opt_sol_mean = sum(opt_sol) / len(opt_sol)
                gap = (score_AM.avg - opt_sol_mean) / opt_sol_mean * 100
                aug_gap = (aug_score_AM.avg - opt_sol_mean) / opt_sol_mean * 100
                if self.tester_params["R3C"]:
                    r3c_gap = (r3c_score_AM.avg - opt_sol_mean) / opt_sol_mean * 100
                    r3c_aug_gap = (r3c_aug_score_AM.avg - opt_sol_mean) / opt_sol_mean * 100


                mean_opt = sum(opt_sol[0:test_num_episode]) / test_num_episode


                print(f" \n*** Test Done on {self.tester_params['problem']} n{self.env.problem_size}*** ")
                print(" NO-AUG SCORE: {:.4f} ({:.4f}%)".format(score_AM.avg, gap))
                print(" AUGMENTATION SCORE: {:.4f} ({:.4f}%) ".format(aug_score_AM.avg,aug_gap))
                if self.tester_params["R3C"]:
                    print(" rrc_NO-AUG SCORE: {:.4f} ({:.4f}%)".format(r3c_score_AM.avg, r3c_gap))
                    print(" rrc_AUGMENTATION SCORE: {:.4f} ({:.4f}%) ".format(r3c_aug_score_AM.avg, r3c_aug_gap))
                print(f"baseline{self.tester_params['baseline']}:{mean_opt} ")

        return scores, aug_scores

    def _test_one_batch(self, model, test_data, env):
        aug_factor = self.tester_params['aug_factor']
        batch_size = test_data.size(0) if isinstance(test_data, torch.Tensor) else test_data[0].size(0)
        sample_size = 1

        # Sampling: augment data based on sample_size: [batch_size, ...] -> [batch_size x sample_size, ...]
        # if self.model_params["model_params"]['eval_type'] == "softmax":
        #     test_data = list(test_data)
        #     for i, data in enumerate(test_data):
        #         if isinstance(data, list):
        #             test_data[i]=data*(sample_size)
        #         else:
        #             if data.dim() == 1:
        #                 test_data[i] = data.repeat(sample_size)
        #             elif data.dim() == 2:
        #                 test_data[i] = data.repeat(sample_size, 1)
        #             elif data.dim() == 3:
        #                 test_data[i] = data.repeat(sample_size, 1, 1)

        # Ready
        model.eval()
        model.set_eval_type( self.model_params["model_params"]['eval_type'])
        with torch.no_grad():
            env.load_problems(batch_size, problems=test_data, aug_factor=aug_factor)
            reset_state, _, _ = env.reset()

            state, reward, done = env.pre_step()
            while not done:
                selected, unselected_indices, prob, stu_probs = model(reset_state, state, returnProbs=True)            # shape: (batch, pomo)
                state, reward, done = env.step(selected)
        # Return
        aug_reward = reward.reshape(aug_factor , batch_size, env.pomo_size)
        # shape: (augmentation, batch, pomo)
        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float()  # negative sign to make positive value
        no_aug_score_mean = no_aug_score.mean()

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float()  # negative sign to make positive value
        aug_score_mean = aug_score.mean()
        if self.tester_params["R3C"]==True:
            print("ori:",aug_score_mean)
            ori_solution=env.selected_node_list
            rrc_reward=self.R3C(env,test_data,ori_solution,self.tester_params["R3C_iter"],model)
            # Return
            rrc_aug_reward = -rrc_reward.reshape(aug_reward.shape)
            # shape: (augmentation, batch, pomo)
            rrc_max_pomo_reward, _ = rrc_aug_reward.max(dim=2)  # get best results from pomo
            # shape: (augmentation, batch)
            rrc_no_aug_score = -rrc_max_pomo_reward[0, :].float()  # negative sign to make positive value
            rrc_no_aug_score_mean = rrc_no_aug_score.mean()

            rrc_max_aug_pomo_reward, _ = rrc_max_pomo_reward.max(dim=0)  # get best results from augmentation
            # shape: (batch,)
            rrc_aug_score = -rrc_max_aug_pomo_reward.float()  # negative sign to make positive value
            rrc_aug_score_mean = rrc_aug_score.mean()
            return no_aug_score_mean.item(), aug_score_mean.item(), no_aug_score, aug_score,rrc_no_aug_score_mean.item(),\
                rrc_aug_score_mean.item(), rrc_no_aug_score, rrc_aug_score


        return no_aug_score_mean.item(), aug_score_mean.item(), no_aug_score, aug_score


    def _solve_cvrplib(self, model, path, env):
        """
            Solving one instance with CVRPLIB format.
        """
        file = open(path, "r")
        lines = [ll.strip() for ll in file]
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("DIMENSION"):
                dimension = int(line.split(':')[1])
            elif line.startswith("CAPACITY"):
                capacity = int(line.split(':')[1])
            elif line.startswith('NODE_COORD_SECTION'):
                locations = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
                i = i + dimension
            elif line.startswith('DEMAND_SECTION'):
                demand = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
                i = i + dimension
            i += 1
        original_locations = locations[:, 1:]
        original_locations = np.expand_dims(original_locations, axis=0)  # [1, n+1, 2]
        loc_scaler = 1000
        assert original_locations.max() <= loc_scaler, ">> Scaler is too small"
        locations = original_locations / loc_scaler  # [1, n+1, 2]: Scale location coordinates to [0, 1]
        depot_xy, node_xy = torch.Tensor(locations[:, :1, :]), torch.Tensor(locations[:, 1:, :])
        node_demand = torch.Tensor(demand[1:, 1:].reshape((1, -1))) / capacity  # [1, n]
        env.problem_size=node_xy.size(1)
        service_time = node_demand.clone() * 0.
        route_limit = node_demand.clone() * 0.
        tw_start = node_demand.clone() * 0.
        tw_end = node_demand.clone() * 0.
        problems_type = ["CVRP" for i in range(service_time.size(0))]
        data= (depot_xy, node_xy, node_demand, route_limit, service_time, tw_start, tw_end, problems_type)


        env.pomo_szie=1
        env.loc_scaler=loc_scaler
        # env_params = {'problem_size': node_xy.size(1), 'pomo_size': node_xy.size(1), 'loc_scaler': loc_scaler, 'device': self.device}
        # env = env_class(**env_params)
        _, _, no_aug_score, aug_score = self._test_one_batch(model, data, env)
        no_aug_score = torch.round(no_aug_score * loc_scaler).long()
        aug_score = torch.round(aug_score * loc_scaler).long()
        print(">> Finish solving {} -> no_aug: {} aug: {}".format(path, no_aug_score, aug_score))

        return no_aug_score, aug_score

    def _solve_cvrptwlib(self, model, path, env):
        """
            Solving one instance with VRPTW benchmark (e.g., Solomon) format.
        """
        file = open(path, "r")
        lines = [ll.strip() for ll in file]
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("NUMBER"):
                line = lines[i+1]
                vehicle_number = int(line.split(' ')[0])  # TODO: check vehicle number constraint
                capacity = int(line.split(' ')[-1])
            elif line.startswith("CUST NO."):
                data = np.loadtxt(lines[i + 1:], dtype=int)
                break
            i += 1
        original_locations = data[:, 1:3]
        original_locations = np.expand_dims(original_locations, axis=0)  # [1, n+1, 2]
        scaler = max(original_locations.max(), data[0, 5] / 3.)  # we set the time window of the depot node as [0, 3]
        # scaler = 1000
        assert original_locations.max() <= scaler, ">> Scaler is too small for {}".format(path)
        locations = original_locations / scaler  # [1, n+1, 2]: Scale location coordinates to [0, 1]
        depot_xy, node_xy = torch.Tensor(locations[:, :1, :]), torch.Tensor(locations[:, 1:, :])
        node_demand = torch.Tensor(data[1:, 3].reshape((1, -1))) / capacity  # [1, n]
        service_time = torch.Tensor(data[1:, -1].reshape((1, -1))) / scaler  # [1, n]
        tw_start = torch.Tensor(data[1:, 4].reshape((1, -1))) / scaler  # [1, n]
        tw_end = torch.Tensor(data[1:, 5].reshape((1, -1))) / scaler  # [1, n]
        depot_xy, node_xy, node_demand, capacity, service_time, tw_start, tw_end = torch.Tensor(
            depot_xy), torch.Tensor(node_xy), torch.Tensor(node_demand), torch.Tensor(capacity), torch.Tensor(
            service_time), torch.Tensor(tw_start), torch.Tensor(tw_end)
        route_limit = node_demand.clone() * 0.
        problems_type = ["VRPTW" for i in range(service_time.size(0))]


        env.pomo_szie=1
        env.loc_scaler=scaler

        env.depot_end = data[0, 5] / scaler
        data= (depot_xy, node_xy, node_demand, route_limit, service_time, tw_start, tw_end, problems_type)
        _, _, no_aug_score, aug_score = self._test_one_batch(model, data, env)

        # Check distance
        original_locations = torch.Tensor(original_locations)
        depot_xy = env.augment_xy_data_by_8_fold(original_locations[:, :1, :])
        node_xy = env.augment_xy_data_by_8_fold(original_locations[:, 1:, :])
        original_locations = torch.cat((depot_xy, node_xy), dim=1)
        gathering_index = env.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        all_xy = original_locations[:, None, :, :].expand(-1, 1, -1, -1)
        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        travel_distances = segment_lengths.sum(2)
        # print(travel_distances.min())

        no_aug_score = torch.round(no_aug_score * scaler).long()
        aug_score = torch.round(aug_score * scaler).long()
        print(">> Finish solving {} -> no_aug: {} aug: {} real: {}".format(path, no_aug_score, aug_score, travel_distances.min()))

        return no_aug_score, aug_score


    def R3C(self,env,test_data,ori_solution,R3C_iter,model):
        solution=ori_solution.clone().squeeze(1)
        for i in range(R3C_iter):
            rearranged=solution
            reload=False
            # if i%10==1:
            #     reload=True
            filp=True
            if 'TW'in test_data[7][0]:
                filp=False
                # print("tw false")
            if "B" in test_data[7][0]:
                filp=False
            if "O" in test_data[7][0]:
                filp = False
                # print("b false")
            rearranged = random_rearrange_subtours(solution,rerange=True,flip=filp,shift=True)#将原始的环路进行重排子环路，翻转，按顺序移动
            cut_len=random.randint(4, min(env.problem_size,500))
            # 切割的时候，要保证start_end包括的客户节点的数量一致，pre是前面的断开的0-start，代表被断的子环路的，start_end是要实际处理的片段
            pre_start_ends,start_ends= parallel_split_solutions(rearranged, cut_len=cut_len)
            with torch.no_grad():
                model.eval()
                model.set_eval_type(self.model_params["model_params"]['eval_type'])
                solution,new_reward= env.RRC_load_step(problems=test_data,ori_solution=rearranged,pre_start_ends=pre_start_ends,start_ends=start_ends,cut_len=cut_len,model=model,reload=reload)
                print(i,new_reward.mean())
                # self.check_solution(env, solution, test_data)

        return new_reward

    def check_solution(self,env,solution,data):
        env=self.check_env
        batch_size=solution.shape[0]
        with torch.no_grad():
            env.load_problems(batch_size, problems=data, aug_factor=1)
            env.pomo_size=1
            reset_state, _, _ = env.reset()
            state, reward, done = env.pre_step()
            i=0
            state, reward, done = env.step(solution[:, [0]])
            while not done:
                i+=1
                # print(i)
                mask=state.ninf_mask
                select=solution[:,[i]]
                selected_mask=torch.gather(mask,2,select.unsqueeze(-1))
                if selected_mask.sum()!=0:
                    a=0
                assert selected_mask.sum()==0,"error solution!"
                state, reward, done = env.step(select)
            print("right solution!")




