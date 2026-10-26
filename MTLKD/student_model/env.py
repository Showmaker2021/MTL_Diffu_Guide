import random
from copy import deepcopy
from dataclasses import dataclass
import torch
import os, pickle
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

def parallel_split_solutions(solutions: torch.Tensor, cut_len: int = 10):
    device = solutions.device
    batch_size, seq_len = solutions.shape
    device = solutions.device
    shape = solutions.shape

    # 生成索引矩阵 [..., dim]
    indices = torch.arange(shape[-1], device=device).view(*([1] * (len(shape) - 1)), -1)
    indices = indices.expand(*shape)

    # 创建替换掩码
    no_zero_mask=(solutions!=0)
    valid_end = (torch.argmax(
        no_zero_mask.int() * torch.arange(seq_len, device=device),
        dim=1
    ) + 1).clamp(max=seq_len)
    mask = indices > valid_end.unsqueeze(1)
    inf_a= torch.where(no_zero_mask, float("inf"), solutions)
    inf_a[mask] = float('inf')
    zero_idx=torch.where(inf_a==0, indices, inf_a)
    zero_idx, _ = torch.sort(zero_idx, dim=-1, stable=True)
    is_inf = torch.isinf(zero_idx)
    first_inf_pos = torch.where(
        is_inf.any(dim=1),
        is_inf.int().argmax(dim=1),
        zero_idx.size(1)
    )
    L_max = first_inf_pos.max().item()
    zero_idx=zero_idx[:,:L_max]
    zero_idx[torch.isinf(zero_idx)] = -1
    start=torch.zeros(shape[0],1)
    pos = torch.arange(zero_idx.size(1), device=device).expand_as(zero_idx)
    start_ends=[]
    #前面0的个数
    num_zeros=torch.arange(zero_idx.shape[1]).unsqueeze(0).repeat(zero_idx.shape[0],1)
    pre_start_ends=[]

    while True:
        cut_bool=(zero_idx -num_zeros- start >= cut_len)
        if all(cut_bool.any(dim=-1)):#存在合适区间
            # 找出第一个True的位置 [batch_size]
            first_pos = torch.where(
                cut_bool.any(dim=1),
                pos.masked_fill(~cut_bool, cut_bool.shape[1]).min(dim=1).values,
                torch.tensor(-1, device=device)
            )
            end=torch.gather(zero_idx,1,first_pos[:,None])
            mask=(indices<end)*(solutions>0)
            reversed_cumsum = torch.flip(torch.cumsum(torch.flip(mask, dims=(1,)), dim=1), dims=(1,))
            cut_start=((reversed_cumsum==(cut_len))*indices).max(dim=1).values.unsqueeze(-1)
            zero_start=((indices<=cut_start)*(solutions==0)*indices).max(dim=1).values.unsqueeze(-1)
            pre_start_end=torch.cat((zero_start,cut_start),dim=-1)
            start_end=torch.cat((cut_start,end),dim=-1)
            start_ends.append(start_end.unsqueeze(1))
            pre_start_ends.append(pre_start_end.unsqueeze(1))
            start=end
        else:
            break
    start_ends=torch.cat(start_ends,dim=1)
    pre_start_ends=torch.cat(pre_start_ends,dim=1)

    return pre_start_ends.to(torch.long),start_ends.to(torch.long)

def batch_flip_with_padding(tensor: torch.Tensor,
                            flip_prob: float = 0.5,
                            padding_value: int = -1) -> torch.Tensor:
    """
    向量化翻转batch中的每一行（以概率flip_prob翻转非padding_value部分）
    """
    # 1. 找到有效长度（第一个padding_value的位置）
    is_padding = (tensor == padding_value)
    has_padding = is_padding.any(dim=1)  # 标记是否有padding_value

    # 对于有padding_value的行，找到第一个padding的位置
    valid_lengths = torch.where(
        has_padding,
        is_padding.int().argmax(dim=1),
        tensor.shape[1]  # 没有padding_value的行，取整个长度
    )

    # 2. 生成翻转掩码
    flip_mask = torch.rand(tensor.shape[0], device=tensor.device) < flip_prob
    # print(flip_mask)
    # 3. 构造翻转索引（只处理非全padding的行）
    rows = torch.arange(tensor.shape[0], device=tensor.device).unsqueeze(1)
    cols = torch.arange(tensor.shape[1], device=tensor.device).unsqueeze(0)

    flipped_cols = torch.where(
        flip_mask.unsqueeze(1) & (cols < valid_lengths.unsqueeze(1)),
        valid_lengths.unsqueeze(1) - 1 - cols,
        cols
    )

    # 4. 使用gather提取翻转后的数据
    return torch.gather(tensor, 1, flipped_cols)
def random_rearrange_subtours(solutions,rerange=False,flip=False,shift=False):
    """向量化随机重排列子环路顺序"""

    device = solutions.device

    batch_size, seq_len = solutions.shape
    non_zero = solutions != 0

    last_non_zero_idx = torch.argmax(
        torch.arange(solutions.shape[1]).expand(solutions.shape[0], -1) * non_zero,
        dim=1
    )+1
    a=torch.arange(solutions.shape[1]).expand(solutions.shape[0], -1)
    solutions[a>last_non_zero_idx.unsqueeze(-1)]=-1
    # solutions=torch.cat((solutions,torch.zeros(batch_size,1)),dim=-1)
    start_from_depot_1=torch.nonzero(solutions == 0)[:,1]#这里包括了子环路后面的0的，开始
    start_from_depot_2 = torch.roll(start_from_depot_1, shifts=-1)#向左移动，结束
    sub_tours_length = start_from_depot_2 - start_from_depot_1#其中负数的代表最后是0-最后一个，其实是不合法的
    max_subtour_length = torch.max(sub_tours_length)
    real_start_end=torch.where(start_from_depot_2>0)[0]


    start_depot=torch.gather(start_from_depot_1,0,real_start_end)
    end_depot=torch.gather(start_from_depot_2,0,real_start_end)

    visit_depot_num = torch.sum(solutions==0, dim=1)-1

    repeat_solutions = solutions.repeat_interleave(visit_depot_num, dim=0)

    x1 = torch.arange(repeat_solutions.shape[1])[None, :].repeat(len(repeat_solutions), 1) \
         >= start_depot[:, None]
    x2 = torch.arange(repeat_solutions.shape[1])[None, :].repeat(len(repeat_solutions), 1) \
         < end_depot[:, None]
    x3 = (x1 * x2).long()
    sub_tourss = (repeat_solutions+1) * x3-1
    #避免后续越界
    sub_tourss=torch.cat((sub_tourss,-torch.ones(sub_tourss.shape[0],max_subtour_length)),dim=1)
    x4 = torch.arange(sub_tourss.shape[1])[None, :].repeat(len(sub_tourss), 1) \
         <(start_depot[:, None] + max_subtour_length)
    x5 = torch.arange(sub_tourss.shape[1])[None, :].repeat(len(sub_tourss), 1) \
         >= start_depot[:, None]
    x5 = x5 * x4

    sub_tours_padding = sub_tourss[x5].reshape(-1, max_subtour_length)
    temp_tri = np.triu(np.ones((len(visit_depot_num), len(visit_depot_num))), k=1)
    visit_depot_num_numpy = visit_depot_num.clone().cpu().numpy()
    temp_index = np.dot(visit_depot_num_numpy, temp_tri)

    temp_index_1 = torch.from_numpy(temp_index).long().cuda()

    temp_index_2 = visit_depot_num + temp_index_1

    x1 = torch.arange(sub_tours_padding.shape[0])[None, :].repeat(batch_size, 1) >= temp_index_1[:, None]
    x2 = torch.arange(sub_tours_padding.shape[0])[None, :].repeat(batch_size, 1) < temp_index_2[:, None]
    x3_ = (x1 * x2).int()
    x3 = x3_[:, :, None].repeat(1, 1, max_subtour_length)#哪些是对应的batch的环路
    sub_tours_padding_repeat = sub_tours_padding[None, :, :].repeat(batch_size, 1, 1)
    subtours_sperate = (sub_tours_padding_repeat+1) * x3-1
    if flip:
        for_flip=subtours_sperate[:,:,1:].reshape(-1,subtours_sperate.shape[-1]-1)#0不参与翻转
        wait_cat=subtours_sperate[:,:,[0]]
        flip_ed=batch_flip_with_padding(for_flip).reshape(sub_tours_padding_repeat.shape[0],sub_tours_padding_repeat.shape[1],-1)
        subtours_sperate=torch.cat((wait_cat,flip_ed),dim=-1)
    if rerange:
        rand_indices = torch.rand(batch_size, x3_.shape[1]).argsort(dim=1)
        subtours_sperate=subtours_sperate.gather(1,rand_indices.unsqueeze(-1).repeat(1,1,subtours_sperate.shape[2]))
    if shift:
        random_shift = random.randint(0, x3_.shape[1])
        shift_idx = torch.arange(x3_.shape[1])[None, :].repeat(x3_.shape[0],1).roll(shifts=random_shift,dims=1)
        subtours_sperate=subtours_sperate.gather(1,shift_idx.unsqueeze(-1).repeat(1,1,subtours_sperate.shape[2]))
    resort_sub_tours = subtours_sperate[subtours_sperate.gt(-1)]#非0结尾的

    solution_max=last_non_zero_idx.max(dim=0).values
    indices = torch.arange(solution_max).expand(batch_size, -1)  # [5,111]
    mask = indices < last_non_zero_idx.unsqueeze(1)  # [5,111] 布尔掩码
    cum_lengths = torch.cat([torch.tensor([0]), last_non_zero_idx.cumsum(dim=0)[:-1]])

    # 计算全局索引
    global_indices = indices + cum_lengths.unsqueeze(1)  # [5,111]
    global_indices[~mask] = 0  # 将无效位置设为0（避免索引越界）

    # 一次性收集所有数据
    result = torch.zeros(batch_size, solution_max)
    result[mask] = resort_sub_tours[global_indices[mask]]
    result=torch.cat((result,torch.zeros(batch_size,1)),dim=-1)
    result=result.to(torch.long)

    return result
def remove_zero_end(x1):
    #移动0到end,默认第一为0
    x=x1[:,1:]
    B, L = x.shape  # B=batch_size, L=sequence_length

    zero_mask = x == 0
    roll_zero_mask=zero_mask.roll(shifts=1,dims=1)
    # 记录非零位置
    first_zero=(~roll_zero_mask)*zero_mask
    non_zero_mask = x != 0  # [B, L]
    final_mask = non_zero_mask | first_zero
    per_num=final_mask.sum(dim=1)
    path0=per_num.max()-per_num


    new_num=path0.max().item()
    new_pad=-torch.ones(B,new_num,dtype=torch.long)
    ar_idx=torch.arange(new_num).unsqueeze(0).repeat(B,1)
    pad_=~(ar_idx<path0.unsqueeze(-1))*new_pad
    final_mask = torch.cat([final_mask, torch.ones(B, new_num, dtype=torch.bool, device=final_mask.device)],
                           dim=1)

    parts_cat=torch.cat((x,pad_),dim=-1)
    result0=parts_cat[final_mask*(parts_cat>=0)].reshape(B,-1)

    result=torch.cat((x1[:,[0]],result0),dim=1)

    return result
def replace_segments_fully_vectorized(solution0, start_ends, replacements):
    #替换解片段
    bs, N = solution0.shape
    k = start_ends.shape[1]
    p = replacements.shape[2]
    device = solution0.device

    # step 1: 构建保留段（不替换的）
    keep_starts = torch.cat([
        torch.zeros((bs, 1), dtype=torch.long, device=device),
        start_ends[:, :, 1] + 1  # 当前段的后一个
    ], dim=1)  # [bs, k+1]

    keep_ends = torch.cat([
        start_ends[:, :, 0],
        torch.full((bs, 1), N, dtype=torch.long, device=device)
    ], dim=1)  # [bs, k+1]

    # 计算保留段的最大长度
    keep_lens = (keep_ends - keep_starts).clamp(min=0)  # [bs, k+1]
    max_keep_len = keep_lens.max().item()
    idx = torch.arange(max_keep_len, device=device)[None, None, :]  # [1,1,L]
    valid_mask = idx < keep_lens.unsqueeze(-1)  # [bs, k+1, L]
    gather_idx = keep_starts.unsqueeze(-1) + idx  # [bs,k+1,L]
    gather_idx = gather_idx * valid_mask  # [bs,k+1,L]

    # gather solution0 中保留的段
    sol_exp = solution0.unsqueeze(1).expand(-1, k+1, -1)  # [bs,k+1,N]
    keep_segs = (torch.gather(sol_exp, 2, gather_idx)+1) * valid_mask-1  # [bs,k+1,L]

    # 展平保留段、替换段
    keep_flat = keep_segs.reshape(bs, -1)                 # [bs, (k+1)*L]
    repl_flat = replacements.reshape(bs, -1)              # [bs, k*p]

    # step 2: 按顺序插入 替换段 + 保留段 → [保留0, 替换0, 保留1, 替换1, ..., 保留k, 替换k, 保留k+1]
    parts = []
    for i in range(k):
        parts.extend([keep_segs[:, i], replacements[:, i]])
    parts.append(keep_segs[:, -1])  # 最后一个保留段

    # step 3: 拼接，自动补0
    parts_cat = torch.cat(parts, dim=1)  # 每个batch长度可能不同

    real_mask=(parts_cat>=0)

    per_num=real_mask.sum(dim=1)
    max_l=per_num.max()
    to_max_num=max_l-per_num

    new_num=to_max_num.max().item()
    new_pad=-torch.ones(bs,new_num,dtype=torch.long)
    ar_idx=torch.arange(new_num).unsqueeze(0).repeat(bs,1)
    pad_=~(ar_idx<to_max_num.unsqueeze(-1))*new_pad

    parts_cat=torch.cat((parts_cat,pad_),dim=-1)
    result0=parts_cat[parts_cat>=0].reshape(bs,-1)

    return result0

@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)
    node_service_time: torch.Tensor = None
    # shape: (batch, problem)
    node_tw_start: torch.Tensor = None
    # shape: (batch, problem)
    node_tw_end: torch.Tensor = None
    # shape: (batch, problem)
    prob_emb: torch.Tensor = None
    # shape: (num_training_prob)
    depot_node_feat: torch.Tensor = None

@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    START_NODE: torch.Tensor = None
    PROBLEM: str = None
    # shape: (batch, pomo)
    selected_count: int = None
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)
    load: torch.Tensor = None
    # shape: (batch, pomo)
    current_time: torch.Tensor = None
    # shape: (batch, pomo)
    length: torch.Tensor = None
    # shape: (batch, pomo)
    open: torch.Tensor = None
    # shape: (batch, pomo)
    current_coord: torch.Tensor = None
    # shape: (batch, pomo, 2)
    prob_emb: torch.Tensor = None
    selected_node_list: torch.Tensor =None
    visited_ninf_flag:torch.Tensor =None
    route_limit: torch.Tensor=None

def get_per_reset_state(reset_State,start_idx,end_idx):
    tem_state=Reset_State()
    tem_state.depot_xy=reset_State.depot_xy[start_idx:end_idx]
    # shape: (batch, 1, 2)
    tem_state.node_xy=reset_State.node_xy[start_idx:end_idx]
    # shape: (batch, problem, 2)
    tem_state.node_demand=reset_State.node_demand[start_idx:end_idx]
    # shape: (batch, problem)
    tem_state.node_service_time=reset_State.node_service_time[start_idx:end_idx]
    # shape: (batch, problem)
    tem_state.node_tw_start=reset_State.node_tw_start[start_idx:end_idx]
    # shape: (batch, problem)
    tem_state.node_tw_end=reset_State.node_tw_end[start_idx:end_idx]
    # shape: (batch, problem)
    tem_state.prob_emb=reset_State.prob_emb[start_idx:end_idx]

    return tem_state
def get_per_state(State,start_idx,end_idx):
    tem_state=Step_State()
    _,pomo_size=State.BATCH_IDX.shape
    bs=end_idx-start_idx
    tem_state.BATCH_IDX = torch.arange(bs)[:, None].expand(bs, pomo_size).to(State.BATCH_IDX.device)
    tem_state.POMO_IDX = torch.arange(pomo_size)[None, :].expand(bs, pomo_size).to(State.BATCH_IDX.device)
    tem_state.START_NODE=State.START_NODE[start_idx:end_idx]
    tem_state.PROBLEM=State.PROBLEM[start_idx:end_idx]
    tem_state.selected_count=State.selected_count

    tem_state.current_node=State.current_node[start_idx:end_idx]
    # shape: (batch, pomo)
    tem_state.ninf_mask=State.ninf_mask[start_idx:end_idx]
    # shape: (batch, pomo, problem+1)
    tem_state.finished=State.finished[start_idx:end_idx]
    # shape: (batch, pomo)
    tem_state.load=State.load[start_idx:end_idx]
    # shape: (batch, pomo)
    tem_state.current_time=State.current_time[start_idx:end_idx]
    # shape: (batch, pomo)
    tem_state.length=State.length[start_idx:end_idx]
    # shape: (batch, pomo)
    tem_state.open=State.open[start_idx:end_idx]
    # shape: (batch, pomo)
    tem_state.current_coord=State.current_coord[start_idx:end_idx]
    # shape: (batch, pomo, 2)
    tem_state.prob_emb=State.prob_emb[start_idx:end_idx]
    return tem_state


def sample_positive_indices(tensor, K):
    """
    Args:
        tensor: shape (b, N), each element is 0 or 1, with ~80% being 1 (positive)
        K: number of positive indices to sample per batch
    Returns:
        indices: shape (b, K), sampled positive indices for each batch (with replacement)
    """
    b, N = tensor.shape
    device = tensor.device

    # 存储每个batch的正样本索引
    positive_indices_per_batch = []

    for i in range(b):
        # 找到当前batch中所有正样本的索引
        positive_indices = torch.where(tensor[i] == 1)[0]
        positive_indices_per_batch.append(positive_indices)

    # 采样
    sampled_indices = torch.zeros((b, K), dtype=torch.long, device=device)

    for i in range(b):
        positive_indices = positive_indices_per_batch[i]
        if len(positive_indices) == 0:
            # 如果没有正样本，可以采样任意索引（或者根据需求处理）
            sampled_indices[i] = torch.randint(0, N, (K,), device=device)
        else:
            # 有放回地采样K次
            sampled = torch.randint(0, len(positive_indices), (K,), device=device)
            sampled_indices[i] = positive_indices[sampled]

    return sampled_indices



class VRP_Env:
    def __init__(self, **env_params):
        # Const @INIT
        ####################################
        self.train_problems=env_params.get("train_problems",["CVRP"])
        self.num_pro=len(self.train_problems)
        self.env_params = env_params
        self.backhaul_ratio = 0.2
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.loc_scaler = env_params['loc_scaler'] if 'loc_scaler' in env_params.keys() else None
        if torch.cuda.is_available():
            self.device = torch.device('cuda', torch.cuda.current_device()) if 'device' not in env_params.keys() else \
            env_params['device']

        else:
            self.device = torch.device('cpu')
        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        self.START_NODE = None
        # IDX.shape: (batch, pomo)
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)
        self.depot_node_service_time = None
        # shape: (batch, problem+1)
        self.depot_node_tw_start = None
        # shape: (batch, problem+1)
        self.depot_node_tw_end = None
        # shape: (batch, problem+1)
        self.speed = 1.0
        self.depot_start, self.depot_end = 0., 3.  # tw for depot [0, 3]

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        # shape: (batch, pomo)
        self.current_time = None
        # shape: (batch, pomo)
        self.length = None
        # shape: (batch, pomo)
        self.open = None
        # shape: (batch, pomo)
        self.current_coord = None
        # shape: (batch, pomo, 2)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

        self.current_only_back=None

    def load_problems(self, batch_size, problems, aug_factor=1,START_NODE=None):
        depot_xy, node_xy, node_demand, route_limit, service_time, tw_start, tw_end ,self.problem_type= problems
        self.problem_size=node_xy.shape[1]

        if depot_xy.dim()==2:
            depot_xy=depot_xy.unsqueeze(1)
        depot_xy = depot_xy.to(self.device)  # 仓库坐标 (batch, 1, 2)
        node_xy = node_xy.to(self.device)  # 节点坐标 (batch, n, 2)
        node_demand = node_demand.to(self.device)  # 节点需求 (batch, n)
        route_limit = route_limit.to(self.device)  # 路径限制 (batch,)
        service_time = service_time.to(self.device)  # 服务时间 (batch, n)
        tw_start = tw_start.to(self.device)  # 时间窗起点 (batch, n)
        tw_end = tw_end.to(self.device)  # 时间窗终点 (batch, n)
        self.batch_size = depot_xy.size(0)
        route_limit = route_limit if route_limit.dim() == 1 else route_limit[:,[0]].squeeze(1)

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = self.augment_xy_data_by_8_fold(depot_xy)
                node_xy = self.augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
                route_limit = route_limit.repeat(8)
                service_time = service_time.repeat(8, 1)
                tw_start = tw_start.repeat(8, 1)
                tw_end = tw_end.repeat(8, 1)
                self.problem_type=self.problem_type*8
            else:
                raise NotImplementedError
        problems_type = self.problem_type
        self.b_indices = torch.tensor([i for i, problem in enumerate(problems_type) if "B" in problem])[:,
                         None]
        self.l_indices = torch.tensor([i for i, problem in enumerate(problems_type) if "L" in problem])[:,
                         None]
        self.tw_indices = torch.tensor([i for i, problem in enumerate(problems_type) if "TW" in problem])[:,
                          None]
        self.o_indices = torch.tensor([i for i, problem in enumerate(problems_type) if "O" in problem])[:,
                         None]
        # reset pomo_size
        self.pomo_size = min(int(self.problem_size * (1 - self.backhaul_ratio)), self.pomo_size)
        if START_NODE is None:
            if self.pomo_size==self.problem_size:
                self.START_NODE = torch.arange(start=1, end=self.problem_size+1)[None, :].expand(self.batch_size, -1).to(self.device)
                if self.b_indices.size(0):
                    START_NODE=self.START_NODE.clone()
                    START_NODE[node_demand < 0]=-1
                    temp_demand=node_demand[self.b_indices].squeeze()
                    other=torch.where(temp_demand>0)
                    inx=other[1].reshape(len(self.b_indices),-1)
                    random_indices = torch.randint(0, inx.size(1),((inx.size(0), self.pomo_size-inx.size(1))))
                    real_ind=torch.gather(inx,dim=1,index=random_indices)
                    sampled_tensor = torch.gather(START_NODE[self.b_indices,:].squeeze(), dim=1, index=real_ind).reshape(-1)
                    mask = START_NODE == -1

                    # 确保 sampled_tensor 的长度与 -1 的数量一致
                    assert mask.sum() == len(sampled_tensor), "sampled_tensor 的长度与 -1 的数量不匹配"

                    # 替换 -1
                    START_NODE[mask] = sampled_tensor
                    self.START_NODE=START_NODE.clone()
            else:
                self.START_NODE=sample_positive_indices(node_demand,self.pomo_size)
        else:
            self.START_NODE=START_NODE
        if depot_xy.dim()==2:
            depot_xy=depot_xy.unsqueeze(1)
        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        self.depot_node_demand = torch.cat((torch.zeros(size=(self.batch_size, 1)).to(self.device), node_demand), dim=1)
        # shape: (batch, problem+1)
        self.route_limit = route_limit
        # shape: (batch, 1)
        self.depot_node_service_time = torch.cat(( torch.zeros(size=(self.batch_size, 1)).to(self.device), service_time), dim=1)
        # shape: (batch, problem+1)
        self.depot_node_tw_start = torch.cat((torch.ones(size=(self.batch_size, 1)).to(self.device) * self.depot_start, tw_start), dim=1)
        # shape: (batch, problem+1)
        self.depot_node_tw_end = torch.cat((torch.ones(size=(self.batch_size, 1)).to(self.device) * self.depot_end, tw_end), dim=1)
        # shape: (batch, problem+1)
        self.current_only_back= torch.zeros(self.batch_size,1, dtype=torch.bool)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size).to(self.device)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size).to(self.device)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand
        self.reset_state.node_service_time = service_time
        self.reset_state.node_tw_start = tw_start
        self.reset_state.node_tw_end = tw_end
        node_demand_service_ts_te=torch.cat((node_demand.unsqueeze(-1),service_time.unsqueeze(-1),tw_start.unsqueeze(-1),tw_end.unsqueeze(-1)),dim=-1)
        depot_node_xy=torch.cat((depot_xy,node_xy),dim=1)
        depot_demand_service_ts_te=torch.zeros(size=(self.batch_size,1,4),dtype=torch.float32).to(self.device)
        depot_node_demand_service_ts_te=torch.cat((depot_demand_service_ts_te,node_demand_service_ts_te),dim=1)
        self.reset_state.depot_node_feat=torch.cat((depot_node_xy,depot_node_demand_service_ts_te),dim=-1)
        self.reset_state.prob_emb =torch.zeros(size=(self.batch_size, 4)).to(self.device)
        # torch.FloatTensor([1, 0, 0, 0, 0])[None,None,:].to(self.device)  # bit vector for [O, B, L, TW]
        if self.o_indices.size(0):self.reset_state.prob_emb[self.o_indices,0]=1
        if self.b_indices.size(0):self.reset_state.prob_emb[self.b_indices,1]=1
        if self.l_indices.size(0):self.reset_state.prob_emb[self.l_indices,2]=1
        if self.tw_indices.size(0):self.reset_state.prob_emb[self.tw_indices,3]=1
        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
        self.step_state.open = torch.zeros(self.batch_size, self.pomo_size).to(self.device)
        if self.o_indices.size(0):self.step_state.open[self.o_indices,:]=1.0
        self.step_state.START_NODE = self.START_NODE
        self.step_state.PROBLEM = self.problem_type
        self.step_state.route_limit=route_limit


    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long).to(self.device)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool).to(self.device)
        # shape: (batch, pomo)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size)).to(self.device)
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1)).to(self.device)
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1)).to(self.device)
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool).to(self.device)
        # shape: (batch, pomo)
        self.current_time = torch.zeros(size=(self.batch_size, self.pomo_size)).to(self.device)
        # shape: (batch, pomo)
        self.length = torch.zeros(size=(self.batch_size, self.pomo_size)).to(self.device)
        # shape: (batch, pomo)
        self.current_coord = self.depot_node_xy[:, :1, :]  # depot
        # shape: (batch, pomo, 2)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.current_time = self.current_time
        self.step_state.length = self.length
        self.step_state.current_coord = self.current_coord
        self.step_state.prob_emb=self.reset_state.prob_emb
        self.step_state.visited_ninf_flag=self.visited_ninf_flag
        reward = None
        done = False
        return self.step_state, reward, done


    def step(self, selected):
        # selected.shape: (batch, pomo)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)
        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)
        self.current_only_back[self.at_the_depot]=False

        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # shape: (batch, pomo, problem+1)
        gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1  # refill loaded at the depot

        current_coord = self.depot_node_xy[torch.arange(self.batch_size)[:, None], selected]
        # shape: (batch, pomo, 2)
        new_length = (current_coord - self.current_coord).norm(p=2, dim=-1)
        # shape: (batch, pomo)
        self.length = self.length + new_length
        self.length[self.at_the_depot] = 0  # reset the length of route at the depot
        self.current_coord = current_coord

        # Mask
        ####################################
        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem+1)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        # Only for VRPB: reset load to 0.
        # >> Old implementation
        #   a. if visit backhaul nodes in the first two POMO moves (i.e., depot -> backhaul, the route is mixed with backhauls and linehauls alternatively);
        #   b. if only backhaul nodes unserved, we relax the load to be 0 (i.e., the vehicle only visit backhauls nodes in the last few routes).
        # if self.selected_node_list.size(-1) == 1:  # POMO first move
        #     depot_backhaul = self.at_the_depot & (self.depot_node_demand[:, 1:self.pomo_size+1] < 0.)
        #     # shape: (batch, pomo)
        #     self.load[depot_backhaul] = 0.
        # else:
        # >> New implementation - Remove constraint a, the POMO start node should be a linehaul.
        unvisited_demand = demand_list + self.visited_ninf_flag
        # shape: (batch, pomo, problem+1)
        linehauls_unserved = torch.where(unvisited_demand > 0., True, False)
        reset_index = self.at_the_depot & (~linehauls_unserved.any(dim=-1))
        # shape: (batch, pomo)
        self.load[reset_index] = 0.


        # capacity constraint
        #   a. the remaining vehicle capacity >= the customer demands
        #   b. the remaining vehicle capacity <= the vehicle capacity (i.e., 1.0)
        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, pomo, problem+1)
        self.ninf_mask[demand_too_large] = float('-inf')
        exceed_capacity = self.load[:, :, None] - demand_list > 1.0 + round_error_epsilon
        self.ninf_mask[exceed_capacity] = float('-inf')
        ####因为RRC可能会把后面的b提前，那么中间0出发的点遇到+demand,-----+++--,因此，only_back的标志，True时候，只让其该环路访问-值
        self.ninf_mask[(demand_list > 0) * self.current_only_back.unsqueeze(-1)]=float('-inf')

        # duration limit constraint
        # Note: different from VRPL, since no need to return to depot for OVRPBLTW
        route_limit = self.route_limit[:, None, None].expand(self.batch_size, self.pomo_size, self.problem_size + 1)
        # shape: (batch, pomo, problem+1)
        # check route limit constraint: length + cur->next->depot <= route_limit
        # unlimit_=route_limit==0
        # route_limit_=route_limit.clone()
        # route_limit_[unlimit_]=float("inf")#对无L约束的全部设置lim=-inf
        if self.l_indices.size(0):
            route_too_large = self.length[:, :, None] + \
                              (self.current_coord[:, :, None, :] - self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)).norm(p=2, dim=-1) \
                              +(self.depot_node_xy[:, None, :1, :] - self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)).norm(p=2, dim=-1) * (1 - self.step_state.open.unsqueeze(-1))  > route_limit + round_error_epsilon
            if self.o_indices.size(0):
                route_too_large[self.o_indices.squeeze(), :, 0] = False
            no_l_indices = torch.arange(self.batch_size, device=self.l_indices.device)
            no_l_indices = no_l_indices[~torch.isin(no_l_indices, self.l_indices)]

            route_too_large[no_l_indices]=False
            # shape: (batch, pomo, problem+1)
            self.ninf_mask[route_too_large] = float('-inf')
            # shape: (batch, pomo, problem+1)

        # time window constraint
        # Note: different from VRPTW, since no need to return to depot for OVRPBLTW
        #   current_time: the end time of serving the current node
        #   a. max(current_time + travel_time, tw_start) or current_time + travel_time <= tw_end
        if self.tw_indices.size(0):
            self.current_time = torch.max(self.current_time + new_length / self.speed, self.depot_node_tw_start[torch.arange(self.batch_size)[:, None], selected]) + self.depot_node_service_time[torch.arange(self.batch_size)[:, None], selected]
            self.current_time[self.at_the_depot] = 0
            # shape: (batch, pomo)
            arrival_time = torch.max(self.current_time[:, :, None] + (self.current_coord[:, :, None, :] - self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)).norm(p=2, dim=-1) / self.speed, self.depot_node_tw_start[:, None, :].expand(-1, self.pomo_size, -1))
            out_of_tw = arrival_time > self.depot_node_tw_end[:, None, :].expand(-1, self.pomo_size, -1) + round_error_epsilon
            # shape: (batch, pomo, problem+1)
            # shape: (batch, pomo, problem+1)
            fail_return_depot = arrival_time + self.depot_node_service_time[:, None, :].expand(-1, self.pomo_size,-1) + (
                                            self.depot_node_xy[:, None, :1, :] - self.depot_node_xy[:, None, :,:].expand(-1, self.pomo_size, -1,
                                                                                           -1)).norm(p=2,dim=-1) / self.speed > self.depot_end + round_error_epsilon
            # shape: (batch, pomo, problem+1)
            if self.o_indices.size(0):
                out_of_tw[self.o_indices.squeeze(), :, 0] = False
                fail_return_depot[self.o_indices.squeeze(),:,:]=False
            self.ninf_mask[fail_return_depot] = float('-inf')

            no_tw_indices = torch.arange(self.batch_size, device=self.tw_indices.device)
            no_tw_indices = no_tw_indices[~torch.isin(no_tw_indices, self.tw_indices)]
            out_of_tw[no_tw_indices,:,:]=False
            self.current_time[no_tw_indices]*=0
            self.ninf_mask[out_of_tw] = float('-inf')

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)

        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.current_time = self.current_time
        self.step_state.length = self.length
        self.step_state.current_coord = self.current_coord
        self.step_state.selected_node_list=self.selected_node_list
        self.step_state.visited_ninf_flag=self.visited_ninf_flag

        # returning values
        done = self.finished.all()
        if done:
            reward = -self.get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def get_travel_distance(self,selected_node_list=None):
        if  selected_node_list is None:
            selected_node_list=self.selected_node_list
        gathering_index = selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)

        not_to_depot = selected_node_list.roll(dims=2, shifts=-1) != 0
        # shape: (batch, pomo, selected_list_length)

        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, selected_list_length)

        if self.loc_scaler:
            segment_lengths = torch.round(segment_lengths * self.loc_scaler) / self.loc_scaler

        if self.o_indices.size(0):
            no_open_idx= torch.arange(self.batch_size, device=self.o_indices.device)
            no_o_indices = no_open_idx[~torch.isin(no_open_idx, self.o_indices)]
            not_to_depot[no_o_indices,:,:]=True
        else:
            not_to_depot[:]=True

        travel_distances = (segment_lengths * not_to_depot).sum(2)
        # shape: (batch, pomo)
        return travel_distances

    def lehd_txt_data(self, data_path, episode):
        raw_data = {
            'depot_xy': [],
            'node_xy': [],
            'node_demand': [],
            'route_limit': [],
            'service_time': [],
            'tw_start': [],
            'tw_end': [],
            'problems_type': [],
            'cost': []
        }

        with open(data_path, "r") as f:
            for line in f.readlines()[:episode]:
                line = line.strip().split(",")

                # 获取各字段的索引位置
                indices = {
                    'depot': int(line.index('depot')),
                    'customer': int(line.index('customer')),
                    'capacity': int(line.index('capacity')),
                    'demand': int(line.index('demand')),
                    'cost': int(line.index('cost')),
                    'node_flag': int(line.index('node_flag'))
                }

                # 解析位置坐标
                depot_coord = [[float(line[indices['depot'] + 1]), float(line[indices['depot'] + 2])]]
                customer_coords = [
                    [float(line[idx]), float(line[idx + 1])]
                    for idx in range(indices['customer'] + 1, indices['capacity'], 2)
                ]
                all_coords = torch.tensor(depot_coord + customer_coords)

                # 解析容量和需求
                capacity = int(float(line[indices['capacity'] + 1]))
                demand_start = indices['demand'] + 1
                demands = [int(line[idx]) for idx in range(demand_start, indices['cost'])]
                demands = demands if int(line[demand_start]) == 0 else [0] + demands

                # 标准化需求
                normalized_demands = torch.tensor(demands[1:]) / capacity

                # 填充默认值
                num_nodes = len(customer_coords)
                raw_data['depot_xy'].append(all_coords[[0],])
                raw_data['node_xy'].append(all_coords[1:])
                raw_data['node_demand'].append(normalized_demands)
                raw_data['route_limit'].append(torch.tensor(0.))
                raw_data['service_time'].append(torch.zeros(num_nodes, dtype=torch.float32))
                raw_data['tw_start'].append(torch.zeros(num_nodes, dtype=torch.float32))
                raw_data['tw_end'].append(torch.zeros(num_nodes, dtype=torch.float32))
                raw_data['problems_type'].append("CVRP")
                raw_data['cost'].append(float(line[indices['cost'] + 1]))

        # 转换为张量（处理不同类型的字段）
        tensor_data = {}
        for key in raw_data:
            if key == 'problems_type':
                tensor_data[key] = raw_data[key]  # 保持字符串列表不变
            elif key == 'cost':
                tensor_data[key] = torch.tensor(raw_data[key])  # 将浮点列表转为张量
            else:
                # 只有已经是张量的数据才进行stack
                if isinstance(raw_data[key][0], torch.Tensor):
                    tensor_data[key] = torch.stack(raw_data[key])
                else:
                    tensor_data[key] = torch.tensor(raw_data[key])

        data1 = (tensor_data['depot_xy'], tensor_data['node_xy'], tensor_data['node_demand']
                 , tensor_data['route_limit'], tensor_data['service_time'], tensor_data['tw_start']
                 , tensor_data['tw_end'], tensor_data['problems_type'],tensor_data['cost'])

        return data1

    def load_dataset(self, path,problem,offset=0, num_samples=1000, disable_print=True):
        assert os.path.splitext(path)[1] == ".pkl", "Unsupported file type (.pkl needed)."
        with open(path, 'rb') as f:
            data = pickle.load(f)[offset: offset+num_samples]
            if not disable_print:
                print(">> Load {} data ({}) from {}".format(len(data), type(data), path))
        if problem in ["CVRP","OVRP","VRPB","OVRPB"]:
            depot_xy, node_xy, node_demand, capacity = [i[0] for i in data], [i[1] for i in data], [i[2] for i in
                                                                                                    data], [i[3] for i
                                                                                                            in data]
            depot_xy, node_xy, node_demand, capacity = torch.Tensor(depot_xy), torch.Tensor(node_xy), torch.Tensor(
                node_demand), torch.Tensor(capacity)
            node_demand = node_demand / capacity.view(-1, 1)
            service_time = node_demand.clone() * 0.
            route_limit = node_demand.clone() * 0.
            tw_start = node_demand.clone() * 0.
            tw_end = node_demand.clone() * 0.
            problems_type = [problem for i in range(service_time.size(0))]
            data1 = (depot_xy, node_xy, node_demand, route_limit, service_time, tw_start, tw_end, problems_type)

        elif problem in  ["VRPL","OVRPL","VRPBL","OVRPBL"]:
            depot_xy, node_xy, node_demand, capacity, route_limit = [i[0] for i in data], [i[1] for i in data], [i[2] for i
                                                                                                                 in
                                                                                                                 data], [
                i[3] for i in data], [i[4] for i in data]
            depot_xy, node_xy, node_demand, capacity, route_limit = torch.Tensor(depot_xy), torch.Tensor(
                node_xy), torch.Tensor(node_demand), torch.Tensor(capacity), torch.Tensor(route_limit)
            node_demand = node_demand / capacity.view(-1, 1)
            service_time = node_demand.clone() * 0.
            tw_start = node_demand.clone() * 0.
            tw_end = node_demand.clone() * 0.
            problems_type = [problem for i in range(service_time.size(0))]
            data1 = (depot_xy, node_xy, node_demand, route_limit, service_time, tw_start, tw_end, problems_type)
        elif problem in  ["VRPTW","OVRPTW","VRPBTW","OVRPBTW"]:
            depot_xy, node_xy, node_demand, capacity, service_time, tw_start, tw_end = [i[0] for i in data], [i[1] for i
                                                                                                              in
                                                                                                              data], [
                i[2] for i in data], [i[3] for i in data], [i[4] for i in data], [i[5] for i in data], [i[6] for i in
                                                                                                        data]
            depot_xy, node_xy, node_demand, capacity, service_time, tw_start, tw_end = torch.Tensor(
                depot_xy), torch.Tensor(node_xy), torch.Tensor(node_demand), torch.Tensor(capacity), torch.Tensor(
                service_time), torch.Tensor(tw_start), torch.Tensor(tw_end)
            node_demand = node_demand / capacity.view(-1, 1)
            route_limit = node_demand.clone() * 0.
            problems_type = [problem for i in range(service_time.size(0))]
            data1 = (depot_xy, node_xy, node_demand, route_limit, service_time, tw_start, tw_end, problems_type)

        else:
            depot_xy, node_xy, node_demand, capacity, route_limit, service_time, tw_start, tw_end = [i[0] for i in
                                                                                                     data], [i[1] for i
                                                                                                             in data], [
                i[2] for i in data], [i[3] for i in data], [i[4] for i in data], [i[5] for i in data], [i[6] for i in
                                                                                                        data], [i[7] for
                                                                                                                i in
                                                                                                                data]
            depot_xy, node_xy, node_demand, capacity, route_limit, service_time, tw_start, tw_end = torch.Tensor(
                depot_xy), torch.Tensor(node_xy), torch.Tensor(node_demand), torch.Tensor(capacity), torch.Tensor(
                route_limit), torch.Tensor(service_time), torch.Tensor(tw_start), torch.Tensor(tw_end)
            node_demand = node_demand / capacity.view(-1, 1)
            problems_type = [problem for i in range(service_time.size(0))]
            data1 = (depot_xy, node_xy, node_demand, route_limit, service_time, tw_start, tw_end, problems_type)
        return data1
    def augment_xy_data_by_8_fold(self, xy_data):
        # xy_data.shape: (batch, N, 2)

        x = xy_data[:, :, [0]]
        y = xy_data[:, :, [1]]
        # x,y shape: (batch, N, 1)

        dat1 = torch.cat((x, y), dim=2)
        dat2 = torch.cat((1 - x, y), dim=2)
        dat3 = torch.cat((x, 1 - y), dim=2)
        dat4 = torch.cat((1 - x, 1 - y), dim=2)
        dat5 = torch.cat((y, x), dim=2)
        dat6 = torch.cat((1 - y, x), dim=2)
        dat7 = torch.cat((y, 1 - x), dim=2)
        dat8 = torch.cat((1 - y, 1 - x), dim=2)

        aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
        # shape: (8*batch, N, 2)

        return aug_xy_data


    def cut_data(self,problems,cut_idx):
        depot_xy, node_xy, node_demand, route_limit, service_time, tw_start, tw_end, problem_type = problems
        bs,bs_per_cut,cut_len=cut_idx.shape
        cut_idx=cut_idx-1
        pro_size=node_xy.shape[1]
        if depot_xy.dim()==2:
            depot_xy=depot_xy.unsqueeze(1)
        new_depot_xy=depot_xy.unsqueeze(1).repeat(1,bs_per_cut,1,1).reshape(-1,1,2)
        if route_limit.dim()==1:
            route_limit=route_limit.unsqueeze(-1).expand_as(service_time)
        new_node_xy=node_xy.unsqueeze(1).repeat(1,bs_per_cut,1,1)
        new_node_demand=node_demand.unsqueeze(1).repeat(1,bs_per_cut,1)
        new_route_limit=route_limit.unsqueeze(1).repeat(1,bs_per_cut,1)
        new_service_time=service_time.unsqueeze(1).repeat(1,bs_per_cut,1)
        new_tw_start=tw_start.unsqueeze(1).repeat(1,bs_per_cut,1)
        new_tw_end=tw_end.unsqueeze(1).repeat(1,bs_per_cut,1)

        new_node_xy=new_node_xy.gather(2,cut_idx.unsqueeze(-1).repeat(1,1,1,2)).reshape(-1,cut_len,2)
        new_node_demand=new_node_demand.gather(2,cut_idx).reshape(-1,cut_len)
        new_route_limit=new_route_limit.gather(2,cut_idx).reshape(-1,cut_len)
        new_service_time=new_service_time.gather(2,cut_idx).reshape(-1,cut_len)
        new_tw_start=new_tw_start.gather(2,cut_idx).reshape(-1,cut_len)
        new_tw_end=new_tw_end.gather(2,cut_idx).reshape(-1,cut_len)
        new_problem_type=problem_type*bs_per_cut
        new_data=(new_depot_xy, new_node_xy, new_node_demand, new_route_limit, new_service_time, new_tw_start, new_tw_end, new_problem_type)

        return new_data

    def get_new_state(self,problems, real_pre_idx):
        self.reset()
        self.pre_step()
        depot_xy, node_xy, node_demand, route_limit, service_time, tw_start, tw_end, problem_type = problems
        bs=depot_xy.size(0)
        _,bs_per_cut,cut_len=real_pre_idx.shape
        if depot_xy.dim()==2:
            depot_xy=depot_xy.unsqueeze(1)
        depot_node_xy=torch.cat((depot_xy,node_xy),dim=1).unsqueeze(1).repeat(1,bs_per_cut,1,1)
        depot_node_demand = torch.cat((torch.zeros(size=(bs, 1)).to(self.device), node_demand), dim=1).unsqueeze(1).repeat(1,bs_per_cut,1)
        depot_node_service_time = torch.cat(( torch.zeros(size=(bs, 1)).to(self.device), service_time), dim=1).unsqueeze(1).repeat(1,bs_per_cut,1)
        # shape: (batch, problem+1)
        depot_node_tw_start = torch.cat((torch.ones(size=(bs, 1)).to(self.device) * self.depot_start, tw_start), dim=1).unsqueeze(1).repeat(1,bs_per_cut,1)
        # shape: (batch, problem+1)
        depot_node_tw_end = torch.cat((torch.ones(size=(bs, 1)).to(self.device) * self.depot_end, tw_end), dim=1).unsqueeze(1).repeat(1,bs_per_cut,1)
        pre_=real_pre_idx[:,:,:-1]
        selected_demand=depot_node_demand.gather(2,pre_).sum(dim=-1).reshape(self.batch_size,-1)
        self.load[selected_demand<(-1e-6)]=0
        self.current_only_back[selected_demand<0]=True
        self.load -= selected_demand

        #对于VRPB问题，如果是0后面访问了-的节点，那么这个环路只能是-的，这该如何处理
        ordered_seq = depot_node_xy.gather(dim=2, index=pre_.unsqueeze(-1).repeat(1,1,1,2))
        # shape: (batch, pomo, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, selected_list_length)
        new_length = (segment_lengths[:,:,:-1]).sum(2).reshape(self.batch_size,-1)
        self.length = self.length + new_length


        self.current_coord = depot_node_xy.gather(2,pre_[:,:,[-1]].unsqueeze(-1).repeat(1,1,1,2)).reshape(self.batch_size,-1,2)
        if self.tw_indices.size(0):
            pre_tw_start=depot_node_tw_start.gather(2,pre_).reshape(-1,pre_.shape[-1])
            pre_tw_end=depot_node_tw_end.gather(2,pre_).reshape(-1,pre_.shape[-1])
            pre_service_time=depot_node_service_time.gather(2,pre_).reshape(-1,pre_.shape[-1])
            for i in range(pre_.shape[-1]-1):
                self.current_time = torch.max(self.current_time + (segment_lengths.reshape(-1,pre_.shape[-1])[:,i] / self.speed).unsqueeze(-1), pre_tw_start[:,[i+1]]) +\
                                    pre_service_time[:,[i+1]]

        selected=torch.ones(self.batch_size,1,dtype=torch.long)
        return self.step(selected)

    def RRC_load_step(self,problems, ori_solution,pre_start_ends,start_ends,cut_len,model,reload=False):

        per_cut=start_ends.shape[1]
        batch_size,solution_len=ori_solution.shape
        ori_so0=ori_solution.clone()

        #########得到等长的客户节点
        ori_solution=ori_solution.unsqueeze(1).repeat(1,per_cut,1)
        s_len_idx=torch.arange(solution_len)[None,None,:].repeat(batch_size,per_cut,1)
        unvisited_node_mask=(s_len_idx<start_ends[:,:,[1]])*(s_len_idx>=start_ends[:,:,[0]])*(ori_solution>0)
        new_data_cust_idx=s_len_idx[unvisited_node_mask].reshape(batch_size,per_cut,-1)
        real_new_data_cust_idx=torch.gather(ori_solution,2,new_data_cust_idx)

        #####处理前面段掉的子环路，把断了的子环路的tensor对齐，在前面补0，可以让其更容易计算
        pre_start_ends_mask=(s_len_idx<=pre_start_ends[:,:,[1]])*(s_len_idx>=pre_start_ends[:,:,[0]])*s_len_idx
        pre_start_ends_mask_double=pre_start_ends_mask.repeat(1,1,2)
        each_num=pre_start_ends[:,:,1]-pre_start_ends[:,:,0]
        max_pre_l=each_num.reshape(-1).max(0).values+1
        pre_end_=pre_start_ends_mask.max(dim=2).values+solution_len+1
        pre_start_=pre_end_-max_pre_l
        indices = torch.arange(max_pre_l).expand(pre_end_.shape[0], pre_end_.shape[1],
                                                 max_pre_l) + pre_start_.unsqueeze(-1)
        result = torch.gather(pre_start_ends_mask_double, dim=2, index=indices)
        real_pre_idx=ori_solution.gather(2,result)

        #把原始解切割，变成新的等长的数据集，以便环境和模型重新处理
        new_data=self.cut_data(problems,real_new_data_cust_idx)
        new_batch_size=new_data[0].shape[0]
        self.load_problems(batch_size=new_batch_size, problems=new_data)
        #让环境处理先前断的子环路先，使状态，需要把load什么处理好
        step_state, reward, done=self.get_new_state(problems,real_pre_idx)
        while not done:
            torch.cuda.empty_cache()
            selected, unselected_indices, prob, stu_probs = model(self.reset_state, step_state, returnProbs=True)  # shape: (batch, pomo)
            step_state, reward, done = self.step(selected)

        ######原始长度的计算
        depot_xy=problems[0]
        if depot_xy.dim()==2:
            depot_xy=depot_xy.unsqueeze(1)
        ori_node_depot_xy=torch.cat((depot_xy,problems[1]),dim=1)
        ori_cut_mask=(s_len_idx <= start_ends[:, :, [1]]) * (s_len_idx >= start_ends[:, :, [0]])
        ori_cut = ori_cut_mask*(ori_solution)
        gathering_index = ori_cut[:, :, :, None].expand(-1, -1, -1, 2)
        all_xy = ori_node_depot_xy[:, None, :, :].expand(-1, ori_cut.shape[1], -1, -1)
        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        not_to_depot = ori_cut.roll(dims=2, shifts=-1) != 0
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        assert len(set(problems[7])) == 1
        if "O" not in problems[7][0]:#开放问题要注意不能计算回仓库节点的距离
            not_to_depot[:]=True
        ori_distances = (segment_lengths *ori_cut_mask* not_to_depot).sum(2)
        ######新长度的计算
        new_unvisited_idx = self.selected_node_list
        ori_idx=torch.cat((torch.zeros(self.batch_size,1),real_new_data_cust_idx.reshape(self.batch_size,-1)),dim=1)
        real_after_inx=ori_idx.unsqueeze(1).repeat(1,self.pomo_size,1).gather(2,new_unvisited_idx).to(torch.long)
        after_xy=ori_node_depot_xy.unsqueeze(1).repeat(1,per_cut,1,1).reshape(self.batch_size,1,-1,2)
        after_xy=after_xy.gather(2,real_after_inx.unsqueeze(-1).repeat(1,1,1,2))
        rolled_after_xy = after_xy.roll(dims=2, shifts=-1)
        segment_after_lengths = ((after_xy-rolled_after_xy)**2).sum(3).sqrt()
        not_to_depot = real_after_inx.roll(dims=2, shifts=-1) != 0
        if "O" not in problems[7][0]:
            not_to_depot[:]=True
        after_distances = (segment_after_lengths * not_to_depot)[:,:,:-1].sum(2)
        ######比较长度，对齐tensor，替换好的
        real_after_inx=real_after_inx.reshape(batch_size,per_cut,-1)
        after_distances=after_distances.reshape(-1,per_cut)
        ori_max_len=(start_ends[:,:,1]-start_ends[:,:,0]).reshape(-1).max(0).values+1
        pa_ori_solution=torch.cat((ori_solution,torch.zeros(batch_size,per_cut,ori_max_len,dtype=torch.long)),dim=2)
        ar_path_idx=torch.arange(pa_ori_solution.shape[-1])[None,None,:].repeat(batch_size,per_cut,1)
        sameL_mask=(ar_path_idx>=start_ends[:,:,[0]])*(ar_path_idx<(start_ends[:,:,[0]]+ori_max_len))
        cutL_mask=(ar_path_idx>=start_ends[:,:,[0]])*(ar_path_idx<=(start_ends[:,:,[1]]))*(pa_ori_solution)
        a=(cutL_mask+1)*sameL_mask-1
        same_l_ori=a[a>=0].reshape(-1,per_cut,ori_max_len)
        new_l=real_after_inx.shape[2]
        if ori_max_len<new_l:
            same_l_ori=torch.cat((same_l_ori,torch.zeros(batch_size,per_cut,new_l-ori_max_len)),dim=-1)
        elif ori_max_len>new_l:
            real_after_inx=torch.cat((real_after_inx,torch.zeros(batch_size,per_cut,ori_max_len-new_l)),dim=-1)
        same_l_ori=same_l_ori.to(torch.long)
        new_max_l=same_l_ori.shape[-1]
        best_mask=(ori_distances > after_distances)[:,:,None].repeat(1,1,new_max_l)
        same_l_ori[best_mask]=real_after_inx[best_mask].long()
        new_sol=replace_segments_fully_vectorized(ori_so0, start_ends, same_l_ori)###替换原始解，在start——end之间
        new_solution=remove_zero_end(new_sol) #由于替换的子环路可能多个0在后面，导致拼接后中间0的数量过多，移动0到后面

        #计算替换后的完整解
        self.load_problems(batch_size=problems[0].shape[0],problems=problems)
        new_reward=self.get_travel_distance(new_solution.unsqueeze(1))
        return new_solution,new_reward

