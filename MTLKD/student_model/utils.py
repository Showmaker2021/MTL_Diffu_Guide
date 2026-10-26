import os, sys
import time
import random
import math
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from scipy.stats import ttest_rel


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += (val * n)
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0


class TimeEstimator:
    def __init__(self):
        self.start_time = time.time()
        self.count_zero = 0

    def reset(self, count=1):
        self.start_time = time.time()
        self.count_zero = count-1

    def get_est(self, count, total):
        curr_time = time.time()
        elapsed_time = curr_time - self.start_time
        remain = total-count
        remain_time = elapsed_time * remain / (count - self.count_zero)

        elapsed_time /= 3600.0
        remain_time /= 3600.0

        return elapsed_time, remain_time

    def get_est_string(self, count, total):
        elapsed_time, remain_time = self.get_est(count, total)

        elapsed_time_str = "{:.2f}h".format(elapsed_time) if elapsed_time > 1.0 else "{:.2f}m".format(elapsed_time*60)
        remain_time_str = "{:.2f}h".format(remain_time) if remain_time > 1.0 else "{:.2f}m".format(remain_time*60)

        return elapsed_time_str, remain_time_str

    def print_est_time(self, count, total):
        elapsed_time_str, remain_time_str = self.get_est_string(count, total)

        print("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(count, total, elapsed_time_str, remain_time_str))





def Rearrange_solution_clockwise( problem, solution):

        problem_size = solution.shape[1]
        coor = problem[:, :, [0, 1]].clone()
        order_node = solution[:, :, 0]
        order_flag = solution[:, :, 1]

        batch_size = solution.shape[0]

        visit_depot_num = torch.sum(order_flag, dim=1)

        all_subtour_num = torch.sum(visit_depot_num)

        fake_solution = torch.cat((order_flag, torch.ones(batch_size)[:, None]), dim=1)

        start_from_depot = fake_solution.nonzero()

        start_from_depot_1 = start_from_depot[:, 1]

        start_from_depot_2 = torch.roll(start_from_depot_1, shifts=-1)

        sub_tours_length = start_from_depot_2 - start_from_depot_1

        max_subtour_length = torch.max(sub_tours_length)

        start_from_depot2 = order_flag.nonzero()

        start_from_depot3 = order_flag.roll(shifts=-1, dims=1).nonzero()

        repeat_solutions_node = order_node.repeat_interleave(visit_depot_num, dim=0)
        double_repeat_solution_node = repeat_solutions_node.repeat(1, 2)

        x1 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             >= start_from_depot2[:, 1][:, None]
        x2 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             <= start_from_depot3[:, 1][:, None]

        x3 = (x1 * x2).long()

        sub_tourss = double_repeat_solution_node * x3

        x4 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             < (start_from_depot2[:, 1][:, None] + max_subtour_length)

        x5 = x1 * x4

        sub_tours_padding = sub_tourss[x5].reshape(all_subtour_num, max_subtour_length)
        subtour_lengths = (sub_tours_padding > 1).int().sum(1)

        repeated_coor = torch.repeat_interleave(coor, repeats=visit_depot_num, dim=0)
        depot_coor = repeated_coor[:, [0], :].clone()
        repeated_coor[:, 0, :] = 0

        subtours_coor = repeated_coor.gather(dim=1, index=sub_tours_padding[:, :, None].repeat(1, 1, 2))
        subtours_coor = torch.cat((subtours_coor, depot_coor), dim=1)
        subtours_coor_sum = torch.sum(subtours_coor, dim=1)
        subtours_centroid = subtours_coor_sum / (subtour_lengths + 1)[:, None]
        subtours_centroid_total_num = subtours_centroid.shape[0]

        temp_tri = np.triu(np.ones((len(visit_depot_num), len(visit_depot_num))), k=1)
        visit_depot_num_numpy = visit_depot_num.clone().cpu().numpy()
        temp_index = np.dot(visit_depot_num_numpy, temp_tri)

        temp_index_1 = torch.from_numpy(temp_index).long().cuda()

        temp_index_2 = visit_depot_num + temp_index_1

        x1 = torch.arange(subtours_centroid_total_num)[None, :].repeat(batch_size, 1) >= temp_index_1[:, None]
        x2 = torch.arange(subtours_centroid_total_num)[None, :].repeat(batch_size, 1) < temp_index_2[:, None]
        x3_ = (x1 * x2).int()
        x3 = x3_[:,:,None].repeat(1,1,2)

        subtours_centroid_repeat = subtours_centroid[None,:,:].repeat(batch_size,1,1)

        subtours_centroid_sperate = subtours_centroid_repeat * x3

        index2 = temp_index_1.clone().unsqueeze(1).unsqueeze(2).repeat(1,1,2)

        based_centroids = subtours_centroid_sperate.gather(dim=1,index=index2)

        single_depot_coor = coor[:, [0], :]

        repeated_depot_coor = coor[:, [0], :].repeat(1,all_subtour_num,1)

        all_centroid_depot_vectors = subtours_centroid_sperate - repeated_depot_coor

        based_centroid_depot_vectors = based_centroids - single_depot_coor

        repeated_based_centroid_depot_vectors = based_centroid_depot_vectors.repeat(1,all_subtour_num,1)

        x1_times_x2 = (repeated_based_centroid_depot_vectors * all_centroid_depot_vectors).sum(2)

        x1_module_length = torch.sqrt((repeated_based_centroid_depot_vectors**2).sum(2))
        x2_module_length = torch.sqrt((all_centroid_depot_vectors**2).sum(2))

        cos_value = x1_times_x2 / (x1_module_length*x2_module_length)
        cos_value[cos_value.ge(1)] =  1 - 1e-5
        cos_value[cos_value.le(-1)] = -1 + 1e-5
        cross_value = np.cross(repeated_based_centroid_depot_vectors.cpu().numpy(), all_centroid_depot_vectors.cpu().numpy())

        cross_value = torch.tensor(cross_value)
        negtivate_sign_2 = torch.ones(size=(cross_value.shape[0],cross_value.shape[1]))
        negtivate_sign_2[cross_value.lt(0)] = -1

        theta_value = torch.arccos(cos_value)
        theta_value = torch.where(torch.isnan(theta_value), torch.full_like(theta_value, 2 * 3.1415926), theta_value)
        theta_value = negtivate_sign_2*theta_value

        theta_value[theta_value.lt(0)] +=2 * 3.1415926

        theta_value[x3_.le(0)] = 6*3.1415926
        theta_value_sort_value, theta_value_sort_index = torch.sort(theta_value,dim=1)

        repeated_sub_tours_padding = sub_tours_padding.unsqueeze(0).repeat(batch_size,1,1)

        gather_theta_value_sort_index = theta_value_sort_index.unsqueeze(2).repeat(1,1,max_subtour_length)

        resort_repeated_sub_tours_padding = repeated_sub_tours_padding.gather(dim=1,index=gather_theta_value_sort_index)

        x4 = torch.arange(all_subtour_num)[None,:].repeat(batch_size,1)

        x5 = (x4 < visit_depot_num[:,None]).int()
        x6 = x5.unsqueeze(2).repeat(1,1,max_subtour_length)

        resort_repeated_sub_tours_padding = resort_repeated_sub_tours_padding*x6

        resort_repeated_sub_tours_padding = resort_repeated_sub_tours_padding.reshape(batch_size,-1)


        resort_sub_tours = resort_repeated_sub_tours_padding[resort_repeated_sub_tours_padding.gt(0)].reshape(batch_size,-1)

        repeated_sub_tours_length = sub_tours_length[sub_tours_length.gt(0)].unsqueeze(0).repeat(batch_size,1)

        resort_repeated_sub_tours_length = repeated_sub_tours_length.gather(dim=1,index= theta_value_sort_index)
        resort_repeated_sub_tours_length = resort_repeated_sub_tours_length*x5
        max_subtour_number = visit_depot_num.max()

        resort_repeated_sub_tours_length = resort_repeated_sub_tours_length[:,:max_subtour_number]

        temp_tri = np.triu(np.ones((batch_size,max_subtour_number.item(), max_subtour_number.item())), k=1)
        resort_repeated_sub_tours_length_numpy = resort_repeated_sub_tours_length.clone().cpu().numpy()
        temp_index = np.dot(resort_repeated_sub_tours_length_numpy, temp_tri)
        temp_index_1 = torch.from_numpy(temp_index).long().cuda()
        index1 = torch.arange(batch_size)
        temp_index_1 = temp_index_1[index1,index1]
        temp_index_1[temp_index_1.ge(problem_size)]=0


        flag = torch.zeros(size=(batch_size,problem_size),dtype=torch.int)
        index1 = torch.arange(batch_size)[:,None].repeat(1,max_subtour_number)


        flag[index1,temp_index_1]=1

        solution_ = torch.cat((resort_sub_tours.unsqueeze(2),flag.unsqueeze(2)),dim=2)
        return solution_



def seed_everything(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def get_opt_sol_path(dir, problem, size):
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
    return os.path.join(dir, BASELINE_SOLVERS.get(problem)+"_"+problem.upper()+"100_1000uni.pkl")


def num_param(model):
    nb_param = 0
    for param in model.parameters():
        nb_param += param.numel()
    print('There are {} ({:.2f} million) parameters in this neural network'.format(nb_param, nb_param / 1e6))


def check_null_hypothesis(a, b):
    print(len(a), a)
    print(len(b), b)
    alpha_threshold = 0.05
    t, p = ttest_rel(a, b)  # Calc p value
    print(t, p)
    p_val = p / 2  # one-sided
    # assert t < 0, "T-statistic should be negative"
    print("p-value: {}".format(p_val))
    if p_val < alpha_threshold:
        print(">> Null hypothesis (two related or repeated samples have identical average values) is Rejected.")
    else:
        print(">> Null hypothesis (two related or repeated samples have identical average values) is Accepted.")


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename, disable_print=False):
    filedir = os.path.split(filename)[0]
    if not os.path.isdir(filedir):
        os.makedirs(filedir)
    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    if not disable_print:
        print(">> Save dataset to {}".format(filename))


def load_dataset(filename, disable_print=False):
    with open(check_extension(filename), 'rb') as f:
        data = pickle.load(f)
    if not disable_print:
        print(">> Load {} data ({}) from {}".format(len(data), type(data), filename))
    return data


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
        Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def run_all_in_pool(func, directory, dataset, opts, use_multiprocessing=True, disable_tqdm=True):
    # # Test
    # res = func((directory, 'test', *dataset[0]))
    # return [res]

    os.makedirs(directory, exist_ok=True)
    num_cpus = os.cpu_count() if opts.cpus is None else opts.cpus

    w = len(str(len(dataset) - 1))
    offset = getattr(opts, 'offset', None)
    if offset is None:
        offset = 0
    ds = dataset[offset:(offset + opts.n if opts.n is not None else len(dataset))]
    pool_cls = (Pool if use_multiprocessing and num_cpus > 1 else ThreadPool)
    with pool_cls(num_cpus) as pool:
        results = list(tqdm(pool.imap(
            func,
            [
                (
                    directory,
                    str(i + offset).zfill(w),
                    *problem
                )
                for i, problem in enumerate(ds)
            ]
        ), total=len(ds), mininterval=opts.progress_bar_mininterval, disable=disable_tqdm))

    failed = [str(i + offset) for i, res in enumerate(results) if res is None]
    assert len(failed) == 0, "Some instances failed: {}".format(" ".join(failed))
    return results, num_cpus


def show(x, y, label, title, xdes, ydes, path, min_y=None, max_y=None, x_scale="linear", dpi=300):
    plt.style.use('fast')  # bmh, fivethirtyeight, Solarize_Light2
    plt.figure(figsize=(8, 8))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan', 'lightpink', 'lightgreen', 'linen', 'slategray', 'darkviolet', 'darkcyan']

    assert len(x) == len(y)
    for i in range(len(x)):
        if i < len(label):
            # plt.scatter(x[i], y[i], color=colors[i], s=50)  # label=label[i]
            plt.plot(x[i], y[i], color=colors[i], label=label[i], linewidth=3)
        else:
            # plt.scatter(x[i], y[i], color=colors[i % len(label)])
            plt.plot(x[i], y[i], color=colors[i % len(label)], linewidth=3)

    plt.gca().get_xaxis().get_major_formatter().set_scientific(False)
    plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
    plt.xlabel(xdes, fontsize=24)
    plt.ylabel(ydes, fontsize=24)

    if min_y and max_y:
        plt.ylim((min_y, max_y))

    plt.title(title, fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(loc='upper right', fontsize=16)
    plt.xscale(x_scale)
    # plt.margins(x=0)

    # plt.grid(True)
    plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close("all")
