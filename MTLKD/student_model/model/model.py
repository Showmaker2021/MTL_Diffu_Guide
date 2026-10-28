import time

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MTLKD_model']


class MTLKD_model(nn.Module):

    def __init__(self,**model_params):
        super().__init__()
        self.model_params = model_params
        self.eval_type = self.model_params['eval_type']
        # self.problem = self.model_params['problem']

        self.encoder =CVRP_Encoder(**model_params)
        self.decoder = CVRP_Decoder(**model_params)
        self.encoded_nodes = None  # shape: (batch, problem+1, EMBEDDING_DIM)
        if torch.cuda.is_available():
            self.device = torch.device('cuda', torch.cuda.current_device()) if 'device' not in model_params.keys() else \
            model_params['device']
        else:
            self.device = torch.device('cpu')
        # assert self.model_params['norm'] == "batch", "The original MTLModel implementation uses batch normalization."
        self.test=model_params.get("test",False)

    def set_eval_type(self, eval_type):
        self.eval_type = eval_type



    def forward(self, reset_state,state,returnProbs=False):
        # self.t1=time.time()
        # print("out_time",self.t1-self.t2)
        ########没选择节点的获取，不齐补0，仓库节点的获取

        ####### encoder

        ####### decoder对未被选择的节点

        #######转换为pomo形式的概率


        probs=None
        can_selected_indices=None
        prob=None
        batch_size ,pomo_size,problem_size = state.ninf_mask.shape
        decoder_atts=None
        if state.selected_count == 0:  # First Move, depot
            real_selected_ind = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long).to(self.device)
            prob = torch.ones(size=(batch_size, pomo_size)).to(self.device)

        #
        # elif state.selected_count == 1:  # Second Move, POMO
        # # selected = torch.arange(start=1, end=pomo_size+1)[None, :].expand(batch_size, -1).to(self.device)
        #     real_selected_ind = state.START_NODE

        else:
            ########没选择节点的获取，不齐补0，仓库节点的获取

            ####### encoder

            ####### decoder对未被选择的节点

            #######转换为pomo形式的概率
            # if RRC_start==True:
            #     self.encoded_nodes=self.encoder(reset_state.depot_node_feat)

            if self.test==False:
                self.encoded_nodes=self.encoder(reset_state.depot_node_feat)
            else:
                if state.selected_count <= 1:
                    self.encoded_nodes=self.encoder(reset_state.depot_node_feat)
                    # self.encoded_nodes = self.encoder(reset_state.depot_node_feat)
            unselected_indices,end0_mask=get_unvisited_indices(state.visited_ninf_flag[:,:,1:])
            # unselected_indices = get_unvisited_nodes_vectorized(state.selected_node_list, problem_size)

            can_selected_indices=torch.cat((torch.zeros(batch_size ,pomo_size,1,device=self.device,dtype=torch.int),unselected_indices),dim=-1) #把仓库节点也加进去了

            unselected_indices=torch.cat((torch.zeros(batch_size ,pomo_size,1,device=self.device,dtype=torch.int),state.current_node.unsqueeze(-1),
                                          can_selected_indices),dim=-1)
            unselected_node_emb=_get_encoding(self.encoded_nodes, unselected_indices)#头尾加canvisited
            # shape: (batch, pomo, embedding)
            attr = torch.cat((state.load[:, :, None], state.current_time[:, :, None], state.route_limit[:,None,None]-state.length[:, :, None], state.open[:, :, None]), dim=2)
            # shape: (batch, pomo, 4)
            probs = self.decoder(unselected_node_emb,unselected_indices, attr, ninf_mask=state.ninf_mask)

            # shape: (batch, pomo, problem+1)


            while True:
                if self.eval_type == 'softmax':
                    try:
                        selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, pomo_size)
                    except Exception as exception:
                        print(">> Catch Exception: {}, on the instances of {}".format(exception, state.PROBLEM))
                        exit(0)
                else:
                    selected = probs.argmax(dim=2)
                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                # shape: (batch, pomo)
                if (prob != 0).all():
                    break
            real_selected_ind = torch.gather(can_selected_indices, 2, selected.unsqueeze(-1)).squeeze(-1)

        # self.t2=time.time()
        # print("in_time",self.t2-self.t1)
        if returnProbs:
            return real_selected_ind,can_selected_indices, prob, probs
        else:
            return real_selected_ind,can_selected_indices, prob


def get_unvisited_indices_vectorized(mask: torch.Tensor, fill_value: int = 0) -> torch.Tensor:
    """
    Args:
        mask: 形状 (batch_size, max_nodes)，-inf 表示已访问节点
        fill_value: 填充值（默认为0）
    Returns:
        unvisited_indices: 形状 (batch_size, N)，N为当前batch中最长的未访问列表长度
    """
    # 1. 获取未访问节点的坐标 (batch_idx, node_idx)
    bs,pomo_s,N=mask.shape
    mask=mask.reshape(bs*pomo_s,N)
    mask[:,0]=0
    batch_indices, node_indices = torch.where(mask != -float('inf'))

    # 2. 计算每个样本的未访问节点数量
    counts = torch.bincount(batch_indices, minlength=mask.size(0))
    max_N = counts.max().item() if mask.size(0) > 0 else 0

    # 3. 为每个样本预分配结果张量，并填充有效索引
    unvisited_indices = torch.full(
        (mask.size(0), max_N),
        fill_value=fill_value,
        dtype=torch.long,
        device=mask.device
    )

    # 4. 使用 scatter 填充有效索引
    # 生成每个样本的写入位置（从0到counts[batch_idx]-1）
    scatter_positions = torch.arange(max_N, device=mask.device).expand(mask.size(0), -1)
    valid_mask = scatter_positions < counts.unsqueeze(1)  # (batch_size, max_N)

    # 将 node_indices 按 batch 分组后填充到对应位置
    unvisited_indices[valid_mask] = node_indices
    unvisited_indices=unvisited_indices.reshape(bs,pomo_s,-1)
    return unvisited_indices


def get_unvisited_indices(mask):
    """
    完全向量化的未访问节点索引恢复（自动计算max_nodes）

    参数:
        mask: [batch, seq, prob_size] (0=未访问, -inf=已访问)

    返回:
        indices: [batch, seq, max_nodes] (补0对齐)
        counts: [batch, seq] (实际有效长度)
    """
    device = mask.device
    batch, seq, prob_size = mask.shape

    # 1. 生成节点索引网格 [batch, seq, prob_size]
    node_ids = torch.arange(prob_size, device=device).view(1, 1, -1)

    # 2. 获取所有未访问节点索引 [batch, seq, prob_size]
    unvisited = (mask == 0) * node_ids

    # 3. 计算每个序列的未访问节点数 [batch, seq]
    counts = (mask == 0).sum(dim=-1)
    max_unvisited = counts.max().item()

    # 4. 创建输出张量（自动初始化为0）
    indices = torch.full((batch, seq, max_unvisited), -1, dtype=torch.long, device=device)

    # 5. 使用scatter_填充有效数据（关键步骤）
    # 生成写入位置索引 [batch, seq, max_unvisited]
    pos_idx = torch.arange(max_unvisited, device=device).view(1, 1, -1).expand(batch, seq, -1)
    valid_mask = pos_idx < counts.unsqueeze(-1)  # 有效位置掩码

    # 6. 压缩未访问节点并填充 [batch, seq, prob_size] -> [batch, seq, max_unvisited]
    compressed = unvisited.sort(dim=-1, descending=True).values
    indices[valid_mask] = compressed[:,:,:max_unvisited][valid_mask]  # 只填充有效索引
    indices+=1
    return indices,valid_mask


def get_unvisited_nodes_vectorized(selected_node_list, node_N):
    batch, pomo, selected = selected_node_list.shape

    # 1. 去重已选节点：生成 (batch, pomo, node_N) 的掩码
    selected_mask = torch.zeros(batch, pomo, node_N, dtype=torch.bool)

    # 用 scatter_ 标记已选节点（并行化）
    selected_mask.scatter_(
        dim=-1,
        index=selected_node_list.clamp(0, node_N - 1),  # 避免越界
        src=torch.ones_like(selected_node_list, dtype=torch.bool)
    )

    # 2. 获取未选节点索引 (batch, pomo, node_N - num_selected)
    unselected_mask = ~selected_mask
    unselected_counts = unselected_mask.sum(dim=-1)  # (batch, pomo)
    max_unselected_num = unselected_counts.max().item()

    # 3. 填充未选节点索引
    unselected_indices = torch.zeros(batch, pomo, max_unselected_num, dtype=torch.long)

    # 并行化填充（利用 masked_select + split）
    flat_unselected_mask = unselected_mask.reshape(-1, node_N)  # (batch*pomo, node_N)
    flat_indices = torch.where(flat_unselected_mask)[1]  # 所有未选节点的全局索引

    # 按 (b, p) 分组
    split_sizes = unselected_counts.flatten().tolist()
    grouped_indices = torch.split(flat_indices, split_sizes)

    # 填充到目标张量
    for i, indices in enumerate(grouped_indices):
        unselected_indices[i // pomo, i % pomo, :len(indices)] = indices
    return unselected_indices


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, node_num, embedding) or (batch, 1, node_num, embedding) or (batch, pomo, node_num, embedding)
    # node_index_to_pick.shape: (batch, pomo, num_to_pick)

    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(-1)
    if encoded_nodes.dim() == node_index_to_pick.dim():
        encoded_nodes_ = encoded_nodes.clone()[:, None, :, :].expand(-1, pomo_size, -1, -1)
    elif encoded_nodes.dim() > node_index_to_pick.dim() and encoded_nodes.size(1) != pomo_size:
        encoded_nodes_ = encoded_nodes.clone().expand(-1, pomo_size, -1, -1)
    else:
        encoded_nodes_ = encoded_nodes.clone()
    gathering_index = node_index_to_pick[:, :, :, None].expand(-1, -1, -1, embedding_dim)
    # shape: (batch, pomo, num_to_pick, embedding)
    picked_nodes = encoded_nodes_.gather(dim=2, index=gathering_index)
    # shape: (batch, pomo, num_to_pick, embedding)

    return picked_nodes



########################################
# ENCODER
########################################
class CVRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num =  1
        self.embedding = nn.Linear(6, embedding_dim, bias=True)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data_):

        data = data_.clone().detach()
        embedded_input = self.embedding(data)

        out = embedded_input  # [B*(V-1), problem_size - current_step +2, embedding_dim]

        layer_count = 0
        for layer in self.layers:
            out = layer(out)
            layer_count += 1
        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.add_norm=Add_And_Normalization_Module(**model_params)
        self.feedForward = FeedForward(**model_params)


    def forward(self, input1):

        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        out_concat = multi_head_attention(q, k, v)  # shape: (B, n, head_num*key_dim)

        multi_head_out = self.multi_head_combine(out_concat)  # shape: (B, n, embedding_dim)
        out1=self.add_norm(multi_head_out,input1)
        # out1 = input1 + multi_head_out
        out2 = self.feedForward(out1)

        out3 = out1 + out2
        return out3
        # shape: (batch, problem, EMBEDDING_DIM)
########################################
# DECODER
########################################
class CVRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        decoder_layer_num = self.model_params['decoder_layer_num']

        self.embedding_first_node = nn.Linear(embedding_dim+4, embedding_dim, bias=True)
        self.embedding_last_node = nn.Linear(embedding_dim+4, embedding_dim, bias=True)

        self.layers = nn.ModuleList([DecoderLayer(**model_params) for _ in range(decoder_layer_num)])
        # self.Linear_final = nn.Linear(embedding_dim, 2, bias=True)
        self.q_re_linear=nn.Linear(embedding_dim*2,embedding_dim,bias=True)



    def forward(self,unselected_node_emb,unselected_indices, attr, ninf_mask=None):
        bs,pomo_size,N,emb_dim=unselected_node_emb.shape
        embedded_first_node=unselected_node_emb[:,:,[0]]
        embedded_last_node=unselected_node_emb[:,:,[1]]
        left_encoded_node=unselected_node_emb[:,:,2:]

        first_node_cat = torch.cat((embedded_first_node,attr.unsqueeze(2)), dim=3)
        last_node_cat = torch.cat((embedded_last_node,attr.unsqueeze(2)), dim=3)
        # ------------------------------------------------
        # ------------------------------------------------
        embedded_first_node_ = self.embedding_first_node(first_node_cat)

        embedded_last_node_ = self.embedding_last_node(last_node_cat)


        embeded_all = torch.cat((embedded_first_node_,embedded_last_node_,left_encoded_node), dim=2)
        out = embeded_all  # [B*(V-1), problem_size - current_step +2, embedding_dim]

        end0_mask = torch.zeros(bs, pomo_size, N, device=unselected_indices.device)
        end0_mask[:, :, 3:].masked_fill_(unselected_indices[:, :, 3:] == 0, -float('inf'))

        end0_mask1=end0_mask.reshape(bs*pomo_size,-1)
        out=out.reshape(bs*pomo_size,-1,emb_dim)
        for layer in self.layers:
            out = layer(out,mask=end0_mask1)

        out_last_depot=out[:,[0,1]].reshape(bs,pomo_size,-1)
        out_last_depot = self.q_re_linear(out_last_depot)
        #  Single-Head Attention, for probability calculation
        #######################################################
        unvistited_ = out.reshape(bs,pomo_size,-1,emb_dim)[:, :, 2:, :]
        score = torch.matmul(out_last_depot.unsqueeze(2), unvistited_.transpose(2, 3)).squeeze(2)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)
        unvistited_mask = torch.gather(ninf_mask, 2, unselected_indices[:,:,2:])


        score_masked = score_clipped + unvistited_mask + end0_mask[:,:,2:]

        probs = F.softmax(score_masked, dim=2)

        return probs


class DecoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.feedForward = FeedForward(**model_params)

    def forward(self, input1,mask=None):

        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        out_concat = multi_head_attention(q, k, v,rank2_ninf_mask=mask)

        multi_head_out = self.multi_head_combine(out_concat)

        out1 = input1 + multi_head_out
        out2 = self.feedForward(out1)
        out3 = out1 + out2
        return out3


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None,return_att=False):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)
    if return_att:
        return out_concat,weights
    else:
        return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.add = True
        if model_params["norm"] == "batch":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=True)
        elif model_params["norm"] == "batch_no_track":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif model_params["norm"] == "instance":
            self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif model_params["norm"] == "layer":
            self.norm = nn.LayerNorm(embedding_dim)
        elif model_params["norm"] == "rezero":
            self.norm = torch.nn.Parameter(torch.Tensor([0.]), requires_grad=True)
        else:
            self.norm = None

    def forward(self, input1=None, input2=None):
        # input.shape: (batch, problem, embedding)
        if isinstance(self.norm, nn.InstanceNorm1d):
            added = input1 + input2 if self.add else input2
            transposed = added.transpose(1, 2)
            # shape: (batch, embedding, problem)
            normalized = self.norm(transposed)
            # shape: (batch, embedding, problem)
            back_trans = normalized.transpose(1, 2)
            # shape: (batch, problem, embedding)
        elif isinstance(self.norm, nn.BatchNorm1d):
            added = input1 + input2 if self.add else input2
            batch, problem, embedding = added.size()
            normalized = self.norm(added.reshape(batch * problem, embedding))
            back_trans = normalized.reshape(batch, problem, embedding)
        elif isinstance(self.norm, nn.LayerNorm):
            added = input1 + input2 if self.add else input2
            back_trans = self.norm(added)
        elif isinstance(self.norm, nn.Parameter):
            back_trans = input1 + self.norm * input2 if self.add else self.norm * input2
        else:
            back_trans = input1 + input2 if self.add else input2

        return back_trans


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))

