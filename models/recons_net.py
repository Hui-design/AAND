import torch.nn as nn
import torch
import torch.nn.functional as F
import math
# from loss import resize_lib, mask_thresh
from utils.vis import *
import pdb

def pair_cosine(a, b):
    cos_sim = torch.einsum('bnc,bmc->bnm', a, b) # [B, N, M]
    B, N, M = cos_sim.shape
    a_norm = torch.sqrt((a**2).sum(dim=-1))         # [B, N]
    a_norm = a_norm.unsqueeze(-1).expand(-1,-1,M)   # [B, N, M]
    b_norm = torch.sqrt((b**2).sum(dim=-1))         # [B, M]
    b_norm = b_norm.unsqueeze(1).expand(-1,N,-1)    # [B, N, M]
    cos_sim = cos_sim / (a_norm*b_norm)
    return cos_sim

resize_lib = {0: torch.nn.AdaptiveAvgPool2d((64, 64)),
              1: torch.nn.AdaptiveAvgPool2d((32, 32)) ,
              2: torch.nn.AdaptiveAvgPool2d((16, 16))}  # 注意：可以尝试不同的下采样方式
mask_thresh = 0.3

def patch_to_tensor(x):
    B, N, C = x.shape
    H, W = int(math.sqrt(N)), int(math.sqrt(N))
    x = x.reshape(B, H, W, C).permute(0,3,1,2)  # [B, C, H, W]
    return x 
    
def tensor_to_patch(x):
    B, C, H, W = x.shape
    x = x.reshape(B, C, H*W).permute(0,2,1)     # [B, N, C] 
    return x

class RAR_single(nn.Module):
    def __init__(self, d_model=[], size_list=[], num_item=50):
        super(RAR_single, self).__init__()
        self.d_model = d_model
        self.num_item = num_item
        self.pos_enc = []
        self.projs = [] 
        self.projs2 = [] 
        self.K_list = []
        hidden_dim = min(2*d_model[0], 1024)
        for i, f_dim in enumerate(d_model):
            self.projs.append(nn.Sequential(  
                nn.Conv2d(f_dim, hidden_dim, 1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, f_dim, 1))
            )
            # self.K_list.append(nn.Parameter(torch.randn(1, num_item+1, f_dim)))  
            # self.V_list.append(nn.Parameter(torch.randn(1, 1, f_dim)))
            self.K_list.append(nn.Parameter(torch.randn(1, num_item*2, f_dim))) 
            self.projs2.append(nn.Sequential(  
                nn.Conv2d(f_dim, hidden_dim, 1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, f_dim, 1)
                )
            )
            self.pos_enc.append(positionalencoding2d(f_dim, size_list[i], size_list[i]))

        self.projs = nn.ModuleList(self.projs)
        self.projs2 = nn.ModuleList(self.projs2)
        self.K_list = nn.ParameterList(self.K_list)  # 如果不加这句话则paramter不会更新
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def regular_score(self,score):
        score = torch.where(torch.isnan(score), torch.zeros_like(score), score)
        score = torch.where(torch.isinf(score), torch.zeros_like(score), score)
        return score
    
    def forward(
        self,
        inputs,
        flag,
        atten_mask=None,
    ):  
        inputs_recons = []
        attention_list = []
        for i in range(len(self.projs)):
            # Query
            # pdb.set_trace()
            pos_enc = self.pos_enc[i].unsqueeze(0).expand(inputs[i].shape[0],-1,-1,-1).to(inputs[i].device)  # [B, C, H, W]
            q = self.projs[i](inputs[i]+pos_enc)       # [B, C, H, W]
            q = tensor_to_patch(q)                     # [B, N, C]
            # Key, Value
            # pdb.set_trace()
            k = self.K_list[i].expand(q.shape[0], -1, -1).to(q.device)      # [B, M, C]
            # v = self.V_list[i].expand(q.shape[0], -1, -1).to(q.device)      # [B, M, C]
            # pdb.set_trace()
            noise = self.projs2[i](inputs[i])   # [B, N, C]
            noise = torch.clamp(self.tanh(noise), min=-1, max=1)
            v = tensor_to_patch(noise * inputs[i])  # [B, N, C]

            cos_sim = pair_cosine(q, k) # [B, N, M]
            attention_scores = F.softmax(cos_sim / 0.2, dim=-1)    # [B, N, M]
            attention_scores = attention_scores[:, :, self.num_item:].sum(dim=-1, keepdim=True)  # [B,N,1]
            attention_list.append(attention_scores)
            # if flag:
            #     mask = (attention_scores >= 0.2).float()
            #     attention_scores = attention_scores * mask

            if atten_mask is not None:
                # pdb.set_trace()
                index = {256:0, 512:1, 1024:2}
                atten_mask = (resize_lib[index[self.d_model[0]]](atten_mask).reshape(attention_scores.shape[0], -1) > 0.3)    # [B,N]
                attention_scores = atten_mask.unsqueeze(-1).float()      # [B,N,1]

            # weighted Value
            # pdb.set_trace()
            # q_recons = torch.matmul(attention_scores, v)     # [B, N, C]
            q_recons = attention_scores * v
            # q_recons = attention_scores * v * torch.exp(attention_scores-0.5)
            q_recons = patch_to_tensor(q_recons)             # [B, C, H, W]
            inputs_recons.append(q_recons)
            

        return inputs_recons[0], attention_list[0]
    


class MLP(nn.Module):
    def __init__(self, f_dim):
        super(MLP, self).__init__()
        self.projs = nn.Sequential(  
                nn.Conv2d(f_dim, 2*f_dim, 1),
                nn.ReLU(),
                nn.Conv2d(2*f_dim, 2*f_dim, 1),
                nn.ReLU(),
                nn.Conv2d(2*f_dim, f_dim, 1)
        )

    def forward(
        self,
        inputs,
        flag,
    ):  
        inputs_recons = self.projs(inputs[0])

        return inputs_recons, None



def positionalencoding2d(D, H, W):
    """
    :param D: dimension of the model
    :param H: H of the positions
    :param W: W of the positions
    :return: DxHxW position matrix
    """
    if D % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D))
    P = torch.zeros(D, H, W)
    # Each dimension use half of D
    D = D // 2
    div_term = torch.exp(torch.arange(0.0, D, 2) * -(math.log(1e4) / D))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D:2, :, :]  = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :]  = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[D::2,  :, :]  = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D+1::2,:, :]  = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    return P