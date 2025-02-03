import torch
from torch.nn import functional as F
import torch.nn as nn
from models.recons_net import pair_cosine
import pdb
import numpy as np
from utils.vis import vis_hotmap_single

resize_lib = {0: torch.nn.AdaptiveAvgPool2d((64, 64)),
              1: torch.nn.AdaptiveAvgPool2d((32, 32)) ,
              2: torch.nn.AdaptiveAvgPool2d((16, 16))}  # 注意：可以尝试不同的下采样方式
mask_thresh = 0.3

def get_recons_loss(a, b):
    # mse_loss = torch.nn.MSELoss()
    # mse_loss = torch.nn.MSELoss(reduction='none')
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    # pdb.set_trace()
    for item in range(len(a)):
        # loss += torch.sqrt(mse_loss(a[item], b[item]).sum(dim=1)).mean()  # [B,C,H,W]->[B,H,W] 
        # loss += torch.mean(1-cos_loss(a[item].reshape(a[item].shape[0],-1),
        #                               b[item].reshape(b[item].shape[0],-1)))
        loss += torch.mean(1-cos_loss(a[item],b[item]))
    return loss

def get_orth_loss(a, b): 
    #mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        #print(a[item].shape)
        #print(b[item].shape)
        #loss += 0.1*mse_loss(a[item], b[item])
        # loss += torch.mean(cos_loss(a[item], b[item]))  # dim=-1  # 为什么用这个的话梯度就很大呢？
        # a: [1, 50, C]
        # loss += torch.mean( F.relu(cos_loss(a[item].view(a[item].shape[0],-1),
        #                             b[item].view(b[item].shape[0],-1)) ) )  # 为什么刚好收敛到了0呢？写错了，写成了1-

        loss += torch.mean( pair_cosine(a[item], b[item])  ) 
    return loss


def loss_fucntion(a, b):
    #mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        #print(a[item].shape)
        #print(b[item].shape)
        #loss += 0.1*mse_loss(a[item], b[item])
        # pdb.set_trace() 
        B, C, H, W = a[item].shape
        f_t = a[item].reshape(B,C,H*W)
        f_s = b[item].reshape(B,C,H*W)
        # weight = 1 / cos_loss(f_t, f_s).detach()  # [B,C,H*W] -> [B,H*W]  相似度越高，权重越小
        # pdb.set_trace()
        loss_i = 1 - cos_loss(f_t, f_s)  # [B,C,H*W] -> [B,H*W], 相似度越高，loss越小
        # loss += torch.mean(loss_i.topk(k=int(0.1*H*W), dim=1)[0]) # 选择损失大的
        loss += torch.mean(loss_i.topk(k=10, dim=1)[0]) # 选择损失大的
        loss += torch.mean(1-cos_loss(a[item].reshape(a[item].shape[0],-1),
                                      b[item].reshape(b[item].shape[0],-1)))
        # loss += torch.mean(weight * loss_i) / 3
        # loss += torch.mean(loss_i)

    return loss


def entropy(p):
    # p: (N*M)
    logits = -p * torch.log(p + 0.0001)
    entropy = torch.sum(logits, dim=-1)  # (N)
    return torch.mean(entropy)  

def get_entropy_loss(attention_list, mask_ori):
    # attention_list[[B,N,M],[]], mask_ori[B,1,H,W]
    entropy_loss = 0.0
    num_item = attention_list[0].shape[-1]
    for i, atten in enumerate(attention_list):
        mask = resize_lib[i](mask_ori).reshape(-1) > 0.3    # [B*N]
        atten = atten.reshape(-1, num_item)                 # [B*N, M]
        # pdb.set_trace()
        if mask.sum() != 0:
            entropy_loss += entropy(atten[mask])   # mask, anomaly  增大 num_item:
    return entropy_loss


def get_focal_loss(pred_list, mask_ori, vit=False):
    # bce_loss = nn.BCELoss()
    bce_loss = BCEFocalLoss()
    total_loss = 0.0
    num_item = pred_list[0].shape[-1]
    P_sum, R_sum = 0, 0
    for i, pred in enumerate(pred_list):
        mask = (resize_lib[i](mask_ori).reshape(-1) > 0.3).float()    # [B*N]
        if vit:
            mask = (resize_lib[2](mask_ori).reshape(-1) > 0.3).float()    # [B*N]
        pred = pred.reshape(-1)
        loss = bce_loss(pred, mask)      
        pred = (pred>0.5).float()
        PT = (pred.bool() & mask.bool()).sum()
        P = PT / (pred.sum()+1e-5)
        R = PT / (mask.sum()+1e-5)
        P_sum += P
        R_sum += R
        total_loss += loss
    return total_loss, P_sum/3, R_sum/3


def get_amplify_loss(inputs, inputs_AT, mask_ori, fore_mask_ori=None, vit=False):
    cos_loss = torch.nn.CosineSimilarity()
    normal_loss = 0
    anomaly_loss = 0
    # 64, 32, 16
    for i in range(len(inputs)):
        # pdb.set_trace()
        B, C, h, w = inputs[i].shape
        mask = resize_lib[i](mask_ori).reshape(-1) > mask_thresh        # [B*h*w]
        if vit:
            mask = resize_lib[2](mask_ori).reshape(-1) > mask_thresh

        if fore_mask_ori is not None:
            fore_mask = resize_lib[i](fore_mask_ori).reshape(-1) > mask_thresh 
            if vit:
                fore_mask = resize_lib[2](fore_mask_ori).reshape(-1) > mask_thresh

        input = inputs[i].permute(0,2,3,1).reshape(-1, C)        # [B,C,h,w]->[B,h,w,C]->[B*h*w,C]
        input_AT = inputs_AT[i].permute(0,2,3,1).reshape(-1, C)  # [B,C,h,w]->[B,h,w,C]->[B*h*w,C]
        # normal unchange
        normal_loss += torch.mean(1-cos_loss(input[~mask], input_AT[~mask]))

        if fore_mask_ori is not None:
            normal_mask = (~mask)&(fore_mask)
        else:
            normal_mask = ~mask
        n_idx = np.random.permutation(normal_mask.sum().item())[:5000]
        a_idx = np.random.permutation(mask.sum().item())[:1000]
        input_normal, input_AT_normal = input[normal_mask][n_idx], input_AT[normal_mask][n_idx]
        if mask.sum() > 0:
            input_anomaly, input_AT_anomaly = input[mask][a_idx], input_AT[mask][a_idx]
            s_anomaly = pair_cosine(input_normal.unsqueeze(0), input_anomaly.unsqueeze(0))[0]
            s_AT_anomaly = pair_cosine(input_normal.unsqueeze(0), input_AT_anomaly.unsqueeze(0))[0]
            weight = 1
            anomaly_loss += torch.mean(F.relu(s_AT_anomaly - (s_anomaly-0.3)) * weight)
        else:
            anomaly_loss += 0

    return normal_loss, anomaly_loss


def get_FnormLoss(inputs, mask_ori, vit=False):
    fnorm_loss = 0.0
    afnorm_loss = 0.0
    count = 0
    for i in range(len(inputs)):
        B, C, h, w = inputs[i].shape
        mask = resize_lib[i](mask_ori).reshape(-1) > mask_thresh         # [B*h*w]
        if vit:
            mask = resize_lib[2](mask_ori).reshape(-1) > mask_thresh      
        input = inputs[i].permute(0,2,3,1).reshape(-1, C)        # [B,C,h,w]->[B,h,w,C]->[B*h*w,C]
        fnorm_loss += (input[~mask] ** 2).mean()
        if mask.sum() != 0: 
            afnorm_loss += (input[mask] ** 2).mean()
            # afnorm_loss += F.relu(math.sqrt(C) - temp)
            count += 1
    fnorm_loss = fnorm_loss / 3
    if count > 0:
        afnorm_loss = afnorm_loss / count
    return fnorm_loss, afnorm_loss


class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """
    def __init__(self, gamma=2, alpha=0.8, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
 
    def forward(self, pt, target):
        # pt = torch.sigmoid(_input)
        # pdb.set_trace()
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss
