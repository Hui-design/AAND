import numpy as np
import random
import torch
from torchvision import transforms
from PIL import ImageFilter
from sklearn.manifold import TSNE 
import seaborn as sns 
import matplotlib.pyplot as plt
import cv2, pdb, os
from PIL import Image
from torch.nn import functional as F
import pandas as pd

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

resize_lib = {0: torch.nn.AdaptiveAvgPool2d((64, 64)),
              1: torch.nn.AdaptiveAvgPool2d((32, 32)) ,
              2: torch.nn.AdaptiveAvgPool2d((16, 16))}  # 注意：可以尝试不同的下采样方式

def vis_anomaly_images(imgs, mask_ori, obj):
    image_mean = np.array(IMAGENET_MEAN).reshape(1,1,3)
    image_std = np.array(IMAGENET_STD).reshape(1,1,3)
    B, C, H, W = imgs.shape
    os.makedirs(f'imgs/{obj}', exist_ok=True)
    masks_list = [resize_lib[i](mask_ori) for i in range(3)]
    for i in range(B):
        img = imgs[i].permute(1,2,0).cpu().numpy() # [H,W,3]
        img = ((img * image_std + image_mean)*255).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(f'imgs/{obj}/img{i}.jpg')

        # pdb.set_trace()
        for level in range(3):
            mask = (masks_list[level][i][0]>0.3).float().cpu().numpy() # [H,W]
            mean = mask.mean()
            mask = (mask * 255).astype(np.uint8)
            mask = Image.fromarray(mask)
            mask.save(f'imgs/{obj}/mask{i}_{level}.jpg')

        mask_ = (mask_ori[i].permute(1,2,0).cpu().numpy().squeeze() * 255).astype(np.uint8)
        mask_ = Image.fromarray(mask_)
        mask_.save(f'imgs/{obj}/mask{i}.jpg')

def vis_gt(gt):
    gt = Image.fromarray(gt.astype(np.uint8) * 255, mode='L')
    gt.save('gt.png')


def vis_hotmap(fs, ss, a_map):
    B, C, H, W = fs.shape
    os.makedirs('imgs/a_map', exist_ok=True)
    # pdb.set_trace()
    fig = plt.figure()
    a_map = a_map.squeeze().cpu().detach()
    sns.heatmap(data=a_map) 
    plt.savefig(f'imgs/a_map/a_map.png')

    tsne = TSNE(n_components=1) 
    fs = fs.squeeze().permute(1,2,0).reshape(H*W, C).detach().cpu()
    ss = ss.squeeze().permute(1,2,0).reshape(H*W, C).detach().cpu()

    # L2-norm
    fs = F.normalize(fs, p=1)
    ss = F.normalize(ss, p=1)
    cat_tsne = tsne.fit_transform(np.vstack((fs, ss)))
    fs_tsne = cat_tsne[:H*W].reshape(H,W)
    ss_tsne = cat_tsne[H*W:].reshape(H,W)

    fig = plt.figure()
    sns.heatmap(data=fs_tsne) 
    plt.savefig(f'imgs/a_map/fs.png')

    fig = plt.figure()
    sns.heatmap(data=ss_tsne) 
    plt.savefig(f'imgs/a_map/ss.png')
    pdb.set_trace()
    plt.close('all')

def vis_hotmap_single(a_map):
    fig = plt.figure()
    sns.heatmap(data=a_map, xticklabels=[], yticklabels=[], cbar=False) 
    plt.savefig(f'atten.png')
    # pdb.set_trace()
    plt.close('all')

def tsne_vis_single(memory, name):
    tsne = TSNE(n_components=2) 
    data = tsne.fit_transform(memory)
    plt.figure() 
    plt.scatter(data[:,0], data[:,1])  
    # plt.legend((s1,s2),('memory','anomaly') ,loc = 'best')
    plt.savefig(f'{name}.png')
    plt.close('all')

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)

def apply_ad_scoremap(rgb_path, scoremap, cls, alpha=0.5):
    img_size = scoremap.shape[0]
    # pdb.set_trace()
    image = cv2.cvtColor(cv2.resize(cv2.imread(rgb_path), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
    mask = normalize(scoremap)
    # vis = apply_ad_scoremap(vis, mask)
    np_image = np.asarray(image, dtype=float)
    scoremap[scoremap>1] =1
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    scoremap = (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)
    
    vis = cv2.cvtColor(scoremap, cv2.COLOR_RGB2BGR)  # BGR
    save_vis = 'imgs'
    if not os.path.exists(save_vis):
        os.makedirs(save_vis)
    cv2.imwrite(f'{save_vis}/{cls}.png', vis)


def plot_hist(normal, anomaly):
    width = 7
    fig = plt.figure(figsize=(width, width*0.7))
    print(f'normal: {normal.min():.4f} {normal.max():.4f}  anomaly: {anomaly.min():.4f} {anomaly.max():.4f}')
    # plt.hist(normal,bins=50,label='normal sample',alpha=0.5)
    # plt.hist(anomaly,bins=50,label='anomalous sample',alpha=0.5)
    # normal = pd.Series(normal)  # 将数据由数组转换成series形式
    # normal.plot(kind = 'kde',label = 'normal')
    # anomaly = pd.Series(anomaly)  # 将数据由数组转换成series形式
    # anomaly.plot(kind = 'kde',label = 'anomaly')
    import seaborn as sn
    sn.kdeplot(normal, color="blue", shade="True", label='normal samples', bw_adjust=0.3)
    sn.kdeplot(anomaly, color="red", shade="True", label='anomalous samples', bw_adjust=0.3)
    plt.legend(fontsize=14)
    plt.tick_params(labelsize=13)
    plt.ylabel('Density',fontsize=12.5)
    plt.xlabel('Anomaly score',fontsize=12.5)
    plt.savefig(f'hist_SS.png')

def plot_line(data, name):
    plt.figure()
    plt.plot(data)  # 绘制 sin(x) 曲线
    print(f'save to visual/{name}.png')
    plt.savefig(f'visual/{name}.png', bbox_inches='tight', dpi=1000)