import torch
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader
from models.resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from models.de_resnet import de_resnet18, de_resnet50, de_wide_resnet50_2
from models.resnet_rar import resnet18, resnet34, resnet50, wide_resnet50_2_rar
# from models.de_resnet_mem import de_wide_resnet50_2_mem
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter
from sklearn import manifold
from matplotlib.ticker import NullFormatter
from scipy.spatial.distance import pdist
import matplotlib
import pickle
import argparse
# from MvTec3D import TestDataset
# from MvTec import TestDataset
import pdb
from utils.vis import *
from utils.utils import PatchMaker, get_dataset_name
from models.loss import *
import time
from models.recons_net import *
from tqdm import tqdm


def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul', vis=False):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    mse_loss = torch.nn.MSELoss(reduction='none')
    for i in range(len(ft_list)):
        # pdb.set_trace()
        fs = fs_list[i]
        ft = ft_list[i]
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        # pdb.set_trace()
        a_map = 1 - F.cosine_similarity(fs, ft)              # cos_loss  [1,C,H,W]->[1,H,W]
        # a_map2 = torch.sqrt(mse_loss(fs, ft).sum(dim=1))      # mse_loss  [1,C,H,W]->[1,H,W]
        # if i == 1 and vis:
        #     vis_hotmap(fs, ft, a_map)
        
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map

    return anomaly_map, None


def evaluation_Stage1(encoder, encoder_AT, dataloader, device, _class_=None):
    encoder.eval()  
    encoder_AT.eval()
    # reconsKV.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    fnorm_loss_list, afnorm_loss_list = [], []
    P_list, R_list = [], []
    with torch.no_grad():
        for img, gt, label, _ in dataloader:
            img = img.to(device)
            gt = gt.to(device)
            inputs = encoder(img)
            # FS
            _, inputs_AT, delta, atten = encoder_AT(img, flag=True)
            # pdb.set_trace()
            loss_atten, P, R = get_focal_loss(atten, gt)
            fnorm_loss, afnorm_loss = get_FnormLoss(delta, gt)
            fnorm_loss_list.append(fnorm_loss.item())
            if afnorm_loss != 0.0: 
                afnorm_loss_list.append(afnorm_loss.item())
            P_list.append(P.item())
            R_list.append(R.item())
            
            anomaly_map, _ = cal_anomaly_map(inputs, inputs_AT, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if label.item()!=0:
                aupro_list.append(0.0)
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))

        # print(afnorm_loss_list, 'len:', len(afnorm_loss_list))

        pr_list_sp = np.array(pr_list_sp)
        gt_list_sp = np.array(gt_list_sp)
        normal = pr_list_sp[gt_list_sp==0]
        anomaly = pr_list_sp[gt_list_sp==1] 
        print(f'normal: {normal.min():.4f} {normal.max():.4f}  anomaly: {anomaly.min():.4f} {anomaly.max():.4f}\t'
              f'P:{np.mean(P_list):.2f}, R:{np.mean(R_list):.2f}')

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
    return auroc_px, auroc_sp, round(np.mean(aupro_list),3), fnorm_loss_list, afnorm_loss_list, np.mean(P_list)


def evaluation_Stage2(encoder_AT, bn, decoder_SS, dataloader, device, _class_=None):
    encoder_AT.eval()  # 注意：现在的encoder是可学习的，所以要加上
    bn.eval()
    decoder_SS.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    time_list = []
    normal_feats, anomaly_feats = [], []
    with torch.no_grad():
        # for img, gt, label, rgb_path, fore_mask in dataloader:
        for img, gt, label, rgb_path in dataloader:
            # break
            img = img.to(device)
            tic = time.time()
            torch.cuda.synchronize()
            # FS
            _, inputs, _, _ = encoder_AT(img, flag=True)
            # inputs = encoder_AT(img)
            # SS
            outputs = decoder_SS(bn(inputs))
            torch.cuda.synchronize()
            time_list.append(time.time() - tic)

            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a', vis=label.bool())
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            # if label.item() == 0:
            #     pdb.set_trace()
                # apply_ad_scoremap(rgb_path[0], anomaly_map, _class_)
            if label.item()!=0:
                # if rgb_path == '/data4/tch/AD_data/mvtec/zipper/test/fabric_interior/007.png':
                #     pdb.set_trace()
                # pdb.set_trace()
                # aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                #                               anomaly_map[np.newaxis,:,:]))
                aupro_list.append(0.0)
            
            anomaly_map = anomaly_map 
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))

        pr_list_px = np.array(pr_list_px)
        gt_list_px = np.array(gt_list_px)
        pr_list_sp = np.array(pr_list_sp)
        gt_list_sp = np.array(gt_list_sp)
        # np.save('results/saved_data/zipper/pr_list_sp_zipper', pr_list_sp)
        # np.save('results/saved_data/zipper/gt_list_sp_zipper', gt_list_sp)

        normal_sp = pr_list_sp[gt_list_sp==0]
        anomaly_sp = pr_list_sp[gt_list_sp==1]

        print(f'normal: {normal_sp.min():.4f} {normal_sp.max():.4f}  anomaly: {anomaly_sp.min():.4f} {anomaly_sp.max():.4f}')

        precision, recall, thresholds = precision_recall_curve(gt_list_sp.astype(int), pr_list_sp)
        F1_scores = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) != 0,
        )
        optimal_threshold = thresholds[np.argmax(F1_scores)]
        fpr_optim = np.mean(normal_sp > optimal_threshold)
        fnr_optim = np.mean(anomaly_sp < optimal_threshold)
        print('thresh:', optimal_threshold)
        print('fpr:', fpr_optim, 'fnr', fnr_optim)

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
    return auroc_px, auroc_sp, round(np.mean(aupro_list),3), time_list, pr_list_sp, gt_list_sp


def test(_class_, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    print(_class_)
    
    Stage1_ckp_path = './checkpoints_Stage1/' + 'wres50_'+_class_+'.pth'
    Stage2_ckp_path = './checkpoints_Stage2/' + 'wres50_'+ _class_+'.pth'
    image_size = 256
    test_data = TestDataset(class_name=_class_, img_size=args.image_size, dataset_path=args.data_root)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    
    # Advanced Teacher
    encoder_AT, bn = wide_resnet50_2_rar(pretrained=False)
    encoder_AT = encoder_AT.to(device)
    # encoder_AT.load_state_dict(torch.load(Stage1_ckp_path)['encoder_AT'])  # 加载Stage1
    # encoder_AT.eval()

    # Vanilla Teacher
    encoder_pre, _ = wide_resnet50_2(pretrained=True)
    encoder_pre = encoder_pre.to(device)
    encoder_pre.eval()

    bn = bn.to(device)
    decoder_SS = de_wide_resnet50_2(pretrained=False)
    decoder_SS = decoder_SS.to(device)

    SS_ckp = torch.load(Stage2_ckp_path)
    for k, v in list(SS_ckp['bn'].items()):
        if 'memory' in k:
            SS_ckp['bn'].pop(k)
    decoder_SS.load_state_dict(SS_ckp['decoder_ss'])
    bn.load_state_dict(SS_ckp['bn'])

    # for sub in range(ord('a'), ord('c')+1):
    #     func = getattr(encoder_AT, f'feat_recons_{chr(sub)}')
    #     mem_cat = func.K_list[0].squeeze().detach().cpu().numpy()
    #     tsne_vis_cat(mem_cat, f'{_class_}_tsne_{sub}')
    # pdb.set_trace()
    # total_params = compute_params([encoder_AT, bn, decoder_SS])
    # print(f'total params: {total_params/1e6} M') #{sum([x.nelement() for x in self.model.parameters()])/1000000.} M

    # baseline, MemKD
    # auroc_px, auroc_sp, aupro_px, time_list = evaluation_Stage2(encoder_pre, bn, decoder_SS, model_reconsV, test_dataloader, device)
    # Ours
    auroc_px, auroc_sp, aupro_px, time_list, preds, labels = evaluation_Stage2(encoder_AT, bn, decoder_SS, test_dataloader, device, _class_)
    # pdb.set_trace()
    print(_class_,':',auroc_px,',',auroc_sp,',',aupro_px)
    return auroc_px, auroc_sp, aupro_px, time_list, preds, labels


def compute_params(nets):
    total_params = 0
    for net in nets:
        print(sum(p.numel() for p in net.parameters()))
        total_params += sum(p.numel() for p in net.parameters())
    return total_params

def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool_)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        # df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
        df = pd.concat([df, pd.DataFrame([{"pro": mean(pros), "fpr": fpr, "threshold": th}])], ignore_index=True)


    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


# if __name__ == '__main__':
#     from utils.utils import setup_seed
#     setup_seed(111)
#     item_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
#                  'transistor', 'metal_nut', 'screw','toothbrush', 'zipper', 'tile', 'wood']
#     # from MvTec3D import mvtec3d_classes
#     # item_list = mvtec3d_classes()
   
#     for i in item_list:
#         test(i)

if __name__ == '__main__':
    # from main import setup_seed
    # setup_seed(111)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/data4/tch/AD_data/mvtec3d')
    parser.add_argument('--image_size', type=int, default=256)
    args = parser.parse_args()
    
    dataset_name = get_dataset_name(args.data_root)
    if dataset_name == 'mvtec':
        from datasets.MvTec import TrainDataset, TestDataset, AnomalyDataset
        from datasets.MvTec import mvtec_classes
        item_list = mvtec_classes()
    elif dataset_name == 'VisA':
        from datasets.VisA import TrainDataset, TestDataset, AnomalyDataset
        from datasets.VisA import visa_classes
        item_list = visa_classes()
    elif dataset_name == 'mvtec3d':
        from datasets.MvTec3D import TrainDataset, TestDataset, AnomalyDataset
        from datasets.MvTec3D import mvtec3d_classes
        item_list = mvtec3d_classes()
    
    # all_time_list = []
    # for i in item_list:
    #     time_list = test(i)
    #     all_time_list += time_list
    # # print(len(all_time_list))
    # print('FPS: ', 1 / np.mean(all_time_list))
        
    p_auc_list, i_auc_list, p_pro_list = [], [], []
    label_list, pred_list = [], []
    for i in item_list:
        p_auc, i_auc, p_pro, _, preds, labels = test(i, args)
        p_auc_list.append(p_auc)
        i_auc_list.append(i_auc)
        p_pro_list.append(p_pro)
        label_list += list(labels)
        pred_list += list(preds)
        # all_time_list += time_list
    # print(len(all_time_list))
    # print('FPS: ', 1 / np.mean(all_time_list))
    i_auc_mean = np.mean(i_auc_list)
    p_auc_mean = np.mean(p_auc_list)
    p_pro_mean = np.mean(p_pro_list)

    normal = np.array(pred_list)[np.array(label_list)==0]
    anomaly = np.array(pred_list)[np.array(label_list)==1] 
    os.makedirs(f'results/{dataset_name}',exist_ok=True)
    np.save(f'results/{dataset_name}/normal_scores', normal)
    np.save(f'results/{dataset_name}/anomaly_scores', anomaly)
    # normal = np.load('normal_scores.npy')
    # anomaly = np.load('anomaly_scores.npy')
    # pdb.set_trace()
    # print(anomaly)

    # width = 7
    # fig = plt.figure(figsize=(width, width*0.75))
    # print(f'normal: {normal.min():.4f} {normal.max():.4f}  anomaly: {anomaly.min():.4f} {anomaly.max():.4f}')
    # plt.hist(normal,bins=20,label='normal sample',alpha=0.5)
    # plt.hist(anomaly,bins=20,label='anomalous sample',alpha=0.5)
    # plt.legend(fontsize=15)
    # plt.tick_params(labelsize=12.5)
    # plt.savefig(f'hist_SS.png')


    with open('results2.txt', 'a') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+'\n')
        f.write('i_auc: ')
        for item in i_auc_list:
            f.write(f"{item:.4f}") 
            f.write(' ')
        f.write(f'avg: {i_auc_mean:.4f}\n')

        f.write('p_auc: ')
        for item in p_auc_list:
            f.write(f"{item:.4f}") 
            f.write(' ')
        f.write(f'avg: {p_auc_mean:.4f}\n')

        f.write('p_pro: ')
        for item in p_pro_list:
            f.write(f"{item:.4f}") 
            f.write(' ')
        f.write(f'avg: {p_pro_mean:.4f}\n')