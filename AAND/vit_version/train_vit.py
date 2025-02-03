import torch
from torchvision.datasets import ImageFolder
import numpy as np
import random
import time
import os
import torch.nn as nn
from torch.utils.data import DataLoader
import os, pdb, sys
sys.path.append(os.getcwd())
from models.resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from models.resnet_rar import resnet18, resnet34, resnet50, wide_resnet50_2_rar
from models.de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
# from models.de_resnet_mem import de_wide_resnet50_2_mem
import torch.backends.cudnn as cudnn
import argparse
from vit_version.test_vit import evaluation_Stage1, evaluation_Stage2
from torch.nn import functional as F
from models.loss import *
from utils.vis import vis_anomaly_images
from models.recons_net import *
from utils.utils import setup_seed, get_dataset_name
import pdb
from utils.vis import *
from tqdm import tqdm
from utils.utils import PatchMaker
# import clip, clip_rar
# from clip_bn import BN_layer, AttnBottleneck
# from clip_decoder import de_VisionTransformer
import vit_version.clip as clip
import vit_version.clip_rar as clip_rar
from vit_version.clip_bn import BN_layer, AttnBottleneck
from vit_version.clip_decoder import de_VisionTransformer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_Stage1(args, _class_, epochs=100, eval_interval=10, lr=0.0002):
    print("training Advanced Teacher...")
    print(_class_)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    Stage1_ckp_path = 'vit_version/checkpoints_Stage1/' + 'wres50_'+_class_+'.pth'
    os.makedirs('vit_version/checkpoints_Stage1', exist_ok=True)
    train_data = AnomalyDataset(class_name=_class_, img_size=args.image_size, dataset_path=args.data_root, aux_path=args.aux_path)  # 注意：既有正常又有合成的异常
    test_data = TestDataset(class_name=_class_, img_size=args.image_size, dataset_path=args.data_root)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    # 初始的预训练模型
    # encoder, _ = clip.load("ViT-L/14@336px", download_root='./clip', device=torch.device("cpu"), image_size=336)  # Modify1
    encoder, _ = clip.load("ViT-B/16", download_root='vit_version/clip', device=torch.device("cpu"), image_size=args.image_size)
    encoder = encoder.to(device)
    encoder.eval()
    # load INet-pretrain, 包含了可学习的RAA模块
    encoder_AT, _ = clip_rar.load("ViT-B/16", download_root='vit_version/clip', device=torch.device("cpu"), image_size=args.image_size)
    encoder_AT = encoder_AT.to(device)

    ## 暂时
    # Stage1_ckp_path = './checkpoints_Stage1/' + 'wres50_'+_class_+'.pth'
    # encoder_AT.load_state_dict(torch.load(Stage1_ckp_path)['encoder_AT'])  # 加载Stage1
    # encoder_AT.eval()

    for name, para in encoder_AT.named_parameters():
        if 'feat_recons' in name:
            para.requires_grad = True
        else:
            para.requires_grad = False
    # print([name for name, para in encoder_AT.named_parameters() if para.requires_grad])
    # pdb.set_trace()
    optimizer_Stage1 = torch.optim.Adam((filter(lambda p: p.requires_grad, encoder_AT.parameters())), lr=lr, betas=(0.5,0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_Stage1, [epochs*0.8, epochs*0.9], gamma=0.2, last_epoch=-1)

    best_auroc_sp = -1
    best_auroc_px = -1
    best_P = -1
    loss_per_epoch = {'fnorm_loss': [], 'afnorm_loss': [], 'test_fnorm_loss': [], 'test_afnorm_loss': []}

    # auroc_px, auroc_sp, aupro_px, test_fnorm_loss, test_afnorm_loss = evaluation_Stage1(encoder, encoder_AT, test_dataloader, device)  
    # print('Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Pixel Aupro{:.3}'.format(auroc_px, auroc_sp, aupro_px))
    # print(f'test_fnorm_loss: {np.mean(test_fnorm_loss):.4f} test_afnorm_loss: {np.mean(test_afnorm_loss):.4f}')
    # pdb.set_trace()
    for epoch in tqdm(range(epochs)):
        encoder_AT.train() # 放到for循环里，验证时进行.eval, 所以每次要重新.train
        # 训练时禁用BN
        for name, module in encoder_AT.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            if isinstance(module, nn.BatchNorm1d):
                module.eval()

        normal_loss_list, anomaly_loss_list, fnorm_loss_list, afnorm_loss_list, focal_loss_list = [], [], [], [], []
        P_list, R_list = [], []
        for batch_data in train_dataloader:
            # load data
            img = batch_data["img"].to(device)
            anomaly_mask = batch_data["anomaly_mask"].to(device)  # 注意：是否要求有一半是正常图像？
            fore_mask = batch_data["fore_mask"].to(device)
            # vis_anomaly_images(img, anomaly_mask, _class_)
            # 预训练特征
            with torch.no_grad():
                _, inputs = encoder.encode_image(img, f_list=[4,8,12])
            # 可学习的特征
            _, inputs_AT, atten, delta = encoder_AT.encode_image(img, f_list=[4,8,12])
            # pdb.set_trace()
            loss_focal, P, R = get_focal_loss(atten, anomaly_mask, vit=True)
            loss_normal, loss_anomaly = get_amplify_loss(inputs, inputs_AT, anomaly_mask, fore_mask, vit=True)
            loss = loss_focal*1.0 + loss_anomaly *0.1 + loss_normal*0 
            # Res Loss: fnorm应减小，afnorm应增大
            fnorm_loss, afnorm_loss = get_FnormLoss(delta, anomaly_mask, vit=True)
            fnorm_loss_list.append(fnorm_loss.item())
            if afnorm_loss != 0.0: 
                afnorm_loss_list.append(afnorm_loss.item())

            optimizer_Stage1.zero_grad()
            loss.backward()
            optimizer_Stage1.step()

            normal_loss_list.append(loss_normal.item())
            loss_anomaly = loss_anomaly.item() if torch.is_tensor(loss_anomaly) else loss_anomaly
            anomaly_loss_list.append(loss_anomaly)
            focal_loss_list.append(loss_focal.item())
            P_list.append(P.item())
            R_list.append(R.item())

        # if np.isnan(np.mean(anomaly_loss_list)):
        #     pdb.set_trace()
        scheduler.step() # modify
        print(f'epoch [{epoch + 1}/{epochs}], focal_loss:{np.mean(focal_loss_list):.4f}\t, anomaly_loss:{np.mean(anomaly_loss_list):.4f}\t'
              f'fnorm_loss:{np.mean(fnorm_loss_list):.4f}, a_fnorm_loss:{np.mean(afnorm_loss_list):.4f}'
              f'P:{np.mean(P_list):.2f}, R:{np.mean(R_list):.2f}')
        loss_per_epoch['fnorm_loss'].append(np.mean(fnorm_loss_list))
        loss_per_epoch['afnorm_loss'].append(np.mean(afnorm_loss_list))

        if (epoch==0) or ((epoch + 1) % eval_interval == 0):
            ## evaluating perfermance to choose the best, common practice in indutrial anomaly detection, such as simplenet, RD++, BGAD ...
            auroc_px, auroc_sp, aupro_px, test_fnorm_loss, test_afnorm_loss, eval_P = evaluation_Stage1(encoder, encoder_AT, test_dataloader, device)  
            print('Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Pixel Aupro{:.3}'.format(auroc_px, auroc_sp, aupro_px))
            print(f'test_fnorm_loss: {np.mean(test_fnorm_loss):.4f} test_afnorm_loss: {np.mean(test_afnorm_loss):.4f}')
            loss_per_epoch['test_fnorm_loss'].append(np.mean(test_fnorm_loss))
            loss_per_epoch['test_afnorm_loss'].append(np.mean(test_afnorm_loss))
            
            # if auroc_sp > best_auroc_sp and epoch > 5:
            if eval_P > best_P and epoch > 5:
                best_P = eval_P
                best_auroc_sp = auroc_sp
                best_auroc_px = auroc_px
                torch.save({'encoder_AT': encoder_AT.state_dict()}, Stage1_ckp_path)
                            # 'reconsKV_fs': recons_rar.state_dict()}, Stage1_ckp_path)
            if auroc_sp > 0.999 and epoch > 5:
                break 
    return best_auroc_px, best_auroc_sp, aupro_px


def train_Stage2(args, _class_, epochs=100, eval_interval=10, lr=0.005):
    print("training Stubborn Student...")
    print(_class_)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    Stage1_ckp_path = 'vit_version/checkpoints_Stage1/' + 'wres50_'+_class_+'.pth'
    Stage2_ckp_path = 'vit_version/checkpoints_Stage2/' + 'wres50_'+ _class_+'.pth'
    os.makedirs('vit_version/checkpoints_Stage2', exist_ok=True)
    # 只有正常数据
    train_data = TrainDataset(class_name=_class_, img_size=args.image_size, dataset_path=args.data_root)
    test_data = TestDataset(class_name=_class_, img_size=args.image_size, dataset_path=args.data_root)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    # 初始的预训练模型
    # encoder, _ = clip.load("ViT-L/14@336px", download_root='./clip', device=torch.device("cpu"), image_size=336)  # Modify1
    encoder_pre, _ = clip.load("ViT-B/16", download_root='vit_version/clip', device=torch.device("cpu"), image_size=args.image_size)
    encoder_pre = encoder_pre.to(device)
    encoder_pre.eval()
    # 加载并固定FS的参数
    encoder_AT, _ = clip_rar.load("ViT-B/16", download_root='vit_version/clip', device=torch.device("cpu"), image_size=args.image_size)
    encoder_AT = encoder_AT.to(device)
    encoder_AT.load_state_dict(torch.load(Stage1_ckp_path)['encoder_AT'])  # 加载Stage1
    encoder_AT.eval()

    # 可学习的参数
    bn = BN_layer(AttnBottleneck, 3)
    bn = bn.to(device)
    decoder_SS = de_VisionTransformer(input_resolution=256, patch_size=16, width=768, layers=6, heads=12, output_dim=512)
    decoder_SS = decoder_SS.to(device)
   
    optimizer_Stage2 = torch.optim.Adam(list(decoder_SS.parameters())+list(bn.parameters()), lr=lr, betas=(0.5,0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_Stage2, [epochs*0.8, epochs*0.9], gamma=0.2, last_epoch=-1)

    best_auroc_sp = -1
    best_auroc_px = -1
    for epoch in tqdm(range(epochs)):
        bn.train()
        decoder_SS.train()
        
        kd_loss_list = []
        recons_loss_list = []
        orth_loss_list = []
        for batch_data in train_dataloader:
            # load data
            img = batch_data[0].to(device)
            # 修改后的特征作为教师，冻结住
            with torch.no_grad():
                _, inputs, atten, delta = encoder_AT.encode_image(img, f_list=[4,8,12])
                # inputs = encoder_pre(img)

            # outputs = decoder_SS(bn(inputs))
            _, outputs = decoder_SS(bn(inputs), f_list=[2,4,6])

            kd_loss = loss_fucntion(inputs, outputs)  # 这句话是不变的
            loss = kd_loss
            
            optimizer_Stage2.zero_grad()
            loss.backward()
            optimizer_Stage2.step()

            kd_loss_list.append(kd_loss.item())

        scheduler.step() # modify
        print('epoch [{}/{}], kd_loss:{:.4f}'.format(epoch + 1, epochs, np.mean(kd_loss_list)))

        if (epoch + 1) % eval_interval == 0:
            auroc_px, auroc_sp, aupro_px, _, _, _ = evaluation_Stage2(encoder_AT, bn, decoder_SS, test_dataloader, device)  
            print('Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Pixel Aupro{:.3}'.format(auroc_px, auroc_sp, aupro_px))
            
            if auroc_sp > best_auroc_sp and epoch > 5:
                best_auroc_sp = auroc_sp
                best_auroc_px = auroc_px
                torch.save({'bn': bn.state_dict(),
                            'decoder_ss': decoder_SS.state_dict()}, Stage2_ckp_path)
            if auroc_sp > 0.999 and epoch > 5:
                break    
    return best_auroc_px, best_auroc_sp, aupro_px


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/data4/tch/AD_data/mvtec')
    parser.add_argument('--aux_path', type=str, default='/data4/tch/AD_data/DRAEM_dtd/dtd/images')
    parser.add_argument('--L_rar', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
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

    setup_seed(111)

    auroc_Stage1_list = []
    auroc_Stage2_list = []
   
    for i in item_list:
        _, auroc_sp, _ = train_Stage1(args, i, epochs=100, eval_interval=5, lr=0.005)
        auroc_Stage1_list.append(auroc_sp)
        _, auroc_sp, _ = train_Stage2(args, i, epochs=120, eval_interval=10, lr=0.005)
        auroc_Stage2_list.append(auroc_sp)
    auroc_Stage1_mean = np.mean(auroc_Stage1_list)
    auroc_Stage2_mean = np.mean(auroc_Stage2_list)

    with open('results.txt', 'a') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+'\n')
        f.write('Stage1: ')
        for item in auroc_Stage1_list:
            f.write(f"{item:.4f}") 
            f.write(' ')
        f.write(f'avg: {auroc_Stage1_mean:.4f}\n')

        f.write('Stage2: ')
        for item in auroc_Stage2_list:
            f.write(f"{item:.4f}") 
            f.write(' ')
        f.write(f'avg: {auroc_Stage2_mean:.4f}\n')
