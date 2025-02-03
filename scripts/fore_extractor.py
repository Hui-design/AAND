import cv2
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os, pdb, sys
sys.path.append(os.getcwd())
from tqdm import tqdm
import json, glob
from scipy.ndimage import gaussian_filter
import argparse


def extract_mvtec(data_root, save_root, obj_names):
    obj_list1 = ['carpet','grid','leather','tile','wood','transistor']
    obj_list2 = ['capsule', 'screw', 'zipper'] 
    obj_list3 = ['hazelnut', 'metal_nut', 'pill', 'toothbrush']
    obj_list4 = ['cabel', 'bottle']

    for obj in tqdm(obj_names): 
        rgb_paths = glob.glob(os.path.join(data_root, obj, 'train', 'good') + "/*.png")
        rgb_paths.sort()
        save_dir = f'{save_root}/{obj}/train'
        os.makedirs(save_dir, exist_ok=True)
        
        for rgb_path in rgb_paths:
            img = Image.open(rgb_path).convert('RGB')
            image_transforms = transforms.Compose([
                    transforms.Grayscale(1)
                ])
            img = image_transforms(img)
            img = np.asarray(img)
            img = gaussian_filter(img, sigma=8)
            # pdb.set_trace()
            if obj in obj_list1:
                ret, new_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
                new_img = Image.fromarray(new_img)
            elif obj in obj_list2:
                ret, new_img = cv2.threshold(img,  150, 255, cv2.THRESH_BINARY)
                new_img = Image.fromarray(255-new_img)
            elif obj in obj_list3:
                ret, new_img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
                new_img = Image.fromarray(new_img)
            elif obj in obj_list4:
                H, W = img.shape
                new_img = np.zeros((H, W), dtype=np.uint8)
                center_x, center_y = W // 2, H // 2
                radius = int(0.45 * H)
                for x in range(W):
                    for y in range(H):
                        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                        if dist <= radius:
                            new_img[y, x] = 255
                new_img = Image.fromarray(new_img)
            file_name = rgb_path.split('/')[-1].split('.')[0]
            new_img.save(f"{save_dir}/{file_name}.png")


def extract_visa(data_root, save_root, obj_names):
    obj_list1 = ['candle', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1']
    obj_list2 = ['pcb2','pcb3','pipe_fryum'] 
    obj_list3 = ['pcb4']
    obj_list4 = ['capsules']

    for obj in obj_names: 
        meta_info = json.load(open(f'{data_root}/meta.json', 'r'))['train']
        rgb_paths = meta_info[obj]
        rgb_paths = [x["img_path"] for x in rgb_paths]
        rgb_paths.sort()
        save_dir = f'{save_root}/{obj}/train'
        os.makedirs(save_dir, exist_ok=True)
        
        for rgb_path in tqdm(rgb_paths):
            rgb_path = f'{data_root}/{rgb_path}'
            img = Image.open(rgb_path).convert('RGB')
            image_transforms = transforms.Compose([
                    transforms.Grayscale(1)
                ])
            img = image_transforms(img)
            img = np.asarray(img)
            img = gaussian_filter(img, sigma=8)
            if obj in obj_list1:
                ret, new_img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
                new_img = Image.fromarray(new_img)
            elif obj in obj_list2:
                ret, new_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
                new_img = Image.fromarray(new_img)
            elif obj in obj_list3:
                ret, new_img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
                new_img = Image.fromarray(new_img)
            elif obj in obj_list4:
                ret, new_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
                new_img = Image.fromarray(255 - new_img)
            file_name = rgb_path.split('/')[-1].split('.')[0]
            new_img.save(f"{save_dir}/{file_name}.png")


def extract_mvtec3d(data_root, save_root, obj_names):
    obj_list1 = ["bagel","cable_gland", "carrot", "cookie", "dowel", "peach","potato", "rope",]
    obj_list2 = ["foam", "tire",]

    for obj in tqdm(obj_names): 
        rgb_paths = glob.glob(os.path.join(data_root, obj, 'train', 'good', 'rgb') + "/*.png")
        rgb_paths.sort()
        save_dir = f'{save_root}/{obj}/train'
        os.makedirs(save_dir, exist_ok=True)
        
        for rgb_path in rgb_paths:
            img = Image.open(rgb_path).convert('RGB')
            image_transforms = transforms.Compose([
                    transforms.Grayscale(1)
                ])
            img = image_transforms(img)
            img = np.asarray(img)
            img = gaussian_filter(img, sigma=8)
            # pdb.set_trace()
            if obj in obj_list1:
                ret, new_img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
                new_img = Image.fromarray(new_img)
            elif obj in obj_list2:
                H, W = img.shape
                new_img = np.zeros((H, W), dtype=np.uint8)
                center_x, center_y = W // 2, H // 2
                half_width = int(0.35 * W)
                half_height = int(0.35 * H)
                for x in range(W):
                    for y in range(H):
                        if (center_x - half_width) <= x <= (center_x + half_width) and (center_y - half_height) <= y <= (center_y + half_height):
                            new_img[y, x] = 255
                new_img = Image.fromarray(new_img)

            file_name = rgb_path.split('/')[-1].split('.')[0]
            new_img.save(f"{save_dir}/{file_name}.png")
            # pdb.set_trace()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/data4/tch/AD_data/mvtec')
    args = parser.parse_args()
    
    if 'mvtec' in args.data_path.lower() and 'mvtec3d' not in args.data_path.lower():
        from datasets.MvTec import mvtec_classes
        obj_names = mvtec_classes()
        save_root = 'fg_mask/mvtec'
        extract_mvtec(args.data_path, save_root, obj_names)

    elif 'visa' in args.data_path.lower():
        from datasets.VisA import visa_classes
        obj_names = visa_classes()
        save_root = 'fg_mask/VisA'
        extract_visa(args.data_path, save_root, obj_names)

    elif 'mvtec3d' in args.data_path.lower():
        from datasets.MvTec3D import mvtec3d_classes
        obj_names = mvtec3d_classes()
        save_root = 'fg_mask/mvtec3d'
        extract_mvtec3d(args.data_path, save_root, obj_names)

    else:
        print('no such dataset')
            