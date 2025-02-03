from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torch.utils.data import Dataset
import numpy as np
import pdb
import os, sys
sys.path.append(os.getcwd())
from torch.utils.data import Dataset
from torchvision import transforms as T
import imgaug.augmenters as iaa
# import albumentations as A
from utils.perlin import rand_perlin_2d_np
from utils.utils import get_dataset_name
import random
from utils.vis import vis_anomaly_images
import cv2
from datasets.database import BaseAnomalyDetectionDataset, SynthesisDataset


def mvtec_classes():
    return [ 
            'carpet',
            'grid',
            'leather',
            'tile',
            'wood',
            'bottle',
            'cable',
            'capsule',
            'hazelnut',
            'metal_nut',
            'pill',
            'screw',
            'toothbrush',
            'transistor',
            'zipper'
            ]
    



class TrainDataset(BaseAnomalyDetectionDataset):
    def __init__(self, class_name, img_size, dataset_path='datasets/eyecandies_preprocessed'):
        super().__init__(split="train", class_name=class_name, img_size=img_size, dataset_path=dataset_path)
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        tot_labels = []
        rgb_paths = glob.glob(os.path.join(self.img_path, 'good') + "/*.png")
        rgb_paths.sort()
        img_tot_paths.extend(rgb_paths)
        tot_labels.extend([0] * len(rgb_paths))
        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        rgb_path, label = self.img_paths[idx], self.labels[idx]
        img = Image.open(rgb_path).convert('RGB')
        img = self.rgb_transform(img)

        return img, label


class TestDataset(BaseAnomalyDetectionDataset):
    def __init__(self, class_name, img_size, dataset_path='datasets/eyecandies_preprocessed'):
        super().__init__(split="test", class_name=class_name, img_size=img_size, dataset_path=dataset_path)
        self.size = img_size
        self.gt_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
        self.img_paths, self.gt_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                rgb_paths.sort()
                img_tot_paths.extend(rgb_paths)
                gt_tot_paths.extend([0] * len(rgb_paths))
                tot_labels.extend([0] * len(rgb_paths))
            else:
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                rgb_paths.sort()
                gt_paths.sort()

                img_tot_paths.extend(rgb_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(rgb_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        rgb_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
        # pdb.set_trace()
        img_original = Image.open(rgb_path).convert('RGB')
        img = self.rgb_transform(img_original)
        
        if gt == 0:
            gt = torch.zeros(
                [1, self.size, self.size])
        else:
            gt = Image.open(gt).convert('L')
            gt = self.gt_transform(gt)
            gt = torch.where(gt > 0.5, 1., .0)

        return img, gt[:1], label, rgb_path
    

class AnomalyDataset(SynthesisDataset):
    def __init__(self, class_name, dataset_path, img_size, aux_path):
        super().__init__(class_name=class_name, img_size=img_size, dataset_path=dataset_path, aux_path=aux_path)
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        tot_labels = []
        rgb_paths = glob.glob(os.path.join(self.img_path, 'good') + "/*.png")
        # tiff_paths = glob.glob(os.path.join(self.img_path, 'good', 'xyz') + "/*.tiff")
        rgb_paths.sort()
        # tiff_paths.sort()
        # sample_paths = list(zip(rgb_paths, tiff_paths))
        # img_tot_paths.extend(sample_paths)
        # tot_labels.extend([0] * len(sample_paths))
        img_tot_paths.extend(rgb_paths)
        tot_labels.extend([0] * len(rgb_paths))
        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        rgb_path = img_path
        # tiff_path = img_path[1]
        class_name = img_path.split("/")[2]

        img = Image.open(rgb_path).convert('RGB')
        img = self.rgb_transform(img)

        img_file = rgb_path.split('/')[-1]
        fg_path = os.path.join(f'fg_mask/{self.dataset_name}/{self.cls}/train', img_file)
        fg_mask = Image.open(fg_path)
        fg_mask = np.asarray(fg_mask)[:, :, np.newaxis]  # [H, W, 1]
        resized_depth_map = resize_organized_pc(fg_mask, img_size=self.size)
        fore_mask = resized_depth_map > 0

        # # pdb.set_trace()
        # vis_anomaly_images(img.unsqueeze(0), fore_mask.unsqueeze(0).float(), class_name)
    
        # modify
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        augmented_image, anomaly_mask, has_anomaly = self.transform_image(img, fore_mask,
                                                                            self.anomaly_source_paths[anomaly_source_idx])
        # # 下采样mask 
        # anomaly_mask = torch.from_numpy(anomaly_mask).unsqueeze(0)
        
        # if has_anomaly == 1.0:
        #     # pdb.set_trace()
        #     vis_anomaly_images(torch.from_numpy(augmented_image)[None], torch.from_numpy(anomaly_mask)[None], class_name)
        
        return {"img": augmented_image, "label": has_anomaly, "anomaly_mask": anomaly_mask, "fore_mask": fore_mask.float()}


if __name__ == '__main__':
    train_dat = AnomalyDataset(class_name='transistor', img_size=256, dataset_path='/data4/tch/AD_data/mvtec', aux_path='/data4/tch/AD_data/DRAEM_dtd/dtd/images')
   
    train_dat.__getitem__(0)