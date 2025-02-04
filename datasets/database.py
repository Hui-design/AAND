from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torch.utils.data import Dataset
import numpy as np
import pdb
from torch.utils.data import Dataset
from torchvision import transforms as T
import imgaug.augmenters as iaa
# import albumentations as A
from utils.perlin import rand_perlin_2d_np
from utils.utils import get_dataset_name
import random
from utils.vis import vis_anomaly_images
import cv2

def resize_organized_pc(organized_pc, img_size=256, tensor_out=True):
    torch_organized_pc = torch.tensor(organized_pc).permute(2, 0, 1).unsqueeze(dim=0).contiguous()
    torch_resized_organized_pc = torch.nn.functional.interpolate(torch_organized_pc, size=img_size,
                                                                 mode='nearest')
    if tensor_out:
        return torch_resized_organized_pc.squeeze(dim=0).contiguous()
    else:
        return torch_resized_organized_pc.squeeze().permute(1, 2, 0).contiguous().numpy()

class BaseAnomalyDetectionDataset(Dataset):
    def __init__(self, split, class_name, img_size, dataset_path='datasets/eyecandies_preprocessed'):
        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]
        self.cls = class_name
        self.size = img_size
        self.dataset_name = get_dataset_name(dataset_path)
        if self.dataset_name == 'mvtec':
            self.img_path = os.path.join(dataset_path, self.cls, split)
            self.gt_path = os.path.join(dataset_path, self.cls, 'ground_truth')
        elif self.dataset_name == 'mvtec3d':
            self.img_path = os.path.join(dataset_path, self.cls, split)
        self.rgb_transform = transforms.Compose(
            [transforms.Resize((self.size, self.size), interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.ToTensor(),
             transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)])


class SynthesisDataset(BaseAnomalyDetectionDataset):
    def __init__(self, class_name, dataset_path, img_size, aux_path):
        super().__init__(split="train", class_name=class_name, img_size=img_size, dataset_path=dataset_path)
        self.anomaly_source_paths = sorted(glob.glob(aux_path + "/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]
        # self.noise_transform = transforms.Compose(
        #     [transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)])

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

        self.resize64 = torch.nn.AdaptiveAvgPool2d((64, 64))
        self.resize32 = torch.nn.AdaptiveAvgPool2d((32, 32))
        self.resize16 = torch.nn.AdaptiveAvgPool2d((16, 16))
    
    # 随机选择3种数据增强方式
    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, anomaly_source_path, fore_mask):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0

        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.size, self.size))

        anomaly_img_augmented = aug(image=anomaly_source_img)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        
        if fore_mask is not None:
            count = 0
            while True:
                count += 1
                perlin_noise = rand_perlin_2d_np((self.size, self.size), (perlin_scalex, perlin_scaley))
                perlin_noise = self.rot(image=perlin_noise)
                threshold = 0.5
                # modify
                perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
                perlin_thr = np.expand_dims(perlin_thr, axis=2)
                perlin_thr = perlin_thr * fore_mask
                # pdb.set_trace()
                if perlin_thr.sum() > 4 or count > 10:
                    break
        else:
            perlin_noise = rand_perlin_2d_np((self.size, self.size), (perlin_scalex, perlin_scaley))
            perlin_noise = self.rot(image=perlin_noise)
            threshold = 0.5
            # modify
            perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
            perlin_thr = np.expand_dims(perlin_thr, axis=2)
        #
        # modify, '/255' 改成 imagenet 的归一化
        # 测试一下img_thr的最大值和image的最小值
        # img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0
        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0
        image_mean = np.array(self.IMAGENET_MEAN).reshape(1,1,3)
        image_std = np.array(self.IMAGENET_STD).reshape(1,1,3)
        img_thr = (img_thr - image_mean) / image_std
        #

        beta = torch.rand(1).numpy()[0] * 0.8
        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)
        
        augmented_image = augmented_image.astype(np.float32)
        msk = (perlin_thr).astype(np.float32)
        augmented_image = msk * augmented_image + (1-msk)*image  # This line is unnecessary and can be deleted
        has_anomaly = 1.0

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.8:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:  # 0.8概率产生异常
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1-msk)*image  # This line is unnecessary and can be deleted
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly=0.0
            return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)


    def transform_image(self, image, fore_mask, anomaly_source_path):
        image = image.permute(1,2,0).numpy()
        if fore_mask is not None:
            fore_mask = fore_mask.permute(1,2,0).numpy()

        # normalize the image to 0.0~1.0
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path, fore_mask)
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return augmented_image, anomaly_mask, has_anomaly



class CutPasteDataSet(BaseAnomalyDetectionDataset):
    def __init__(self, class_name, img_size, dataset_path='datasets/eyecandies_preprocessed'):
        super().__init__(split="train", class_name=class_name, img_size=img_size, dataset_path=dataset_path)
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.cutpaste_transform = CutPaste_fg(type='3way')
        self.gt_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])

    def load_dataset(self):
        img_tot_paths = []
        tot_labels = []
        rgb_paths = glob.glob(os.path.join(self.img_path, 'good', 'rgb') + "/*.png")
        tiff_paths = glob.glob(os.path.join(self.img_path, 'good', 'xyz') + "/*.tiff")
        rgb_paths.sort()
        tiff_paths.sort()
        sample_paths = list(zip(rgb_paths, tiff_paths))
        img_tot_paths.extend(sample_paths)
        tot_labels.extend([0] * len(sample_paths))
        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]
        img = Image.open(rgb_path).convert('RGB')
        # img.save("imgs/ori_img.jpg")
        # pdb.set_trace()
        organized_pc = read_tiff_organized_pc(tiff_path)
        depth_map = organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis]
        resized_depth_map = resize_organized_pc(depth_map, img_size=[img.size[1],img.size[0]]).squeeze(0).numpy()
        fore_mask = resized_depth_map > 0

        cutpaste, anomaly_mask = self.cutpaste_transform(img, fore_mask)
        # cutpaste = self.copy_paste(img, fore_mask)
        # cutpaste.save("imgs/aug_img1.jpg")
        # cutpaste_scar.save("imgs/aug_img2.jpg")
        cutpaste = Image.fromarray(cutpaste)
        anomaly_mask = Image.fromarray(anomaly_mask)
        aug_img = self.rgb_transform(cutpaste)
        anomaly_mask = self.gt_transform(anomaly_mask)

        return {"img": aug_img, "anomaly_mask": anomaly_mask, "fore_mask": fore_mask}

