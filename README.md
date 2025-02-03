
<h2 align="center">
<a href="https://arxiv.org/abs/2405.02068" target="_blank">Advancing Pre-trained Teacher: Towards Robust Feature Discrepancy for Anomaly Detection</a>
</h2>

## Overview


<!-- <div align="center">
    <img src="assets/framework.png" alt="framework" width="700" >
</div> -->

<!-- The experimental results of PARE-Net
<div align="center">
    <img src="assets/image.png" alt="framework" width="800" >
</div> -->




## News
<!-- 
2024.07.20: Code and pretrained models on 3DMatch and KITTI are released.

2024.07.14: Paper is available at [arXiv](https://arxiv.org/abs/2407.10142).

2024.07.04: Our paper is accepted by ECCV 2024!. -->

## 🔧  Installation

Please use the following command for installation.

```bash
# It is recommended to create a new environment
conda create -n AAND python==3.8
conda activate AAND

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install packages and other dependencies
pip install -r requirements.txt
```


## 💾 Dataset and Preprocessing

- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [VisA](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar), use [visa.py](https://github.com/ByChelsea/VAND-APRIL-GAN/blob/master/data/visa.py) to generate meta.json
- [MVTec-3D](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad), we only use the rgb images, called **MVTec3D-RGB** in our paper.
- [DRAEM_dtd](https://www.robots.ox.ac.uk/~vgg/data/dtd/), used as the auxillary texture datasets for synthesizing anomalies like [DRAEM](https://github.com/VitjanZ/DRAEM). 
```
<your_path>
├── mvtec
    ├── bottle
        ├── train
        ├── test
        ├── ground_truth
    ├── ...

├── VisA
    ├── meta.json
    ├── candle
        ├── Data
        ├── image_anno.csv
    ├── ...

├── mvtec3d
    ├── bagel
        ├── train
            ├── good
                ├── rgb (we only use rgb)
                ├── xyz
        ├── test
        ├── ...

├── DRAEM_dtd
    ├── dtd
        ├── images
            ├── ...
```

- Extract foreground mask for **training** images.

```bash
python scripts/fore_extractor.py --data_path <your_path>/<dataset_name>/  # the <dataset_name> is mvtec, VisA, or mvtec3d
```

<!-- ## ⚽ Demo
After installation, you can run the demo script in `experiments/3DMatch` by:
```bash
cd experiments/3DMatch
python demo.py
```

To test your own data, you can downsample the point clouds with 2.5cm and specify the data path:
```bash
python demo.py --src_file=your_data_path/src.npy --ref_file=your_data_path/ref.npy --gt_file=your_data_path/gt.npy --weights=../../pretrain/3dmatch.pth.tar
``` -->

## 🚅 Training
<!-- You can train models on mvtec, VisA, or mvtec3d by the following commands:

```bash
cd experiments/3DMatch (or KITTI)
CUDA_VISIBLE_DEVICES=0 python trainval.py
```
You can also use multiple GPUs by:
```bash
CUDA_VISIBLE_DEVICES=GPUS python -m torch.distributed.launch --nproc_per_node=NGPUS trainval.py
```
For example,
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 trainval.py
``` -->

## ⛳ Testing
<!-- To test a pre-trained models on 3DMatch, use the following commands:
```bash
# 3DMatch
python test.py --benchmark 3DMatch --snapshot ../../pretrain/3dmatch.pth.tar
python eval.py --benchmark 3DMatch
```
To test the model on 3DLoMatch, just change the argument `--benchmark 3DLoMatch`.

To test a pre-trained models on KITTI, use the similar commands:
```bash
# KITTI
python test.py --snapshot ../../pretrain/kitti.pth.tar
python eval.py
``` -->

## Citation

<!-- ```bibtex
@inproceedings{yao2024parenet,
    title={PARE-Net: Position-Aware Rotation-Equivariant Networks for Robust Point Cloud Registration},
    author={Runzhao Yao and Shaoyi Du and Wenting Cui and Canhui Tang and Chengwu Yang},
    journal={arXiv preprint arXiv:2407.10142},
    year={2024}
}
``` -->

## Acknowledgements
<!-- Our code is heavily brought from
- [GeoTransformer](https://github.com/qinzheng93/GeoTransformer)
- [VectorNeurons](https://github.com/FlyingGiraffe/vnn)
- [PAConv](https://github.com/CVMI-Lab/PAConv) -->



