
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
There are two underlying assumptions in KD-based anomaly detection framework. **Assumption I**: The teacher model can represent two separable distributions for the normal and abnormal patterns; **Assumption II**: the student model can only reconstruct the normal distribution. In this paper, we propose a simple yet effective two-stage anomaly detection framework, termed AAND, which comprises an Anomaly Amplification Stage **Stage I** to address Assumption I and a Normality Distillation Stage **Stage II** to address Assumption II. 



## News

🔥 2024.06: Our another KD-based Project [VAND-GNL](https://github.com/Hui-design/VAND-GNL) won the 2nd Place of CVPR 2024 [VAND2.0 Challenge](https://www.hackster.io/contests/openvino2024#challengeNav)

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


## 💾 Dataset

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

## Preprocessing
- Extract foreground mask for **training** images.

```bash
python scripts/fore_extractor.py --data_path <your_path>/<dataset_name>/ --aux_path <your_path>/dtd/images/  # the <dataset_name> is mvtec, VisA, or mvtec3d
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
You can train models on mvtec, VisA, or mvtec3d by the following commands:
```bash
python train.py --data_root <your_path>/<dataset_name>/  # the <dataset_name> is mvtec, VisA, or mvtec3d
```


## ⛳ Testing
You can test the trained models on mvtec, VisA, or mvtec3d by the following commands:
```bash
python test.py --data_root <your_path>/<dataset_name>/  # the <dataset_name> is mvtec, VisA, or mvtec3d
```

## Citation

```bibtex
@article{tang2024advancing,
  title={Advancing Pre-trained Teacher: Towards Robust Feature Discrepancy for Anomaly Detection},
  author={Tang, Canhui and Zhou, Sanping and Li, Yizhe and Dong, Yonghao and Wang, Le},
  journal={arXiv preprint arXiv:2405.02068},
  year={2024}
}
```


## Acknowledgements
- [RD4AD](https://github.com/hq-deng/RD4AD)
- [DRAEM](https://github.com/VitjanZ/DRAEM)



