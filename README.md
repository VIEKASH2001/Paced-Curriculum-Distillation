<a>
    <img src="https://readme-typing-svg.demolab.com?font=Georgia&size=50&duration=2000&pause=500&multiline=true&width=1600&height=80&lines=Paced Curriculum Distillation" alt="Typing SVG" />
</a>
Paced Curriculum Distillation with Prediction and Label Uncertainty for Image Segmentation
International Journal of Computer Assisted Radiology and Surgery

Purpose
In curriculum learning, the idea is to train on easier samples first and gradually increase the difficulty, while in self-paced learning, a pacing function defines the speed to adapt the training progress. While both methods heavily rely on the ability to score the difficulty of data samples, an optimal scoring function is still under exploration.

Methodology
Distillation is a knowledge transfer approach where a teacher network guides a student network by feeding a sequence of random samples. We argue that guiding student networks with an efficient curriculum strategy can improve model generalization and robustness. For this purpose, we design an uncertainty-based paced curriculum learning in self-distillation for medical image segmentation. We fuse the prediction uncertainty and annotation boundary uncertainty to develop a novel paced-curriculum distillation (P-CD). We utilize the teacher model to obtain prediction uncertainty and spatially varying label smoothing with Gaussian kernel to generate segmentation boundary uncertainty from the annotation. We also investigate the robustness of our method by applying various types and severity of image perturbation and corruption.

Results
The proposed technique is validated on two medical datasets of breast ultrasound image segmentation and robot-assisted surgical scene segmentation and achieved significantly better performance in terms of segmentation and robustness.

Conclusion
P-CD improves the performance and obtains better generalization and robustness over the dataset shift. While curriculum learning requires extensive tuning of hyper-parameters for pacing function, the level of performance improvement suppresses this limitation.


# Table of Contents

- [Table of Contents](#table-of-contents)
- [Commands for Training and Testing](#commands-for-training-and-testing)
  - [Setup](#setup)
  - [BUS Dataset](#bus-dataset)
    - [UNet](#unet)
    - [KD](#kd)
    - [KD + CPL](#kd--cpl)
    - [KD + TPL](#kd--tpl)
    - [CPL + TPL](#cpl--tpl)
- [Dataset Directory Structure [`Dataset Link`]](#dataset-directory-structure-dataset-link)
  - [Number of Files](#number-of-files)

# Commands for Training and Testing

## Setup

```shell
cd /media/mobarak/data/CKD-Plus-Plus/cpl-tpl
conda activate ckd
```

## BUS Dataset

### UNet

- Training

```shell
python train_model.py \
    --model-path outputs/bus/unet/unet.pth \
    --model-name UNet
```

- Testing

```shell
python test_model.py \
    --model-path outputs/bus/unet/unet.pth
```

### KD

- Training

```shell
python train_model.py \
    --model-path outputs/bus/kd/kd.pth \
    --model-name KD
```

- Testing

```shell
python test_model.py \
    --model-path outputs/bus/kd/kd.pth
```

### KD + CPL

- Training

```shell
python train_model.py \
    --model-path outputs/bus/kd_cpl/kd_cpl.pth \
    --model-name CKD \
    --teacher-path outputs/bus/unet/unet-teacher.pth \
    --alpha 0.3
```

- Testing

```shell
python test_model.py \
    --model-path outputs/bus/kd_cpl/kd_cpl.pth
```

### KD + TPL

- Training

```shell
python train_model.py \
    --epochs 40 \
    --model-path outputs/bus/kd_cpl/kd_tpl.pth \
    --model-name CKD \
    --teacher-path outputs/bus/unet/unet-teacher.pth \
    --spl \
    --mu-update 0.183 \
    --initial-mu 0.39 \
    --ckd-loss-type both_weighted_spl_per_px_no_alpha_no_weights
```

- Testing

```shell
python test_model.py \
    --model-path outputs/bus/kd_tpl/kd_tpl.pth
```

### CPL + TPL

- Training

```shell
python train_model.py \
    --epochs 40 \
    --model-path outputs/bus/kd_cpl/kd_tpl.pth \
    --model-name CKD \
    --teacher-path outputs/bus/unet/unet-teacher.pth \
    --spl \
    --mu-update 0.183 \
    --initial-mu 0.39 \
    --ckd-loss-type both_weighted_spl_per_px_no_alpha
```

- Testing

```shell
python test_model.py \
    --model-path outputs/bus/cpl_tpl/cpl_tpl.pth
```

# Dataset Directory Structure [[`Dataset Link`](https://drive.google.com/file/d/1QZhJudi99giidfB6Uej6qQ4SyUc6Irrr/view?usp=sharing)]

```
❯ tree -dl
.
├── bus
│   ├── benign
│   ├── malignant
│   └── normal
└── needle
    ├── test
    ├── test_mask
    ├── train
    └── train_mask

9 directories
```

## Number of Files

```
❯ find -L . -type f | cut -d/ -f2 | sort | uniq -c
   1578 bus
      1 bus_train.json
      1 bus_val.json
   1918 needle
      1 needle_train.json
      1 needle_val.json
```
