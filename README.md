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
