# Eidos: Efficient, Imperceptible Adversarial 3D Point Clouds

This repository provides the official PyTorch implementation of the following conference paper:

Eidos: Efficient, Imperceptible Adversarial 3D Point Clouds

## Quick start

a. To setup a conda environment:

```shell
conda env create -f environment.yml
```

b. We use the same model from paper *Shape-invariant 3D Adversarial Point Clouds (CVPR 2022)*, download checkpoints from [https://drive.google.com/file/d/1L25i0l6L_b1Vw504WQR8-Z0oh2FJA0G9/view]() and set it properly in `./checkpoint/`:

c. start eidos attack:

```shell
conda activate eidos_attack
python main.py --transfer_attack_method ifgm_bp_ours --task_name adv_result --exponential_step
```

## Eidos Attack

```shell
python main.py --transfer_attack_method ifgm_bp_ours --task_name adv_result --exponential_step --surrogate_model pointnet_cls
```

## BP2 Attack

```shell
python main.py --transfer_attack_method ifgm_ours --task_name adv_result --bp_version bp2 --l2_weight 1.0 --hd_weight 0.0 --curve_weight 0.0 --cd_weight 0.0 --exponential_step --surrogate_model pointnet_cls
```

## GSDA-BP Attack

```shell
python main.py --transfer_attack_method gsda_bp --task_name adv_result --surrogate_model pointnet_cls
```

## BP Attack

```shell
python main.py --transfer_attack_method ifgm_bp --task_name adv_result --surrogate_model pointnet_cls
```

## Eidos query-based Attack

```shell
python main.py --transfer_attack_method ifgm_bp_ours_query --task_name adv_result --surrogate_model dgcnn --target_model paconv --step_size 0.32
```

## Point-Transformer

a. following the instruction at [https://github.com/lulutang0608/Point-BERT]() to install CPP extensions

b. download `PointTransformer_ModelNet1024points.pth` from [https://cloud.tsinghua.edu.cn/f/9be5d9dcbaeb48adb360/?dl=1]() and make sure it is in `checkpoint/ModelNet40/PointTransformer_ModelNet1024points.pth`
