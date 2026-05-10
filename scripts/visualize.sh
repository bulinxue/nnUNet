#!/bin/bash

# ===================== 可视化脚本 =====================
# 同时绘制真值（蓝色）和预测（绿色）轮廓到原始CT上

python autodl-tmp/Project/nnUNet/src/nnunetv2/scripts/visualize.py --batch --ct autodl-tmp/Project/nnUNet/nnUNet_raw/Task44_READ/imagesTr --pred autodl-tmp/Project/nnUNet/nnUNet_results/Dataset044_READ/nnUNetTrainer_100epochs__nnUNetPlans__2d/fold_all/validation --output autodl-tmp/Project/nnUNet/nnUNet_results/Dataset044_READ/nnUNetTrainer_100epochs__nnUNetPlans__2d/fold_all/vis --gt autodl-tmp/Project/nnUNet/nnUNet_raw/Task44_READ/labelsTr
