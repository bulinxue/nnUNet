import os
import shutil
import random
import json
import nibabel as nib
import numpy as np

# ===================== 路径配置 =====================
IMG_SOURCE = "/root/autodl-tmp/data/READ/sample_raw_convert_1"
LABEL_SOURCE = "/root/autodl-tmp/data/READ/sample_raw/raw label nll"

TASK_NAME = "Task44_READ"
OUTPUT_BASE = "/root/autodl-tmp/Project/nnUNet/nnUNet_raw"
TASK_DIR = os.path.join(OUTPUT_BASE, TASK_NAME)

# ===================== 先删掉旧的任务文件夹 =====================
if os.path.exists(TASK_DIR):
    shutil.rmtree(TASK_DIR)
    print(f"已删除旧目录: {TASK_DIR}")

# ===================== 创建目录 =====================
imagesTr = os.path.join(TASK_DIR, "imagesTr")
imagesTs = os.path.join(TASK_DIR, "imagesTs")
labelsTr = os.path.join(TASK_DIR, "labelsTr")

os.makedirs(imagesTr, exist_ok=True)
os.makedirs(imagesTs, exist_ok=True)
os.makedirs(labelsTr, exist_ok=True)

# ===================== 获取所有病例 =====================
VAL_SPLIT = 0.2

all_cases = [f.replace(".nii.gz", "") for f in os.listdir(IMG_SOURCE) if f.endswith(".nii.gz")]
random.seed(42)
random.shuffle(all_cases)

n_val = int(len(all_cases) * VAL_SPLIT)
val_cases = all_cases[:n_val]
train_cases = all_cases[n_val:]

print(f"总病例: {len(all_cases)}, 训练集: {len(train_cases)}, 验证集: {len(val_cases)}")

# ===================== 复制训练集 =====================
for case in train_cases:
    # 图像（不加 _0000）
    src_img = os.path.join(IMG_SOURCE, f"{case}.nii.gz")
    dst_img = os.path.join(imagesTr, f"{case}.nii.gz")
    shutil.copy(src_img, dst_img)

    # 标签
    src_lab = os.path.join(LABEL_SOURCE, f"{case}.nii.gz")
    dst_lab = os.path.join(labelsTr, f"{case}.nii.gz")
    shutil.copy(src_lab, dst_lab)

# ===================== 复制验证集 =====================
for case in val_cases:
    src_img = os.path.join(IMG_SOURCE, f"{case}.nii.gz")
    dst_img = os.path.join(imagesTs, f"{case}.nii.gz")
    shutil.copy(src_img, dst_img)

# ===================== 修复标签方向 =====================
print("\n正在修复标签方向...")
fixed_count = 0
for f in sorted(os.listdir(labelsTr)):
    if not f.endswith('.nii.gz'):
        continue
    
    case = f.replace('.nii.gz', '')
    img_file = os.path.join(imagesTr, case + '.nii.gz')
    lab_file = os.path.join(labelsTr, f)
    
    if not os.path.exists(img_file):
        print(f"  跳过 {case}：找不到对应图像")
        continue
    
    # 读取图像和标签
    img_nii = nib.load(img_file)
    lab_data = nib.load(lab_file).get_fdata()
    
    # 把标签的方向/原点改成和图像一致
    new_lab = nib.Nifti1Image(lab_data.astype(np.uint8), img_nii.affine, img_nii.header)
    nib.save(new_lab, lab_file)
    fixed_count += 1
    print(f"  已修复: {case}")

print(f"标签修复完成: {fixed_count}/{len(train_cases)}")

# ===================== 生成 dataset.json =====================
dataset_json = {
    "modality": {"0": "CT"},
    "labels": {"0": "background", "1": "tumor"},
    "numTraining": len(train_cases),
    "file_ending": ".nii.gz",
    
    "training": [
        {"image": f"./imagesTr/{c}.nii.gz", "label": f"./labelsTr/{c}.nii.gz"}
        for c in train_cases
    ],
    "test": [f"./imagesTs/{c}.nii.gz" for c in val_cases]
}

with open(os.path.join(TASK_DIR, "dataset.json"), "w") as f:
    json.dump(dataset_json, f, indent=4)

print("\n✅ nnUNet 任务目录构建完成！")
print(f"路径: {TASK_DIR}")