#!/usr/bin/env python3
"""
将模型分割结果轮廓与原始标注同时绘制到原始CT图像上
真值=红色，预测=绿色，一目了然对比
"""
import os
import argparse
import numpy as np
import nibabel as nib
import cv2
from pathlib import Path


def normalize_ct(ct_slice):
    """将CT切片归一化到0-255用于显示"""
    ct_min = ct_slice.min()
    ct_max = ct_slice.max()
    if ct_max > ct_min:
        normalized = (ct_slice - ct_min) / (ct_max - ct_min) * 255
    else:
        normalized = np.zeros_like(ct_slice)
    return normalized.astype(np.uint8)


def extract_contours(seg_slice, min_area=10):
    """从分割切片中提取轮廓"""
    binary = (seg_slice > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def visualize_with_gt(
    ct_path,
    pred_path,
    gt_path=None,
    output_path=None,
    slice_idx=None,
    axis=0,
    thickness=2
):
    """
    可视化分割结果 + 原始标注叠加到CT上
    真值=蓝色，预测=绿色
    """
    # 加载数据
    ct_img = nib.load(ct_path)
    ct_data = ct_img.get_fdata()
    pred_img = nib.load(pred_path)
    pred_data = pred_img.get_fdata()

    has_gt = gt_path is not None and os.path.exists(gt_path)
    if has_gt:
        gt_img = nib.load(gt_path)
        gt_data = gt_img.get_fdata()

    # 选择切片
    if slice_idx is None:
        slice_idx = ct_data.shape[axis] // 2

    # 提取切片
    if axis == 0:
        ct_slice = ct_data[slice_idx, :, :]
        pred_slice = pred_data[slice_idx, :, :]
        if has_gt:
            gt_slice = gt_data[slice_idx, :, :]
    elif axis == 1:
        ct_slice = ct_data[:, slice_idx, :]
        pred_slice = pred_data[:, slice_idx, :]
        if has_gt:
            gt_slice = gt_data[:, slice_idx, :]
    else:
        ct_slice = ct_data[:, :, slice_idx]
        pred_slice = pred_data[:, :, slice_idx]
        if has_gt:
            gt_slice = gt_data[:, :, slice_idx]

    # 旋转
    ct_slice = np.rot90(ct_slice, k=1)
    pred_slice = np.rot90(pred_slice, k=1)
    if has_gt:
        gt_slice = np.rot90(gt_slice, k=1)

    # 归一化 CT
    ct_normalized = normalize_ct(ct_slice)
    result = cv2.cvtColor(ct_normalized, cv2.COLOR_GRAY2BGR)

    # 画真值轮廓 = 蓝色
    if has_gt:
        gt_contours = extract_contours(gt_slice)
        cv2.drawContours(result, gt_contours, -1, (255, 0, 0), thickness)  # BGR: 蓝色
        cv2.putText(result, f"GT (blue): {len(gt_contours)} regions",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # 画预测轮廓 = 绿色
    pred_contours = extract_contours(pred_slice)
    cv2.drawContours(result, pred_contours, -1, (0, 255, 0), thickness)  # BGR: 绿色
    cv2.putText(result, f"Pred (green): {len(pred_contours)} regions",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 信息文本
    case_name = Path(ct_path).stem.replace('.nii', '').replace('_0000', '')
    cv2.putText(result, f"Case: {case_name} | Slice: {slice_idx}/{ct_data.shape[axis]-1}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 保存
    if output_path is None:
        output_dir = Path(ct_path).parent / 'visualizations'
        output_dir.mkdir(exist_ok=True)
        axis_names = ['axial', 'coronal', 'sagittal']
        output_path = output_dir / f"{case_name}_{axis_names[axis]}_slice{slice_idx}.png"

    cv2.imwrite(str(output_path), result)
    print(f"✅ 已保存: {output_path}")
    return result


def visualize_batch(
    ct_dir,
    pred_dir,
    gt_dir=None,
    output_dir=None,
    thickness=2
):
    """批量可视化"""
    ct_files = sorted(Path(ct_dir).glob('*.nii.gz'))
    if not ct_files:
        ct_files = sorted(Path(ct_dir).glob('*.nii'))

    if output_dir is None:
        output_dir = Path(ct_dir).parent / 'visualizations_all'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ct_path in ct_files:
        case_name = ct_path.name.replace('.nii.gz', '').replace('.nii', '').replace('_0000', '')

        # 找预测文件（多种后缀尝试）
        pred_path = None
        for suffix in ['.nii.gz', '_0000.nii.gz', '.nii', '_0000.nii']:
            p = Path(pred_dir) / f"{case_name}{suffix}"
            if p.exists():
                pred_path = p
                break
        if pred_path is None:
            print(f"⚠️ 跳过 {case_name}：找不到预测文件")
            continue

        # 找真值文件
        gt_path = None
        if gt_dir:
            for suffix in ['.nii.gz', '.nii']:
                g = Path(gt_dir) / f"{case_name}{suffix}"
                if g.exists():
                    gt_path = g
                    break

        try:
            visualize_with_gt(
                str(ct_path), str(pred_path), str(gt_path) if gt_path else None,
                output_path=str(output_dir / f"{case_name}_overlay.png"),
                thickness=thickness
            )
        except Exception as e:
            print(f"❌ {case_name} 处理失败: {e}")

    print(f"\n全部完成！结果保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='可视化分割结果 + 原始标注叠加到CT上')
    parser.add_argument('--ct', type=str, help='原始CT图像路径或文件夹')
    parser.add_argument('--pred', type=str, help='预测结果路径或文件夹')
    parser.add_argument('--gt', type=str, default=None, help='原始标注（真值）路径或文件夹')
    parser.add_argument('--output', type=str, default=None, help='输出路径')
    parser.add_argument('--slice', type=int, default=None, help='切片索引')
    parser.add_argument('--axis', type=int, default=0, choices=[0, 1, 2])
    parser.add_argument('--thickness', type=int, default=2, help='轮廓线宽')
    parser.add_argument('--batch', action='store_true', help='批量处理模式')

    args = parser.parse_args()

    if args.batch:
        if not args.ct or not args.pred:
            print("❌ 批处理需要 --ct 和 --pred")
            return
        visualize_batch(args.ct, args.pred, args.gt, args.output, args.thickness)
    else:
        if not args.ct or not args.pred:
            print("示例:")
            print("  单文件: python visualize.py --ct CT.nii.gz --pred pred.nii.gz --gt label.nii.gz")
            print("  批处理: python visualize.py --batch --ct ./imagesTs --pred ./predictions --gt ./labelsTr")
            return
        visualize_with_gt(
            args.ct, args.pred, args.gt, args.output,
            slice_idx=args.slice, axis=args.axis, thickness=args.thickness
        )


if __name__ == '__main__':
    main()