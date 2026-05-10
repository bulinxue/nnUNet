import numpy as np
import os

# 修改为你预处理文件存放的路径
preprocessed_dir = '/root/autodl-tmp/Project/nnUNet/nnUNet_preprocessed/Dataset044_example/nnUNetPlans_2d'

# 找一个训练用的 .npy 文件（图像）和对应的 seg 文件（标签）
for f in os.listdir(preprocessed_dir):
    if f.endswith('.npy') and 'seg' not in f:  # 找一个图像文件
        case_id = f.split('.')[0]  # 提取病例名
        seg_file = f.replace('.npy', '_seg.npy')  # 推断标签文件名
        
        if os.path.exists(os.path.join(preprocessed_dir, seg_file)):
            img = np.load(os.path.join(preprocessed_dir, f))
            seg = np.load(os.path.join(preprocessed_dir, seg_file))
            
            print(f"病例: {case_id}")
            print(f"  图像形状: {img.shape}, 范围: {img.min():.2f} / {img.max():.2f}")
            print(f"  标签形状: {seg.shape}, 唯一值: {np.unique(seg)}")
            print(f"  标签前景体素占比: {(seg == 1).sum() / seg.size * 100:.4f}%")
            
            # 保存第一个病例的中间切片图像和标签，方便用工具直接观看
            mid_slice = img.shape[1] // 2
            np.save(f'{case_id}_slice.npy', img[0, mid_slice])
            np.save(f'{case_id}_seg_slice.npy', seg[0, mid_slice])
            print(f"  已保存中间切片到当前文件夹")
        else:
            print(f"警告：找不到 {case_id} 的标签文件 {seg_file}")
        break