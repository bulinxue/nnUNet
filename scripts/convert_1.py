import os
import subprocess

# 路径配置
ROOT_DCM_DIR = "/root/autodl-tmp/data/READ/sample_raw/raw DCM data"
OUTPUT_DIR = "/root/autodl-tmp/data/READ/sample_raw_convert_1"

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 遍历 raw DCM data 下所有一级子文件夹
for item in os.listdir(ROOT_DCM_DIR):
    dcm_folder = os.path.join(ROOT_DCM_DIR, item)
    
    # 只处理文件夹，跳过文件
    if not os.path.isdir(dcm_folder):
        continue
    
    # 文件夹名作为输出文件名
    out_name = item
    print(f"正在转换: {out_name}")

    # dcm2niix 参数：压缩、不生成json、按文件夹名命名
    cmd = [
        "dcm2niix",
        "-z", "y",
        "-b", "n",
        "-f", out_name,
        "-o", OUTPUT_DIR,
        dcm_folder
    ]
    # 执行转换
    subprocess.run(cmd)

print("\n✅ 全部转换完成！")
print(f"输出目录: {OUTPUT_DIR}")