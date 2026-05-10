#!/bin/bash

# ===================== 环境变量 =====================
export nnUNet_raw="/root/autodl-tmp/Project/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/root/autodl-tmp/Project/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/root/autodl-tmp/Project/nnUNet/nnUNet_results"

# ===================== 配置 =====================
DATASET_ID=44
DATASET_NAME="Task${DATASET_ID}_READ"
TRAINER="nnUNetTrainer_100epochs"   # 训练器，50 epochs 用于快速测试
CONFIG="2d"

# ===================== 菜单 =====================
show_menu() {
    echo ""
    echo "=========================================="
    echo "  nnU-Net 流程控制脚本"
    echo "  当前数据集: $DATASET_NAME (ID: $DATASET_ID)"
    echo "=========================================="
    echo "  1. 转换数据集 (convert_MSD_dataset)"
    echo "  2. 计划与预处理 (plan_and_preprocess)"
    echo "  3. 训练 (train)"
    echo "  4. 预测 (predict)"
    echo "  5. 一键跑完 1→2→3"
    echo "  6. 查看预处理后的标签是否正常"
    echo "  0. 退出"
    echo "=========================================="
}

# ===================== 各步骤函数 =====================

step1_convert() {
    echo ""
    echo "▶ 第 1 步：转换数据集"
    echo "------------------------------"
    nnUNetv2_convert_MSD_dataset -i "${nnUNet_raw}/${DATASET_NAME}"
    if [ $? -eq 0 ]; then
        echo "✅ 转换完成"
    else
        echo "❌ 转换失败，请检查"
    fi
}

step2_preprocess() {
    echo ""
    echo "▶ 第 2 步：计划与预处理"
    echo "------------------------------"
    nnUNetv2_plan_and_preprocess -d ${DATASET_ID} --verify_dataset_integrity --verbose
    if [ $? -eq 0 ]; then
        echo "✅ 预处理完成"
    else
        echo "❌ 预处理失败，请检查 dataset.json 和文件完整性"
    fi
}

step3_train() {
    echo ""
    echo "▶ 第 3 步：训练"
    echo "------------------------------"
    echo "训练器: ${TRAINER}"
    echo "配置: ${CONFIG}"
    echo "折: all (全量数据训练)"
    echo ""
    
    # 检查预处理是否完成
    if [ ! -d "${nnUNet_preprocessed}/Dataset$(printf '%03d' ${DATASET_ID})_${DATASET_NAME#Task*_}" ] && \
       [ ! -d "${nnUNet_preprocessed}/Dataset$(printf '%03d' ${DATASET_ID})_READ" ]; then
        echo "⚠️  预处理目录不存在，请先运行第 2 步"
        return
    fi
    
    nnUNetv2_train ${DATASET_ID} ${CONFIG} all -tr ${TRAINER}
    if [ $? -eq 0 ]; then
        echo "✅ 训练完成"
    else
        echo "❌ 训练失败"
    fi
}

step4_predict() {
    echo ""
    echo "▶ 第 4 步：预测"
    echo "------------------------------"
    
    # 默认输入输出路径
    DEFAULT_INPUT="${nnUNet_raw}/${DATASET_NAME}/imagesTs"
    DEFAULT_OUTPUT="${nnUNet_results}/Dataset$(printf '%03d' ${DATASET_ID})_READ/${TRAINER}__nnUNetPlans__${CONFIG}/fold_all/validation"
    
    echo ""
    echo "请输入预测输入文件夹路径（直接回车使用默认值）:"
    echo "默认: ${DEFAULT_INPUT}"
    read -r INPUT_DIR
    INPUT_DIR="${INPUT_DIR:-$DEFAULT_INPUT}"
    
    echo "请输入预测输出文件夹路径（直接回车使用默认值）:"
    echo "默认: ${DEFAULT_OUTPUT}"
    read -r OUTPUT_DIR
    OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT}"
    
    echo ""
    echo "输入: ${INPUT_DIR}"
    echo "输出: ${OUTPUT_DIR}"
    echo ""
    
    nnUNetv2_predict -i "${INPUT_DIR}" -o "${OUTPUT_DIR}" -d ${DATASET_ID} -c ${CONFIG} -f all -tr ${TRAINER}
    if [ $? -eq 0 ]; then
        echo "✅ 预测完成"
        echo "结果保存在: ${OUTPUT_DIR}"
    else
        echo "❌ 预测失败"
    fi
}

step_verify() {
    echo ""
    echo "▶ 校验数据集完整性"
    echo "------------------------------"
    nnUNetv2_plan_and_preprocess -d ${DATASET_ID} --verify_dataset_integrity
}

step_check_labels() {
    echo ""
    echo "▶ 检查预处理后的标签"
    echo "------------------------------"
    
    # 找到预处理目录
    PREPROC_DIR=$(ls -d ${nnUNet_preprocessed}/Dataset$(printf '%03d' ${DATASET_ID})_* 2>/dev/null | head -1)
    
    if [ -z "$PREPROC_DIR" ]; then
        echo "❌ 找不到预处理目录，请先运行第 2 步"
        return
    fi
    
    PREPROC_DIR="${PREPROC_DIR}/nnUNetPlans_2d"
    
    if [ ! -d "$PREPROC_DIR" ]; then
        echo "❌ 找不到 ${PREPROC_DIR}"
        return
    fi
    
    cd "$PREPROC_DIR" || return
    
    # 找第一个 seg 文件
    FIRST_SEG=$(ls *_seg.b2nd 2>/dev/null | head -1)
    
    if [ -z "$FIRST_SEG" ]; then
        echo "❌ 找不到标签文件"
        return
    fi
    
    echo "检查文件: ${FIRST_SEG}"
    python3 -c "
import blosc2, numpy as np
seg = blosc2.open(urlpath='${FIRST_SEG}', mode='r', dparams={'nthreads': 1})
arr = seg[:]
print(f'dtype: {arr.dtype}')
print(f'shape: {arr.shape}')
print(f'unique: {np.unique(arr)}')
print(f'has nan: {np.any(np.isnan(arr.astype(float)))}')
print(f'非零占比: {(arr != 0).sum() / arr.size * 100:.4f}%')
"
    cd - > /dev/null || return
}

# ===================== 主循环 =====================
while true; do
    show_menu
    echo -n "请输入选项 [0-7]: "
    read -r choice
    
    case $choice in
        1) step1_convert ;;
        2) step2_preprocess ;;
        3) step3_train ;;
        4) step4_predict ;;
        5) 
            step1_convert && step2_preprocess && step3_train
            ;;
        6) step_check_labels ;;
        0) 
            echo "退出"
            exit 0
            ;;
        *) echo "无效选项，请输入 0-7" ;;
    esac
    
    echo ""
    echo "按回车键继续..."
    read -r
done