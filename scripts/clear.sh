#!/bin/bash

# ===================== nnU-Net 清理脚本 =====================
# 菜单编号与生产脚本 step3_run.sh 完全对应
#   生产 1 = convert_MSD_dataset  → 清理 1
#   生产 2 = plan_and_preprocess  → 清理 2
#   生产 3 = train                → 清理 3
#   生产 4 = predict              → 清理 4

export nnUNet_raw="/root/autodl-tmp/Project/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/root/autodl-tmp/Project/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/root/autodl-tmp/Project/nnUNet/nnUNet_results"

DATASET_ID=44
DATASET_NAME="Task${DATASET_ID}_READ"

show_menu() {
    echo ""
    echo "=========================================="
    echo "  nnU-Net 清理脚本（编号与生产脚本一致）"
    echo "  当前数据集: $DATASET_NAME (ID: $DATASET_ID)"
    echo "=========================================="
    echo "  1. 清理原始数据 (对应 convert_MSD_dataset)"
    echo "  2. 清理预处理   (对应 plan_and_preprocess)"
    echo "  3. 清理训练结果 (对应 train)"
    echo "  4. 清理预测结果 (对应 predict)"
    echo "  5. 清理指定数据集全部（1+2+3+4）"
    echo "  6. 清理所有数据集（危险！）"
    echo "  7. 查看磁盘占用"
    echo "  8. 清理 pip/conda 缓存"
    echo "  0. 退出"
    echo "=========================================="
}

show_usage() {
    echo ""
    echo "--- 磁盘使用情况 ---"
    df -h /root/autodl-tmp
    echo ""
    echo "--- 各目录大小 ---"
    for dir in "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"; do
        [ -d "$dir" ] && echo "  $dir : $(du -sh "$dir" 2>/dev/null | cut -f1)" || echo "  $dir : 不存在"
    done
    echo ""
    echo "--- 数据集列表 ---"
    [ -d "$nnUNet_raw" ] && ls -d "$nnUNet_raw"/Task* 2>/dev/null | while read l; do echo "  $(basename "$l") (原始数据)"; done
    [ -d "$nnUNet_preprocessed" ] && ls -d "$nnUNet_preprocessed"/Dataset* 2>/dev/null | while read l; do echo "  $(basename "$l") (预处理)"; done
    [ -d "$nnUNet_results" ] && ls -d "$nnUNet_results"/Dataset* 2>/dev/null | while read l; do echo "  $(basename "$l") (训练/预测)"; done
}

clean_raw() {
    echo ""
    echo "▶ 清理原始数据 (对应 convert_MSD_dataset)"
    [ -d "$nnUNet_raw" ] && ls -d "$nnUNet_raw"/Task* 2>/dev/null || echo "  无"
    echo -n "任务名 (如 Dataset044_READ, all=全部, 回车取消): "
    read -r t
    [ -z "$t" ] && return
    [ "$t" = "all" ] && rm -rf "${nnUNet_raw:?}/Dataset"* || rm -rf "${nnUNet_raw:?}/${t}"
    echo "✅ 已清理"
}

clean_preprocessed() {
    echo ""
    echo "▶ 清理预处理 (对应 plan_and_preprocess)"
    [ -d "$nnUNet_preprocessed" ] && ls -d "$nnUNet_preprocessed"/Dataset* 2>/dev/null || echo "  无"
    echo -n "数据集名 (如 Dataset044_READ, all=全部, 回车取消): "
    read -r t
    [ -z "$t" ] && return
    [ "$t" = "all" ] && rm -rf "${nnUNet_preprocessed:?}/Dataset"* || rm -rf "${nnUNet_preprocessed:?}/${t}"
    echo "✅ 已清理"
}

clean_results() {
    echo ""
    echo "▶ 清理训练结果 (对应 train)"
    [ -d "$nnUNet_results" ] && ls -d "$nnUNet_results"/Dataset* 2>/dev/null || echo "  无"
    echo -n "数据集名 (如 Dataset044_READ, all=全部, 回车取消): "
    read -r t
    [ -z "$t" ] && return
    [ "$t" = "all" ] && rm -rf "${nnUNet_results:?}/Dataset"* || rm -rf "${nnUNet_results:?}/${t}"
    echo "✅ 已清理"
}

clean_predictions() {
    echo ""
    echo "▶ 清理预测结果 (对应 predict)"
    [ -d "$nnUNet_results" ] && for d in "$nnUNet_results"/Dataset*; do
        [ -d "$d" ] && for tr in "$d"/*; do
            [ -d "${tr}/fold_all/validation" ] && echo "  ${tr}/fold_all/validation"
        done
    done
    echo -n "数据集名 (如 Dataset044_READ, all=全部, 回车取消): "
    read -r t
    [ -z "$t" ] && return
    if [ "$t" = "all" ]; then
        for d in "$nnUNet_results"/Dataset*; do
            [ -d "$d" ] && for tr in "$d"/*; do rm -rf "${tr}/fold_all/validation" 2>/dev/null; done
        done
    else
        for tr in "$nnUNet_results/${t}"/*; do rm -rf "${tr}/fold_all/validation" 2>/dev/null; done
    fi
    echo "✅ 已清理"
}

clean_dataset_all() {
    echo ""
    echo "▶ 清理指定数据集全部（1+2+3+4）"
    echo -n "数据集 ID (如 44, 回车取消): "
    read -r did
    [ -z "$did" ] && return
    printf -v pd '%03d' "$did" 2>/dev/null
    echo "将清理: Task${did}_* / Dataset${pd}_*"
    echo -n "确认? (y/N): "
    read -r cf
    [ "$cf" != "y" ] && [ "$cf" != "Y" ] && return
    rm -rf "${nnUNet_raw:?}/Task${did}_"* "${nnUNet_preprocessed:?}/Dataset${pd}_"* "${nnUNet_results:?}/Dataset${pd}_"*
    echo "✅ 已清理"
}

clean_all() {
    echo ""
    echo "⚠️ 删除所有 nnU-Net 数据"
    echo -n "输入 YES 确认: "
    read -r cf
    [ "$cf" = "YES" ] && rm -rf "${nnUNet_raw:?}"/* "${nnUNet_preprocessed:?}"/* "${nnUNet_results:?}"/* && echo "✅ 已清空"
}

clean_caches() {
    pip cache purge 2>/dev/null
    conda clean -a -y 2>/dev/null
    rm -rf /tmp/* 2>/dev/null
    echo "✅ 缓存已清理"
}

while true; do
    show_menu
    echo -n "请输入选项 [0-8]: "
    read -r choice
    case $choice in
        1) clean_raw ;;
        2) clean_preprocessed ;;
        3) clean_results ;;
        4) clean_predictions ;;
        5) clean_dataset_all ;;
        6) clean_all ;;
        7) show_usage ;;
        8) clean_caches ;;
        0) echo "退出"; exit 0 ;;
        *) echo "无效选项" ;;
    esac
    read -p "按回车继续..."
done
