#!/bin/bash

# ================== 输入与输出路径 ==================
INPUT_DIR="./datasets/original"
OUTPUT_DIR="./datasets/test"

# ================== 数据集文件名 ==================
# 之前的数据集
# DATASETS=("Humanities.json" "WP.json")

# 当前要跑的数据集
DATASETS=("Essay.json" "Reuters.json" "Social_Sciences.json" "WP.json")

# ================== 参数配置 ==================
METHOD="entropy"
GENERATOR="lspg"
KENLM_PATH="merge_Ngram.bin"
CANDIDATE_TOP_K=10
REPLACE_RATIO=0.15

# ================== 循环跑每个数据集 ==================
for dataset in "${DATASETS[@]}"; do
    dataset_name=$(basename "$dataset" .json)
    echo "=== Processing $dataset_name ==="

    python3 ./attack/main.py \
        --method $METHOD \
        --generator $GENERATOR \
        --input_file "$INPUT_DIR/$dataset" \
        --kenlm_path $KENLM_PATH \
        --candidate_top_k $CANDIDATE_TOP_K \
        --replace_ratio $REPLACE_RATIO \
        --output_file "$OUTPUT_DIR/attacked_${dataset_name}_0.15.json"
done
