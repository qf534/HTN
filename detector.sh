#!/bin/bash
# 跑六个数据集 (0.05 / 0.25) 用 detector = dgpt

# 数据集名字
datasets=("STEM" "Essay" "WP" "Reuters" "Social_Sciences" "Humanities")
# 替换比例
ratios=("0.05" "0.25")

detector="dgpt"

mkdir -p logs

for dataset in "${datasets[@]}"; do
  for ratio in "${ratios[@]}"; do
    json_path="./datasets/ablation/ablation_${dataset}_${ratio}.json"
    echo "Running detector=${detector} on dataset=${dataset}, ratio=${ratio}"
    python detector.py \
      --detector "${detector}" \
      --json_path "${json_path}" \
      --device cuda \
     
  done
done
