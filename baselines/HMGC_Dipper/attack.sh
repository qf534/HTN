#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

# 输出日志目录
LOG_DIR="./logs"
mkdir -p $LOG_DIR

# 数据集路径
DATA_DIR="./dataset/MGT_datasets"

# 输出目录
OUTPUT_DIR="./mis_results"

# surrogate 模型路径
MODEL_PATH="./outputs_STEM,SS,HUMAN/surrogate_roberta"

# 循环步长数据集
for STEP in 100 300 500 700 900; do
  for PREFIX in Social_Sciences_tokens Humanities_tokens; do
    DATA_FILE="${DATA_DIR}/${PREFIX}_${STEP}.jsonl"
    LOG_FILE="${LOG_DIR}/${PREFIX}_${STEP}.log"

    echo "==== Running on ${DATA_FILE} ====" | tee "$LOG_FILE"

    # 开始时间
    echo "Start time: $(date)" | tee -a "$LOG_FILE"

    # 开始时显存
    echo "GPU memory at start:" | tee -a "$LOG_FILE"
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total --format=csv,noheader,nounits | tee -a "$LOG_FILE"

    # 运行并记录时间
    /usr/bin/time -v \
    python attack/multi_flint_attack.py \
      --model_name_or_path "$MODEL_PATH" \
      --data_file "$DATA_FILE" \
      --output_dir "$OUTPUT_DIR" \
      --attacking_method dualir \
      --num_workers 1 \
      --num_gpu_per_process 2 \
      --text_key text \
      --label_key label \
    2>&1 | tee -a "$LOG_FILE"

    # 结束时显存
    echo "GPU memory at end:" | tee -a "$LOG_FILE"
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total --format=csv,noheader,nounits | tee -a "$LOG_FILE"

    # 结束时间
    echo "End time: $(date)" | tee -a "$LOG_FILE"
  done
done
