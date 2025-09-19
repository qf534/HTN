#!/usr/bin/env bash
# run_attacks.sh: 批量对多个数据集执行 TextFlint Dipper-Paraphrase 攻击脚本
# 会记录运行时间和 GPU 显存情况

# flint_transform.py 脚本路径
FLINT_SCRIPT="./attack/baseline/flint_dipper_paraphrase.py"

# 数据集目录
DATA_DIR="./dataset/MGT_datasets"

# 输出目录
OUT_ROOT="./results_tokens"
mkdir -p "$OUT_ROOT"

# 日志目录
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$LOG_DIR"

# 数据集前缀和 step
PREFIXES=("Social_Sciences_tokens" "Humanities_tokens")
STEPS=(100 300 500 700 900)

# 遍历数据集
for PREFIX in "${PREFIXES[@]}"; do
  for STEP in "${STEPS[@]}"; do
    TEST_FILE="${DATA_DIR}/${PREFIX}_${STEP}.jsonl"
    OUT_DIR="${OUT_ROOT}/${PREFIX}_${STEP}"
    LOG_FILE="${LOG_DIR}/${PREFIX}_${STEP}.log"

    mkdir -p "$OUT_DIR"

    echo "==========================================" | tee "$LOG_FILE"
    echo "Processing dataset: $TEST_FILE" | tee -a "$LOG_FILE"
    echo "Output directory: $OUT_DIR" | tee -a "$LOG_FILE"

    # 记录开始时间
    START_TIME=$(date +%s)
    echo "Start time: $(date)" | tee -a "$LOG_FILE"

    # 开始时 GPU 显存
    echo "GPU memory at start:" | tee -a "$LOG_FILE"
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total \
      --format=csv,noheader,nounits | tee -a "$LOG_FILE"

    # 执行攻击并记录详细时间信息
    /usr/bin/time -v \
      python3 "$FLINT_SCRIPT" \
        --test_file "$TEST_FILE" \
        --out_dir "$OUT_DIR" \
      2>&1 | tee -a "$LOG_FILE"

    # 结束时 GPU 显存
    echo "GPU memory at end:" | tee -a "$LOG_FILE"
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total \
      --format=csv,noheader,nounits | tee -a "$LOG_FILE"

    # 记录结束时间和耗时
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "End time: $(date)" | tee -a "$LOG_FILE"
    echo "Duration: ${DURATION} seconds" | tee -a "$LOG_FILE"

    echo "Finished $PREFIX-$STEP" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
  done
done

echo "All datasets processed. Logs and outputs are under $OUT_ROOT/"
