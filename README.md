# Human-Style Transformation for Robust Evasion of Machine-Generated Text Detection

## Setup
### Requirements

- **Python**: version > 3.9  
- **CUDA**: version 12.4  
It is recommended to use `conda` to create an isolated Python environment.

```
pip install -r requirements.txt
```

## LSPG

This project uses the synonym generator **LSPG**, which can be found here:

* LSPG repository: [https://github.com/KpKqwq/LSPG](https://github.com/KpKqwq/LSPG)

Follow the instructions in the LSPG repo to prepare its models, vocab, and dependencies.

## Attack

### Running the attack module

The attack module entry point is located in `attack/main.py`. Below is a typical usage example:

```
python attack/main.py --method [entropy | random] --generator [lspg | api] --input_file ./datasets/original/STEM.json \
  --kenlm_path ./merge_Ngram.bin --candidate_top_k [5 | 10] --replace_ratio [0.05 | 0.1 | 0.15 | 0.2] \
  --output_file ./outputs/attacked_STEM.json
```
## Batch Processing

If you want to process multiple datasets at the same time, you can use:

```
bash main.sh
```
```
METHOD="entropy"
GENERATOR="lspg"
KENLM_PATH="merge_Ngram.bin"
CANDIDATE_TOP_K=10
REPLACE_RATIO=0.15

python3 ./attack/main.py \
  --method ${METHOD} \
  --generator ${GENERATOR} \
  --input_file "$INPUT_DIR/$dataset" \
  --kenlm_path ${KENLM_PATH} \
  --candidate_top_k ${CANDIDATE_TOP_K} \
  --replace_ratio ${REPLACE_RATIO} \
  --output_file "./outputs/attacked_mydataset.json"
```
## Detector
For Logrank, Logprob, DetectGPT, FastDetectGPT, and ChatGPTDetector
```
python detector.py --detector [logrank | logprob | dgpt | fdgpt | chatgpt-roberta] --json_path ./datasets/original/STEM.json \
--device cuda 
```

For RAIDAR
```
./raidar_llm_detect/detect.py
```
For Binoculars (supports batch processing of multiple JSON files)
```
./detectors/Binoculars/main.py
```

If you find our paper/resources useful, please cite:


---

