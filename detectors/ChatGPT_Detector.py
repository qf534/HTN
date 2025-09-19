import json
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from tqdm import tqdm

# 判断是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载模型和分词器
print("Loading model...")
model_path = "chatgpt-detector-roberta"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()


def predict_text(text):
    """
    返回模型预测为“机器生成”的概率（标签为1的概率）
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # 数据送到GPU
    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=-1).squeeze()
        # 假设标签0是Human，标签1是Machine
        return probs[1].item()


def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def calculate_auroc(data):
    y_true = []
    y_scores = []

    # 人类文本（标签0）
    print("\nProcessing human texts...")
    for text in tqdm(data["original"]):
        if not text.strip():
            continue
        prob = predict_text(text)
        y_scores.append(prob)
        y_true.append(0)

    # 机器文本（标签1）
    print("\nProcessing machine texts...")
    for text in tqdm(data["sampled"]):
        if not text.strip():
            continue
        prob = predict_text(text)
        y_scores.append(prob)
        y_true.append(1)

    auroc = roc_auc_score(y_true, y_scores)
    return auroc


if __name__ == "__main__":
    json_file_path = ""

    print(f"Loading data from {json_file_path}...")
    dataset = load_json_data(json_file_path)

    if "original" not in dataset or "sampled" not in dataset:
        raise ValueError("JSON文件必须包含'original'和'sampled'字段")

    auroc_score = calculate_auroc(dataset)
    print(f"\nAUROC score: {auroc_score:.4f}")
