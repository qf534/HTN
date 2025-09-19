import json
from bert_score import score

model_path = "deberta-xlarge-mnli"

#2. 读取两个 JSON 文件
with open("file1.json", "r", encoding="utf-8") as f1:
    data1 = json.load(f1)

with open("file2.json", "r", encoding="utf-8") as f2:
    data2 = json.load(f2)

sampled1 = data1["sampled"]
sampled2 = data2["sampled"]

assert len(sampled1) == len(sampled2)


P, R, F1 = score(
    cands=sampled1,
    refs=sampled2,
    model_type=model_path,
    num_layers=24,
    lang="en",
    device="cuda"
)

# 4. 打印平均 F1
print(f"平均 BERTScore F1 分数：{F1.mean().item():.4f}")


