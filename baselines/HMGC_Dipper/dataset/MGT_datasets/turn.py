import json

# 假设 data.json 是你的输入文件，包含 {"original": [...], "sampled": [...]}
with open("/home/qf/RAFT/generated_datasets_tokens/Social_Sciences_tokens_900.json", "r", encoding="utf-8") as f:
    data = json.load(f)

output_lines = []

# 处理 original（人类文本）
for item in data.get("original", []):
    output_lines.append(json.dumps({"text": item.strip(), "label": "human"}, ensure_ascii=False))

# 处理 sampled（机器文本）
for item in data.get("sampled", []):
    output_lines.append(json.dumps({"text": item.strip(), "label": "gpt"}, ensure_ascii=False))

# 写入输出文件，每行一个 JSON 对象
with open("/home/qf/HMGC/HMGC-main/dataset/MGT_datasets/Social_Sciences_tokens_900.jsonl", "w", encoding="utf-8") as f:
    for line in output_lines:
        f.write(line + "\n")
