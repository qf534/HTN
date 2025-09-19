import csv
import json

# 修改为你的 CSV 文件路径
csv_file_path = '/home/qf/HMGC/HMGC-main/dataset/MGT_datasets/WP_LLMs.csv'
json_file_path = '/home/qf/HMGC/HMGC-main/dataset/MGT_datasets/WP.json'

original = []
sampled = []

# 读取CSV并提取C3和C4列内容
with open(csv_file_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)  # 跳过表头
    for row in reader:
        if len(original) < 300 and len(row) > 2:
            original.append(row[2][:400])  # C3列
        if len(sampled) < 300 and len(row) > 3:
            sampled.append(row[3][:400])  # C4列
        if len(original) >= 300 and len(sampled) >= 300:
            break

# 构造最终 JSON 对象
output_data = {
    "original": original,
    "sampled": sampled
}

# 写入 JSON 文件
with open(json_file_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print("✅ JSON 文件已生成：", json_file_path)
