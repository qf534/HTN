import json

input_file = '/home/qf/HMGC/HMGC-main/mis_results/attack_WP_Mis_HMGC.jsonl'      # 输入文件路径（.jsonl 格式）
output_file = '/home/qf/RAFT/main/Mis_HMGC/attack_WP_Mis_HMGC.json'   # 输出所有文本为 sampled 字段

sampled_texts = []

with open(input_file, 'r', encoding='utf-8') as fin:
    for line in fin:
        data = json.loads(line)
        sampled_texts.append(data.get('x', ''))  # 不再判断 y，直接提取 x

# 保存为 {"sampled": [ ... ]}
with open(output_file, 'w', encoding='utf-8') as fout:
    json.dump({'sampled': sampled_texts}, fout, ensure_ascii=False, indent=4)

print(f"提取完成，共提取 {len(sampled_texts)} 条文本，输出到 {output_file}")
