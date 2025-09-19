import pandas as pd
import json
import random
from pathlib import Path


def load_and_extract(csv_path):
    """
    从 CSV 中提取第 3 列 (human) 和第 5–10 列 (gpt) 里的文本，返回统一格式的列表。
    假设 CSV 文件有表头或没有表头均可（若有表头会自动取列索引；若无表头，请加 header=None）。
    """
    # 如果你的 CSV 没有表头，可以加上 header=None，并在下面通过索引访问
    df = pd.read_csv(csv_path, header=0)  # 如果没有表头，改成 header=None

    # 确保至少有 10 列
    if df.shape[1] < 10:
        raise ValueError(f"{csv_path} 的列数不足 10 列，无法提取 C3–C10")

    data = []

    # 提取 C3（索引 2）列的 human 文本
    human_col_idx = 2
    for text in df.iloc[:, human_col_idx].dropna():
        text = str(text).strip()
        if text:
            data.append({'text': text, 'label': 'human'})

    # 提取 C5–C10（索引 4 到 9）列的 gpt 文本
    gpt_col_indices = list(range(4, 10))  # [4, 5, 6, 7, 8, 9]
    for col_idx in gpt_col_indices:
        for text in df.iloc[:, col_idx].dropna():
            text = str(text).strip()
            if text:
                data.append({'text': text, 'label': 'gpt'})

    return data


def save_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    print(f"✅ 保存文件: {filename} ({len(data)} 条样本)")


def main():
    csv_files = ["Essay_LLMs.csv", "WP_LLMs.csv", "Reuters_LLMs.csv"]

    all_data = []
    for file in csv_files:
        all_data.extend(load_and_extract(file))

    # 打乱顺序
    random.seed(42)
    random.shuffle(all_data)

    # 按 8:1:1 分割
    total = len(all_data)
    train_end = int(total * 0.8)
    valid_end = int(total * 0.9)

    train_data = all_data[:train_end]
    valid_data = all_data[train_end:valid_end]
    test_data = all_data[valid_end:]

    # 保存为 jsonl 文件
    save_jsonl(train_data, "/home/qf/HMGC/HMGC-main/dataset/MGT/train.jsonl")
    save_jsonl(valid_data, "/home/qf/HMGC/HMGC-main/dataset/MGT/valid.jsonl")
    save_jsonl(test_data, "/home/qf/HMGC/HMGC-main/dataset/MGT/test.jsonl")


if __name__ == "__main__":
    main()
