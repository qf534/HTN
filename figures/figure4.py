import os
import json
from bert_score import score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

# 配置
model_path = "/homes/qf/models/deberta-xlarge-mnli"
device = "cuda"  # 如果用 CPU 可改为 "cpu"

datasets = ["Social_Sciences", "STEM", "Humanities","Essay", "Reuters", "WP"]  # 扩展到6个数据集
methods = ["Ours", "Dipper", "Mis_HMGC", "RAFT"]

# 文件路径结构
base_path = {
    "Original": "/home/qf/RAFT/main/original",
    "Ours": "/home/qf/RAFT/main/Ours",
    "Dipper": "/home/qf/RAFT/main/Dipper",
    "HMGC": "/home/qf/RAFT/main/HMGC",
    "Mis_HMGC": "/home/qf/RAFT/main/Mis_HMGC",
    "RAFT": "/home/qf/RAFT/main/RAFT"
}

# 文件名构造函数
file_names = {
    "Original": lambda ds: f"{ds}.json",
    "Ours": lambda ds: f"attacked_{ds}_Ours.json",
    "Dipper": lambda ds: f"attacked_{ds}_Dipper.json",
    "HMGC": lambda ds: f"attacked_{ds}_HMGC.json",
    "Mis_HMGC": lambda ds: f"attacked_{ds}_Mis_HMGC.json",
    "RAFT": lambda ds: f"attacked_{ds}_RAFT.json"
}

# 收集全部 BERTScore F1 分数
records = []

for dataset in datasets:
    print(f"Processing dataset: {dataset}")

    # 读取原始文本
    with open(os.path.join(base_path["Original"], file_names["Original"](dataset)), "r", encoding="utf-8") as f:
        original_texts = json.load(f)["sampled"]

    for method in methods:
        file_path = os.path.join(base_path[method], file_names[method](dataset))
        with open(file_path, "r", encoding="utf-8") as f:
            method_texts = json.load(f)["sampled"]

        assert len(original_texts) == len(method_texts), f"{dataset} - {method} 数量不一致"

        # 计算 BERTScore（F1）
        _, _, F1 = score(
            cands=method_texts,
            refs=original_texts,
            model_type=model_path,
            num_layers=24,
            lang="en",
            device=device
        )

        for f1 in F1.tolist():
            records.append({
                "Dataset": dataset,
                "Method": method,
                "BERTScore": f1
            })

# 转为 DataFrame
df = pd.DataFrame(records)

sns.set(style="whitegrid", font_scale=1.2)
fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharey=True)  # 改为2x3
palette = {
    "Ours": "#1f77b4",
    "Dipper": "#ff7f0e",
    "HMGC": "#2ca02c",
    "Mis_HMGC": "#d62728",
    "RAFT": "#e377c2",
}

# 字体设置
title_font = {'fontsize': 16, 'fontweight': 'bold'}
label_font = {'fontsize': 14, 'fontweight': 'bold'}
tick_fontsize = 12

for i, dataset in enumerate(datasets):
    row, col = divmod(i, 3)   # 计算子图位置
    ax = axes[row, col]
    sub_df = df[df["Dataset"] == dataset]

    sns.boxplot(
        x="Method", y="BERTScore", data=sub_df,
        ax=ax, palette=palette, width=0.6, fliersize=2, linewidth=1.5
    )

    ax.set_title(dataset.replace("_", " "), **title_font)
    ax.set_xlabel("", **label_font)
    if col == 0:
        ax.set_ylabel("BERTScore F1", **label_font)
    else:
        ax.set_ylabel("")

    # 固定纵轴范围和间隔
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

    # 横纵轴刻度
    ax.tick_params(axis='x', labelrotation=15, labelsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(14)
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
        tick.set_fontsize(12)

# 图例
handles = [mpatches.Patch(color=palette[m], label=m) for m in methods]
fig.legend(
    handles=handles,
    loc='lower center',
    ncol=len(methods),
    frameon=False,
    bbox_to_anchor=(0.5, -0.05),
    fontsize=16,       # 字体更大
    title_fontsize=18, # 标题更大
    handlelength=2.5,
    handleheight=1.2
)

plt.tight_layout(rect=[0, 0.07, 1, 1])  # 留出下方空间给图例
plt.savefig("bertscore.pdf", bbox_inches="tight")
plt.show()
