import json
import torch
from transformers import GPTNeoForCausalLM, GPT2TokenizerFast
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from matplotlib.patches import Patch

# === 配置部分 ===
datasets = ["Social_Sciences", "STEM", "Humanities","Essay", "Reuters", "WP"]  # 6个数据集
attack_methods = ["Ours", "Dipper", "Mis_HMGC", "RAFT"]
base_dir = ""
original_dir = ""

# 自定义颜色
custom_palette = {
    "Ours": "#1f77b4",
    "Dipper": "#ff7f0e",
    "HMGC": "#2ca02c",
    "Mis_HMGC": "#d62728",
    "RAFT": "#e377c2",
}

# === 模型初始化 ===
model_path = "gpt-neo-2.7B"
tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
model = GPTNeoForCausalLM.from_pretrained(model_path)
model.eval().cuda()


# === PPL 计算函数 ===
def calculate_perplexity(text, model, tokenizer, device="cuda"):
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    max_length = model.config.max_position_embeddings
    stride = 512
    nlls = []
    seq_len = input_ids.size(1)
    for i in range(0, seq_len, stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, seq_len)
        trg_len = end_loc - i
        input_ids_slice = input_ids[:, begin_loc:end_loc]
        target_ids = input_ids_slice.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids_slice, labels=target_ids)
            log_likelihood = outputs.loss * trg_len
        nlls.append(log_likelihood)
    total_nll = torch.stack(nlls).sum()
    avg_nll = total_nll / seq_len
    return torch.exp(avg_nll).item()


# === 图像准备 ===
fig, axs = plt.subplots(2, 3, figsize=(18, 9), sharey=True)  # 2x3 布局
sns.set(style="whitegrid")

title_font = {'fontsize': 16, 'fontweight': 'bold'}
label_font = {'fontsize': 14, 'fontweight': 'bold'}

for idx, dataset in enumerate(datasets):
    print(f"Processing dataset: {dataset}")
    original_path = f"{original_dir}/{dataset}.json"

    with open(original_path, "r", encoding="utf-8") as f:
        original_data = json.load(f)
    original_texts = original_data["sampled"]

    all_data = []

    for method in attack_methods:
        attack_path = f"{base_dir}/{method}/attacked_{dataset}_{method}.json"
        try:
            with open(attack_path, "r", encoding="utf-8") as f:
                attacked_data = json.load(f)
            rewritten_texts = attacked_data["sampled"]
        except:
            print(f"Missing file: {attack_path}")
            continue

        ratios = []
        for o, r in tqdm(zip(original_texts, rewritten_texts), total=len(original_texts), desc=f"{dataset}-{method}"):
            try:
                ppl_o = calculate_perplexity(o, model, tokenizer)
                ppl_r = calculate_perplexity(r, model, tokenizer)
                if ppl_o > 0:
                    ratio = (ppl_r - ppl_o) / ppl_o
                    ratios.append(ratio)
            except:
                continue

        for ratio in ratios:
            all_data.append({
                "Attack": method,
                "PPL Change Ratio": ratio
            })

    df = pd.DataFrame(all_data)

    # 找到 2x3 的位置
    row, col = divmod(idx, 3)
    ax = axs[row, col]

    sns.boxplot(
        data=df,
        x="Attack",
        y="PPL Change Ratio",
        palette=custom_palette,
        ax=ax,
        linewidth=1.5
    )

    ax.set_title(dataset.replace("_", " "), **title_font)
    ax.set_xlabel("")
    if col == 0:  # 第一列显示 ylabel
        ax.set_ylabel("PPL Change Ratio", **label_font)
    else:
        ax.set_ylabel("")

    # 横轴刻度（只调大小，不加粗）
    ax.tick_params(axis='x', rotation=15, labelsize=16, labelcolor='black')
    ax.tick_params(axis='y', labelsize=15, labelcolor='black')

    for tick in ax.get_xticklabels():
        tick.set_fontsize(14)   # 不加粗
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
        tick.set_fontsize(12)

# 图例
legend_elements = [Patch(facecolor=custom_palette[name], label=name) for name in attack_methods]
fig.legend(
    handles=legend_elements,
    fontsize=18,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.05),
    ncol=5,
    frameon=False
)

plt.tight_layout(rect=[0, 0.07, 1, 1])  # 留出下方空间给 legend
plt.savefig("PPL.pdf", bbox_inches="tight")
plt.show()
