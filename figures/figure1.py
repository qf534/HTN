import json
import kenlm
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from transformers import AutoTokenizer, AutoModelForCausalLM

# === 路径设置 ===
kenlm_model_path = "./kenlm_Ngram.bin"
input_json_path = ""

# === 加载 KenLM 模型 ===
kenlm_model = kenlm.Model(kenlm_model_path)

# === 加载数据集 ===
with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 三类句子
original_sentences = data["original"][:500]   # human
sampled_sentences = data["sampled"][:500]    # GPT-4o-mini
para_sentences = data["para"][:500]          # deepseek-chat

# === KenLM 得分函数 ===
def score_sentence(sentence):
    return kenlm_model.score(sentence.strip(), bos=True, eos=True)

# === 计算 KenLM 分数 ===
original_scores = [score_sentence(sent) for sent in original_sentences]
sampled_scores = [score_sentence(sent) for sent in sampled_sentences]
para_scores = [score_sentence(sent) for sent in para_sentences]

# === 分组取平均函数 ===
def average_every_n(scores, n):
    return [sum(scores[i:i+n]) / len(scores[i:i+n]) for i in range(0, len(scores), n)]

step = 25
avg_orig_scores = average_every_n(original_scores, step)
avg_samp_scores = average_every_n(sampled_scores, step)
avg_para_scores = average_every_n(para_scores, step)
x_avg = list(range(1, len(avg_orig_scores) + 1))

# === 绘制折线图 ===
plt.figure(figsize=(12, 6))
plt.plot(x_avg, avg_orig_scores, label="Human", color="green", marker='o')
plt.plot(x_avg, avg_samp_scores, label="GPT-4o-mini", color="blue", marker='o')
plt.plot(x_avg, avg_para_scores, label="DeepSeek-V3.1", color="red", marker='o')

plt.xlabel("Group Index (Each group = 25 sentences)",fontsize=14)
plt.ylabel("Average N-gram Score",fontsize=14)


# 横轴强制显示整数
plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Ngram_score_avg25_line_3.pdf")
plt.show()

#DeepSeek-V3.1