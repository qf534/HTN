import matplotlib.pyplot as plt

# 横坐标
x = [0, 0.05, 0.1, 0.15, 0.2, 0.25]

# 6个数据集的 AUROC
# auroc1 = [0.8951, 0.7303, 0.6329, 0.4460, 0.4018, 0.2719]
# auroc2 = [0.8981, 0.6487, 0.4774, 0.3175, 0.1870, 0.1059]
# auroc3 = [0.8131, 0.5856, 0.4331, 0.2916, 0.1952, 0.1309]
# auroc4 = [0.8707, 0.6947, 0.5867, 0.4747, 0.4092, 0.3898]
# auroc5 = [0.9683, 0.8249, 0.7073, 0.6045, 0.5252, 0.4785]
# auroc6 = [0.8834, 0.7029, 0.5584, 0.4541, 0.3858, 0.3554]
auroc1 = [0.7302, 0.5354, 0.4282, 0.3254, 0.2401, 0.1849]
auroc2 = [0.7015, 0.5046, 0.3881, 0.2820, 0.1933, 0.1354]
auroc3 = [0.7453, 0.5599, 0.4494, 0.3435, 0.2565, 0.1992]
auroc4 = [0.7022, 0.4743, 0.3812, 0.2957, 0.2510, 0.2462]
auroc5 = [0.7068, 0.4272, 0.3016, 0.2240, 0.1837, 0.1791]
auroc6 = [0.7246, 0.5337, 0.4273, 0.3563, 0.3064, 0.2930]
# 6个数据集的 PPL
# ppl1 = [13.3490, 19.4613, 23.0230, 31.6824, 33.1145, 42.2190]
# ppl2 = [11.6635, 18.2764, 23.0748, 28.5744, 35.0950, 41.7161]
# ppl3 = [16.8657, 24.9764, 31.8226, 39.5900, 47.7311, 55.9574]
# ppl4 = [13.3490, 19.4613, 23.0230, 31.6824, 33.1145, 42.2190]
# ppl5 = [11.6635, 18.2764, 23.0748, 28.5744, 35.0950, 41.7161]
# ppl6 = [16.8657, 24.9764, 31.8226, 39.5900, 47.7311, 55.9574]
ppl1 = [16.6751, 24.5628, 30.0352, 36.0864, 42.7511, 49.1117]
ppl2 = [13.5981, 20.1872, 25.1360, 30.8975, 37.2743, 43.6528]
ppl3 = [18.7011, 26.1612, 31.2989, 37.2500, 43.3094, 49.1582]
ppl4 = [16.6751, 24.5628, 30.0352, 36.0864, 42.7511, 49.1117]
ppl5 = [13.5981, 20.1872, 25.1360, 30.8975, 37.2743, 43.6528]
ppl6 = [18.7011, 26.1612, 31.2989, 37.2500, 43.3094, 49.1582]
# 创建两张子图 (1行2列)
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(14, 5))

# ----------------- 左边子图 (Dataset1-3) -----------------
ax2 = ax1.twinx()

ax1.plot(x, auroc1, marker='o', color='red', label="Social_Sciences")
ax1.plot(x, auroc2, marker='o', color='orange', label="STEM")
ax1.plot(x, auroc3, marker='o', color='green', label="Humanities")
ax1.set_ylabel("AUROC",fontsize=14)
ax1.set_xlabel("Replaced Ratio",fontsize=14)

ax2.plot(x, ppl1, marker='o', linestyle="--", color='blue', label="Social_Sciences PPL")
ax2.plot(x, ppl2, marker='o', linestyle="--", color='purple', label="STEM PPL")
ax2.plot(x, ppl3, marker='o', linestyle="--", color='brown', label="Humanities PPL")
ax2.set_ylabel("Perplexity",fontsize=14)

# AUROC纵轴固定 0-1，间隔0.2
ax1.set_ylim(0, 1)
ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax1.set_xticks(x)
ax1.set_title("(a) Log Rank", fontsize=14)

# ----------------- 右边子图 (Dataset4-6) -----------------
ax4 = ax3.twinx()

ax3.plot(x, auroc4, marker='o', color='red', label="Social_Sciences")
ax3.plot(x, auroc5, marker='o', color='orange', label="STEM")
ax3.plot(x, auroc6, marker='o', color='green', label="Humanities")
ax3.set_ylabel("AUROC",fontsize=14)
ax3.set_xlabel("Replaced Ratio",fontsize=14)

ax4.plot(x, ppl4, marker='o', linestyle="--", color='blue', label="Social_Sciences PPL")
ax4.plot(x, ppl5, marker='o', linestyle="--", color='purple', label="STEM PPL")
ax4.plot(x, ppl6, marker='o', linestyle="--", color='brown', label="Humanities PPL")
ax4.set_ylabel("Perplexity",fontsize=14)

# AUROC纵轴固定 0-1，间隔0.2
ax3.set_ylim(0, 1)
ax3.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax3.set_xticks(x)
ax3.set_title("(b) Fast-DetectGPT", fontsize=14)

lines = []
labels = []

for ax in [ax1, ax2]:
    l, lab = ax.get_legend_handles_labels()
    lines += l
    labels += lab

fig.legend(lines, labels, loc='lower center', ncol=6, framealpha=1, fontsize=14)

# 美化布局
plt.tight_layout(rect=[0, 0.1, 1, 1])  # 给图例留出下方空间
plt.savefig("rate_0.pdf", format="pdf", bbox_inches="tight")
plt.show()