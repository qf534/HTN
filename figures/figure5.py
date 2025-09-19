import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
# === 数据准备 ===
tokens = [100, 300, 500, 700, 900]
x = np.arange(len(tokens))  # x轴位置

methods = ["HMGC", "Dipper", "RAFT", "Ours"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

# 显存数据 (GB)
humanities_mem = {
    "HMGC": [29.39, 25.73, 25.73, 25.73, 25.73],
    "Dipper": [41.02, 41.02, 41.02, 41.02, 41.02],
    "RAFT": [32.5, 32.5, 32.5, 32.5, 32.5],
    "Ours": [20.21, 20.21, 20.21, 20.21, 20.21],
}
social_mem = {
    "HMGC": [29.57, 25.73, 25.73, 25.73, 25.73],
    "Dipper": [41.02, 41.02, 41.02, 41.02, 41.02],
    "RAFT": [32.5, 32.5, 32.5, 32.5, 32.5],
    "Ours": [20.21, 20.21, 20.21, 20.21, 20.21],
}

# 时间数据 (min)
humanities_time = {
    "HMGC": [7.67, 30.92, 32.63, 32.6, 33.0],
    "Dipper": [18.8, 68.18, 68.87, 64.92, 67.75],
    "RAFT": [14.59, 35.97, 36.76, 35.1, 30.04],
    "Ours": [11.93, 33.72, 35.1, 35.17, 35.2],
}
social_time = {
    "HMGC": [7.68, 27.57, 31.02, 31.02, 31.0],
    "Dipper": [20.02, 65.52, 73.23, 72.05, 74.67],
    "RAFT": [10.68, 25.28, 24.54, 24.17, 24.58],
    "Ours": [12.15, 32.4, 35.55, 35.48, 35.55],
}

# === 画图 ===
fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

datasets = [("Humanities", humanities_mem, humanities_time),
            ("Social Sciences", social_mem, social_time)]

bar_width = 0.18

for ax, (title, mem_data, time_data) in zip(axes, datasets):
    # 左y轴: 显存柱状图
    for i, method in enumerate(methods):
        ax.bar(x + i * bar_width, mem_data[method], width=bar_width,
               color=colors[i], alpha=0.7)
    ax.set_ylabel("GPU Usage (GB)", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_ylim(0, 48)  # 最大值固定为48GB

    # 右y轴: 时间折线图
    ax2 = ax.twinx()
    for i, method in enumerate(methods):
        ax2.plot(x + i * bar_width + bar_width / 2, time_data[method],
                 marker='o', color=colors[i], linestyle='--')
    ax2.set_ylabel("Time (min)", fontsize=14)

# X轴设置
plt.xticks(x + bar_width * 1.5, tokens, fontsize=14)
axes[-1].set_xlabel("Token Length", fontsize=14)

# 图例: 只显示方法和颜色
legend_handles = []
legend_labels = []

for i, method in enumerate(methods):
    # GPU: 用柱状矩形 Patch
    gpu_patch = mpatches.Patch(color=colors[i], alpha=0.7, label=f"{method} (GPU)")
    # Time: 用虚线 + marker Line2D
    time_line = plt.Line2D([0], [0], color=colors[i], lw=2, linestyle="--", marker="o",
                           label=f"{method} (Time)")

    legend_handles.extend([gpu_patch, time_line])
    legend_labels.extend([f"{method} (GPU)", f"{method} (Time)"])

fig.legend(handles=legend_handles, labels=legend_labels,
           loc="lower center", ncol=4, fontsize=14, bbox_to_anchor=(0.5, -0.08))

plt.tight_layout()

plt.savefig("GPU_TIME.pdf", bbox_inches="tight")
plt.show()
