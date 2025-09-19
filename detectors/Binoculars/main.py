import os
import json
from sklearn.metrics import roc_auc_score
from binoculars import Binoculars

root_dir = ""

bino = Binoculars()

for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith(".json"):
            file_path = os.path.join(dirpath, filename)
            try:
                # 读取 JSON 文件
                with open(file_path, "r", encoding="utf-8") as f:
                    dataset = json.load(f)

                original_texts = dataset.get("original", [])
                sampled_texts = dataset.get("sampled", [])
                labels = [0] * len(original_texts) + [1] * len(sampled_texts)
                all_texts = original_texts + sampled_texts

                if not all_texts:
                    print(f"[跳过] 空文件: {file_path}")
                    continue

                scores = [-bino.compute_score(text) for text in all_texts]

                auroc = roc_auc_score(labels, scores)
                print(f"[{file_path}] AUROC: {auroc:.4f}")

            except Exception as e:
                print(f"[错误] 处理 {file_path} 时出错: {e}")
