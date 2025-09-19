import os
import json
import logging
import argparse
import torch
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

from tqdm import tqdm
from datetime import datetime
from textflint.adapter import auto_dataset
from textattack.shared import AttackedText
from textattack.goal_function_results import GoalFunctionResultStatus
from pathlib import Path
from attack.methods.models import SurrogateDetectionModel
from utils.conf_util import setup_logger

# 全局变量，用于存储攻击器实例
attacking = None

# 初始化 logger
logger = logging.getLogger()
setup_logger(logger)

def init(model_path, attacking_method="dualir", n_gpu=1):
    global attacking

    # 当前进程索引，用于选择 GPU
    p_idx = int(mp.current_process()._identity[0])  # Pool 进程标号从 1 开始
    gpu_i = (p_idx - 1) * n_gpu

    # 构造 PyTorch 设备对象
    victim_dev = torch.device(f"cuda:{gpu_i + 0}")
    ppl_dev    = torch.device(f"cuda:{gpu_i + 1}") if n_gpu > 1 else victim_dev
    ta_dev     = torch.device(f"cuda:{gpu_i + 2}") if n_gpu > 2 else victim_dev

    # 设置环境变量，供后续调用使用
    os.environ["VICTIM_DEVICE"] = str(victim_dev)
    os.environ["PPL_DEVICE"]    = str(ppl_dev)
    os.environ["TA_DEVICE"]     = str(ta_dev)

    # TextAttack 设备设置
    from textattack.shared import utils
    utils.device = str(ta_dev)

    # 设置 PyTorch 显存按需增长 / 限制
    for dev in {victim_dev, ppl_dev, ta_dev}:
        torch.cuda.set_device(dev)
        try:
            # 限制每个进程最多占 90% 显存
            torch.cuda.set_per_process_memory_fraction(0.9, device=dev)
        except Exception:
            pass  # 如果当前 torch 版本不支持，忽略

    # 根据攻击方法动态加载 recipe
    if attacking_method == "dualir":
        from attack.recipes.rspmu_mlm_dualir import get_recipe
        recipe_func = get_recipe
    elif attacking_method == "wir":
        from attack.recipes.rspmu_mlm_wir import get_recipe
        recipe_func = get_recipe
    elif attacking_method == "greedy":
        from attack.recipes.rspmu_mlm_greedy import get_recipe
        recipe_func = get_recipe
    elif attacking_method == "no_pos":
        from attack.recipes.ablation_rsmu_mlm_dualir_no_pos import get_recipe
        recipe_func = get_recipe
    elif attacking_method == "no_use":
        from attack.recipes.ablation_rspm_mlm_dualir_no_use import get_recipe
        recipe_func = get_recipe
    elif attacking_method == "no_max_perturbed":
        from attack.recipes.ablation_rspu_mlm_dualir_no_max_perturbed import get_recipe
        recipe_func = get_recipe
    else:
        raise NotImplementedError(f"Not supported attacking recipe: {attacking_method}")

    label2id = {"gpt": 0, "human": 1, "tied": 1}
    target_cls = 1

    # 初始化代理检测模型和攻击器
    victim_model = SurrogateDetectionModel(model_path, batch_size=128, label2id=label2id)
    attacking = recipe_func(target_cls)
    attacking.init_goal_function(victim_model)

    logger.info(f"Initialized process {p_idx} on {victim_dev}, TA device {ta_dev}, using recipe {recipe_func}")


class MultiProcessingHelper:
    def __init__(self):
        self.total = None

    def __call__(self, data_samples, trans_save_path, func, workers=None, init_fn=None, init_args=None):
        self.total = len(data_samples)
        with mp.Pool(workers, initializer=init_fn, initargs=init_args) as pool, \
             tqdm(pool.imap(func, data_samples), total=self.total, dynamic_ncols=True) as pbar, \
             open(trans_save_path, "wt") as w_trans:
            for trans_res in pbar:
                if trans_res is None:
                    continue
                w_trans.write(json.dumps(trans_res.dump(), ensure_ascii=False) + "\n")


def init_sample_from_textattack(ori):
    text_input, label_str = ori.to_tuple()
    label_output = attacking.goal_function.model.label2id[label_str]
    attacked_text = AttackedText(text_input)
    if attacked_text.num_words <= 2:
        logger.debug(f"Skipping short text: {attacked_text.text}")
        return None
    goal_function_result, _ = attacking.goal_function.init_attack_example(attacked_text, label_output)
    return goal_function_result


def do_attack_one(ori_one):
    goal_function_result = init_sample_from_textattack(ori_one)
    if goal_function_result is None or goal_function_result.goal_status == GoalFunctionResultStatus.SKIPPED:
        return None
    result = attacking.attack_one(goal_function_result)
    train_data_dict = result.perturbed_result.attacked_text._text_input
    return ori_one.replace_fields(list(train_data_dict.keys()), list(train_data_dict.values()))


def main(args):
    # 加载测试数据
    sample_list = []
    with open(args.data_file, "r") as rf:
        for line in rf:
            r_j = json.loads(line.strip())
            sample_list.append({"x": r_j[args.text_key], "y": r_j[args.label_key]})

    # 使用 TextFlint 构造 dataset，如果不需要可直接使用 sample_list
    dataset = auto_dataset(sample_list, task="SA")

    dataset_name = Path(args.data_file).stem

    # 改成 attack_{数据集名称}_HMGC.jsonl
    output_file = os.path.join(
        args.output_dir,
        f"attack_{dataset_name}_Mis_HMGC.jsonl"
    )

    # —— 在这里确保目录存在 ——
    os.makedirs(args.output_dir, exist_ok=True)
    # 多进程执行攻击
    worker = MultiProcessingHelper()
    worker(
        dataset,
        output_file,
        func=do_attack_one,
        workers=args.num_workers,
        init_fn=init,
        init_args=(args.model_name_or_path, args.attacking_method, args.num_gpu_per_process),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--attacking_method", default="dualir")
    parser.add_argument("--num_gpu_per_process", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--text_key", default="text")
    parser.add_argument("--label_key", default="label")
    args = parser.parse_args()
    main(args)
