import argparse
import spacy
import kenlm
import json
from tqdm import tqdm
from replacer import replace_words
from words_selectors import random_based, entropy_based, maskdiff_based
from candidate_generators import lspg, api
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM

def load_selector(method, device=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    if method == "random":
        return random_based.select_indices
    elif method == "entropy":
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained("opt-2.7b")
        model = AutoModelForCausalLM.from_pretrained("opt-2.7b").to(device)
        model.eval()

        def selector(doc, tokens, replace_ratio, **kwargs):
            return entropy_based.select_indices(
                doc, tokens, replace_ratio,
                tokenizer=tokenizer,
                model=model,
                device=device
            )

        return selector

    elif method == "maskdiff":
        return maskdiff_based.select_indices
    else:
        raise ValueError("Unknown method: choose from [random, entropy, maskdiff]")

def load_generator(generator_name):
    if generator_name == "lspg":
        return lspg.substitutes_generator
    elif generator_name == "api":
        return api.substitutes_generator
    else:
        raise ValueError("Unknown generator: choose from [lspg, api]")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["random","entropy","maskdiff"], default="entropy")
    parser.add_argument("--generator", type=str, choices=["lspg","api"], default="lspg",
                        help="Candidate_word_generation_method")
    parser.add_argument("--input_file", type=str, default="../datasets/original/STEM.json")
    parser.add_argument("--kenlm_path", type=str, default="merge_Ngram.bin")
    parser.add_argument("--candidate_top_k", type=int, default=10)
    parser.add_argument("--replace_ratio", type=float, default=0.15)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_file", type=str, default="./attacked_STEM.json")
    parser.add_argument("--log_file", type=str, default="./logs.json",
                        help="File to store replacement logs separately")
    args = parser.parse_args()

    # NLP 和 KenLM
    nlp = spacy.load("en_core_web_sm")
    kenlm_model = kenlm.Model(args.kenlm_path)

    # 读取数据集
    with open(args.input_file, encoding="utf-8") as f:
        dataset = json.load(f)
    sampled = dataset.get("sampled", [])
    original = dataset.get("original", [])
    if args.max_samples:
        sampled = sampled[:args.max_samples]
        original = original[:args.max_samples]

    # 选择策略和生成器
    selector = load_selector(args.method)
    generator_func = load_generator(args.generator)

    attacked_list = []
    logs_list = []
    # 处理句子，显示进度条
    for sent in tqdm(sampled, desc="Processing sentences"):
        doc = nlp(sent)
        tokens = [tok.text for tok in doc]

        # 选词
        selected_indices = selector(
            doc, tokens, args.replace_ratio,
            kenlm_model=kenlm_model
        )

        # 替换词
        attacked = replace_words(
            doc, tokens, selected_indices,
            kenlm_model=kenlm_model,
            candidate_top_k=args.candidate_top_k,
            generator_func=generator_func
        )
        attacked_list.append(attacked)

    # 保存结果，保留 original + sampled
    output_data = {"original": original, "sampled": attacked_list}
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    with open(args.log_file, "w", encoding="utf-8") as f:
        json.dump(logs_list, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
