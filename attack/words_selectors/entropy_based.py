import math
import string
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def select_indices(doc, tokens, replace_ratio, tokenizer=None, model=None, device=None, **kwargs):
    """Next-token generation: 选预测概率最低的 token"""

    if tokenizer is None or model is None:
        raise ValueError("This method needs tokenizer and model (OPT causal LM)")
    if replace_ratio is None:
        raise ValueError("replace_ratio must be specified")

    # 候选词索引（过滤掉专有名词、标点、太短的 token）
    candidate_indices = [
        tok.i for tok in doc
        if tok.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]
           and tok.pos_ != "PROPN"
           and len(tok.text) > 2
           and tok.text not in string.punctuation
    ]
    if not candidate_indices:
        return []

    # ======== 用 OPT 计算每个 token 的预测概率 ========
    input_ids = tokenizer(" ".join(tokens), return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
    with torch.no_grad():
        logits = model(input_ids).logits  # shape [1, seq_len, vocab_size]
        probs = torch.nn.functional.softmax(logits, dim=-1)

    def get_prob(tokens, idx):
        # 第 idx 个 token 的概率 = 模型在 idx-1 位置预测它的概率
        token_id = tokenizer.convert_tokens_to_ids(tokens[idx])
        if idx == 0:  # 第一个 token 没有前文
            return 1.0
        return probs[0, idx-1, token_id].item()

    probs_list = [(idx, get_prob(tokens, idx)) for idx in candidate_indices]

    # 按概率升序（不确定性高的优先）
    probs_sorted = sorted(probs_list, key=lambda x: x[1])

    # 选前 replace_ratio 部分
    total_to_replace = max(1, math.ceil(len(tokens) * replace_ratio))
    final_selection = [idx for idx, _ in probs_sorted[:total_to_replace]]

    return final_selection
