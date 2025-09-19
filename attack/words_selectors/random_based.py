import random
import string

def select_indices(doc, tokens, replace_ratio, **kwargs):
    """随机选择 15% 左右的词"""
    candidate_indices = [
        tok.i for tok in doc
        if tok.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]
           and tok.pos_ != "PROPN"
           and len(tok.text) > 2
           and tok.text not in string.punctuation
    ]

    if not candidate_indices:
        return []

    total_to_replace = max(1, int(len(tokens) * replace_ratio))
    return random.sample(candidate_indices, min(total_to_replace, len(candidate_indices)))
