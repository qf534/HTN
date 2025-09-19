import re
import requests, json, logging
import torch
import math

def fix_space_before_punct(sentence: str) -> str:
    """
    清理替换完成后的句子：
      1. 去除标点前的空格；
      2. 去除连字符两侧的空格；
      3. 合并多余的空格为单个空格。
    """
    sentence = re.sub(r'\s+([,\.!?;:])', r'\1', sentence)  # 标点前空格
    sentence = re.sub(r'\s*-\s*', r'-', sentence)  # 连字符两侧空格
    sentence = re.sub(r' {2,}', ' ', sentence)  # 多余空格
    return sentence


def replace_words(doc, tokens, selected_indices, kenlm_model, candidate_top_k=10, generator_func=None):
    """
    替换 selected_indices 对应的词，返回攻击后的句子和替换日志
    """
    modified = tokens.copy()
    logs = []  # 用于记录每次替换的信息

    if generator_func is None:
        generator_func = substitutes_generator

    for idx in selected_indices:
        original = tokens[idx]

        # 获取当前 token 所在句子 span
        sent_span = next(s for s in doc.sents if s.start <= idx < s.end)
        sent_tokens = [tok.text for tok in sent_span]
        local_idx = idx - sent_span.start

        # 候选词生成
        cands = generator_func(sent_tokens, local_idx, top_k=candidate_top_k)
        cands = [w for w in cands if w != original]
        if not cands:
            continue

        # KenLM 打分选择最优词
        best_score, best_word = float("-inf"), None
        for w in cands:
            tmp_tokens = sent_tokens.copy()
            tmp_tokens[local_idx] = w
            tmp_sent = fix_space_before_punct(" ".join(tmp_tokens))
            score = kenlm_model.score(tmp_sent)
            if score > best_score:
                best_score = score
                best_word = w

        if best_word is not None:
            modified[idx] = best_word

            # 记录替换信息
            logs.append({
                "index": idx,
                "original": original,
                "candidates": cands,
                "chosen": best_word
            })

    return fix_space_before_punct(" ".join(modified)), logs

