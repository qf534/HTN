import re
import requests, json, logging

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


def substitutes_generator(text_words, mask_word_index, top_k=None):
    """
    调用外部接口获取候选替换词
    """
    prefix = text_words[:mask_word_index]
    prefix_s = " ".join(prefix).strip()
    target = text_words[mask_word_index]
    suffix = text_words[mask_word_index + 1:]
    suffix_s = " ".join(suffix).strip()

    try:
        response = requests.get(
            f"http://10.135.128.18:8085/get_candidates?prefix={prefix_s}&target={target}&suffix={suffix_s}",
            timeout=30
        )
        if response.status_code == 200:
            j = response.json()
            candidates = j.get("substitutes", [])
            if top_k is not None:
                candidates = candidates[:top_k]
            return candidates
        else:
            logging.warning(f"ciwater error: {response.status_code}")
            return []
    except Exception as e:
        logging.warning(f"候选词生成失败: {str(e)}")
        return []