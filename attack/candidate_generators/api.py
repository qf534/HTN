#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import time
import logging
from openai import OpenAI

# 配置日志
logging.basicConfig(level=logging.WARNING)

# 初始化 OpenAI 客户端
client = OpenAI(
    base_url="",
    api_key=""
)

def openai_backoff(**kwargs):
    retries, wait_time = 0, 10
    while retries < 10:
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            logging.warning(f"[Retry] OpenAI API 调用失败: {e}")
            time.sleep(wait_time)
            wait_time *= 2
            retries += 1
    return None

def generate_text(query):
    response = openai_backoff(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": query}]
    )
    if response:
        return response.choices[0].message.content
    return ""

def substitutes_generator(text_words, mask_word_index, top_k=10):
    """
    调用大模型生成候选词
    接口与 local_generator 保持一致
    """
    target_word = text_words[mask_word_index]

    # 构造 prompt，把 target 用方括号标记
    sentence = " ".join(text_words)
    query = (
        f"Given the input sentence, highlight a word using brackets. "
        f"List {top_k} alternative words for it. Output words only.\n"
        f"{sentence.replace(target_word, '[' + target_word + ']')}"
    )

    try:
        output = generate_text(query)
        if not isinstance(output, str):
            logging.warning(f"[Warning] generate_text returned non-string: {output}")
            return []

        # 只提取单词，去掉非字母字符
        predicted_words = re.findall(r"\b[a-zA-Z]+\b", output)
        predicted_words = [w for w in predicted_words if w.lower() != target_word.lower()]
        return predicted_words[:top_k]

    except Exception as e:
        logging.warning(f"[Error] API substitutes_generator failed: {e}")
        return []
