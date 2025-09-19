# Adapted from https://github.com/cvlab-columbia/RaidarLLMDetect
import json
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(
    base_url="https://api.openai-proxy.org/v1",
    api_key="sk-XmGVCSM8OERssBmOHPbZX6Ri3a12g1WU6KlkGncYCu089hzR"
)


def GPT_self_prompt(prompt_str, content_to_be_detected):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"{prompt_str}: {content_to_be_detected}",
            }
        ],
    )

    return completion.choices[0].message.content


prompt_list = [
    "Revise this with your best effort",
    "Help me polish this",
    "Rewrite this for me",
    "Make this fluent while doing minimal change",
    "Refine this for me please",
    "Concise this for me and keep all the information",
    "Improve this in GPT way",
]


def rewrite_json(input_json, prompt_list, output_dir):
    all_data = []
    for cc in tqdm(range(len(input_json))):
        tmp_dict = {}
        tmp_dict["input"] = input_json[cc]
        for ep in prompt_list:
            tmp_dict[ep] = GPT_self_prompt(ep, tmp_dict["input"])
        all_data.append(tmp_dict)
        with open(output_dir, "w") as file:
            json.dump(all_data, file, indent=4)


human_GPT_dir  = '/home/qf/RAFT/main/New/ablation_STEM_0.1.json'
GPT_attack_dir = '/home/qf/RAFT/main/New/ablation_STEM_0.1.json'
output_dir = '/home/qf/RAFT/raidar_llm_detect/results/New/ablation_STEM_0.1_GPT.json'
with open(human_GPT_dir, "r") as fp:
    fp = json.load(fp)
    human, GPT = fp["original"], fp["sampled"]

with open(GPT_attack_dir, "r") as fp:
    GPT_attack = json.load(fp)

rewrite_json(GPT, prompt_list, output_dir)
