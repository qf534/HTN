import json
import torch
import torch.nn.functional as F
import transformers
import numpy as np
from openai import OpenAI
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm
# ---------------------- Configurations ----------------------
client = OpenAI(
    base_url="https://api.openai-proxy.org/v1",
    api_key="sk-XmGVCSM8OERssBmOHPbZX6Ri3a12g1WU6KlkGncYCu089hzR"
)

device = "cuda:1"
radar_model_id = "/homes/qf/models/RADAR-Vicuna-7B"
input_file = "/home/qf/RAFT/main/Latest/attacked_STEM_ours.json"  # JSON file with {"original": [...], "sampled": [...]} format

# ---------------------- Load Data ----------------------
with open(input_file, "r") as f:
    data = json.load(f)
human_texts = data["original"]
ai_texts = data["sampled"]

# ---------------------- Rewriting via GPT ----------------------
def gpt_paraphrase(text):
    messages = [
        {"role": "system", "content": "Enhance the word choices in the sentence to sound more like that of a human."},
        {"role": "user", "content": text}
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response.choices[0].message.content.strip()

print("Rewriting AI texts using GPT-3.5-turbo...")
# paraphrased_ai_texts = []
# for t in tqdm(ai_texts, desc="GPT Rewriting"):
#     rewritten = gpt_paraphrase(t)
#     paraphrased_ai_texts.append(rewritten)

# ---------------------- Load RADAR Detector ----------------------
print("Loading RADAR detector...")
detector = transformers.AutoModelForSequenceClassification.from_pretrained(radar_model_id)
tokenizer = transformers.AutoTokenizer.from_pretrained(radar_model_id)
detector.eval()
detector.to(device)

# ---------------------- Detection ----------------------
def get_llm_probabilities(texts):
    all_probs = []
    batch_size = 8  # 适当设置 batch size
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = detector(**inputs).logits
            probs = F.log_softmax(logits, dim=-1)[:, 0].exp().tolist()  # Class 0 = AI-generated
            all_probs.extend(probs)
    return all_probs


print("Detecting probabilities for each category...")
human_preds = get_llm_probabilities(human_texts)
ai_preds = get_llm_probabilities(ai_texts)
# paraphrased_ai_preds = get_llm_probabilities(paraphrased_ai_texts)

# ---------------------- Compute AUROC ----------------------
def get_roc_metrics(human_preds, ai_preds):
    fpr, tpr, _ = roc_curve([0] * len(human_preds) + [1] * len(ai_preds), human_preds + ai_preds)
    roc_auc = auc(fpr, tpr)
    return float(roc_auc)

print("\nResults:")
print("W/O Paraphrase Detection AUROC:", get_roc_metrics(human_preds, ai_preds))
# print("W/ Paraphrase Detection AUROC:", get_roc_metrics(human_preds, paraphrased_ai_preds))
