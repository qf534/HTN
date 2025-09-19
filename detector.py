import json
import numpy as np
import torch
import argparse
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from sklearn.metrics import roc_auc_score
from detectors.baselines import Baselines
from detectors.ghostbuster import Ghostbuster
from detectors.detect_gpt import Detect_GPT
from detectors.fast_detect_gpt import Fast_Detect_GPT
from detectors.roberta_gpt2_detector import GPT2RobertaDetector
from transformers import GPTNeoForCausalLM
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

class AIGCDetector:
    def __init__(self, detector, target_detector_device="cpu"):
        self.detector = detector
        self.target_detector_device = target_detector_device
        self.detector_model = self.load_detector()

    def load_detector(self):
        print(f"Loading detector: {self.detector}")
        if self.detector == "dgpt":
            return Detect_GPT(
                "./detectors/dgpt/*perturbation_100.json",
                0.3, 1.0, 2, 10,
                "gpt2-xl", "t5-3b",
                device0=self.target_detector_device,
                device1=self.target_detector_device,
            )
        elif self.detector == "fdgpt":
            return Fast_Detect_GPT(
                "gpt-j-6b",
                "gpt-neo-2.7B",
                "Social_Sciences",
                "./detectors/fast_detecte_gpt/*sampling_discrepancy.json",
                "cuda:0", "cuda:1",
            )
        elif self.detector == "ghostbuster":
            return Ghostbuster()
        elif self.detector == "logrank":
            return Baselines("logrank", "gpt-neo-2.7B", device=self.target_detector_device)
        elif self.detector == "logprob":
            return Baselines("likelihood", "gpt-neo-2.7B", device=self.target_detector_device)
        elif self.detector == "roberta-base":
            return GPT2RobertaDetector("roberta-base", self.target_detector_device, "./assets/detector-base.pt")
        elif self.detector == "roberta-large":
            return GPT2RobertaDetector("roberta-large", self.target_detector_device, "./assets/detector-large.pt")
        elif self.detector == "chatgpt-roberta":
            model_path = "chatgpt-detector-roberta"
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.to(self.target_detector_device)
            model.eval()
            return {"model": model, "tokenizer": tokenizer}
        else:
            raise ValueError(f"Unsupported detector: {self.detector}")

    def detect(self, text):
        try:
            if self.detector in ["roberta-base", "roberta-large", "ghostbuster"]:
                score = self.detector_model.crit(text)
                prediction = 1 if score > 0.5 else 0
                confidence = float(score)
                return prediction, confidence

            elif self.detector == "chatgpt-roberta":
                tokenizer = self.detector_model["tokenizer"]
                model = self.detector_model["model"]
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
                inputs = {k: v.to(self.target_detector_device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = softmax(outputs.logits, dim=-1).squeeze()
                    confidence = probs[1].item()  # 机器概率
                    prediction = 1 if confidence > 0.5 else 0
                return prediction, confidence

            else:
                prediction, _, score = self.detector_model.run(text)
                confidence = float(score)
                return prediction, confidence

        except Exception as e:
            print(f"Detection failed for text: {text}. Error: {str(e)}")
            return 0, 0.5


def calculate_auroc(labels, scores):
    return roc_auc_score(labels, scores)


def calculate_perplexity(text, model, tokenizer, device):
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    max_length = model.config.max_position_embeddings
    stride = 512
    nlls = []
    seq_len = input_ids.size(1)
    for i in range(0, seq_len, stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, seq_len)
        trg_len = end_loc - i
        input_ids_slice = input_ids[:, begin_loc:end_loc]
        target_ids = input_ids_slice.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids_slice, labels=target_ids)
            log_likelihood = outputs.loss * trg_len
        nlls.append(log_likelihood)
    total_nll = torch.stack(nlls).sum()
    avg_nll = total_nll / seq_len
    return torch.exp(avg_nll).item()


def calculate_tpr_at_fpr(labels, scores, target_fpr=0.05):
    labels = np.array(labels)
    scores = np.array(scores)
    # 计算 FPR 阈值
    neg_scores = scores[labels == 0]
    thresh = np.percentile(neg_scores, 100 * (1 - target_fpr))
    tpr = np.mean(scores[labels == 1] >= thresh)
    return tpr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", type=str, default="logrank")
    parser.add_argument(
        "--json_path",
        type=str,
        default="./datasets/Ours/attacked_Social_Sciences_ours.json",
        help="Path to dataset json file"
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()


    detector_name = args.detector
    target_detector_device = args.device

    detector = AIGCDetector(detector=detector_name, target_detector_device=target_detector_device)

    with open(args.json_path) as f:
        data = json.load(f)

    combined_data = []
    for text in data.get("original", []):
        combined_data.append({"text": text, "label": 0})
    for text in data.get("sampled", []):
        combined_data.append({"text": text, "label": 1})

    # ppl 仅在 logrank/logprob 检测器启用
    ppl_model, ppl_tokenizer, device = None, None, None
    enable_ppl = detector_name in ["logrank", "logprob"]
    if enable_ppl:
        device = target_detector_device if torch.cuda.is_available() else "cpu"
        ppl_model = GPTNeoForCausalLM.from_pretrained("gpt-neo-2.7B")
        ppl_model.to(device)
        ppl_model.eval()
        ppl_tokenizer = GPT2TokenizerFast.from_pretrained("gpt-neo-2.7B")

    labels, scores, predictions, perplexities = [], [], [], []
    scores_label_1, scores_label_0 = [], []

    for item in tqdm(combined_data, desc="Detecting", unit="text"):
        text = item["text"]
        true_label = item["label"]
        prediction, confidence = detector.detect(text)
        if prediction is not None:
            labels.append(true_label)
            scores.append(confidence)
            predictions.append(prediction)

            if true_label == 1:
                scores_label_1.append(confidence)
                if enable_ppl:
                    try:
                        ppl = calculate_perplexity(text, ppl_model, ppl_tokenizer, device)
                        perplexities.append(ppl)
                    except Exception as e:
                        print(f"Perplexity calculation failed for text: {text}. Error: {str(e)}")
            else:
                scores_label_0.append(confidence)

    print(f"Confidence distribution: Min={min(scores):.4f}, Max={max(scores):.4f}, "
          f"Mean={np.mean(scores):.4f}, Std={np.std(scores):.4f}")

    try:
        auroc = calculate_auroc(labels, scores)
        print(f"AUROC: {auroc:.4f}")
    except ValueError as e:
        print(f"Error calculating AUROC: {str(e)}")

    if enable_ppl and perplexities:
        avg_perplexity = np.mean(perplexities)
        print(f"Average Perplexity for label==1 texts: {avg_perplexity:.4f}")

    if scores_label_1:
        avg_score_label_1 = np.mean(scores_label_1)
        print(f"Average confidence score for label=1 (machine-generated): {avg_score_label_1:.4f}")

    if scores_label_0:
        avg_score_label_0 = np.mean(scores_label_0)
        print(f"Average confidence score for label=0 (human-written): {avg_score_label_0:.4f}")


if __name__ == "__main__":
    main()
