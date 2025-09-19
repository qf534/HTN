# Copyright (c) Guangsheng Bao.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from .model import load_tokenizer, load_model
from .prob_estimator import ProbEstimator
from .detector import Detector


class Fast_Detect_GPT(Detector):
    def __init__(
        self,
        reference_model_name,
        scoring_model_name,
        dataset,
        ref_path,
        device_ref="cuda:0",
        device_score="cuda:1",
        cache_dir="../cache",
    ):
        self.reference_model_name = reference_model_name
        self.scoring_model_name = scoring_model_name
        self.dataset = dataset
        self.ref_path = ref_path
        self.device_ref = device_ref
        self.device_score = device_score
        self.cache_dir = cache_dir


        # 加载评分模型
        self.scoring_tokenizer = load_tokenizer(
            self.scoring_model_name, self.dataset, self.cache_dir
        )
        self.scoring_model = load_model(
            self.scoring_model_name, self.device_score, self.cache_dir
        )
        self.scoring_model.eval()

        # 加载采样模型（reference model）
        if self.reference_model_name != self.scoring_model_name:
            self.reference_tokenizer = load_tokenizer(
                self.reference_model_name, self.dataset, self.cache_dir
            )
            self.reference_model = load_model(
                self.reference_model_name, self.device_ref, self.cache_dir
            )
            self.reference_model.eval()

        self.criterion_fn = self.get_sampling_discrepancy_analytic
        self.prob_estimator = ProbEstimator(self.ref_path)

    def get_sampling_discrepancy_analytic(self, logits_ref, logits_score, labels):
        assert logits_ref.shape[0] == 1
        assert logits_score.shape[0] == 1
        assert labels.shape[0] == 1
        if logits_ref.size(-1) != logits_score.size(-1):
            vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
            logits_ref = logits_ref[:, :, :vocab_size]
            logits_score = logits_score[:, :, :vocab_size]

        labels = (
            labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
        )
        lprobs_score = torch.log_softmax(logits_score, dim=-1)
        probs_ref = torch.softmax(logits_ref, dim=-1)
        log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
        mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
        var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(
            mean_ref
        )
        discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(
            dim=-1
        ).sqrt()
        discrepancy = discrepancy.mean()
        return discrepancy.item()

    def get_tokens(self, query):
        tokenized = self.scoring_tokenizer(
            query, return_tensors="pt", padding=True, return_token_type_ids=False
        ).to(self.device_score)
        input, output = tokenized["input_ids"].detach().cpu().numpy().tolist(), []
        for i in input:
            token = self.scoring_tokenizer.convert_ids_to_tokens(i)
            output.append(token)
        return output[0]

    def run(self, query, indexes=None):
        # tokenize for scoring model
        tokenized_score = self.scoring_tokenizer(
            query, return_tensors="pt", padding=True, return_token_type_ids=False
        ).to(self.device_score)
        labels = tokenized_score.input_ids[:, 1:]

        with torch.no_grad():
            if indexes:
                mask = torch.ones_like(tokenized_score["input_ids"])
                for i in range(len(mask[0])):
                    mask[0][i] = 0 if i in indexes else 1
                tokenized_score["attention_mask"] = mask.to(self.device_score)

            logits_score = self.scoring_model(**tokenized_score).logits[:, :-1]

            # reference model
            if self.reference_model_name == self.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized_ref = self.reference_tokenizer(
                    query,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False,
                ).to(self.device_ref)

                assert torch.all(
                    tokenized_ref.input_ids[:, 1:].cpu() == labels.cpu()
                ), "Tokenizer mismatch."

                logits_ref = self.reference_model(**tokenized_ref).logits[:, :-1]

                # 将 reference logits 转到 scoring model 所在设备，避免跨设备计算错误
                logits_ref = logits_ref.to(self.device_score)

            # 最终 labels 也确保在 scoring model 所在设备上
            crit = self.criterion_fn(logits_ref, logits_score, labels.to(self.device_score))

        llm_likelihood = self.prob_estimator.crit_to_prob(crit)
        human_likelihood = 1 - llm_likelihood

        return llm_likelihood, human_likelihood, crit

    def llm_likelihood(self, query, indexes=None):
        return self.run(query, indexes)[0]

    def human_likelihood(self, query: str, indexes=None):
        return self.run(query, indexes)[1]

    def crit(self, query: str, indexes=None):
        return self.run(query, indexes)[2]


if __name__ == "__main__":
    fast_detect_gpt = Fast_Detect_GPT(
        "gpt-j-6b",
        "gpt-neo-2.7B",
        "Social_Sciences",
        "./detectors/fast_detecte_gpt/*sampling_discrepancy.json",
        "cuda:0", "cuda:1"
    )

    original = (
        "In a move that could have a serious impact on the Russian economy, the government is considering freezing domestic energy prices. This decision risks undermining recent oil industry reforms and the broader economic recovery that has been taking place in the country.The potential freeze on energy prices would aim to reduce costs for consumers struggling with the impact of the Covid-19 pandemic.",
        "By freezing energy prices, the government wants to help consumers survive the influenza epidemic, thus weakening the energy reforms that have been introduced. In a decision that may have a severe impact on the Russian economy, the government is mulling a decision to freeze energy prices. Such a decision may endanger the oil reforms and the overall economic recovery that has been going on in Russia. It may also help the victims of the influenza epidemic but it may also put the economy at risk.",
        "In a move that could have a serious bearing in the Russian economy, the government is considering freezing domestic energy prices. This decision risks undermining recent oil industry reforms and the broader economic recovery that has been taking shape in the country.The potential freeze on energy rates would aim to reduce costs for consumers struggling with the impact because the Covid-19 pandemic.",
        "In a state that cannot have a serious effects on the Russian society, the un is considering freezing domestic energy prices. this decision risks undermining recent oil industry reforms and the broader economic recovery that has been taken place in the eu.that potential consequences on energy prices would aim to lowers costs for consumers struggling with the impact of the Covid-19 pandemic.",
        "In a decision that could be a serious impact on the local economy, the government is considering freezing domestic oil prices. This decision risks undermining our oil industry reforms and the broader economic recovery that has been taking part in the world. The potential freeze on fuel prices would attempt to reduce costs for consumers struggling with the effects of the Covid-19 pandemic."
    )

    print(fast_detect_gpt.llm_likelihood(original))
