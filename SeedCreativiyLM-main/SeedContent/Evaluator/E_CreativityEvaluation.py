from typing import Dict, List
from tqdm import tqdm

import torch
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from SeedContent.Pipeline.P_MultistageGeneration import P_MultiStageGeneration
from SeedContent.Pipeline.P_ControlGroupGeneration import P_ControlGroupGeneration
from SeedContent.Evaluator.E_ScoreEvaluation import E_ScoreEvaluation
from SeedContent.Prompts.PromptTemplates import PromptTemplates
from SeedCore.Config.ConfigInstance import ConfigSubsystem
from SeedCore.LLM.LLMEvaluator import LLMEvaluator
from SeedCore.SharedFunctionLibrary import LOG_TEXT


class E_CreativityEvaluation(LLMEvaluator):
    """
    LitBench 데이터셋을 활용한 시드 아이디어 창의성 평가 클래스입니다.

    평가 모델 옵션:
    1. HuggingFace 모델 (기본): dmnsh/Qwen3-4b-W0-GenRM
    2. OpenAI 모델: gpt-4o, gpt-4-turbo 등 (use_openai_evaluator=true)
    """

    def __init__(self):
        super().__init__()

        self.generator = P_MultiStageGeneration()
        self.control_group_generator = P_ControlGroupGeneration()
        self.tokenizer = None
        self.model = None
        self.openai_client = None

    def _get_config(self):
        return ConfigSubsystem.get_config("config/Evaluator/E_CreativityEvaluation.json")

    def _is_openai_model(self, model_name: str) -> bool:
        """OpenAI 모델인지 확인"""
        openai_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "o1", "o1-mini"]
        return any(model_name.startswith(m) for m in openai_models)

    def _load_model(self):
        """평가 모델을 로드합니다."""
        eval_config = self._get_config()
        model_name = eval_config.get("model_name")

        # OpenAI 모델인 경우
        if self._is_openai_model(model_name):
            if self.openai_client is None:
                api_key = ConfigSubsystem.get_config("config/excluded/credentials.json").get("OPENAI_API_KEY")
                self.openai_client = openai.OpenAI(api_key=api_key)
                LOG_TEXT(f"OpenAI client initialized for evaluation model: {model_name}", "GREEN", "INFO")
            return

        # HuggingFace 모델인 경우
        if self.model is not None:
            return

        LOG_TEXT("Loading creativity evaluation model...", "GREEN", "INFO")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        LOG_TEXT("Tokenizer loaded.", "GREEN", "INFO")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        LOG_TEXT("Creativity evaluation model loaded.", "GREEN", "INFO")

    def _evaluate_with_model(self, eval_question: str, max_new_tokens: int) -> str:
        """평가 모델로 응답 생성"""
        eval_config = self._get_config()
        model_name = eval_config.get("model_name")

        if self._is_openai_model(model_name):
            # OpenAI API 호출
            response = self.openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": eval_question}],
                max_tokens=max_new_tokens,
                temperature=0.3
            )
            return response.choices[0].message.content
        else:
            # HuggingFace 모델
            inputs = self.tokenizer(eval_question, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def load_eval_dataset(self):
        """평가용 데이터셋을 로드합니다."""
        LOG_TEXT("Loading evaluation dataset...", "GREEN", "INFO")

        eval_config = self._get_config()
        dataset_name = eval_config.get("dataset_name")

        dataset = load_dataset(dataset_name)
        test_data = dataset["train"]

        filtered_data = test_data.select_columns(["prompt", "chosen_story", "rejected_story"])

        LOG_TEXT("Evaluation dataset loaded.", "GREEN", "INFO")

        return filtered_data

    def evaluate(self,
                 num_stages: int = 2,
                 include_baseline: bool = False,
                 model_override: str = None,
                 compute_scores: bool = False,
                 batch_size: int = 8,
                 prefix_prompt: str = ""
                 ) -> Dict:
        """
        창의성 평가를 수행합니다.

        Args:
            num_stages: 생성에 사용할 스테이지 수
            include_baseline: Control group baseline 비교 포함 여부
            model_override: Generator 모델 오버라이드 (예: "gpt-4o", "gpt-4-turbo")
            compute_scores: PPL과 Diversity 점수 계산 여부
            batch_size: 배치 평가 크기 (현재는 순차 처리만 지원)

        Returns:
            평가 결과 딕셔너리 (win_count, total_count, win_rate, results, baseline_results, avg_ppl, avg_diversity)
        """
        self._load_model()

        eval_config = self._get_config()
        eval_sample_size = eval_config.get("eval_sample_size")
        max_new_tokens = eval_config.get("max_new_tokens")
        model_override = model_override or eval_config.get("generator_model_override")

        score_evaluator = E_ScoreEvaluation() if compute_scores else None
        dataset = self.load_eval_dataset().select(range(eval_sample_size))

        results, baseline_results = [], []
        win_count, baseline_win_count = 0, 0
        ppl_scores, diversity_scores = [], []
        baseline_ppl_scores, baseline_diversity_scores = [], []

        # Process in batches
        for batch_start in tqdm(range(0, eval_sample_size, batch_size), desc="Evaluating creativity"):
            batch_end = min(batch_start + batch_size, eval_sample_size)
            batch = dataset.select(range(batch_start, batch_end))

            for entry in batch:
                prompt = entry['prompt']
                chosen_story = entry['chosen_story']

                # Attach an evaluation instruction to the prompt
                prompt = prefix_prompt + prompt

                # ===== SEED-BASED GENERATION AND EVALUATION =====
                # Generate using multistage pipeline
                # If we need scores, request logprobs from generation
                if compute_scores:
                    stage_outputs, stage_responses = self.generator.generate(
                        prompt, num_stages, model_override, return_logprobs=True
                    )
                else:
                    stage_outputs = self.generator.generate(prompt, num_stages, model_override)
                    stage_responses = None

                # Concatenate all stage outputs (stage1 + stage2 + ... + stageN)
                generated_story = "".join(stage_outputs)

                # Evaluate creativity
                eval_question = PromptTemplates.get_creativity_evaluation_prompt(chosen_story, generated_story)
                eval_response = self._evaluate_with_model(eval_question, max_new_tokens)
                preferred = self._parse_preferred(eval_response)
                is_win = preferred == "B"

                if is_win:
                    win_count += 1

                result = {
                    "prompt": prompt,
                    "chosen_story": chosen_story,
                    "generated_story": generated_story,
                    "stage_outputs": stage_outputs,
                    "eval_response": eval_response,
                    "preferred": preferred,
                    "is_win": is_win
                }

                # Compute PPL and Diversity for seed-based generation
                if compute_scores and stage_responses:
                    combined_logprobs = self._combine_stage_logprobs(stage_responses)
                    ppl_score, diversity_score, logprobs_dict = score_evaluator.evaluate(response_choice=combined_logprobs)

                    ppl_scores.append(ppl_score)
                    diversity_scores.append(diversity_score)
                    result["ppl_score"] = ppl_score
                    result["diversity_score"] = diversity_score
                    result["logprobs_data"] = logprobs_dict  # Store logprobs data

                results.append(result)

                # ===== BASELINE EVALUATION =====
                if include_baseline:
                    baseline_response_choice = None
                    if compute_scores:
                        baseline_response_choice = self.control_group_generator.generate(
                            prompt, logprobs=True, return_full_response=True
                        )
                        baseline_story = baseline_response_choice.message.content if baseline_response_choice else None
                    else:
                        baseline_story = self.control_group_generator.generate(prompt)

                    if baseline_story:
                        baseline_eval_question = PromptTemplates.get_creativity_evaluation_prompt(
                            chosen_story, baseline_story
                        )
                        baseline_eval_response = self._evaluate_with_model(baseline_eval_question, max_new_tokens)
                        baseline_preferred = self._parse_preferred(baseline_eval_response)
                        baseline_is_win = baseline_preferred == "B"

                        if baseline_is_win:
                            baseline_win_count += 1

                        baseline_result = {
                            "prompt": prompt,
                            "chosen_story": chosen_story,
                            "generated_story": baseline_story,
                            "eval_response": baseline_eval_response,
                            "preferred": baseline_preferred,
                            "is_win": baseline_is_win
                        }

                        # Compute PPL and Diversity for baseline
                        if compute_scores and baseline_response_choice:
                            baseline_ppl, baseline_div, baseline_logprobs_dict = score_evaluator.evaluate(
                                response_choice=baseline_response_choice
                            )
                            baseline_ppl_scores.append(baseline_ppl)
                            baseline_diversity_scores.append(baseline_div)
                            baseline_result["ppl_score"] = baseline_ppl
                            baseline_result["diversity_score"] = baseline_div
                            baseline_result["logprobs_data"] = baseline_logprobs_dict  # Store logprobs data

                        baseline_results.append(baseline_result)

        # Calculate metrics
        win_rate = win_count / len(results) if results else 0.0
        baseline_win_rate = baseline_win_count / len(baseline_results) if baseline_results else 0.0
        avg_ppl = sum(ppl_scores) / len(ppl_scores) if ppl_scores else None
        avg_diversity = sum(diversity_scores) / len(diversity_scores) if diversity_scores else None
        baseline_avg_ppl = sum(baseline_ppl_scores) / len(baseline_ppl_scores) if baseline_ppl_scores else None
        baseline_avg_diversity = sum(baseline_diversity_scores) / len(baseline_diversity_scores) if baseline_diversity_scores else None

        LOG_TEXT(f"\n===== EVALUATION RESULTS =====", "GREEN", "INFO")
        LOG_TEXT(f"Seed-based approach:", "CYAN", "INFO")
        LOG_TEXT(f"  - Win rate: {win_rate:.2%} ({win_count}/{len(results)})", "CYAN", "INFO")
        if avg_ppl is not None:
            LOG_TEXT(f"  - Average PPL: {avg_ppl:.4f}", "CYAN", "INFO")
        if avg_diversity is not None:
            LOG_TEXT(f"  - Average Diversity: {avg_diversity:.4f}", "CYAN", "INFO")

        if include_baseline:
            LOG_TEXT(f"Baseline approach:", "YELLOW", "INFO")
            LOG_TEXT(f"  - Win rate: {baseline_win_rate:.2%} ({baseline_win_count}/{len(baseline_results)})", "YELLOW", "INFO")
            if baseline_avg_ppl is not None:
                LOG_TEXT(f"  - Average PPL: {baseline_avg_ppl:.4f}", "YELLOW", "INFO")
            if baseline_avg_diversity is not None:
                LOG_TEXT(f"  - Average Diversity: {baseline_avg_diversity:.4f}", "YELLOW", "INFO")

        return {
            "win_count": win_count,
            "total_count": len(results),
            "win_rate": win_rate,
            "results": results,
            "baseline_win_count": baseline_win_count if include_baseline else None,
            "baseline_win_rate": baseline_win_rate if include_baseline else None,
            "baseline_results": baseline_results if include_baseline else None,
            "avg_ppl": avg_ppl,
            "avg_diversity": avg_diversity,
            "baseline_avg_ppl": baseline_avg_ppl,
            "baseline_avg_diversity": baseline_avg_diversity
        }

    def _combine_stage_logprobs(self, stage_responses: List):
        """
        Combine logprobs from multiple stages into a single response object.
        """
        class CombinedResponse:
            def __init__(self):
                self.logprobs = None

        class CombinedLogprobs:
            def __init__(self):
                self.content = []

        combined = CombinedResponse()
        combined.logprobs = CombinedLogprobs()

        for stage_response in stage_responses:
            combined.logprobs.content.extend(stage_response.logprobs.content)

        return combined

    def _parse_preferred(self, eval_response: str) -> str:
        """평가 응답에서 선호 결과를 파싱합니다."""
        response_lower = eval_response.lower()

        if "preferred: b" in response_lower or "preferred:b" in response_lower:
            return "B"
        elif "preferred: a" in response_lower or "preferred:a" in response_lower:
            return "A"

        # 마지막 줄에서 A 또는 B 찾기
        lines = eval_response.strip().split('\n')
        for line in reversed(lines):
            line_lower = line.lower().strip()
            if line_lower.endswith('b') or 'b' in line_lower.split()[-1:]:
                return "B"
            elif line_lower.endswith('a') or 'a' in line_lower.split()[-1:]:
                return "A"

        return "Unknown"
