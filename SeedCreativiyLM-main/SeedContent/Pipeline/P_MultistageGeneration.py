from typing import Optional, List

from SeedContent.Generator.G_Answer import G_Answer
from SeedContent.Pipeline.P_SeedStageGeneration import P_SeedStageGeneration
from SeedContent.Pruner.SeedPruner import SeedPruner
from SeedCore.Config.ConfigInstance import ConfigSubsystem
from SeedCore.SharedFunctionLibrary import LOG_TEXT


class P_MultiStageGeneration:

    def __init__(self) -> None:
        api_key = ConfigSubsystem.get_config("config/excluded/credentials.json").get("OPENAI_API_KEY")

        # Initialize components
        self.seed_generation_sub_pipeline = P_SeedStageGeneration()
        self.answer_generator = G_Answer(api_key)
        self.seed_pruner = SeedPruner()

    def _get_config(self):
        return ConfigSubsystem.get_config("config/Pipeline/P_MultiStageGenerator.json")

    def _create_default_stages(self, num_stages: int) -> List:
        stages = []

        seed_generation_pipeline_config = self._get_config().get("seed_generation_pipeline")

        core_seed_params = dict(seed_generation_pipeline_config["core_seed_params"])
        interpolation_seed_params = dict(seed_generation_pipeline_config["interpolation_seed_params"])
        descendant_seed_params = dict(seed_generation_pipeline_config["descendant_seed_params"])
        enable_pruning = seed_generation_pipeline_config.get("enable_pruning")

        for i in range(num_stages):
            stage_config = {
                "stage_index": i,
                "total_stages": num_stages,
                "core_seed_params": core_seed_params,
                "interpolation_seed_params": interpolation_seed_params,
                "descendant_seed_params": descendant_seed_params,
                "enable_pruning": enable_pruning
            }
            stages.append(stage_config)
        return stages

    def generate(
            self,
            question: str,
            num_stages: int = 3,
            model_override: str = None,
            return_logprobs: bool = False,
    ):
        """
        Generate response using multi-stage seed-based approach.

        Args:
            question: Input question/prompt
            num_stages: Number of generation stages
            model_override: Optional model name to override config setting
            return_logprobs: If True, return (content_list, stage_responses) for PPL calculation

        Returns:
            If return_logprobs=False: List of generated content strings
            If return_logprobs=True: Tuple of (content_list, stage_responses_with_logprobs)
        """
        accumulated_content = []
        stage_responses = []  # Store full response objects with logprobs
        previous_seeds = []

        # create stages
        stages = self._create_default_stages(num_stages)

        # Iterate through each stage, generating content and seeds
        for i in range(num_stages):
            stage = stages[i]

            stage_content, current_seeds = self.generate_stage(
                question=question,
                stage=stage,
                stage_index=i,
                num_stages=num_stages,
                accumulated_content=accumulated_content,
                previous_seeds=previous_seeds,
                model_override=model_override
            )

            accumulated_content.append(stage_content.message.content)
            stage_responses.append(stage_content)  # Store full response with logprobs
            previous_seeds = current_seeds

        if return_logprobs:
            return accumulated_content, stage_responses
        else:
            return accumulated_content

    def generate_stage(self, question: str, stage: dict, stage_index: int, num_stages: int, accumulated_content: list[str], previous_seeds: List[str], model_override: str = None):

        pruner_config = self._get_config().get("seed_pruner")
        pruner_enabled = pruner_config.get("b_enabled", True)
        pruner_similarity_threshold = pruner_config.get("similarity_threshold")

        answer_config = self._get_config().get("answer_generator")

        stage_ratio = stage_index / max(num_stages - 1, 1)

        # Generate seeds for the current stage
        core_seeds, interpolation_seeds, descendant_seeds = self.seed_generation_sub_pipeline.generate_seed_stage(
            question=question,
            stage_ratio=stage_ratio,
            core_seed_params=stage["core_seed_params"],
            interpolation_seed_params=stage["interpolation_seed_params"],
            descendant_seed_params=stage["descendant_seed_params"],
            previous_seeds=previous_seeds,
            previous_content=accumulated_content
        )

        # Prune seeds if enabled
        all_seeds = core_seeds + interpolation_seeds + descendant_seeds

        if pruner_enabled:
            all_seeds = self.seed_pruner.prune_seeds(
                all_seeds,
                pruner_similarity_threshold
            )

        LOG_TEXT(f"Stage {stage['stage_index']} - Generated seeds: {str(all_seeds)}", colorKey="CYAN", verbosity="Verbose")

        # Use model_override if provided, otherwise use config
        model_name = model_override if model_override else answer_config.get("model_name")

        # Generate content for the current stage
        stage_result = self.answer_generator.generate(
            question=question,
            seeds=all_seeds,
            stage_index=stage_index,
            num_stages=num_stages,
            final_stage_recommended_max_tokens_multiplier=answer_config.get("final_stage_recommended_max_tokens_multiplier", 0.2),
            max_tokens=int(answer_config.get("max_tokens") / num_stages),
            recommended_max_tokens=int(answer_config.get("recommended_max_tokens") / num_stages),
            temperature=answer_config.get("temperature"),
            model_name=model_name,
            previous_content=accumulated_content
        )

        LOG_TEXT(f"Stage {stage['stage_index']} - Generated content: {stage_result.message.content}", colorKey="CYAN", verbosity="Verbose")

        return stage_result, all_seeds
