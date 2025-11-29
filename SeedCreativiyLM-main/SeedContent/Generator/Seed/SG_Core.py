from typing import List, Iterator
from pydantic import BaseModel

from SeedCore.LLM.LLMGenerator import LLMGenerator
from SeedCore.Maths.BezierCurve import BezierCurve
from SeedCore.Config.ConfigInstance import ConfigSubsystem
from SeedContent.Prompts.PromptTemplates import PromptTemplates


class SG_Core(LLMGenerator):

    class CoreSeedParams:
        num_seeds: int
        model_name: str
        max_tokens: int
        temperature_curve: BezierCurve
        temperature_range: dict

    def __init_(self, api_key: str):
        super().__init__(api_key)

    def _compute_temperature(self, stage_ratio: float, temperature_curve, temperature_range) -> float:
        min_temp = temperature_range["min"]
        max_temp = temperature_range["max"]

        curve_value = temperature_curve.get_value_at(stage_ratio)

        return curve_value[1] * (max_temp - min_temp) + min_temp

    def _create_prompt(self, question, previous_seeds, previous_content, num_seeds) -> str:

        context_info = ""
        if previous_seeds:
            context_info += f"\nPrevious seeds: {', '.join(previous_seeds)}"
        if previous_content:
            context_info += f"\nPrevious content: {previous_content[:200]}..."

        return PromptTemplates.get_core_seed_prompt(question, context_info, num_seeds)

    def generate(self, question: str, **kwargs) -> Iterator[str]:

        """
        A generator class for Core seeds: High-temperature creative tokens.

        Args:
            question: The main topic or question to generate seeds for.
            **kwargs: Additional parameters such as:
                - stage_ratio: Float between 0 and 1 indicating the progress through the multi-stage
                - previous_seeds: List of previously generated seeds for context.
                - previous_content: Previously generated content for context.
                - core_seed_params: CoreSeedParams object containing generation parameters.
        """

        # CoreSeeds class for response parsing
        class CoreSeeds(BaseModel):
            ideas: List[str]



        # extract parameters from kwargs
        stage_ratio = kwargs.get("stage_ratio")
        previous_seeds = kwargs.get("previous_seeds", [])
        previous_content = kwargs.get("previous_content", "")
        core_seed_params = kwargs.get("core_seed_params")

        model = core_seed_params["model_name"]
        max_tokens = core_seed_params.get("max_tokens")
        num_seeds = core_seed_params.get("num_seeds")
        temperature_curve = BezierCurve.make_from_points(core_seed_params.get("temperature_curve_points"))
        temperature_range = core_seed_params.get("temperature_range")

        # create prompt - pass question and kwargs right through
        prompt = self._create_prompt(
            question=question,
            previous_seeds=previous_seeds,
            previous_content=previous_content,
            num_seeds=num_seeds
        )

        completion = self.client.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format=CoreSeeds,
            temperature=self._compute_temperature(stage_ratio, temperature_curve, temperature_range),
            max_tokens=max_tokens
        )

        seeds_obj = completion.choices[0].message.parsed

        return seeds_obj.ideas[:num_seeds]
