from typing import List, Iterator
from pydantic import BaseModel

from SeedCore.LLM.LLMGenerator import LLMGenerator
from SeedCore.Config.ConfigInstance import ConfigSubsystem
from SeedContent.Prompts.PromptTemplates import PromptTemplates


class SG_Interpolation(LLMGenerator):

    class InterpolationSeedParams:
        num_seeds: int
        model_name: str
        max_tokens: int
        temperature: float

    def __init__(self, api_key: str):
        super().__init__(api_key)

    def _create_prompt(self, core_seeds, num_seeds) -> str:

        return PromptTemplates.get_interpolation_seed_prompt(core_seeds, num_seeds).strip()

    def generate(self, question: str, **kwargs) -> Iterator[str]:
        """
        Interpolation seed generator: creates bridge concepts between core seeds.

        Args:
            question: The original question or topic (not heavily used here)
            **kwargs:
                - core_seeds : List[str]
                - interpolation_seed_params: InterpolationParams
        """

        # response parsing model
        class InterpolationSeeds(BaseModel):
            Interpolations: List[str]

        # extract parameters
        interpolation_seed_params = kwargs.get("interpolation_seed_params")
        core_seeds = kwargs.get("core_seeds")

        # require at least 2 seeds
        if len(core_seeds) < 2:
            return []

        prompt = self._create_prompt(
            core_seeds=core_seeds,
            num_seeds=interpolation_seed_params["num_seeds"]
        )

        completion = self.client.chat.completions.parse(
            model=interpolation_seed_params["model_name"],
            messages=[{"role": "user", "content": prompt}],
            response_format=InterpolationSeeds,
            temperature=interpolation_seed_params["temperature"],
            max_tokens=interpolation_seed_params["max_tokens"]
        )

        seeds_obj = completion.choices[0].message.parsed

        return seeds_obj.Interpolations[:interpolation_seed_params["num_seeds"]]
