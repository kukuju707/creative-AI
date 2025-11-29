from typing import List, Iterator
from pydantic import BaseModel

from SeedCore.LLM.LLMGenerator import LLMGenerator
from SeedCore.Config.ConfigInstance import ConfigSubsystem
from SeedContent.Prompts.PromptTemplates import PromptTemplates


class SG_Descendant(LLMGenerator):

    """
    A generator class for Descendant seeds: Deep conceptual extensions from existing seeds

    Args:
        question: The main topic or question to generate seeds for.
        **kwargs: Additional parameters such as:
            - descendant_seed_params: DescendantSeedParams

    """

    class DescendantSeedParams:
        num_seeds : int
        model_name: str
        max_tokens: int
        temperature: float

    def __init__(self, api_key: str):
        super().__init__(api_key)

    def _create_prompt(self, core_seeds, interpolation_seeds, num_seeds) -> str:

        return PromptTemplates.get_descendant_seed_prompt(core_seeds, interpolation_seeds, num_seeds)

    def generate(self, question: str, **kwargs) -> Iterator[str]:

        # CoreSeeds class for response parsing
        class DescendantSeeds(BaseModel):
            deep_concepts: List[str]

        # extract parameters

        descendant_seed_params = kwargs.get("descendant_seed_params")

        core_seeds = kwargs.get("core_seeds")
        interpolation_seeds = kwargs.get("interpolation_seeds")

        completion = self.client.chat.completions.parse(
            model=descendant_seed_params["model_name"],
            messages=[{"role": "user", "content": self._create_prompt(
                core_seeds,
                interpolation_seeds,
                descendant_seed_params["num_seeds"]
            )}],
            response_format=DescendantSeeds,
            temperature=descendant_seed_params["temperature"],
            max_tokens=descendant_seed_params["max_tokens"]
        )

        seeds_obj = completion.choices[0].message.parsed
        return seeds_obj.deep_concepts[:descendant_seed_params["num_seeds"]]






