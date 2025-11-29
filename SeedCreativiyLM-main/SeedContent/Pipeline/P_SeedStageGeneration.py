from typing import Optional, List

from SeedContent.Generator.Seed.SG_Core import SG_Core
from SeedContent.Generator.Seed.SG_Descendant import SG_Descendant
from SeedContent.Generator.Seed.SG_Interpolation import SG_Interpolation
from SeedCore.Config.ConfigInstance import ConfigSubsystem
from SeedCore.SharedFunctionLibrary import LOG_TEXT


class P_SeedStageGeneration:

    """
    A pipeline that generates all seed types for a single stage in the multi-stage pipeline.
    """

    def __init__(self):

        # Load API key from configuration
        api_key = ConfigSubsystem.get_config("config/excluded/credentials.json").get("OPENAI_API_KEY")

        # Initialize individual seed generators
        self.core_seed_generator = SG_Core(api_key)
        self.interpolation_seed_generator = SG_Interpolation(api_key)
        self.descendant_seed_generator = SG_Descendant(api_key)

    def generate_seed_stage(
        self,
        question: str,
        stage_ratio: float,
        core_seed_params: SG_Core.CoreSeedParams,
        interpolation_seed_params: SG_Interpolation.InterpolationSeedParams,
        descendant_seed_params: SG_Descendant.DescendantSeedParams,
        previous_seeds: Optional[List[str]] = None,
        previous_content: Optional[str] = None,
    ):
        core_seeds = self.core_seed_generator.generate(
            question,
            stage_ratio=stage_ratio,
            core_seed_params=core_seed_params,
            previous_seeds=previous_seeds,
            previous_content=previous_content,
        )

        LOG_TEXT(f"Generated core seeds: {str(core_seeds)}", colorKey="GREEN", verbosity="Verbose")

        interpolation_seeds = self.interpolation_seed_generator.generate(
            question,
            interpolation_seed_params=interpolation_seed_params,
            core_seeds=core_seeds
        )

        LOG_TEXT(f"Generated interpolation seeds: {str(interpolation_seeds)}", colorKey="GREEN", verbosity="Verbose")

        descendant_seeds = self.descendant_seed_generator.generate(
            question,
            descendant_seed_params=descendant_seed_params,
            core_seeds=core_seeds,
            interpolation_seeds=interpolation_seeds
        )

        LOG_TEXT(f"Generated descendant seeds: {str(descendant_seeds)}",colorKey="GREEN", verbosity="Verbose")

        return core_seeds, interpolation_seeds, descendant_seeds
