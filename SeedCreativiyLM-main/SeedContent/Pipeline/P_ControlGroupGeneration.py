from SeedContent.Generator.G_ControlGroup import G_ControlGroup
from SeedCore.Config.ConfigInstance import ConfigSubsystem


class P_ControlGroupGeneration:
    """
    Control group baseline generation pipeline.
    Simple direct generation without seed-based approach.
    """

    def __init__(self):
        api_key = ConfigSubsystem.get_config("config/excluded/credentials.json").get("OPENAI_API_KEY")
        self.control_group_gen = G_ControlGroup(api_key)

    def _get_config(self):
        return ConfigSubsystem.get_config("config/Pipeline/P_MultiStageGenerator.json").get("control_group")

    def generate(self, question: str, logprobs: bool = False, return_full_response: bool = False):
        """
        Generate response using control group (baseline) approach.

        Args:
            question: Input question/prompt
            logprobs: Whether to return logprobs for PPL evaluation
            return_full_response: If True, return full response_choice object instead of just content

        Returns:
            Response content (str) or full response_choice object if return_full_response=True
        """
        config = self._get_config()

        response = self.control_group_gen.generate(
            question,
            model_name=config.get("model_name", "gpt-4o-mini"),
            temperature=config.get("temperature", 0.7),
            recommended_max_tokens=config.get("recommended_max_tokens", 50),
            max_tokens=config.get("max_tokens", 100),
            logprobs=logprobs
        )

        if response is None:
            return None

        if return_full_response:
            return response

        return response.message.content