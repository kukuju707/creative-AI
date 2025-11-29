from SeedCore.LLM.LLMGenerator import LLMGenerator
from SeedContent.Prompts.PromptTemplates import PromptTemplates


class G_ControlGroup(LLMGenerator):
    def __init__(self, api_key: str):
        super().__init__(api_key)

    def generate(self, question: str, **kwargs):
        """
        Control group generation without seed-based approach.
        Supports logprobs for PPL evaluation.
        """
        model_name = kwargs.get("model_name", "gpt-4o-mini")
        temperature = kwargs.get("temperature", 0.7)
        recommended_max_tokens = kwargs.get("recommended_max_tokens", 50)
        max_tokens = kwargs.get("max_tokens", 100)
        logprobs = kwargs.get("logprobs", False)

        prompt = PromptTemplates.get_control_group_prompt(question, recommended_max_tokens)

        try:
            completion_params = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            # Add logprobs parameters if requested
            if logprobs:
                completion_params["logprobs"] = True
                completion_params["top_logprobs"] = 1

            response = self.client.chat.completions.create(**completion_params)

            return response.choices[0]
        except Exception as e:
            print(f"답변 생성 중 오류 발생: {e}")
            return None