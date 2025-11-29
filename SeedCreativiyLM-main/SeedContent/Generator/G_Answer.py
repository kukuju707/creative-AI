import openai

from SeedCore.LLM.LLMGenerator import LLMGenerator
from SeedContent.Prompts.PromptTemplates import PromptTemplates
from typing import AsyncIterator, Iterator

from SeedCore.SharedFunctionLibrary import LOG_TEXT


class G_Answer(LLMGenerator):

    def __init__(self, api_key: str):
        super().__init__(api_key)

    def generate(self, question: str, **kwargs) -> Iterator[str]:

        model_name = kwargs.get("model_name")
        temperature = kwargs.get("temperature")
        recommended_max_tokens = int(kwargs.get("recommended_max_tokens"))
        max_tokens = int(kwargs.get("max_tokens"))
        seeds = kwargs.get("seeds")
        previous_content = kwargs.get("previous_content")
        num_stages = kwargs.get("num_stages")
        stage_index = kwargs.get("stage_index")

        final_stage_recommended_max_tokens_multiplier = float(kwargs.get("final_stage_recommended_max_tokens_multiplier", 0.2))

        seed_str = ", ".join(seeds)

        prompt = ""
        if stage_index == 0: # First stage
            prompt = PromptTemplates.get_answer_first_stage_prompt(seed_str, question, recommended_max_tokens)
        elif stage_index == num_stages - 1: # Final stage
            prompt = PromptTemplates.get_answer_last_stage_prompt(seed_str, question, int(recommended_max_tokens * final_stage_recommended_max_tokens_multiplier), previous_content)
        else:
            prompt = PromptTemplates.get_answer_middle_stage_prompt(seed_str, question, recommended_max_tokens, "".join(previous_content))

        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                logprobs=True
            )

            #return response.choices[0].message.content
            return response.choices[0]

        except Exception as e:
            print(f"답변 생성 중 오류 발생: {e}")
            return ""