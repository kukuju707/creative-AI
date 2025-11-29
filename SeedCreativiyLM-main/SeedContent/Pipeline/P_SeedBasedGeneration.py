import os

from SeedContent.Generator.G_Answer import G_Answer
from SeedContent.Generator.Seed.SG_Core import SG_Core

from SeedCore.Config.ConfigInstance import ConfigSubsystem


class P_SeedBasedGeneration:
    """
    시드 토큰을 기반으로 콘텐츠를 생성하는 LLM 파이프라인 클래스.
    LLMG_SeedGenerator와 LLMG_AnswerGenerator를 저장하고 관리하며, 질문에 대한 시드 토큰을 생성하고 이를 바탕으로 답변을 생성하는 기능을 제공합니다.
    """

    def __init__(self):

        # 시드 생성기 + 답변 생성기를 먼저 초기화합니다.
        api_key = ConfigSubsystem.get_config("config/excluded/credentials.json").get("OPENAI_API_KEY")
        self.seed_gen = SG_Core(api_key)
        self.answer_gen = G_Answer(api_key)


    def generate(self, question: str, num_seeds: int = 5) -> str:
        """
        주어진 질문에 대해 시드 토큰을 생성하고, 이를 바탕으로 답변을 생성합니다.

        :param question: 답변을 생성할 질문.
        :param num_seeds: 생성할 시드 토큰의 개수.
        :return: 생성된 답변 문자열.
        """
        seeds = self.seed_gen.generate(question, num_seeds=num_seeds)
        answer = self.answer_gen.generate(question, seeds=seeds).message.content
        return answer