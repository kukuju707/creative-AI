
from SeedContent.Pipeline.P_MultistageGeneration import P_MultiStageGeneration


def main():
    generator = P_MultiStageGeneration()

    generator.generate(
        #question="야구에 대해서 한국어로 설명해줘.",
        #question="인공지능의 미래에 대해 설명해줘.",
        question="만약 당신이 타임머신을 타고 과거로 돌아갈 수 있다면, 어느 시대로 가고 싶나요? 그리고 그 이유는 무엇인가요?",
        num_stages=4
    )


if __name__ == "__main__":
    main()
