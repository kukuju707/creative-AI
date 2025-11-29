# Multi-Stage Seed-Based Generation System

다중 시드(seed) 기반의 창의적 텍스트 생성 시스템입니다.

## 설치

```bash
pip install -r requirements.txt
```

환경 변수 설정 (`.env` 파일 또는 시스템 환경 변수):
```
OPENAI_API_KEY=your-api-key-here
```

## 실행법

### 텍스트 생성 실행

```bash
python main.py
```

`main.py`에서 질문과 stage 수를 수정할 수 있습니다:

```python
from SeedContent.Pipeline.P_MultistageGeneration import P_MultiStageGeneration

generator = P_MultiStageGeneration()
generator.generate(
    question="Write a poem about the sea.",  # 원하는 프롬프트로 변경
    num_stages=2  # stage 수 조절
)
```

설정 변경: `config/Pipeline/P_MultiStageGenerator.json`

## 평가 실행법

LitBench 데이터셋을 활용한 창의성 평가:

```bash
python evaluate.py
```

`evaluate.py`에서 평가 설정을 수정할 수 있습니다:

```python
from SeedContent.Evaluator.E_CreativityEvaluation import E_CreativityEvaluation

evaluator = E_CreativityEvaluation()
results = evaluator.evaluate(
    num_stages=2,           # 생성 stage 수
    include_baseline=True   # baseline(control group) 비교 포함 여부
)

print(f"Win Rate: {results['win_rate']:.2%}")
print(f"Baseline Win Rate: {results['baseline_win_rate']:.2%}")
```

설정 변경: `config/Evaluator/E_CreativityEvaluation.json`

```json
{
  "model_name": "dmnsh/Qwen3-4b-W0-GenRM",  // 평가 모델 (HuggingFace 또는 OpenAI)
  "dataset_name": "SAA-Lab/LitBench-Train",
  "eval_sample_size": 3  // 평가 샘플 수
}
```
