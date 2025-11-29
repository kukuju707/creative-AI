import json
from datetime import datetime
from SeedContent.Evaluator.E_CreativityEvaluation import E_CreativityEvaluation


def main():
    evaluator = E_CreativityEvaluation()

    results = evaluator.evaluate(
        num_stages=3,
        include_baseline=True,
        compute_scores=True,
        prefix_prompt="Provide a creative, witty, engaging, catchy, immersive story based on the following prompt:\n"
    )

    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"evaluation_results_{timestamp}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
