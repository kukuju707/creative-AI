from SeedContent.Evaluator.E_ScoreEvaluation import E_ScoreEvaluation
from SeedContent.Pipeline.P_MultistageGeneration import P_MultiStageGeneration
from SeedContent.Pipeline.P_ControlGroupGeneration import P_ControlGroupGeneration
from SeedCore.SharedFunctionLibrary import LOG_TEXT


class P_MultiStageG_Evaluation:
    """
    Multi-stage generation evaluation pipeline.
    Supports PPL, Diversity score evaluation with baseline comparison.
    """

    def __init__(self) -> None:
        self.generator = P_MultiStageGeneration()
        self.control_generator = P_ControlGroupGeneration()
        self.score_evaluator = E_ScoreEvaluation()

    def evaluate(self, question: str = "Write a poem about the sea.", num_stages: int = 2, include_baseline: bool = False):
        """
        Evaluate generation with PPL and Diversity scores.

        Args:
            question: Input prompt
            num_stages: Number of generation stages
            include_baseline: Include control group comparison

        Returns:
            Dictionary containing evaluation results
        """
        LOG_TEXT(f"Evaluating: {question[:50]}...", colorKey="CYAN")

        # Seed-based generation
        seed_result = self.generator.generate(
            question=question,
            num_stages=num_stages
        )

        # concat the list of str to a single str
        seed_result = " ".join(seed_result)

        LOG_TEXT(f"Seed-based result: {seed_result[:100]}...", colorKey="BLUE")

        results = {
            "seed_result": seed_result,
            "baseline_result": None
        }

        # Baseline generation
        if include_baseline:
            baseline_result = self.control_generator.generate(question=question)
            results["baseline_result"] = baseline_result
            LOG_TEXT(f"Baseline result: {baseline_result[:100]}...", colorKey="YELLOW")

        return results