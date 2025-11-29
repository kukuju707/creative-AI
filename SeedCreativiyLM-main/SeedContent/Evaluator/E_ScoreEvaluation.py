from typing import List, Dict, Tuple

import torch
import openai
import math

from SeedCore.Config.ConfigInstance import ConfigSubsystem
from SeedCore.LLM.LLMEvaluator import LLMEvaluator


class E_ScoreEvaluation(LLMEvaluator):

    def __init__(self):
        super().__init__()
        api_key = ConfigSubsystem.get_config("config/excluded/credentials.json").get("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=api_key)

    def evaluate(self, **kwargs) -> Tuple[float, float, Dict]:
        """
        PPL score, Diversity score 등의 지표를 활용한 시드 아이디어 평가를 진행함.

        Diversity = 1 - Average(Pairwise Cosine Similarity) at token level
        PPL(X) = exp( - (1/t) * sum_{i=1}^{t} log p_theta(x_i | x_{<i}) )

        Returns:
            Tuple of (ppl_score, diversity_score, logprobs_dict)
        """
        response_choice = kwargs.get("response_choice")

        ppl_score, logprobs_dict = self.eval_PPL(response_choice)
        diversity_score = self.eval_diversity_token_level(response_choice)

        return ppl_score, diversity_score, logprobs_dict

    def eval_PPL(self, response_choice) -> Tuple[float, Dict]:
        """
        Perplexity 점수 계산 (Sliding window approach for robust calculation)
        PPL(X) = exp( - (1/t) * sum_{i=1}^{t} log p_theta(x_i | x_{<i}) )

        Uses weighted accumulation to prevent numerical overflow and handles edge cases.
        Stores individual logprobs for further analysis.

        Returns:
            Tuple of (ppl_score, logprobs_dict) where logprobs_dict contains:
                - 'logprobs': list of individual logprobs
                - 'tokens': list of token strings
                - 'avg_logprob': average logprob value
                - 'num_tokens': total number of tokens
        """
        # Check if logprobs exists and has content
        if not response_choice.logprobs or not response_choice.logprobs.content:
            print("Warning: No logprobs available in response_choice")
            return float('inf'), self._create_empty_logprobs_dict()

        # Extract logprobs and tokens
        logprobs_content = response_choice.logprobs.content

        if len(logprobs_content) == 0:
            print("Warning: Empty logprobs list")
            return float('inf'), self._create_empty_logprobs_dict()

        # Extract individual logprobs and tokens
        logprobs = [tok.logprob for tok in logprobs_content]
        tokens = [tok.token for tok in logprobs_content]

        # Filter out any None or invalid logprobs
        valid_pairs = [(lp, tok) for lp, tok in zip(logprobs, tokens) if lp is not None and math.isfinite(lp)]

        if len(valid_pairs) == 0:
            print("Warning: No valid logprobs after filtering")
            return float('inf'), self._create_empty_logprobs_dict()

        valid_logprobs, valid_tokens = zip(*valid_pairs)
        valid_logprobs = list(valid_logprobs)
        valid_tokens = list(valid_tokens)

        # Calculate average negative log-likelihood
        # Note: logprobs from API are already log probabilities (negative values)
        # So we need to negate them to get negative log-likelihoods (positive values)
        sum_neg_log_likelihood = -sum(valid_logprobs)
        num_tokens = len(valid_logprobs)
        avg_neg_log_likelihood = sum_neg_log_likelihood / num_tokens

        # Compute perplexity: PPL = exp(avg_neg_log_likelihood)
        # Use clipping to prevent overflow
        avg_neg_log_likelihood = min(avg_neg_log_likelihood, 50.0)  # exp(50) ~ 5e21, reasonable upper bound

        ppl = math.exp(avg_neg_log_likelihood)

        # Create logprobs dictionary for storage
        logprobs_dict = {
            'logprobs': valid_logprobs,
            'tokens': valid_tokens,
            'avg_logprob': -avg_neg_log_likelihood,  # Store as logprob (negative)
            'num_tokens': num_tokens,
            'sum_neg_log_likelihood': sum_neg_log_likelihood
        }

        return ppl, logprobs_dict

    def _create_empty_logprobs_dict(self) -> Dict:
        """Create empty logprobs dictionary for error cases"""
        return {
            'logprobs': [],
            'tokens': [],
            'avg_logprob': None,
            'num_tokens': 0,
            'sum_neg_log_likelihood': None
        }

    def eval_diversity_token_level(self, response_choice) -> float:
        """
        Token-level diversity using pairwise cosine similarity.
        Diversity = 1 - Average(Pairwise Cosine Similarity)
        Uses matrix operations for efficiency.
        """
        # Check if logprobs exists and has content
        if not response_choice.logprobs or not response_choice.logprobs.content:
            print("Warning: No logprobs available for diversity calculation")
            return 0.0

        # Extract tokens from logprobs
        tokens = [tok_data.token for tok_data in response_choice.logprobs.content]

        if len(tokens) < 2:
            print("Warning: Not enough tokens for diversity calculation")
            return 0.0

        try:
            # Get embeddings for all tokens
            embeddings_response = self.client.embeddings.create(
                model="text-embedding-3-large",
                input=tokens
            )
            embeddings = [d.embedding for d in embeddings_response.data]

            # Convert to tensor for efficient matrix operations (num_tokens, embedding_dim)
            embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

            # Normalize embeddings for cosine similarity
            embeddings_normalized = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1)

            # Compute pairwise cosine similarity: similarity_matrix[i,j] = cos_sim(i,j)
            similarity_matrix = torch.mm(embeddings_normalized, embeddings_normalized.t())

            # Extract upper triangular part (excluding diagonal)
            num_tokens = similarity_matrix.size(0)
            mask = torch.triu(torch.ones(num_tokens, num_tokens), diagonal=1).bool()
            pairwise_similarities = similarity_matrix[mask]

            # Diversity = 1 - avg_similarity
            avg_similarity = pairwise_similarities.mean().item()
            diversity_score = 1.0 - avg_similarity

            return diversity_score

        except Exception as e:
            print(f"Error calculating diversity: {e}")
            return 0.0
