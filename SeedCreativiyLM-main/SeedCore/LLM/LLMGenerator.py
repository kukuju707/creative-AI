from abc import abstractmethod
from typing import AsyncIterator, Iterator

import openai


class LLMGenerator:
    """
    Base class for LLM-based generators.
    Provides common functionality for OpenAI API interactions.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the LLM generator with API key.
        
        Args:
            api_key: OpenAI API key for authentication.
        """
        self.client = openai.OpenAI(api_key=api_key)

    def generate(self, question: str, **kwargs) -> Iterator[str]:
        """
        Execute prompt generation with provided arguments.
        
        Args:
            question: The input question or prompt.
            **kwargs: Additional keyword arguments for generation.
            
        Returns:
            Iterator of generated text strings.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement generate method")

    async def agenerate(self, question: str, **kwargs) -> AsyncIterator[str]:
        """
        Asynchronously execute prompt generation with provided arguments.
        
        Args:
            question: The input question or prompt.
            **kwargs: Additional keyword arguments for generation.
            
        Returns:
            Async iterator of generated text strings.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement agenerate method")
