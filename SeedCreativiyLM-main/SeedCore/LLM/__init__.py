"""
SeedCore.LLM - Core LLM framework and type definitions.

This module contains base classes and type definitions for LLM-based generation.
"""

from SeedCore.LLM.LLMGenerator import LLMGenerator
from SeedCore.LLM.LLMEvaluator import LLMEvaluator

__all__ = [
    "LLMGenerator",
    "LLMEvaluator",
]
