from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, List

# Attempt to import LLMCallStats from aot_dataclasses, if not found, define a placeholder.
# This is to avoid circular dependency if aot_dataclasses also needs to import from here later,
# though for this step, direct import should be fine.
try:
    from src.aot_dataclasses import LLMCallStats
except ImportError:
    @dataclass
    class LLMCallStats: # Placeholder if not found
        completion_tokens: int = 0
        prompt_tokens: int = 0
        call_duration_seconds: float = 0.0
        model_name: Optional[str] = None

@dataclass
class CoTConfig:
    """Base configuration for any Chain-of-Thought process."""
    # Common configuration parameters can be added here if identified
    # For now, it serves as a base class.
    pass

@dataclass
class CoTResult:
    """Standardized result for any Chain-of-Thought process."""
    final_answer: Optional[str] = None
    succeeded: bool = False
    error_message: Optional[str] = None
    
    total_completion_tokens: int = 0
    total_prompt_tokens: int = 0
    total_llm_interaction_time_seconds: float = 0.0
    total_process_wall_clock_time_seconds: float = 0.0 # Wall clock for the CoT process itself

    process_specific_data: Optional[Any] = None # For AoTResult, L2TResult, etc.
    
    # Optional: If a CoT process generates a trace or similar structured output
    # that is common enough to be promoted from process_specific_data.
    reasoning_trace: Optional[List[str]] = field(default_factory=list)


class CoTProcess(ABC):
    """Abstract base class for a Chain-of-Thought process."""

    def __init__(self, config: CoTConfig):
        self.config = config

    @abstractmethod
    def run(self, problem_text: str) -> CoTResult:
        """Executes the CoT process."""
        pass

    @abstractmethod
    def get_summary(self, result: CoTResult) -> str:
        """Generates a summary string for the CoT process execution."""
        pass
