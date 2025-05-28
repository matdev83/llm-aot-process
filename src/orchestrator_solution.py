from dataclasses import dataclass, field
from typing import Optional, Any, List # Added Any, List

# Attempt to import from new/modified files.
# It's assumed LLMCallStats will be available, e.g. from aot_dataclasses
try:
    from src.aot_dataclasses import LLMCallStats
except ImportError:
    @dataclass # Define placeholder if necessary, though aot_dataclasses should provide it
    class LLMCallStats:
        completion_tokens: int = 0
        prompt_tokens: int = 0
        call_duration_seconds: float = 0.0
        model_name: Optional[str] = None

try:
    from src.cot_process import CoTResult
except ImportError:
    @dataclass # Define placeholder if necessary
    class CoTResult:
        final_answer: Optional[str] = None
        succeeded: bool = False
        error_message: Optional[str] = None
        total_completion_tokens: int = 0
        total_prompt_tokens: int = 0
        total_llm_interaction_time_seconds: float = 0.0
        total_process_wall_clock_time_seconds: float = 0.0
        process_specific_data: Optional[Any] = None
        reasoning_trace: Optional[List[str]] = field(default_factory=list)


# It's assumed AssessmentDecision will be available from aot_enums
try:
    from src.aot_enums import AssessmentDecision
except ImportError:
    from enum import Enum
    class AssessmentDecision(Enum): # Placeholder
        AOT = "AOT"
        ONESHOT = "ONESHOT"
        ERROR = "ERROR"


@dataclass
class OrchestratorSolution:
    """Unified solution dataclass for the CoTOrchestrator."""
    final_answer: Optional[str] = None
    total_wall_clock_time_seconds: float = 0.0 # Overall wall clock for the orchestrator

    # CoT process related
    cot_result: Optional[CoTResult] = None
    cot_summary_output: Optional[str] = None # Summary from the CoTProcess itself

    # Assessment related
    assessment_stats: Optional[LLMCallStats] = None
    assessment_decision: Optional[AssessmentDecision] = None # e.g. AOT, ONESHOT

    # Direct one-shot call related (if CoT process is bypassed or as initial attempt)
    main_oneshot_call_stats: Optional[LLMCallStats] = None

    # Fallback one-shot call related (if CoT process fails or assessment leads to error)
    fallback_oneshot_call_stats: Optional[LLMCallStats] = None
    
    process_failed_and_fell_back: bool = False # True if CoT process failed and fallback was used

    # Trigger mode used by the orchestrator for this solution
    trigger_mode_used: Optional[str] = None # Store the string representation of the trigger mode

    @property
    def total_completion_tokens(self) -> int:
        tokens = 0
        if self.assessment_stats: tokens += self.assessment_stats.completion_tokens
        if self.main_oneshot_call_stats: tokens += self.main_oneshot_call_stats.completion_tokens
        if self.cot_result: tokens += self.cot_result.total_completion_tokens
        if self.fallback_oneshot_call_stats: tokens += self.fallback_oneshot_call_stats.completion_tokens
        return tokens

    @property
    def total_prompt_tokens(self) -> int:
        tokens = 0
        if self.assessment_stats: tokens += self.assessment_stats.prompt_tokens
        if self.main_oneshot_call_stats: tokens += self.main_oneshot_call_stats.prompt_tokens
        if self.cot_result: tokens += self.cot_result.total_prompt_tokens
        if self.fallback_oneshot_call_stats: tokens += self.fallback_oneshot_call_stats.prompt_tokens
        return tokens

    @property
    def grand_total_tokens(self) -> int:
        return self.total_completion_tokens + self.total_prompt_tokens

    @property
    def total_llm_interaction_time_seconds(self) -> float:
        time_sum = 0.0
        if self.assessment_stats: time_sum += self.assessment_stats.call_duration_seconds
        if self.main_oneshot_call_stats: time_sum += self.main_oneshot_call_stats.call_duration_seconds
        if self.cot_result: time_sum += self.cot_result.total_llm_interaction_time_seconds
        if self.fallback_oneshot_call_stats: time_sum += self.fallback_oneshot_call_stats.call_duration_seconds
        return time_sum
