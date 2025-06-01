from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from src.aot.dataclasses import LLMCallStats # Reusing for consistency
# Assuming AssessmentDecision will be imported where ToTSolution is used, or add:
from src.aot.enums import AssessmentDecision # For ToTSolution field
from .enums import ToTSearchStrategy, ToTScoringMethod
from .constants import (
    DEFAULT_TOT_THOUGHT_GENERATION_MODEL_NAMES,
    DEFAULT_TOT_EVALUATION_MODEL_NAMES,
    DEFAULT_K_THOUGHTS, DEFAULT_B_BEAM_WIDTH, DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_THOUGHTS_IN_STATE, DEFAULT_MAX_TOTAL_THOUGHTS_GENERATED,
    DEFAULT_MAX_TIME_SECONDS
)

@dataclass
class ToTConfig:
    thought_generation_model_names: List[str] = field(default_factory=lambda: list(DEFAULT_TOT_THOUGHT_GENERATION_MODEL_NAMES))
    evaluation_model_names: List[str] = field(default_factory=lambda: list(DEFAULT_TOT_EVALUATION_MODEL_NAMES))
    k_thoughts: int = DEFAULT_K_THOUGHTS
    b_beam_width: int = DEFAULT_B_BEAM_WIDTH
    max_depth: int = DEFAULT_MAX_DEPTH
    max_thoughts_in_state: int = DEFAULT_MAX_THOUGHTS_IN_STATE
    max_total_thoughts_generated: int = DEFAULT_MAX_TOTAL_THOUGHTS_GENERATED
    max_time_seconds: int = DEFAULT_MAX_TIME_SECONDS
    search_strategy: ToTSearchStrategy = ToTSearchStrategy.BEAM
    scoring_method: ToTScoringMethod = ToTScoringMethod.LLM
    # Add other relevant configs from AoTRunnerConfig if needed for consistency (e.g. max_reasoning_tokens)

@dataclass
class ToTThought:
    text: str
    raw_generation_response: Optional[str] = None # Full response from LLM for this thought
    generation_stats: Optional[LLMCallStats] = None # Stats for generating this thought

@dataclass
class ToTNode:
    id: str # Unique identifier for the node
    thoughts_path: List[ToTThought] # Sequence of thoughts from root to this node (the "state")
    score: Optional[float] = None # Evaluation score for this state
    evaluation_details: Optional[str] = None # Justification or details from evaluation
    parent_id: Optional[str] = None # ID of the parent node
    depth: int = 0
    children_ids: List[str] = field(default_factory=list)
    evaluation_stats: Optional[LLMCallStats] = None # Stats for evaluating this node's state

    @property
    def current_state_text(self) -> str:
        return "\n".join([t.text for t in self.thoughts_path])

@dataclass
class ToTResult:
    final_answer: Optional[str] = None
    best_solution_path: Optional[List[ToTThought]] = None # Best sequence of thoughts found
    all_generated_nodes: Dict[str, ToTNode] = field(default_factory=dict) # Store all nodes for inspection
    total_thoughts_generated: int = 0
    total_nodes_evaluated: int = 0
    total_completion_tokens: int = 0 # Sum of all completion tokens for this process
    total_prompt_tokens: int = 0 # Sum of all prompt tokens
    total_llm_interaction_time_seconds: float = 0.0
    total_process_wall_clock_time_seconds: float = 0.0
    succeeded: bool = False
    error_message: Optional[str] = None

    @property
    def grand_total_tokens(self) -> int:
        return self.total_completion_tokens + self.total_prompt_tokens

@dataclass
class ToTSolution: # Similar to AoTSolution, but for ToT
    final_answer: Optional[str] = None
    # Reusing from AoT for consistency, can be made generic later
    assessment_stats: Optional[LLMCallStats] = None
    assessment_decision: Optional[AssessmentDecision] = None # If orchestrator uses it
    tot_result: Optional[ToTResult] = None
    fallback_call_stats: Optional[LLMCallStats] = None # If fallback to one-shot occurs
    total_wall_clock_time_seconds: float = 0.0 # Overall time including orchestration
    tot_failed_and_fell_back: bool = False
    tot_summary_output: Optional[str] = None # Summary from ToTProcess

    @property
    def total_completion_tokens(self) -> int:
        tokens = 0
        if self.assessment_stats: tokens += self.assessment_stats.completion_tokens
        if self.tot_result: tokens += self.tot_result.total_completion_tokens
        if self.fallback_call_stats: tokens += self.fallback_call_stats.completion_tokens
        return tokens

    @property
    def total_prompt_tokens(self) -> int:
        tokens = 0
        if self.assessment_stats: tokens += self.assessment_stats.prompt_tokens
        if self.tot_result: tokens += self.tot_result.total_prompt_tokens
        if self.fallback_call_stats: tokens += self.fallback_call_stats.prompt_tokens
        return tokens

    @property
    def grand_total_tokens(self) -> int:
        return self.total_completion_tokens + self.total_prompt_tokens

    @property
    def total_llm_interaction_time_seconds(self) -> float:
        time_sum = 0.0
        if self.assessment_stats: time_sum += self.assessment_stats.call_duration_seconds
        if self.tot_result: time_sum += self.tot_result.total_llm_interaction_time_seconds
        if self.fallback_call_stats: time_sum += self.fallback_call_stats.call_duration_seconds
        return time_sum
