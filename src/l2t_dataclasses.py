from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any # Added Any for placeholder

# Assuming CoTConfig is in src.cot_process
try:
    from src.cot_process import CoTConfig 
except ImportError:
    # Minimal placeholder if running this part in isolation
    @dataclass
    class CoTConfig: # type: ignore
        pass

# LLMCallStats is imported from aot_dataclasses
try:
    from src.aot_dataclasses import LLMCallStats
except ImportError:
    @dataclass
    class LLMCallStats: # Placeholder
        completion_tokens: int = 0
        prompt_tokens: int = 0
        call_duration_seconds: float = 0.0
        model_name: Optional[str] = None

from src.l2t_constants import (
    DEFAULT_L2T_CLASSIFICATION_MODEL_NAMES,
    DEFAULT_L2T_CLASSIFICATION_TEMPERATURE,
    DEFAULT_L2T_INITIAL_PROMPT_MODEL_NAMES,
    DEFAULT_L2T_INITIAL_PROMPT_TEMPERATURE,
    DEFAULT_L2T_MAX_STEPS,
    DEFAULT_L2T_MAX_TIME_SECONDS,
    DEFAULT_L2T_MAX_TOTAL_NODES,
    DEFAULT_L2T_THOUGHT_GENERATION_MODEL_NAMES,
    DEFAULT_L2T_THOUGHT_GENERATION_TEMPERATURE,
    DEFAULT_L2T_X_EVA_DEFAULT,
    DEFAULT_L2T_X_FMT_DEFAULT,
)


class L2TNodeCategory(Enum): # Remains unchanged
    CONTINUE = "CONTINUE"
    TERMINATE_BRANCH = "TERMINATE_BRANCH"
    FINAL_ANSWER = "FINAL_ANSWER"
    BACKTRACK = "BACKTRACK"


@dataclass
class L2TNode: # Remains unchanged
    id: str
    content: str
    parent_id: Optional[str]
    generation_step: int
    children_ids: List[str] = field(default_factory=list)
    category: Optional[L2TNodeCategory] = None


@dataclass
class L2TGraph: # Remains unchanged
    nodes: Dict[str, L2TNode] = field(default_factory=dict)
    v_pres: List[str] = field(default_factory=list)
    v_hist: List[str] = field(default_factory=list)
    root_node_id: Optional[str] = None

    def add_node(self, node: L2TNode, is_root: bool = False) -> None:
        if node.id in self.nodes:
            raise ValueError(f"Node with id {node.id} already exists.")
        self.nodes[node.id] = node
        self.v_pres.append(node.id)
        if is_root:
            if self.root_node_id is not None:
                raise ValueError(
                    f"Root node already set to {self.root_node_id}. Cannot set {node.id} as root."
                )
            self.root_node_id = node.id
        if node.parent_id and node.parent_id in self.nodes:
            parent_node = self.nodes[node.parent_id]
            if node.id not in parent_node.children_ids:
                parent_node.children_ids.append(node.id)

    def classify_node(self, node_id: str, category: L2TNodeCategory) -> None:
        if node_id not in self.nodes:
            raise ValueError(f"Node with id {node_id} not found.")
        self.nodes[node_id].category = category

    def move_to_hist(self, node_id: str) -> None:
        if node_id not in self.nodes:
            raise ValueError(f"Node with id {node_id} not found.")
        if node_id not in self.v_pres:
            if node_id not in self.v_hist:
                 raise ValueError(f"Node {node_id} not in v_pres to move to v_hist.")
            return 
        self.v_pres.remove(node_id)
        if node_id not in self.v_hist:
            self.v_hist.append(node_id)

    def get_node(self, node_id: str) -> Optional[L2TNode]:
        return self.nodes.get(node_id)

    def get_parent(self, node_id: str) -> Optional[L2TNode]:
        node = self.get_node(node_id)
        if node and node.parent_id:
            return self.get_node(node.parent_id)
        return None

    def get_children(self, node_id: str) -> List[L2TNode]:
        node = self.get_node(node_id)
        if node:
            return [self.nodes[child_id] for child_id in node.children_ids if child_id in self.nodes]
        return []


@dataclass
class L2TConfig(CoTConfig): # Inherits from CoTConfig
    classification_model_names: List[str] = field(
        default_factory=lambda: DEFAULT_L2T_CLASSIFICATION_MODEL_NAMES
    )
    thought_generation_model_names: List[str] = field(
        default_factory=lambda: DEFAULT_L2T_THOUGHT_GENERATION_MODEL_NAMES
    )
    initial_prompt_model_names: List[str] = field(
        default_factory=lambda: DEFAULT_L2T_INITIAL_PROMPT_MODEL_NAMES
    )
    classification_temperature: float = DEFAULT_L2T_CLASSIFICATION_TEMPERATURE
    thought_generation_temperature: float = DEFAULT_L2T_THOUGHT_GENERATION_TEMPERATURE
    initial_prompt_temperature: float = DEFAULT_L2T_INITIAL_PROMPT_TEMPERATURE
    max_steps: int = DEFAULT_L2T_MAX_STEPS # Max steps for the L2T process
    max_total_nodes: int = DEFAULT_L2T_MAX_TOTAL_NODES
    max_time_seconds: int = DEFAULT_L2T_MAX_TIME_SECONDS # Max time for the L2T process
    x_fmt_default: str = DEFAULT_L2T_X_FMT_DEFAULT
    x_eva_default: str = DEFAULT_L2T_X_EVA_DEFAULT


@dataclass
class L2TResult: # Remains largely unchanged, will be part of CoTResult.process_specific_data
    final_answer: Optional[str] = None
    reasoning_graph: Optional[L2TGraph] = None # Specific to L2T

    # Stats specific to the L2T process execution
    total_llm_calls: int = 0 
    total_completion_tokens: int = 0 # Tokens for L2T process steps
    total_prompt_tokens: int = 0 # Tokens for L2T process steps
    total_llm_interaction_time_seconds: float = 0.0 # LLM time for L2T process
    total_process_wall_clock_time_seconds: float = 0.0 # Wall clock for L2T process
    
    succeeded: bool = False
    error_message: Optional[str] = None
    # Note: The summary string that was part of L2TSolution will now be generated by L2TProcessor.get_summary()

# The old 'L2TSolution' dataclass is removed from this file.
# It is replaced by 'OrchestratorSolution' in 'orchestrator_solution.py'.
