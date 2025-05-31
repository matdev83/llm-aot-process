from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from src.aot.dataclasses import LLMCallStats
from src.aot.enums import AssessmentDecision
from .enums import L2TTriggerMode, L2TNodeCategory, L2TTerminationReason

# Import all default constants, including GNN/RL ones
from .constants import *

@dataclass
class L2TNode:
    id: str
    content: str
    parent_id: Optional[str]
    generation_step: int
    children_ids: List[str] = field(default_factory=list)
    category: Optional['L2TNodeCategory'] = None
    prompt_messages: Optional[Any] = None
    llm_response: Optional[str] = None
    classification_prompt: Optional[Any] = None
    classification_response: Optional[str] = None

@dataclass
class L2TGraph:
    nodes: Dict[str, L2TNode] = field(default_factory=dict)
    v_pres: List[str] = field(default_factory=list)
    v_hist: List[str] = field(default_factory=list)
    root_node_id: Optional[str] = None

    def add_node(self, node: L2TNode, is_root: bool = False) -> None:
        if node.id in self.nodes:
            print(f"Warning: Node with id {node.id} already exists. Overwriting/Updating.")
        self.nodes[node.id] = node
        if node.id not in self.v_pres and node.id not in self.v_hist:
            self.v_pres.append(node.id)
        if is_root:
            if self.root_node_id is not None and self.root_node_id != node.id:
                raise ValueError(f"Root node already set to {self.root_node_id}. Cannot set {node.id} as root.")
            self.root_node_id = node.id
        if node.parent_id and node.parent_id in self.nodes:
            parent_node = self.nodes[node.parent_id]
            if node.id not in parent_node.children_ids:
                parent_node.children_ids.append(node.id)

    def classify_node(self, node_id: str, category: L2TNodeCategory) -> None:
        if node_id not in self.nodes: raise ValueError(f"Node with id {node_id} not found.")
        self.nodes[node_id].category = category

    def move_to_v_hist(self, node_id: str) -> None:
        if node_id not in self.nodes:
            print(f"Warning: Node {node_id} not found in graph when trying to move to v_hist. Skipping.")
            if node_id in self.v_pres: self.v_pres.remove(node_id)
            return
        if node_id in self.v_pres: self.v_pres.remove(node_id)
        if node_id not in self.v_hist: self.v_hist.append(node_id)

    def re_add_to_v_pres(self, node_id: str) -> None:
        if node_id not in self.nodes: raise ValueError(f"Node with id {node_id} not found in graph.")
        if node_id in self.v_hist: self.v_hist.remove(node_id)
        if node_id not in self.v_pres: self.v_pres.append(node_id)
        self.nodes[node_id].category = L2TNodeCategory.UNCATEGORIZED
        self.nodes[node_id].classification_prompt = None
        self.nodes[node_id].classification_response = None

    def get_node(self, node_id: str) -> Optional[L2TNode]: return self.nodes.get(node_id)
    def get_parent(self, node_id: str) -> Optional[L2TNode]:
        node = self.get_node(node_id)
        return self.get_node(node.parent_id) if node and node.parent_id else None
    def get_children(self, node_id: str) -> List[L2TNode]:
        node = self.get_node(node_id)
        return [self.nodes[child_id] for child_id in node.children_ids if child_id in self.nodes] if node else []

    def get_graph_context_for_node_classification(
        self, node_id: str, max_prompt_node_references: int,
        max_prompt_node_context_tokens: int, max_sibling_thoughts_to_include: int
    ) -> str:
        context_lines = []
        node = self.get_node(node_id)
        if not node: return "Error: Node not found for context."
        context_lines.append(f"Current Node ({node.id}): {node.content}")
        parent = self.get_parent(node_id)
        if parent: context_lines.append(f"Parent Node ({parent.id}): {parent.content}")
        siblings_added = 0
        if parent:
            for child_id in parent.children_ids:
                if child_id != node_id and siblings_added < max_sibling_thoughts_to_include:
                    sibling_node = self.get_node(child_id)
                    if sibling_node:
                        context_lines.append(f"Sibling Node ({sibling_node.id}): {sibling_node.content}")
                        siblings_added +=1
        return "\n".join(context_lines)

@dataclass
class L2TFullConfig:
    # Core LLM parameters
    classification_model_names: List[str] = field(default_factory=lambda: DEFAULT_L2T_CLASSIFICATION_MODEL_NAMES)
    thought_generation_model_names: List[str] = field(default_factory=lambda: DEFAULT_L2T_THOUGHT_GENERATION_MODEL_NAMES)
    initial_prompt_model_names: List[str] = field(default_factory=lambda: DEFAULT_L2T_INITIAL_PROMPT_MODEL_NAMES)
    classification_temperature: float = DEFAULT_L2T_CLASSIFICATION_TEMPERATURE
    thought_generation_temperature: float = DEFAULT_L2T_THOUGHT_GENERATION_TEMPERATURE
    initial_prompt_temperature: float = DEFAULT_L2T_INITIAL_PROMPT_TEMPERATURE
    initial_thought_max_tokens: int = DEFAULT_L2T_INITIAL_THOUGHT_MAX_TOKENS
    classification_max_tokens: int = DEFAULT_L2T_CLASSIFICATION_MAX_TOKENS
    generation_max_tokens: int = DEFAULT_L2T_GENERATION_MAX_TOKENS

    # Process limits
    max_steps: int = DEFAULT_L2T_MAX_STEPS
    max_total_nodes: int = DEFAULT_L2T_MAX_TOTAL_NODES
    max_time_seconds: int = DEFAULT_L2T_MAX_TIME_SECONDS

    # Prompting related configs
    max_prompt_node_references: int = DEFAULT_L2T_MAX_PROMPT_NODE_REFERENCES
    max_prompt_node_context_tokens: int = DEFAULT_L2T_MAX_PROMPT_NODE_CONTEXT_TOKENS
    max_sibling_thoughts_to_include: int = DEFAULT_L2T_MAX_SIBLING_THOUGHTS_TO_INCLUDE

    # GNN Architecture related:
    gnn_input_dim: int = DEFAULT_GNN_INPUT_NODE_DIM
    gcn_hidden_dim: int = DEFAULT_GNN_GCN_HIDDEN_DIM
    mlp_hidden_dim: int = DEFAULT_GNN_MLP_HIDDEN_DIM
    action_dim: int = DEFAULT_GNN_ACTION_DIM
    gnn_subgraph_beta: int = DEFAULT_GNN_SUBGRAPH_BETA # New field

    # Critic Network Architecture related (for RL):
    critic_input_dim: int = DEFAULT_CRITIC_INPUT_DIM
    critic_hidden_dim: int = DEFAULT_CRITIC_HIDDEN_DIM

    # PPO Hyperparameters:
    ppo_learning_rate: float = DEFAULT_PPO_LEARNING_RATE
    ppo_gamma: float = DEFAULT_PPO_GAMMA
    ppo_epsilon: float = DEFAULT_PPO_EPSILON
    ppo_gae_lambda: float = DEFAULT_PPO_GAE_LAMBDA
    ppo_epochs: int = DEFAULT_PPO_EPOCHS
    ppo_batch_size: int = DEFAULT_PPO_BATCH_SIZE
    ppo_entropy_coefficient: float = DEFAULT_PPO_ENTROPY_COEFFICIENT

    # Original L2T fields
    x_fmt_default: str = DEFAULT_L2T_X_FMT_DEFAULT
    x_eva_default: str = DEFAULT_L2T_X_EVA_DEFAULT
    pass_remaining_steps_pct: Optional[float] = None

@dataclass
class L2TResult:
    problem: str
    start_time: float
    end_time: Optional[float] = None
    final_answer: Optional[str] = None
    reasoning_graph: Optional[L2TGraph] = None
    total_llm_calls: int = 0
    total_completion_tokens: int = 0
    total_prompt_tokens: int = 0
    total_llm_interaction_time_seconds: float = 0.0
    total_processing_time_seconds: float = 0.0
    succeeded: Optional[bool] = None
    error_message: Optional[str] = None
    solution_node_id: Optional[str] = None
    termination_reason: Optional[L2TTerminationReason] = None
    total_nodes: int = 0
    total_steps: int = 0

@dataclass
class L2TFullSolution:
    final_answer: Optional[str] = None
    total_wall_clock_time_seconds: float = 0.0
    assessment_stats: Optional[LLMCallStats] = None
    assessment_decision: Optional[AssessmentDecision] = None
    main_call_stats: Optional[LLMCallStats] = None
    l2t_full_result: Optional[L2TResult] = None
    l2t_full_process_summary: Optional[str] = None
    l2t_failed_and_fell_back: bool = False
    fallback_call_stats: Optional[LLMCallStats] = None

    @property
    def succeeded(self) -> bool: return self.final_answer is not None
    @property
    def total_completion_tokens(self) -> int:
        total = 0
        if self.assessment_stats: total += self.assessment_stats.completion_tokens
        if self.main_call_stats: total += self.main_call_stats.completion_tokens
        if self.l2t_full_result: total += self.l2t_full_result.total_completion_tokens
        if self.fallback_call_stats: total += self.fallback_call_stats.completion_tokens
        return total
    @property
    def total_prompt_tokens(self) -> int:
        total = 0
        if self.assessment_stats: total += self.assessment_stats.prompt_tokens
        if self.main_call_stats: total += self.main_call_stats.prompt_tokens
        if self.l2t_full_result: total += self.l2t_full_result.total_prompt_tokens
        if self.fallback_call_stats: total += self.fallback_call_stats.prompt_tokens
        return total
    @property
    def grand_total_tokens(self) -> int: return self.total_completion_tokens + self.total_prompt_tokens
    @property
    def total_llm_interaction_time_seconds(self) -> float:
        total = 0.0
        if self.assessment_stats: total += self.assessment_stats.call_duration_seconds
        if self.main_call_stats: total += self.main_call_stats.call_duration_seconds
        if self.l2t_full_result: total += self.l2t_full_result.total_llm_interaction_time_seconds
        if self.fallback_call_stats: total += self.fallback_call_stats.call_duration_seconds
        return total
