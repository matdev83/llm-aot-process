"""
Dataclasses for L2T-Full System.

This module defines the core data structures used throughout the L2T-Full
implementation, including configuration objects, graph elements (nodes, graph),
and result containers. These structures help maintain a clear and consistent
data flow between different components like the orchestrator, processor,
GNN, and RL agent.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# Imports from other L2T-Full modules
from .enums import L2TTriggerMode, L2TNodeCategory, L2TTerminationReason
from .constants import * # Imports all default constant values

# Imports from related projects (e.g., AOT for comparison or shared elements)
from src.aot.dataclasses import LLMCallStats
from src.aot.enums import AssessmentDecision


@dataclass
class L2TNode:
    """
    Represents a single node in the L2T reasoning graph.
    Each node typically contains a piece of thought or reasoning content.

    Attributes:
        id: Unique identifier for the node.
        content: Textual content of the thought or reasoning step.
        parent_id: Identifier of the parent node, if any.
        generation_step: The step number in the L2T process when this node was generated.
        children_ids: List of IDs of child nodes.
        category: Classification of the node's purpose (e.g., CONTINUE, FINAL_ANSWER),
                  determined by LLM classification or other logic.
        prompt_messages: The prompt messages used to generate this node's content (for audit).
        llm_response: The raw LLM response that led to this node's content (for audit).
        classification_prompt: Prompt used to classify this node (for audit).
        classification_response: Raw LLM response for this node's classification (for audit).
    """
    id: str
    content: str
    parent_id: Optional[str] # Must be provided, can be None for root.
    generation_step: int
    children_ids: List[str] = field(default_factory=list)
    category: Optional['L2TNodeCategory'] = field(default=L2TNodeCategory.UNCATEGORIZED) # Default to Uncategorized

    # Audit fields for LLM interactions
    prompt_messages: Optional[Any] = None
    llm_response: Optional[str] = None
    classification_prompt: Optional[Any] = None
    classification_response: Optional[str] = None

@dataclass
class L2TGraph:
    """
    Represents the entire reasoning graph constructed during the L2T process.

    Attributes:
        nodes: Dictionary mapping node IDs to L2TNode objects.
        v_pres: List of node IDs that are pending processing (present exploration frontier).
        v_hist: List of node IDs that have already been processed (historical path).
        root_node_id: Identifier of the root node of the graph.
    """
    nodes: Dict[str, L2TNode] = field(default_factory=dict)
    v_pres: List[str] = field(default_factory=list) # Nodes to be processed
    v_hist: List[str] = field(default_factory=list) # Nodes already processed
    root_node_id: Optional[str] = None

    def add_node(self, node: L2TNode, is_root: bool = False) -> None:
        """Adds a node to the graph and manages its links."""
        if node.id in self.nodes:
            # This could be an update or a re-addition (e.g. from backtracking)
            # For now, simple overwrite with warning. Consider specific update logic if needed.
            logging.warning(f"Node with id {node.id} already exists. Overwriting/Updating.")
        self.nodes[node.id] = node
        if node.id not in self.v_pres and node.id not in self.v_hist :
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
        """Sets the category of a specified node."""
        if node_id not in self.nodes: raise ValueError(f"Node with id {node_id} not found.")
        self.nodes[node_id].category = category

    def move_to_v_hist(self, node_id: str) -> None:
        """Moves a node from v_pres (to be processed) to v_hist (processed)."""
        if node_id not in self.nodes:
            logging.warning(f"Node {node_id} not found in graph when trying to move to v_hist. Removing from v_pres if present.")
            if node_id in self.v_pres: self.v_pres.remove(node_id)
            return
        if node_id in self.v_pres:
            self.v_pres.remove(node_id)
        if node_id not in self.v_hist:
            self.v_hist.append(node_id)

    def re_add_to_v_pres(self, node_id: str) -> None:
        """Re-adds a node to v_pres, typically for backtracking. Resets its category."""
        if node_id not in self.nodes: raise ValueError(f"Node with id {node_id} not found in graph.")
        if node_id in self.v_hist: self.v_hist.remove(node_id)
        if node_id not in self.v_pres: self.v_pres.append(node_id)

        # Reset category and classification audit fields for re-processing
        self.nodes[node_id].category = L2TNodeCategory.UNCATEGORIZED
        self.nodes[node_id].classification_prompt = None
        self.nodes[node_id].classification_response = None

    def get_node(self, node_id: str) -> Optional[L2TNode]:
        """Retrieves a node by its ID."""
        return self.nodes.get(node_id)

    def get_parent(self, node_id: str) -> Optional[L2TNode]:
        """Retrieves the parent of a specified node."""
        node = self.get_node(node_id)
        return self.get_node(node.parent_id) if node and node.parent_id else None

    def get_children(self, node_id: str) -> List[L2TNode]:
        """Retrieves all children of a specified node."""
        node = self.get_node(node_id)
        return [self.nodes[child_id] for child_id in node.children_ids if child_id in self.nodes] if node else []

    def get_graph_context_for_node_classification(
        self, node_id: str, max_prompt_node_references: int,
        max_prompt_node_context_tokens: int, max_sibling_thoughts_to_include: int
    ) -> str:
        """
        Generates a textual context of the graph portion relevant for classifying a node.
        Includes parent, siblings (up to a limit), and the node itself.

        Note: Token counting for max_prompt_node_context_tokens is not implemented here;
        this is a simplified context assembly.
        """
        # TODO: Implement robust token counting and context truncation.
        context_lines = []
        node = self.get_node(node_id)
        if not node: return "Error: Node not found for context generation."

        # Current node's content is primary
        context_lines.append(f"Current Node to Classify (ID: {node.id}):\nThought: {node.content}")

        parent = self.get_parent(node_id)
        if parent:
            context_lines.append(f"\nParent Node (ID: {parent.id}):\nThought: {parent.content}")
            # Siblings (other children of the parent)
            siblings_added = 0
            for child_id in parent.children_ids:
                if child_id != node_id and siblings_added < max_sibling_thoughts_to_include:
                    sibling_node = self.get_node(child_id)
                    if sibling_node:
                        context_lines.append(f"Sibling Node (ID: {sibling_node.id}):\nThought: {sibling_node.content}")
                        siblings_added +=1

        # Could also include children of the current node if deemed relevant for classification
        # children = self.get_children(node_id) ...

        # logging.debug(f"Graph context for node {node_id}: {' | '.join(context_lines)[:300]}...")
        return "\n\n".join(context_lines)

@dataclass
class L2TFullConfig:
    """
    Configuration for the L2T-Full system, including LLM parameters,
    process limits, GNN architecture, and RL hyperparameters.
    """
    # Core LLM parameters
    classification_model_names: List[str] = field(default_factory=lambda: DEFAULT_L2T_CLASSIFICATION_MODEL_NAMES)
    thought_generation_model_names: List[str] = field(default_factory=lambda: DEFAULT_L2T_THOUGHT_GENERATION_MODEL_NAMES)
    initial_prompt_model_names: List[str] = field(default_factory=lambda: DEFAULT_L2T_INITIAL_PROMPT_MODEL_NAMES)
    classification_temperature: float = DEFAULT_L2T_CLASSIFICATION_TEMPERATURE
    thought_generation_temperature: float = DEFAULT_L2T_THOUGHT_GENERATION_TEMPERATURE # Base temperature
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
    action_dim: int = DEFAULT_GNN_ACTION_DIM # Number of continuous actions from GNN
    gnn_subgraph_beta: int = DEFAULT_GNN_SUBGRAPH_BETA

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

    # Original L2T fields (may or may not be used by L2T-Full directly)
    x_fmt_default: str = DEFAULT_L2T_X_FMT_DEFAULT
    x_eva_default: str = DEFAULT_L2T_X_EVA_DEFAULT
    pass_remaining_steps_pct: Optional[float] = None

@dataclass
class L2TResult:
    """
    Holds the results of a single run of the L2TFullProcessor.
    """
    problem: str
    start_time: float # Timestamp of when processing started
    end_time: Optional[float] = None # Timestamp of when processing ended
    final_answer: Optional[str] = None
    reasoning_graph: Optional[L2TGraph] = None
    total_llm_calls: int = 0
    total_completion_tokens: int = 0
    total_prompt_tokens: int = 0
    total_llm_interaction_time_seconds: float = 0.0
    total_processing_time_seconds: float = 0.0 # Wall-clock time for the run
    succeeded: Optional[bool] = None # True if final_answer found, False otherwise
    error_message: Optional[str] = None
    solution_node_id: Optional[str] = None # ID of the node containing the final answer
    termination_reason: Optional[L2TTerminationReason] = None
    total_nodes: int = 0 # Total nodes created in the graph
    total_steps: int = 0 # Total reasoning steps taken

@dataclass
class L2TFullSolution:
    """
    Represents the overall solution from the L2TFullOrchestrator,
    which might involve an assessment, a direct one-shot call, or the full L2T-Full process.
    """
    final_answer: Optional[str] = None
    total_wall_clock_time_seconds: float = 0.0 # Total time for orchestrator.solve()

    # Assessment related (if ASSESS_FIRST mode is used)
    assessment_stats: Optional[LLMCallStats] = None
    assessment_decision: Optional[AssessmentDecision] = None

    # Direct one-shot related (if L2T-Full is bypassed or not triggered by orchestrator)
    main_call_stats: Optional[LLMCallStats] = None

    # L2T-Full process related results
    l2t_full_result: Optional[L2TResult] = None
    l2t_full_process_summary: Optional[str] = None # Summary string from L2TFullProcess

    # Fallback related (if L2T-Full process itself fails and orchestrator falls back)
    l2t_failed_and_fell_back: bool = False
    fallback_call_stats: Optional[LLMCallStats] = None

    @property
    def succeeded(self) -> bool:
        """Determines if the overall orchestration process yielded a final answer."""
        return self.final_answer is not None

    # Properties to aggregate token counts and times from different stages
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

# Ensure logging is imported for L2TGraph warning if not already top-level
import logging
