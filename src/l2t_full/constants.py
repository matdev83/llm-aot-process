from typing import List

# Default model names for L2T operations
DEFAULT_L2T_CLASSIFICATION_MODEL_NAMES: List[str] = ["claude-3-haiku-20240307"]
DEFAULT_L2T_THOUGHT_GENERATION_MODEL_NAMES: List[str] = ["claude-3-sonnet-20240229"]
DEFAULT_L2T_INITIAL_PROMPT_MODEL_NAMES: List[str] = ["claude-3-sonnet-20240229"]

# Default temperatures for LLM calls
DEFAULT_L2T_CLASSIFICATION_TEMPERATURE: float = 0.5
DEFAULT_L2T_THOUGHT_GENERATION_TEMPERATURE: float = 0.7
DEFAULT_L2T_INITIAL_PROMPT_TEMPERATURE: float = 0.7

# Default max tokens for different LLM calls
DEFAULT_L2T_INITIAL_THOUGHT_MAX_TOKENS: int = 1500
DEFAULT_L2T_CLASSIFICATION_MAX_TOKENS: int = 150
DEFAULT_L2T_GENERATION_MAX_TOKENS: int = 1000 # For subsequent thought generation

# Default limits for the L2T process
DEFAULT_L2T_MAX_STEPS: int = 10
DEFAULT_L2T_MAX_TOTAL_NODES: int = 30
DEFAULT_L2T_MAX_TIME_SECONDS: int = 300  # 5 minutes

# Default values for formatting or evaluation flags/templates
DEFAULT_L2T_X_FMT_DEFAULT: str = "standard_format"
DEFAULT_L2T_X_EVA_DEFAULT: str = "standard_eval_v1" # Example evaluation template name

# Default config values for prompt construction in L2T
DEFAULT_L2T_MAX_PROMPT_NODE_REFERENCES: int = 3
DEFAULT_L2T_MAX_PROMPT_NODE_CONTEXT_TOKENS: int = 2000
DEFAULT_L2T_MAX_SIBLING_THOUGHTS_TO_INCLUDE: int = 2


# --- L2T-Full Specific Constants (GNN, RL) ---

# GNN Architecture related:
DEFAULT_GNN_INPUT_NODE_DIM: int = 384 # Changed to match all-MiniLM-L6-v2 output
DEFAULT_GNN_GCN_HIDDEN_DIM: int = 64
DEFAULT_GNN_MLP_HIDDEN_DIM: int = 32
# Example: temp_adj, num_thoughts_adj (0 to generate 1, 1 to gen 2, etc.), another_param_adj
DEFAULT_GNN_ACTION_DIM: int = 3
DEFAULT_GNN_SUBGRAPH_BETA: int = 2 # For subgraph extraction neighborhood size

# Critic Network Architecture related (for RL):
DEFAULT_CRITIC_INPUT_DIM: int = DEFAULT_GNN_GCN_HIDDEN_DIM
DEFAULT_CRITIC_HIDDEN_DIM: int = 32

# PPO Hyperparameters:
DEFAULT_PPO_LEARNING_RATE: float = 3e-4
DEFAULT_PPO_GAMMA: float = 0.99
DEFAULT_PPO_EPSILON: float = 0.2
DEFAULT_PPO_GAE_LAMBDA: float = 0.95
DEFAULT_PPO_EPOCHS: int = 10
DEFAULT_PPO_BATCH_SIZE: int = 32
DEFAULT_PPO_ENTROPY_COEFFICIENT: float = 0.01

# Mock log probability for RL agent (until actor can provide it)
MOCK_ACTION_LOG_PROB: float = -0.5

# Reward related
REWARD_FINAL_ANSWER = 100.0
REWARD_CONTINUE_SUCCESSFUL_GENERATION = 5.0
REWARD_CONTINUE_FAILED_GENERATION = -10.0
REWARD_TERMINATE_BRANCH = -2.0
REWARD_BACKTRACK = -1.0
REWARD_UNCATEGORIZED_FAILURE = -20.0
# LLM_REWARD_SCALE_FACTOR (e.g. if LLM gives 0-10, scale it to 0-50 for instance)
# TODO: Define if LLM-based reward is added.
LLM_REWARD_SCALE_FACTOR = 5.0 # Example
MIN_REWARD_SCORE = 0.0
MAX_REWARD_SCORE = 10.0 # Assuming x_eva returns a score in this range
NORMALIZED_MIN_REWARD = 0.0
NORMALIZED_MAX_REWARD = 10.0 # Max possible reward from heuristic or LLM for a single step, excluding FINAL_ANSWER

# Action interpretation scale factors
TEMP_ADJUSTMENT_SCALE = 0.2 # e.g. GNN output of 1 means +/- 0.2 to base temp
NUM_THOUGHTS_ADJUSTMENT_SCALE = 1 # e.g. GNN output of 1 means generate 1+1=2 thoughts (if applicable)
MIN_TEMP = 0.1
MAX_TEMP = 1.5
MIN_NUM_THOUGHTS = 1
MAX_NUM_THOUGHTS = 3 # If GNN controls this; current setup implies 1 per step.

DEFAULT_BASE_TOP_P = 0.9
TOP_P_ADJUSTMENT_SCALE = 0.1 # e.g. GNN output of 1 means +/- 0.1 to base top_p
MIN_TOP_P = 0.5
MAX_TOP_P = 1.0
