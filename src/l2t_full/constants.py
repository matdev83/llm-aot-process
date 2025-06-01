"""
Constants for the L2T-Full (Learn to Think - Full Implementation) System.

This module defines shared constants and default hyperparameter values used across
the L2T-Full components, including the core reasoning processor, GNN, and RL agent.
"""
from typing import List

# Default model names for L2T operations
DEFAULT_L2T_CLASSIFICATION_MODEL_NAMES: List[str] = ["claude-3-haiku-20240307"]
DEFAULT_L2T_THOUGHT_GENERATION_MODEL_NAMES: List[str] = ["claude-3-sonnet-20240229"]
DEFAULT_L2T_INITIAL_PROMPT_MODEL_NAMES: List[str] = ["claude-3-sonnet-20240229"]

# Default temperatures for LLM calls
DEFAULT_L2T_CLASSIFICATION_TEMPERATURE: float = 0.5
DEFAULT_L2T_THOUGHT_GENERATION_TEMPERATURE: float = 0.7 # Base temperature for thoughts
DEFAULT_L2T_INITIAL_PROMPT_TEMPERATURE: float = 0.7

# Default max tokens for different LLM calls
DEFAULT_L2T_INITIAL_THOUGHT_MAX_TOKENS: int = 1500
DEFAULT_L2T_CLASSIFICATION_MAX_TOKENS: int = 150
DEFAULT_L2T_GENERATION_MAX_TOKENS: int = 1000

# Default limits for the L2T process
DEFAULT_L2T_MAX_STEPS: int = 10
DEFAULT_L2T_MAX_TOTAL_NODES: int = 30
DEFAULT_L2T_MAX_TIME_SECONDS: int = 300  # 5 minutes

# Default values for formatting or evaluation flags/templates
DEFAULT_L2T_X_FMT_DEFAULT: str = "standard_format"
DEFAULT_L2T_X_EVA_DEFAULT: str = "standard_eval_v1"

# Default config values for prompt construction in L2T
DEFAULT_L2T_MAX_PROMPT_NODE_REFERENCES: int = 3
DEFAULT_L2T_MAX_PROMPT_NODE_CONTEXT_TOKENS: int = 2000
DEFAULT_L2T_MAX_SIBLING_THOUGHTS_TO_INCLUDE: int = 2


# --- L2T-Full Specific Constants (GNN, RL) ---

# GNN Architecture related:
DEFAULT_GNN_INPUT_NODE_DIM: int = 384 # Matches all-MiniLM-L6-v2 sentence transformer output
DEFAULT_GNN_GCN_HIDDEN_DIM: int = 128 # Hidden dimension after GCN-like processing
DEFAULT_GNN_MLP_HIDDEN_DIM: int = 64  # Hidden dimension in subsequent MLPs
DEFAULT_GNN_ACTION_DIM: int = 3 # Number of continuous actions GNN outputs (e.g., temp_adj, top_p_adj, custom_adj)
DEFAULT_GNN_SUBGRAPH_BETA: int = 2 # Neighborhood depth for subgraph extraction for GNN input

# Critic Network Architecture related (for RL):
# Input to Critic is typically a state representation. If it's the output of GNN's GCN layer:
DEFAULT_CRITIC_INPUT_DIM: int = DEFAULT_GNN_GCN_HIDDEN_DIM
DEFAULT_CRITIC_HIDDEN_DIM: int = 64 # Hidden dimension for Critic's MLP

# PPO Hyperparameters:
DEFAULT_PPO_LEARNING_RATE: float = 3e-5 # Common learning rate for Adam
DEFAULT_PPO_GAMMA: float = 0.99 # Discount factor for future rewards
DEFAULT_PPO_EPSILON: float = 0.2 # PPO clipping parameter
DEFAULT_PPO_GAE_LAMBDA: float = 0.95 # Lambda for Generalized Advantage Estimation
DEFAULT_PPO_EPOCHS: int = 10 # Number of optimization epochs per PPO update
DEFAULT_PPO_BATCH_SIZE: int = 32 # Batch size for PPO updates
DEFAULT_PPO_ENTROPY_COEFFICIENT: float = 0.01 # Encourages exploration

# Mock log probability for RL agent (used when actual log_prob from GNN isn't fully integrated for PPO yet)
MOCK_ACTION_LOG_PROB: float = -0.5 # Example value, represents log(0.606) approx.

# Reward related constants
REWARD_FINAL_ANSWER: float = 100.0
REWARD_CONTINUE_SUCCESSFUL_GENERATION: float = 5.0 # Base reward for generating a new thought
REWARD_CONTINUE_FAILED_GENERATION: float = -10.0 # Penalty for failing to generate a thought on CONTINUE
REWARD_TERMINATE_BRANCH: float = -2.0 # Penalty for terminating a branch (could be neutral if GNN decides well)
REWARD_BACKTRACK: float = -1.0 # Small penalty, as it's a strategic move
REWARD_UNCATEGORIZED_FAILURE: float = -20.0 # Larger penalty for parsing failures or unexpected states

# LLM-based reward scaling (if/when implemented)
# TODO: Define and use if LLM-based reward scoring is added for CONTINUE steps.
# LLM_REWARD_SCALE_FACTOR: float = 5.0 # Example: if LLM gives 0-10, scale it to 0-50.
# MIN_REWARD_SCORE: float = 0.0 # Min score from LLM evaluation
# MAX_REWARD_SCORE: float = 10.0 # Max score from LLM evaluation
# NORMALIZED_MIN_REWARD: float = 0.0 # For heuristic rewards if they need normalization
# NORMALIZED_MAX_REWARD: float = 10.0

# Action interpretation scale factors and limits for GNN outputs
TEMP_ADJUSTMENT_SCALE: float = 0.2 # GNN output of 1.0 -> +/- 0.2 adjustment
MIN_TEMP: float = 0.1 # Minimum allowed temperature
MAX_TEMP: float = 1.5 # Maximum allowed temperature

NUM_THOUGHTS_ADJUSTMENT_SCALE: int = 1 # GNN output of 1.0 -> base_thoughts + 1
MIN_NUM_THOUGHTS: int = 1 # Minimum number of thoughts to generate (processor currently does 1)
MAX_NUM_THOUGHTS: int = 3 # Maximum if GNN were to control parallel thought generation

DEFAULT_BASE_TOP_P: float = 0.9 # Default base top_p for LLM calls
TOP_P_ADJUSTMENT_SCALE: float = 0.1 # GNN output of 1.0 -> +/- 0.1 adjustment
MIN_TOP_P: float = 0.5 # Minimum allowed top_p
MAX_TOP_P: float = 1.0 # Maximum allowed top_p (often 0.99 or 1.0)
