from typing import List

# Default model names for different ToT tasks
DEFAULT_TOT_THOUGHT_GENERATION_MODEL_NAMES: List[str] = ["openai/gpt-3.5-turbo"]
DEFAULT_TOT_EVALUATION_MODEL_NAMES: List[str] = ["openai/gpt-3.5-turbo"] # Can be a smaller/faster model

# Default temperatures for LLM calls in ToT
DEFAULT_TOT_THOUGHT_GENERATION_TEMPERATURE: float = 0.7
DEFAULT_TOT_EVALUATION_TEMPERATURE: float = 0.5
DEFAULT_TOT_DIRECT_ONESHOT_TEMPERATURE: float = 0.7 # For ToT's fallback/direct one-shot
DEFAULT_TOT_ASSESSMENT_TEMPERATURE: float = 0.4    # For ToT's assessment step

# Default ToT algorithm parameters
DEFAULT_K_THOUGHTS: int = 3  # Number of thoughts to generate per node
DEFAULT_B_BEAM_WIDTH: int = 2 # Beam width for beam search
DEFAULT_MAX_DEPTH: int = 5    # Maximum depth of the tree
DEFAULT_MAX_THOUGHTS_IN_STATE: int = 10 # Max thoughts allowed in a single state path to prevent overly long states
DEFAULT_MAX_TOTAL_THOUGHTS_GENERATED: int = 50 # Overall limit on thoughts generated to control cost/time
DEFAULT_MAX_TIME_SECONDS: int = 300 # Max time for the ToT process

# Fallback values for dynamic step prediction (if needed, similar to AoT)
MIN_PREDICTED_STEP_TOKENS_FALLBACK_TOT: int = 50
MIN_PREDICTED_STEP_DURATION_FALLBACK_TOT: float = 5.0
