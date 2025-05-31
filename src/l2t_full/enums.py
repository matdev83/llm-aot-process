from enum import Enum

class L2TTriggerMode(Enum):
    """
    Defines how the L2T process is triggered.
    - ALWAYS_L2T: Always run the L2T process.
    - NEVER_L2T: Never run the L2T process, always use a direct one-shot LLM call.
    - ASSESS_FIRST: Use an assessor model (with optional heuristic shortcut)
                    to decide between a one-shot LLM call and the L2T process.
    """
    ALWAYS_L2T = "always_l2t"
    NEVER_L2T = "never_l2t"
    ASSESS_FIRST = "assess_first"

class L2TNodeCategory(Enum):
    """
    Represents the classification of a node in the L2T graph.
    """
    CONTINUE = "CONTINUE"  # Further processing or thought generation is needed.
    FINAL_ANSWER = "FINAL_ANSWER"  # This node contains the final answer.
    TERMINATE_BRANCH = "TERMINATE_BRANCH"  # This branch of reasoning should be stopped.
    BACKTRACK = "BACKTRACK"  # Backtrack to a previous node and explore a different path.
    UNCATEGORIZED = "UNCATEGORIZED" # Default state before classification

class L2TTerminationReason(Enum):
    """
    Reason why the L2T process terminated.
    """
    SOLUTION_FOUND = "SOLUTION_FOUND"
    MAX_STEPS_REACHED = "MAX_STEPS_REACHED"
    MAX_TIME_REACHED = "MAX_TIME_REACHED"
    MAX_NODES_REACHED = "MAX_NODES_REACHED"
    NO_NODES_TO_PROCESS = "NO_NODES_TO_PROCESS" # v_pres became empty
    ERROR = "ERROR" # An unexpected error occurred
    USER_INTERRUPT = "USER_INTERRUPT" # If user interaction was possible
    LLM_FAILURE = "LLM_FAILURE" # Unrecoverable LLM call failure
    UNKNOWN = "UNKNOWN" # Default or unspecified reason
    CONFIG_VALIDATION_FAILED = "CONFIG_VALIDATION_FAILED" # Invalid L2T configuration
    INITIAL_THOUGHT_FAILED = "INITIAL_THOUGHT_FAILED" # Could not generate initial thought
    CLASSIFICATION_FAILED_TERMINAL = "CLASSIFICATION_FAILED_TERMINAL" # Classification failed and no recovery

# Example of how these might be used by L2TFullProcessor:
# node.category = L2TNodeCategory.CONTINUE
# result.termination_reason = L2TTerminationReason.MAX_STEPS_REACHED
