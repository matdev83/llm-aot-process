"""
Enumerations for the L2T-Full System.

This module defines various Enum types used to represent states, categories,
and modes within the L2T-Full reasoning process.
"""
from enum import Enum

class L2TTriggerMode(Enum):
    """
    Defines how the L2T-Full process is triggered by an orchestrator.
    - ALWAYS_L2T: Always run the full L2T-Full process.
    - NEVER_L2T: Never run L2T-Full; typically use a direct one-shot LLM call instead.
    - ASSESS_FIRST: Use an assessor model (with optional heuristic shortcut)
                    to decide between a one-shot LLM call and the L2T-Full process.
    """
    ALWAYS_L2T = "always_l2t"
    NEVER_L2T = "never_l2t"
    ASSESS_FIRST = "assess_first"

class L2TNodeCategory(Enum):
    """
    Represents the semantic classification of a node in the L2T reasoning graph.
    This category guides the L2TFullProcessor on how to proceed after a node is evaluated.
    """
    CONTINUE = "CONTINUE"  # Further processing or thought generation is needed from this node.
    FINAL_ANSWER = "FINAL_ANSWER"  # This node contains the final answer to the problem.
    TERMINATE_BRANCH = "TERMINATE_BRANCH"  # This branch of reasoning should be stopped; it's unpromising.
    BACKTRACK = "BACKTRACK"  # The current path is stuck or suboptimal; backtrack to a previous node.
    UNCATEGORIZED = "UNCATEGORIZED" # Default state for a new node before classification.

class L2TTerminationReason(Enum):
    """
    Reason why the L2T-Full process (specifically L2TFullProcessor.run()) terminated.
    """
    SOLUTION_FOUND = "SOLUTION_FOUND"           # A node was classified as FINAL_ANSWER.
    MAX_STEPS_REACHED = "MAX_STEPS_REACHED"       # Maximum configured reasoning steps exceeded.
    MAX_TIME_REACHED = "MAX_TIME_REACHED"        # Maximum configured wall-clock time exceeded.
    MAX_NODES_REACHED = "MAX_NODES_REACHED"       # Maximum configured graph nodes exceeded.
    NO_NODES_TO_PROCESS = "NO_NODES_TO_PROCESS" # v_pres (nodes to process) became empty.
    ERROR = "ERROR"                             # An unexpected internal error occurred.
    USER_INTERRUPT = "USER_INTERRUPT"           # Process was interrupted by user (if applicable).
    LLM_FAILURE = "LLM_FAILURE"                 # Unrecoverable LLM call failure (e.g., persistent API errors).
    CONFIG_VALIDATION_FAILED = "CONFIG_VALIDATION_FAILED" # L2TFullConfig validation failed.
    INITIAL_THOUGHT_FAILED = "INITIAL_THOUGHT_FAILED" # Could not generate or parse the initial thought.
    CLASSIFICATION_FAILED_TERMINAL = "CLASSIFICATION_FAILED_TERMINAL" # Node classification failed critically.
    UNKNOWN = "UNKNOWN"                         # Default or unspecified reason for termination.
