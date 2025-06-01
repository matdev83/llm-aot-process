"""
L2T-Full Package Initialization.

This package implements the full "Learn to Think" (L2T-Full) system,
which integrates Graph Neural Networks (GNNs) for policy guidance and
Reinforcement Learning (RL) for training the reasoning process.

Modules:
    - orchestrator: Manages the overall L2T-Full execution flow.
    - processor: Implements the core step-by-step reasoning graph construction.
    - gnn_module: Defines the GNN model (Actor) for action selection.
    - rl_module: Defines the Critic model and PPO learning algorithm.
    - dataclasses: Data structures for configuration, graph, results, etc.
    - constants: Shared constants and default hyperparameters.
    - enums: Enumerations used throughout the L2T-Full system.
    - prompt_generator: (Potentially specialized) prompt generation logic.
    - response_parser: (Potentially specialized) response parsing logic.
"""

# This makes it easier to import classes directly from src.l2t_full
# e.g., from src.l2t_full import L2TFullOrchestrator

from .orchestrator import L2TFullOrchestrator, L2TFullProcess
from .processor import L2TFullProcessor
from .gnn_module import GNNReasoningActor
from .rl_module import CriticValueNetwork, PPOAgent
from .dataclasses import L2TFullConfig, L2TResult, L2TFullSolution, L2TNode, L2TGraph
from .enums import L2TTriggerMode, L2TNodeCategory, L2TTerminationReason
from .constants import * # Import all constants for easier access if needed at package level

__all__ = [
    "L2TFullOrchestrator",
    "L2TFullProcess",
    "L2TFullProcessor",
    "GNNReasoningActor",
    "CriticValueNetwork",
    "PPOAgent",
    "L2TFullConfig",
    "L2TResult",
    "L2TFullSolution",
    "L2TNode",
    "L2TGraph",
    "L2TTriggerMode",
    "L2TNodeCategory",
    "L2TTerminationReason",
    # Add other important classes/functions if they should be part of the public API
]

# TODO: Add dependency notes for PyTorch and sentence-transformers to project README
#       or requirements.txt / pyproject.toml.
# PyTorch: Used in gnn_module.py, rl_module.py, and processor.py (for tensors).
# Sentence-Transformers: Used in processor.py for graph featurization.
