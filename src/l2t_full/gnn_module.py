"""
Graph Neural Network (GNN) Module for L2T-Full.

This module defines the GNNReasoningActor, which serves as the 'Actor' in an
Actor-Critic Reinforcement Learning setup. It uses PyTorch to implement a network
that processes graph features (representing the state of the reasoning process)
and outputs parameters for an action distribution. These actions can influence
subsequent steps in the L2T-Full reasoning process, such as adjusting LLM
parameters or guiding thought generation.

Current Implementation Notes:
- Uses nn.Linear as a stand-in for true Graph Convolutional Network (GCN) layers.
  Actual graph topology (adjacency matrix) is not explicitly used by these linear layers.
  The input `graph_features` tensor is expected to be pre-processed by
  `L2TFullProcessor._featurize_graph_for_gnn` to already incorporate relevant graph context
  (e.g., features of a node and its neighbors, potentially aggregated).
- Outputs parameters for a Normal distribution, suitable for continuous actions.
- Requires PyTorch to be installed.
- TODO: Consider replacing nn.Linear GCN stand-in with actual GCN layers from a library
        like PyTorch Geometric (PyG) or Deep Graph Library (DGL) if more complex
        graph-based message passing is needed. This would require updating dependencies
        and the forward pass signature to include graph connectivity data.
- TODO: Ensure `requirements.txt` or `pyproject.toml` includes `torch`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Distribution

from typing import Any, Tuple, Dict, List # Retained for potential future use with Any if type complexity increases

class GNNReasoningActor(nn.Module):
    """
    The GNN Actor model. It processes state features (derived from the reasoning graph)
    and outputs parameters for an action distribution.
    Inherits from torch.nn.Module.
    """
    def __init__(
        self,
        input_node_dim: int,    # Dimension of the input state feature vector
        gcn_hidden_dim: int,    # Hidden dimension for the GCN stand-in layer
        mlp_hidden_dim: int,    # Hidden dimension for MLP layers
        output_action_dim: int, # Dimension of the continuous action space
    ):
        """
        Initializes the GNNReasoningActor.

        Args:
            input_node_dim: Dimensionality of the input feature vector for a state.
            gcn_hidden_dim: Output dimension of the GCN-like layer.
            mlp_hidden_dim: Hidden dimension for the MLP layers.
            output_action_dim: The number of continuous action parameters to output.
        """
        super().__init__()
        self.input_node_dim = input_node_dim
        self.gcn_hidden_dim = gcn_hidden_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.action_dim = output_action_dim

        # Define network layers
        # This layer processes the input state features.
        self.gcn_layer = nn.Linear(input_node_dim, gcn_hidden_dim)

        # MLP layers process the features further
        self.mlp_layer1 = nn.Linear(gcn_hidden_dim, mlp_hidden_dim)

        # Output layers for the Normal distribution parameters (mean and standard deviation)
        self.mean_layer = nn.Linear(mlp_hidden_dim, self.action_dim)
        self.log_std_layer = nn.Linear(mlp_hidden_dim, self.action_dim) # Output log_std for numerical stability

    def forward(
        self,
        state_features: torch.Tensor # Expected shape: [batch_size, input_node_dim]
    ) -> Tuple[Distribution, torch.Tensor]:
        """
        Performs a forward pass of the GNN model to get an action distribution.

        Args:
            state_features: A tensor representing the current state, typically derived
                            from graph features. Shape: [batch_size, input_node_dim].

        Returns:
            A tuple containing:
                - action_dist (torch.distributions.Normal): The action distribution.
                - means (torch.Tensor): The means of the action distribution (for direct use if needed).
        """
        # Pass features through GCN-like layer, then activation
        x = F.relu(self.gcn_layer(state_features)) # Output: [batch_size, gcn_hidden_dim]

        # Pass through MLP layer, then activation
        x = F.relu(self.mlp_layer1(x)) # Output: [batch_size, mlp_hidden_dim]

        # Calculate means and standard deviations for the Normal distribution
        means = self.mean_layer(x) # Output: [batch_size, action_dim]
        log_stds = self.log_std_layer(x) # Output: [batch_size, action_dim]
        stds = torch.exp(log_stds) # Ensure stds are positive
        # Clamp stds for numerical stability during training
        stds = torch.clamp(stds, min=1e-3, max=10.0)

        action_dist = Normal(means, stds) # Creates a Normal distribution for each action component
        return action_dist, means

    def evaluate_actions(
        self,
        state_features: torch.Tensor,    # Shape: [batch_size, input_node_dim]
        actions_taken: torch.Tensor      # Shape: [batch_size, action_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates actions taken in given states under the current policy.
        This is primarily used during PPO updates.

        Args:
            state_features: Batch of state features.
            actions_taken: Batch of actions that were actually taken in those states.

        Returns:
            A tuple containing:
                - log_probs (torch.Tensor): Log probabilities of the actions_taken. Shape: [batch_size].
                - entropy (torch.Tensor): Entropy of the action distribution for each state. Shape: [batch_size].
        """
        action_dist, _ = self.forward(state_features) # Get current distribution for the states
        # log_prob returns log_prob for each action component. Sum for joint log_prob of the action tuple.
        log_probs = action_dist.log_prob(actions_taken).sum(dim=-1)
        # entropy also returns per-component entropy. Sum for total entropy of the action tuple.
        entropy = action_dist.entropy().sum(dim=-1)
        return log_probs, entropy

if __name__ == "__main__":
    # This section provides a basic example of how to use the GNNReasoningActor.
    # It's useful for quick debugging or understanding the module's standalone behavior.
    print("PyTorch GNNReasoningActor Example (Updated for Distributions)")
    print(f"Torch version: {torch.__version__}")
    print("Note: This example uses nn.Linear as a stand-in for GCN layers.")
    print("Ensure PyTorch is installed (`pip install torch`). Update requirements.txt or pyproject.toml.")

    # Define example dimensions consistent with constants if possible
    INPUT_DIM = 384 # Assuming using sentence transformer embeddings
    GCN_HIDDEN = 64
    MLP_HIDDEN = 32
    ACTION_DIM = 3

    gnn_actor = GNNReasoningActor(
        input_node_dim=INPUT_DIM, gcn_hidden_dim=GCN_HIDDEN,
        mlp_hidden_dim=MLP_HIDDEN, output_action_dim=ACTION_DIM
    )
    print("\nGNN Actor Model Structure:")
    print(gnn_actor)

    # Example for a single state input (batch size 1)
    mock_single_state_features = torch.randn(1, INPUT_DIM)
    print(f"\nMock input for single state tensor shape: {mock_single_state_features.shape}")

    gnn_actor.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculations for inference
        action_distribution, action_means = gnn_actor(mock_single_state_features)
        sampled_action = action_distribution.sample()
        log_prob_sampled = action_distribution.log_prob(sampled_action).sum(dim=-1)

    print(f"\nOutput action_means tensor (single state): {action_means}")
    print(f"Sampled action from distribution (single state): {sampled_action}")
    print(f"Log probability of sampled action (single state): {log_prob_sampled}")

    # Example for a batch of states
    batch_size = 5
    batch_states = torch.randn(batch_size, INPUT_DIM)
    # Actions that might have been taken (e.g., from experience buffer)
    batch_actions_taken = torch.randn(batch_size, ACTION_DIM)

    with torch.no_grad():
        log_probs, entropy = gnn_actor.evaluate_actions(batch_states, batch_actions_taken)

    print(f"\n--- Testing evaluate_actions (batch_size={batch_size}) ---")
    print(f"Input states shape: {batch_states.shape}")
    print(f"Input actions_taken shape: {batch_actions_taken.shape}")
    print(f"Output log_probs shape: {log_probs.shape}")
    print(f"Output entropy shape: {entropy.shape}")
    print(f"Log_probs: {log_probs}")
    print(f"Entropy: {entropy}")
    print("\nReminder: Standard deviations are learned. Clamping is applied for stability.")
