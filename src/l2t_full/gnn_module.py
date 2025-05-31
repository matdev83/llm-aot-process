# TODO: This module requires a graph neural network library (e.g., PyTorch Geometric or DGL)
# for true graph convolutions. For now, nn.Linear is used as a placeholder for GCN effect.
# Ensure PyTorch (torch) is added to project dependencies.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Distribution

from typing import Any, Tuple, Dict, List

class GNNReasoningActor(nn.Module):
    def __init__(
        self,
        input_node_dim: int,
        gcn_hidden_dim: int,
        mlp_hidden_dim: int,
        output_action_dim: int, # This now represents the dimension of the continuous action space
    ):
        super().__init__()
        self.input_node_dim = input_node_dim
        self.gcn_hidden_dim = gcn_hidden_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.action_dim = output_action_dim # Number of continuous actions

        self.gcn_layer = nn.Linear(input_node_dim, gcn_hidden_dim)
        self.mlp_layer1 = nn.Linear(gcn_hidden_dim, mlp_hidden_dim)

        # Output layer for means of the Normal distribution
        self.mean_layer = nn.Linear(mlp_hidden_dim, self.action_dim)
        # Output layer for standard deviations (log_stds for stability, then softplus)
        self.log_std_layer = nn.Linear(mlp_hidden_dim, self.action_dim)

    def forward(
        self,
        graph_features: torch.Tensor
    ) -> Tuple[Distribution, torch.Tensor]: # Returns distribution and raw mean params for interpretation
        """
        Performs a forward pass of the GNN model.
        Outputs parameters for a Normal distribution over continuous actions.
        Assumes graph_features is of shape [batch_size, input_node_dim].
        If batch_size is 1, it processes a single state.
        """
        # graph_features is expected to be [batch_size, input_node_dim]
        # Each row is an already featurized state (e.g., aggregated from a subgraph by the processor)
        x = F.relu(self.gcn_layer(graph_features)) # Output: [batch_size, gcn_hidden_dim]
        # The aggregation logic for multiple nodes forming ONE state is now handled
        # in the processor's _featurize_graph_for_gnn method before calling this.
        # This 'x' is now the "state embedding" for each item in the batch.
        x = F.relu(self.mlp_layer1(x)) # Output: [batch_size, mlp_hidden_dim]

        means = self.mean_layer(x) # Output: [batch_size, action_dim]
        log_stds = self.log_std_layer(x) # Output: [batch_size, action_dim]
        stds = torch.exp(log_stds)
        stds = torch.clamp(stds, min=1e-3, max=10.0)

        action_dist = Normal(means, stds) # Distribution over a batch of actions
        return action_dist, means

    def evaluate_actions(
        self,
        state_features: torch.Tensor,
        actions_taken: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates actions taken in given states.
        Used in PPO update to calculate log_probs of actions under the current policy
        and the policy's entropy.
        """
        action_dist, _ = self.forward(state_features) # Get current distribution for the state
        log_probs = action_dist.log_prob(actions_taken).sum(dim=-1) # Sum over action dimensions if multi-dim action
        entropy = action_dist.entropy().sum(dim=-1) # Sum over action dimensions
        return log_probs, entropy

if __name__ == "__main__":
    print("PyTorch GNNReasoningActor Example (Updated for Distributions)")
    print(f"Torch version: {torch.__version__}")

    INPUT_DIM = 128
    GCN_HIDDEN = 64
    MLP_HIDDEN = 32
    ACTION_DIM = 3

    gnn_actor = GNNReasoningActor(
        input_node_dim=INPUT_DIM, gcn_hidden_dim=GCN_HIDDEN,
        mlp_hidden_dim=MLP_HIDDEN, output_action_dim=ACTION_DIM
    )
    print("\nGNN Actor Model Structure:")
    print(gnn_actor)

    mock_node_features = torch.randn(1, INPUT_DIM)
    print(f"\nMock input graph_features tensor shape: {mock_node_features.shape}")

    gnn_actor.eval()
    with torch.no_grad():
        # Forward now returns a distribution and the means
        action_distribution, action_means = gnn_actor(mock_node_features)
        sampled_action = action_distribution.sample()
        log_prob_sampled = action_distribution.log_prob(sampled_action).sum(dim=-1) # Sum if multi-dim

    print(f"\nOutput action_means tensor: {action_means}")
    print(f"Sampled action from distribution: {sampled_action}")
    print(f"Log probability of sampled action: {log_prob_sampled}")

    # Test evaluate_actions
    # Assume we have a batch of states and actions
    batch_size = 5
    batch_states = torch.randn(batch_size, INPUT_DIM)
    # Actions that might have been taken (these would come from experience buffer)
    # For Normal distribution, actions are continuous values
    batch_actions_taken = torch.randn(batch_size, ACTION_DIM)

    with torch.no_grad(): # Typically evaluation is no_grad unless it's part of loss computation graph
        log_probs, entropy = gnn_actor.evaluate_actions(batch_states, batch_actions_taken)

    print(f"\n--- Testing evaluate_actions (batch_size={batch_size}) ---")
    print(f"Input states shape: {batch_states.shape}")
    print(f"Input actions_taken shape: {batch_actions_taken.shape}")
    print(f"Output log_probs shape: {log_probs.shape}") # Should be [batch_size]
    print(f"Output entropy shape: {entropy.shape}")   # Should be [batch_size]
    print(f"Log_probs: {log_probs}")
    print(f"Entropy: {entropy}")
    print("\nReminder: Standard deviations are now learned. Clamping is applied for stability.")
