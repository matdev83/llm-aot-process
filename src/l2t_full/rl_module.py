"""
Reinforcement Learning (RL) Module for L2T-Full.

This module defines the RL components used in the L2T-Full system, including:
- CriticValueNetwork: Estimates the value of a state V(s).
- PPOAgent: Implements the Proximal Policy Optimization (PPO) algorithm
            to train the GNNReasoningActor (policy) and the CriticValueNetwork.

It uses PyTorch for neural network implementations and RL algorithm calculations.

TODO:
- Ensure `requirements.txt` or `pyproject.toml` includes `torch`.
- Consider more sophisticated replay buffer strategies if needed.
- Hyperparameter tuning for PPO will be crucial for effective training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Distribution

from typing import Any, List, Tuple, Optional, Dict
import logging

from .gnn_module import GNNReasoningActor # GNN Actor is now a PyTorch nn.Module

logger = logging.getLogger(__name__)

class CriticValueNetwork(nn.Module):
    """
    The Critic Network (V(s)).
    Estimates the value of a given state, which is used to calculate advantages in PPO.
    Inherits from torch.nn.Module.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initializes the CriticValueNetwork.

        Args:
            input_dim: Dimensionality of the input state features. This should typically
                       align with the state representation used by the Actor.
            hidden_dim: Dimensionality of the hidden layer in the MLP.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Define network layers
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1) # Outputs a single scalar value (the state value)

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the Critic network.

        Args:
            state_features: Numerical representation of the state (PyTorch Tensor).
                            Expected shape: [batch_size, input_dim].

        Returns:
            value: The estimated value of the input state(s) (PyTorch Tensor).
                   Shape: [batch_size, 1].
        """
        x = F.relu(self.layer1(state_features))
        value = self.layer2(x)
        return value

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent.
    Manages the Actor (GNNReasoningActor) and Critic (CriticValueNetwork),
    collects experiences, and performs PPO updates to train both networks.
    """
    def __init__(
        self,
        actor: GNNReasoningActor,
        critic: CriticValueNetwork,
        learning_rate: float = 3e-4, # Note: Separate LRs for actor/critic can be used
        gamma: float = 0.99,          # Discount factor
        epsilon: float = 0.2,         # PPO clipping parameter
        gae_lambda: float = 0.95,     # Lambda for Generalized Advantage Estimation
        ppo_epochs: int = 10,         # Number of optimization epochs per PPO update
        batch_size: int = 64,         # Minibatch size for PPO updates from buffer
        entropy_coefficient: float = 0.01 # Weight for entropy bonus in actor loss
    ):
        """
        Initializes the PPOAgent.

        Args:
            actor: The GNNReasoningActor instance (policy).
            critic: The CriticValueNetwork instance (value function).
            learning_rate: Learning rate for Adam optimizers.
            gamma: Discount factor for future rewards.
            epsilon: Clipping parameter for PPO's surrogate objective.
            gae_lambda: Factor for Generalized Advantage Estimation.
            ppo_epochs: Number of epochs to train on the collected data per update.
            batch_size: Number of experiences to sample from the buffer for each PPO epoch.
            entropy_coefficient: Coefficient for the entropy bonus in the actor's loss.
        """
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.epsilon = epsilon
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.entropy_coefficient = entropy_coefficient

        # Initialize PyTorch optimizers for actor and critic networks
        self.actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)

        # Buffer to store experiences (tuples of tensors and other data)
        self.experience_buffer: List[Dict[str, Any]] = []
        logger.info(f"PPOAgent initialized with PyTorch optimizers. Actor LR: {learning_rate}, Critic LR: {learning_rate}")


    def store_experience(
        self,
        state: torch.Tensor,        # State tensor, shape e.g., [1, feature_dim]
        action: torch.Tensor,       # Action tensor, shape e.g., [1, action_dim]
        old_log_prob: torch.Tensor, # Log probability of the action, shape e.g., [1, 1] or [1]
        reward: float,              # Scalar reward
        next_state: torch.Tensor,   # Next state tensor, shape e.g., [1, feature_dim]
        done: bool                  # Boolean done flag
    ):
        """
        Stores a single step of experience in the buffer.
        Tensors are detached from the computation graph before storage.
        """
        # Ensure tensors are on CPU and detached for storage if they came from GPU or have grads.
        # .detach() is important to prevent gradients from flowing back further than intended during PPO updates.
        self.experience_buffer.append({
            "state": state.detach().cpu() if state is not None else None,
            "action": action.detach().cpu() if action is not None else None,
            "old_log_prob": old_log_prob.detach().cpu() if old_log_prob is not None else None,
            "reward": reward,
            "next_state": next_state.detach().cpu() if next_state is not None else None,
            "done": done
        })
        if len(self.experience_buffer) % 10 == 0: # Log less frequently
             logger.info(f"PPOAgent: Stored experience. Current buffer size: {len(self.experience_buffer)}")


    def compute_advantages_and_returns(
        self,
        rewards: torch.Tensor,        # 1D Tensor of rewards, shape [num_steps]
        values: torch.Tensor,         # 1D Tensor of state values V(s_t), shape [num_steps]
        dones: torch.Tensor,          # 1D Tensor of done flags (0.0 or 1.0), shape [num_steps]
        next_values: torch.Tensor     # 1D Tensor of next state values V(s_{t+1}), shape [num_steps]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes advantages using Generalized Advantage Estimation (GAE) and returns (targets for value function).
        All inputs are expected to be 1D PyTorch tensors.
        """
        n_steps = len(rewards)
        advantages = torch.zeros_like(rewards)
        # `returns` here are the targets for the value function V(s_t) = A_t + V(s_t)
        # which are often called Q-value estimates or discounted future rewards.
        returns = torch.zeros_like(rewards)
        gae = 0.0 # Current GAE estimate, reset at 'done'

        for t in reversed(range(n_steps)):
            # If current state 't' is terminal (dones[t]=1.0), then next_values[t] (V(s_{t+1})) is effectively 0.
            # The (1.0 - dones[t]) handles this by zeroing out future value/GAE if current state is terminal.
            delta = rewards[t] + self.gamma * next_values[t] * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = gae + values[t] # Target for value function update

        return advantages, returns

    def update(self):
        """
        Performs PPO updates for Actor and Critic networks using experiences from the buffer.
        Processes the entire buffer in minibatches over several epochs.
        """
        if not self.experience_buffer: # Check if buffer is empty first
            logger.info("PPOAgent: Experience buffer is empty. Skipping update.")
            return

        # Filter out any experiences where state or next_state might be None (e.g. if featurization failed)
        valid_experiences = [exp for exp in self.experience_buffer if exp['state'] is not None and exp['next_state'] is not None]

        if not valid_experiences or len(valid_experiences) < self.batch_size : # Check if enough valid experiences for at least one batch
            logger.info(f"PPOAgent: Not enough valid experiences ({len(valid_experiences)}) for a batch of size {self.batch_size}. Skipping update.")
            # self.experience_buffer.clear() # Removed: Do not clear if update is skipped due to insufficient batch size
            return

        logger.info(f"PPOAgent: Starting update with {len(valid_experiences)} valid experiences.")

        try:
            # Convert list of dicts to tensors for batch processing
            states_tensor = torch.cat([exp["state"] for exp in valid_experiences], dim=0)
            actions_tensor = torch.cat([exp["action"] for exp in valid_experiences], dim=0)
            old_log_probs_tensor = torch.cat([exp["old_log_prob"] for exp in valid_experiences], dim=0).squeeze()
            rewards_tensor = torch.tensor([exp["reward"] for exp in valid_experiences], dtype=torch.float32).squeeze()
            next_states_tensor = torch.cat([exp["next_state"] for exp in valid_experiences], dim=0)
            dones_tensor = torch.tensor([exp["done"] for exp in valid_experiences], dtype=torch.float32).squeeze()
        except Exception as e:
            logger.error(f"PPOAgent: Error converting buffer to tensors: {e}. Buffer content example: {valid_experiences[0] if valid_experiences else 'N/A'}", exc_info=True)
            self.experience_buffer.clear(); return

        # Get current and next state values from Critic (no gradients needed for these targets)
        with torch.no_grad():
            values_tensor = self.critic(states_tensor).squeeze()
            next_values_tensor = self.critic(next_states_tensor).squeeze()

        advantages, returns_tensor = self.compute_advantages_and_returns(
            rewards_tensor, values_tensor, dones_tensor, next_values_tensor
        )

        # Normalize advantages (common practice)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_experiences = len(valid_experiences)
        for epoch in range(self.ppo_epochs):
            # Create minibatches by shuffling indices
            indices = torch.randperm(num_experiences)
            for start_idx in range(0, num_experiences, self.batch_size):
                end_idx = start_idx + self.batch_size
                minibatch_indices = indices[start_idx:end_idx]

                # Slice data for the minibatch
                batch_states = states_tensor[minibatch_indices]
                batch_actions = actions_tensor[minibatch_indices]
                batch_old_log_probs = old_log_probs_tensor[minibatch_indices]
                batch_advantages = advantages[minibatch_indices]
                batch_returns = returns_tensor[minibatch_indices]

                # Get new log_probs and entropy from Actor for the current policy
                # This involves a forward pass through the actor, so requires gradients.
                log_probs, entropy = self.actor.evaluate_actions(batch_states, batch_actions)

                # Get new values from Critic for the current policy (requires gradients)
                new_values = self.critic(batch_states).squeeze()

                # PPO Ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)

                # Surrogate objectives for PPO
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages

                # Actor Loss (policy loss + entropy bonus)
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coefficient * entropy.mean()

                # Critic Loss (value loss) - typically Mean Squared Error
                critic_loss = F.mse_loss(new_values, batch_returns)

                # Update Actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                # Optional: Gradient clipping for actor
                # nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.actor_optimizer.step()

                # Update Critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                # Optional: Gradient clipping for critic
                # nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_optimizer.step()

            logger.debug(f"  PPO Epoch {epoch + 1}/{self.ppo_epochs}: Last minibatch Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")

        self.experience_buffer.clear() # Clear buffer after all epochs and minibatches
        logger.info(f"PPOAgent: Update finished. Processed {num_experiences} experiences over {self.ppo_epochs} epochs. Buffer cleared.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    print("\n--- PyTorch RL Module Example ---")
    INPUT_DIM_GNN = 384; GCN_HIDDEN_GNN = 64; MLP_HIDDEN_GNN = 32; ACTION_DIM_GNN = 3
    actor_model = GNNReasoningActor(
        input_node_dim=INPUT_DIM_GNN, gcn_hidden_dim=GCN_HIDDEN_GNN,
        mlp_hidden_dim=MLP_HIDDEN_GNN, output_action_dim=ACTION_DIM_GNN)
    # Critic input dim should match state feature dim. Assuming it's the GNN's input dim.
    critic_model = CriticValueNetwork(input_dim=INPUT_DIM_GNN, hidden_dim=32)

    # Use smaller batch size for this test example to ensure update runs
    ppo_agent = PPOAgent(actor=actor_model, critic=critic_model, batch_size=2, ppo_epochs=2)

    print("\n--- Simulating Experience Collection (PyTorch Tensors) ---")
    num_experiences_to_store = 5 # Store more than batch_size to test batching
    for i in range(num_experiences_to_store):
        mock_state_tensor = torch.randn(1, INPUT_DIM_GNN)
        action_dist, _ = actor_model(mock_state_tensor) # Get distribution from actor
        mock_action_tensor = action_dist.sample() # Sample action
        mock_old_log_prob_tensor = action_dist.log_prob(mock_action_tensor).sum(dim=-1, keepdim=True) # Get log_prob
        mock_reward = float(i) * 0.2
        mock_next_state_tensor = torch.randn(1, INPUT_DIM_GNN)
        done = (i == num_experiences_to_store - 1) # Last experience is 'done'
        ppo_agent.store_experience(
            mock_state_tensor, mock_action_tensor, mock_old_log_prob_tensor,
            mock_reward, mock_next_state_tensor, done)

    print("\n--- Simulating PPO Update ---")
    ppo_agent.update()
    print("\nPyTorch RL Module Example Finished.")
