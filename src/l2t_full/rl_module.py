# TODO: This module requires a deep learning library (e.g., PyTorch or TensorFlow)
# for the Critic network and PPO updates. Ensure these are added to project dependencies.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Distribution

from typing import Any, List, Tuple, Optional, Dict
import logging # Import logging

# Import GNNReasoningActor from .gnn_module (now a PyTorch nn.Module)
from .gnn_module import GNNReasoningActor

logger = logging.getLogger(__name__) # Define module-level logger

class CriticValueNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1)

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(state_features))
        value = self.layer2(x)
        return value

class PPOAgent:
    def __init__(
        self,
        actor: GNNReasoningActor,
        critic: CriticValueNetwork,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        gae_lambda: float = 0.95,
        ppo_epochs: int = 10,
        batch_size: int = 64,
        entropy_coefficient: float = 0.01
    ):
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.epsilon = epsilon
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.entropy_coefficient = entropy_coefficient

        self.actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)

        self.experience_buffer: List[Dict[str, Any]] = []
        logger.info(f"PPOAgent initialized with PyTorch optimizers.")


    def store_experience(
        self,
        state: torch.Tensor, action: torch.Tensor, old_log_prob: torch.Tensor,
        reward: float, next_state: torch.Tensor, done: bool
    ):
        self.experience_buffer.append({
            "state": state.detach() if state is not None else None,
            "action": action.detach() if action is not None else None,
            "old_log_prob": old_log_prob.detach() if old_log_prob is not None else None,
            "reward": reward,
            "next_state": next_state.detach() if next_state is not None else None,
            "done": done
        })
        if len(self.experience_buffer) % 10 == 0:
             logger.info(f"PPOAgent: Stored experience. Current buffer size: {len(self.experience_buffer)}")


    def compute_advantages_and_returns(
        self,
        rewards: torch.Tensor, values: torch.Tensor,
        dones: torch.Tensor, next_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_steps = len(rewards)
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(n_steps)):
            delta = rewards[t] + self.gamma * next_values[t] * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
        return advantages, returns

    def update(self):
        if not self.experience_buffer or len(self.experience_buffer) < self.batch_size:
            logger.info(f"PPOAgent: Buffer size ({len(self.experience_buffer)}) < batch size ({self.batch_size}). Skip update.")
            return

        logger.info(f"PPOAgent: Starting update with {len(self.experience_buffer)} experiences.")

        valid_experiences = [exp for exp in self.experience_buffer if exp['state'] is not None and exp['next_state'] is not None]
        if not valid_experiences or len(valid_experiences) < self.batch_size:
            logger.info(f"PPOAgent: Not enough valid experiences ({len(valid_experiences)}) for a batch. Skipping update.")
            self.experience_buffer.clear()
            return

        try:
            states_list = [exp["state"] for exp in valid_experiences]
            actions_list = [exp["action"] for exp in valid_experiences]
            old_log_probs_list = [exp["old_log_prob"] for exp in valid_experiences]
            rewards_list = [exp["reward"] for exp in valid_experiences]
            next_states_list = [exp["next_state"] for exp in valid_experiences]
            dones_list = [exp["done"] for exp in valid_experiences]

            states_tensor = torch.cat(states_list, dim=0)
            actions_tensor = torch.cat(actions_list, dim=0)
            old_log_probs_tensor = torch.cat(old_log_probs_list, dim=0).squeeze()
            rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32).squeeze()
            next_states_tensor = torch.cat(next_states_list, dim=0)
            dones_tensor = torch.tensor(dones_list, dtype=torch.float32).squeeze()
        except Exception as e:
            logger.error(f"PPOAgent: Error converting buffer to tensors: {e}. Skipping update.", exc_info=True)
            self.experience_buffer.clear()
            return

        with torch.no_grad():
            values_tensor = self.critic(states_tensor).squeeze()
            next_values_tensor = self.critic(next_states_tensor).squeeze()

        advantages, returns_tensor = self.compute_advantages_and_returns(
            rewards_tensor, values_tensor, dones_tensor, next_values_tensor
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for epoch in range(self.ppo_epochs):
            log_probs, entropy = self.actor.evaluate_actions(states_tensor, actions_tensor)
            new_values = self.critic(states_tensor).squeeze()
            ratio = torch.exp(log_probs - old_log_probs_tensor)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coefficient * entropy.mean()
            critic_loss = F.mse_loss(new_values, returns_tensor)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            logger.debug(f"  PPO Epoch {epoch + 1}: Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")

        self.experience_buffer.clear()
        logger.info(f"PPOAgent: Update finished. Processed {len(valid_experiences)} experiences. Buffer cleared.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) # Ensure logger is configured for __main__
    logger = logging.getLogger(__name__)

    print("\n--- PyTorch RL Module Example ---")
    INPUT_DIM_GNN = 128; GCN_HIDDEN_GNN = 64; MLP_HIDDEN_GNN = 32; ACTION_DIM_GNN = 3
    actor_model = GNNReasoningActor(
        input_node_dim=INPUT_DIM_GNN, gcn_hidden_dim=GCN_HIDDEN_GNN,
        mlp_hidden_dim=MLP_HIDDEN_GNN, output_action_dim=ACTION_DIM_GNN
    )
    CRITIC_INPUT_DIM = INPUT_DIM_GNN; CRITIC_HIDDEN_DIM = 32
    critic_model = CriticValueNetwork(input_dim=CRITIC_INPUT_DIM, hidden_dim=CRITIC_HIDDEN_DIM)
    ppo_agent = PPOAgent(actor=actor_model, critic=critic_model, batch_size=2, ppo_epochs=3)

    print("\n--- Simulating Experience Collection (PyTorch Tensors) ---")
    num_experiences_to_store = 5
    for i in range(num_experiences_to_store):
        mock_state_tensor = torch.randn(1, INPUT_DIM_GNN)
        action_dist, _ = actor_model(mock_state_tensor)
        mock_action_tensor = action_dist.sample()
        mock_old_log_prob_tensor = action_dist.log_prob(mock_action_tensor).sum(dim=-1, keepdim=True)
        mock_reward = i * 0.2
        mock_next_state_tensor = torch.randn(1, INPUT_DIM_GNN)
        done = (i == num_experiences_to_store - 1)
        ppo_agent.store_experience(
            mock_state_tensor, mock_action_tensor, mock_old_log_prob_tensor,
            mock_reward, mock_next_state_tensor, done
        )

    print("\n--- Simulating PPO Update ---")
    ppo_agent.update()
    print("\nPyTorch RL Module Example Finished.")
