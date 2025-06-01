import unittest
import torch
import torch.nn as nn
import torch.optim as optim # Added for PPOAgent tests
from torch.distributions import Normal

from src.l2t_full.gnn_module import GNNReasoningActor
from src.l2t_full.rl_module import CriticValueNetwork, PPOAgent # Added PPOAgent
from src.l2t_full.constants import (
    DEFAULT_GNN_INPUT_NODE_DIM,
    DEFAULT_GNN_GCN_HIDDEN_DIM,
    DEFAULT_GNN_MLP_HIDDEN_DIM,
    DEFAULT_GNN_ACTION_DIM,
    DEFAULT_CRITIC_INPUT_DIM,
    DEFAULT_CRITIC_HIDDEN_DIM,
    # PPO constants for agent initialization
    DEFAULT_PPO_LEARNING_RATE,
    DEFAULT_PPO_GAMMA,
    DEFAULT_PPO_EPSILON,
    DEFAULT_PPO_GAE_LAMBDA,
    DEFAULT_PPO_EPOCHS,
    DEFAULT_PPO_BATCH_SIZE,
    DEFAULT_PPO_ENTROPY_COEFFICIENT
)

class TestGNNReasoningActor(unittest.TestCase):
    def setUp(self):
        self.input_dim = DEFAULT_GNN_INPUT_NODE_DIM
        self.action_dim = DEFAULT_GNN_ACTION_DIM
        self.model = GNNReasoningActor(
            input_node_dim=self.input_dim,
            gcn_hidden_dim=DEFAULT_GNN_GCN_HIDDEN_DIM,
            mlp_hidden_dim=DEFAULT_GNN_MLP_HIDDEN_DIM,
            output_action_dim=self.action_dim
        )
        self.model.eval()

    def test_initialization(self):
        self.assertIsInstance(self.model.gcn_layer, nn.Linear)
        self.assertIsInstance(self.model.mlp_layer1, nn.Linear)
        self.assertIsInstance(self.model.mean_layer, nn.Linear)
        self.assertIsInstance(self.model.log_std_layer, nn.Linear)
        self.assertEqual(self.model.mean_layer.out_features, self.action_dim)

    def test_forward_pass(self):
        mock_input_features_single = torch.randn(1, self.input_dim)
        with torch.no_grad():
            action_dist_single, means_single = self.model.forward(mock_input_features_single)
        self.assertIsInstance(action_dist_single, Normal)
        self.assertEqual(means_single.shape, (1, self.action_dim))
        action_single = action_dist_single.sample()
        self.assertEqual(action_single.shape, (1, self.action_dim))
        log_prob_single = action_dist_single.log_prob(action_single)
        self.assertEqual(log_prob_single.shape, (1, self.action_dim))

        batch_size = 5
        mock_input_features_batch = torch.randn(batch_size, self.input_dim)
        with torch.no_grad():
            action_dist_batch, means_batch = self.model.forward(mock_input_features_batch)
        self.assertIsInstance(action_dist_batch, Normal)
        self.assertEqual(means_batch.shape, (batch_size, self.action_dim))
        action_batch = action_dist_batch.sample()
        self.assertEqual(action_batch.shape, (batch_size, self.action_dim))
        log_prob_batch = action_dist_batch.log_prob(action_batch)
        self.assertEqual(log_prob_batch.shape, (batch_size, self.action_dim))

    def test_evaluate_actions(self):
        batch_size = 5
        state_features = torch.randn(batch_size, self.input_dim)
        actions_taken = torch.randn(batch_size, self.action_dim)
        with torch.no_grad():
            log_probs, entropy = self.model.evaluate_actions(state_features, actions_taken)
        self.assertEqual(log_probs.shape, (batch_size,))
        self.assertEqual(entropy.shape, (batch_size,))

class TestCriticValueNetwork(unittest.TestCase):
    def setUp(self):
        self.input_dim = DEFAULT_CRITIC_INPUT_DIM
        self.model = CriticValueNetwork(
            input_dim=self.input_dim,
            hidden_dim=DEFAULT_CRITIC_HIDDEN_DIM
        )
        self.model.eval()

    def test_initialization(self):
        self.assertIsInstance(self.model.layer1, nn.Linear)
        self.assertIsInstance(self.model.layer2, nn.Linear)
        self.assertEqual(self.model.layer2.out_features, 1)

    def test_forward_pass(self):
        mock_input_features_single = torch.randn(1, self.input_dim)
        with torch.no_grad():
            value_single = self.model.forward(mock_input_features_single)
        self.assertEqual(value_single.shape, (1, 1))
        batch_size = 5
        mock_input_features_batch = torch.randn(batch_size, self.input_dim)
        with torch.no_grad():
            value_batch = self.model.forward(mock_input_features_batch)
        self.assertEqual(value_batch.shape, (batch_size, 1))

class TestPPOAgent(unittest.TestCase):
    def setUp(self):
        self.gnn_input_dim = DEFAULT_GNN_INPUT_NODE_DIM # 384
        self.gnn_action_dim = DEFAULT_GNN_ACTION_DIM # 3
        # Critic input dim should match GNN's feature representation for state value
        # If critic processes GNN's gcn_layer output: DEFAULT_GNN_GCN_HIDDEN_DIM
        # If critic processes same input as GNN actor: self.gnn_input_dim
        # Current PPOAgent setup implies critic takes a state representation,
        # which is the same input GNN actor takes for its policy.
        self.critic_input_dim = self.gnn_input_dim

        self.actor = GNNReasoningActor(
            input_node_dim=self.gnn_input_dim,
            gcn_hidden_dim=DEFAULT_GNN_GCN_HIDDEN_DIM,
            mlp_hidden_dim=DEFAULT_GNN_MLP_HIDDEN_DIM,
            output_action_dim=self.gnn_action_dim
        )
        self.critic = CriticValueNetwork(
            input_dim=self.critic_input_dim, # Ensure this matches what PPO will feed it
            hidden_dim=DEFAULT_CRITIC_HIDDEN_DIM
        )
        self.agent = PPOAgent(
            actor=self.actor,
            critic=self.critic,
            learning_rate=DEFAULT_PPO_LEARNING_RATE,
            gamma=DEFAULT_PPO_GAMMA,
            epsilon=DEFAULT_PPO_EPSILON,
            gae_lambda=DEFAULT_PPO_GAE_LAMBDA,
            ppo_epochs=DEFAULT_PPO_EPOCHS, # Use a smaller number for faster tests e.g. 1 or 2
            batch_size=DEFAULT_PPO_BATCH_SIZE, # Use a smaller batch size for easier testing e.g. 2 or 4
            entropy_coefficient=DEFAULT_PPO_ENTROPY_COEFFICIENT
        )
        # Override for faster tests
        self.agent.ppo_epochs = 2
        self.agent.batch_size = 2


    def test_initialization(self):
        self.assertIsInstance(self.agent.actor_optimizer, optim.Adam, "Actor optimizer should be Adam")
        self.assertIsInstance(self.agent.critic_optimizer, optim.Adam, "Critic optimizer should be Adam")

    def test_store_experience(self):
        state = torch.randn(1, self.gnn_input_dim)
        action_dist, _ = self.actor(state)
        action = action_dist.sample()
        old_log_prob = action_dist.log_prob(action).sum(dim=-1, keepdim=True)
        reward = 1.0
        next_state = torch.randn(1, self.gnn_input_dim)
        done = False

        self.agent.store_experience(state, action, old_log_prob, reward, next_state, done)
        self.assertEqual(len(self.agent.experience_buffer), 1)

        stored_exp = self.agent.experience_buffer[0]
        self.assertTrue(torch.equal(stored_exp["state"], state.detach()))
        self.assertTrue(torch.equal(stored_exp["action"], action.detach()))
        self.assertTrue(torch.equal(stored_exp["old_log_prob"], old_log_prob.detach()))
        self.assertEqual(stored_exp["reward"], reward)
        self.assertTrue(torch.equal(stored_exp["next_state"], next_state.detach()))
        self.assertEqual(stored_exp["done"], done)

        # Test with None state/next_state (though PPOAgent update filters these)
        self.agent.experience_buffer.clear()
        self.agent.store_experience(None, action, old_log_prob, reward, None, done)
        self.assertIsNone(self.agent.experience_buffer[0]["state"])
        self.assertIsNone(self.agent.experience_buffer[0]["next_state"])


    def test_compute_advantages_and_returns_simple(self):
        # rewards: [r1, r2], values: [V(s1), V(s2)], dones: [F, T], next_values: [V(s2), 0]
        rewards = torch.tensor([1.0, 2.0], dtype=torch.float32)
        values = torch.tensor([0.5, 1.0], dtype=torch.float32) # V(s1), V(s2)
        dones = torch.tensor([0.0, 1.0], dtype=torch.float32)  # s2 is terminal
        next_values = torch.tensor([1.0, 0.0], dtype=torch.float32) # V(s2), V(s3)=0

        gamma = 0.99
        gae_lambda = 0.95

        advantages, returns = self.agent.compute_advantages_and_returns(
            rewards, values, dones, next_values
        )
        self.assertEqual(advantages.shape, rewards.shape)
        self.assertEqual(returns.shape, rewards.shape)

        # Manual GAE calculation for this simple case:
        # For t=1 (last step, s2 is terminal):
        # delta_1 = r_1 + gamma * V(s_3) * (1-done_1) - V(s_2)
        #         = 2.0 + 0.99 * 0.0 * (1-1.0) - 1.0 = 1.0
        # gae_1   = delta_1 = 1.0
        # return_1 = gae_1 + V(s_2) = 1.0 + 1.0 = 2.0
        self.assertAlmostEqual(advantages[1].item(), 1.0, places=4)
        self.assertAlmostEqual(returns[1].item(), 2.0, places=4)

        # For t=0 (s1):
        # delta_0 = r_0 + gamma * V(s_2) * (1-done_0) - V(s_1)
        #         = 1.0 + 0.99 * 1.0 * (1-0.0) - 0.5 = 1.0 + 0.99 - 0.5 = 1.49
        # gae_0   = delta_0 + gamma * gae_lambda * (1-done_0) * gae_1
        #         = 1.49    + 0.99  * 0.95     * (1-0.0)    * 1.0
        #         = 1.49    + 0.9405 = 2.4305
        # return_0 = gae_0 + V(s_1) = 2.4305 + 0.5 = 2.9305
        self.assertAlmostEqual(advantages[0].item(), 2.4305, places=4)
        self.assertAlmostEqual(returns[0].item(), 2.9305, places=4)


    def test_update_sufficient_experiences(self):
        # Populate buffer
        for _ in range(self.agent.batch_size):
            state = torch.randn(1, self.gnn_input_dim)
            action_dist, _ = self.actor(state)
            action = action_dist.sample()
            old_log_prob = action_dist.log_prob(action).sum(-1, keepdim=True)
            self.agent.store_experience(state, action, old_log_prob, 1.0, torch.randn(1, self.gnn_input_dim), False)

        initial_actor_param = list(self.agent.actor.parameters())[0].clone().detach()
        initial_critic_param = list(self.agent.critic.parameters())[0].clone().detach()

        self.agent.update()

        self.assertEqual(len(self.agent.experience_buffer), 0, "Buffer should be cleared after update")

        # Check if parameters changed (indicating optimizer step)
        # This test can be flaky if learning rate is too small or gradients are zero by chance.
        # A more robust test would check loss reduction or specific gradient values.
        self.assertFalse(torch.equal(list(self.agent.actor.parameters())[0], initial_actor_param), "Actor params should change")
        self.assertFalse(torch.equal(list(self.agent.critic.parameters())[0], initial_critic_param), "Critic params should change")


    def test_update_insufficient_experiences(self):
        # Populate with fewer than batch_size experiences
        num_exps = self.agent.batch_size - 1
        if num_exps <= 0: # Handle if batch_size is 1
            self.skipTest("Batch size is 1, cannot test insufficient experiences meaningfully this way.")
            return

        for _ in range(num_exps):
            state = torch.randn(1, self.gnn_input_dim)
            action_dist, _ = self.actor(state)
            action = action_dist.sample()
            old_log_prob = action_dist.log_prob(action).sum(-1, keepdim=True)
            self.agent.store_experience(state, action, old_log_prob, 1.0, torch.randn(1, self.gnn_input_dim), False)

        initial_actor_param = list(self.agent.actor.parameters())[0].clone().detach()

        # The PPOAgent.update() currently skips if buffer < batch_size
        self.agent.update()

        self.assertEqual(len(self.agent.experience_buffer), num_exps, "Buffer should not be cleared if update is skipped")
        self.assertTrue(torch.equal(list(self.agent.actor.parameters())[0], initial_actor_param), "Actor params should NOT change if update is skipped")


if __name__ == '__main__':
    unittest.main()
