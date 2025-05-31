import unittest
import torch
import torch.nn as nn
from torch.distributions import Normal

# Adjust imports based on project structure. Assuming 'src' is in PYTHONPATH.
from src.l2t_full.gnn_module import GNNReasoningActor
from src.l2t_full.rl_module import CriticValueNetwork
from src.l2t_full.constants import (
    DEFAULT_GNN_INPUT_NODE_DIM,
    DEFAULT_GNN_GCN_HIDDEN_DIM,
    DEFAULT_GNN_MLP_HIDDEN_DIM,
    DEFAULT_GNN_ACTION_DIM,
    DEFAULT_CRITIC_INPUT_DIM,
    DEFAULT_CRITIC_HIDDEN_DIM
)

class TestGNNReasoningActor(unittest.TestCase):
    def setUp(self):
        """Initialize a GNNReasoningActor instance for testing."""
        self.input_dim = DEFAULT_GNN_INPUT_NODE_DIM
        self.action_dim = DEFAULT_GNN_ACTION_DIM
        self.model = GNNReasoningActor(
            input_node_dim=self.input_dim,
            gcn_hidden_dim=DEFAULT_GNN_GCN_HIDDEN_DIM,
            mlp_hidden_dim=DEFAULT_GNN_MLP_HIDDEN_DIM,
            output_action_dim=self.action_dim
        )
        self.model.eval() # Set model to evaluation mode for consistency in tests

    def test_initialization(self):
        """Test if the model layers are initialized correctly."""
        self.assertIsInstance(self.model.gcn_layer, nn.Linear, "GCN layer should be nn.Linear")
        self.assertIsInstance(self.model.mlp_layer1, nn.Linear, "MLP layer1 should be nn.Linear")
        self.assertIsInstance(self.model.mean_layer, nn.Linear, "Mean layer should be nn.Linear")
        self.assertIsInstance(self.model.log_std_layer, nn.Linear, "Log_std layer should be nn.Linear")

        self.assertEqual(self.model.gcn_layer.out_features, DEFAULT_GNN_GCN_HIDDEN_DIM)
        self.assertEqual(self.model.mlp_layer1.out_features, DEFAULT_GNN_MLP_HIDDEN_DIM)
        self.assertEqual(self.model.mean_layer.out_features, self.action_dim)
        self.assertEqual(self.model.log_std_layer.out_features, self.action_dim)

    def test_forward_pass(self):
        """Test the forward pass of the GNNReasoningActor."""
        mock_input_features_single = torch.randn(1, self.input_dim)
        with torch.no_grad():
            action_dist_single, means_single = self.model.forward(mock_input_features_single)

        self.assertIsInstance(action_dist_single, Normal, "Action distribution should be Normal")
        self.assertEqual(means_single.shape, (1, self.action_dim), "Means shape mismatch for single input")
        self.assertEqual(action_dist_single.mean.shape, (1, self.action_dim))
        self.assertEqual(action_dist_single.stddev.shape, (1, self.action_dim))

        action_single = action_dist_single.sample()
        self.assertEqual(action_single.shape, (1, self.action_dim), "Sampled action shape mismatch for single input")

        log_prob_single = action_dist_single.log_prob(action_single)
        self.assertEqual(log_prob_single.shape, (1, self.action_dim), "Log_prob shape mismatch for single input")

        batch_size = 5
        mock_input_features_batch = torch.randn(batch_size, self.input_dim)
        with torch.no_grad():
            action_dist_batch, means_batch = self.model.forward(mock_input_features_batch)

        self.assertIsInstance(action_dist_batch, Normal)
        # The forward pass is now batch-aware, no internal aggregation to a single output.
        self.assertEqual(means_batch.shape, (batch_size, self.action_dim), "Means shape mismatch for batched input")
        self.assertEqual(action_dist_batch.mean.shape, (batch_size, self.action_dim))
        self.assertEqual(action_dist_batch.stddev.shape, (batch_size, self.action_dim))

        action_batch = action_dist_batch.sample()
        self.assertEqual(action_batch.shape, (batch_size, self.action_dim)) # Shape of sampled actions

        log_prob_batch = action_dist_batch.log_prob(action_batch)
        self.assertEqual(log_prob_batch.shape, (batch_size, self.action_dim)) # Shape of log_probs per action component


    def test_evaluate_actions(self):
        """Test the evaluate_actions method."""
        batch_size = 5
        state_features = torch.randn(batch_size, self.input_dim)
        actions_taken = torch.randn(batch_size, self.action_dim)

        with torch.no_grad(): # evaluate_actions should not affect gradients unless explicitly training
            log_probs, entropy = self.model.evaluate_actions(state_features, actions_taken)

        self.assertEqual(log_probs.shape, (batch_size,), "Log_probs shape mismatch in evaluate_actions")
        self.assertEqual(entropy.shape, (batch_size,), "Entropy shape mismatch in evaluate_actions")
        self.assertFalse(log_probs.requires_grad, "Log_probs should not require grad")
        self.assertFalse(entropy.requires_grad, "Entropy should not require grad")


class TestCriticValueNetwork(unittest.TestCase):
    def setUp(self):
        """Initialize a CriticValueNetwork instance for testing."""
        self.input_dim = DEFAULT_CRITIC_INPUT_DIM
        self.model = CriticValueNetwork(
            input_dim=self.input_dim,
            hidden_dim=DEFAULT_CRITIC_HIDDEN_DIM
        )
        self.model.eval() # Set model to evaluation mode

    def test_initialization(self):
        """Test if the model layers are initialized correctly."""
        self.assertIsInstance(self.model.layer1, nn.Linear, "Layer1 should be nn.Linear")
        self.assertIsInstance(self.model.layer2, nn.Linear, "Layer2 should be nn.Linear")
        self.assertEqual(self.model.layer1.out_features, DEFAULT_CRITIC_HIDDEN_DIM)
        self.assertEqual(self.model.layer2.out_features, 1)

    def test_forward_pass(self):
        """Test the forward pass of the CriticValueNetwork."""
        mock_input_features_single = torch.randn(1, self.input_dim)
        with torch.no_grad():
            value_single = self.model.forward(mock_input_features_single)
        self.assertEqual(value_single.shape, (1, 1), "Value shape mismatch for single input")

        batch_size = 5
        mock_input_features_batch = torch.randn(batch_size, self.input_dim)
        with torch.no_grad():
            value_batch = self.model.forward(mock_input_features_batch)
        self.assertEqual(value_batch.shape, (batch_size, 1), "Value shape mismatch for batched input")


if __name__ == '__main__':
    unittest.main()
