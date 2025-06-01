import unittest
import torch

# Assuming 'src' is in PYTHONPATH or discoverable
from src.l2t_full.processor import L2TFullProcessor
from src.l2t_full.dataclasses import L2TGraph, L2TNode, L2TFullConfig
from src.l2t_full.constants import (
    DEFAULT_GNN_INPUT_NODE_DIM, # Should be 384
    DEFAULT_GNN_SUBGRAPH_BETA
)

class TestFeaturizeGraphForGNN(unittest.TestCase):
    def setUp(self):
        """Initialize processor and graph for testing."""
        self.config = L2TFullConfig()
        # Ensure gnn_input_dim matches the sentence transformer model for most tests
        self.config.gnn_input_dim = 384 # all-MiniLM-L6-v2 output dim
        # Suppress INFO logs from sentence_transformers during tests unless debugging
        import logging
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

        self.processor = L2TFullProcessor(api_key="test_key_dummy", config=self.config)
        self.graph = L2TGraph()

    def test_empty_graph_or_no_node(self):
        """Test with an empty graph or if the target node_id is None or not found."""
        # Test with None node_id
        features_none_node = self.processor._featurize_graph_for_gnn(self.graph, None)
        self.assertIsInstance(features_none_node, torch.Tensor)
        self.assertEqual(features_none_node.shape, (1, self.config.gnn_input_dim))
        self.assertTrue(torch.all(features_none_node.eq(0)), "Features should be zero for None node_id")

        # Test with empty graph and a valid (but not found) node ID
        features_empty_graph = self.processor._featurize_graph_for_gnn(self.graph, "node1")
        self.assertIsInstance(features_empty_graph, torch.Tensor)
        self.assertEqual(features_empty_graph.shape, (1, self.config.gnn_input_dim))
        self.assertTrue(torch.all(features_empty_graph.eq(0)), "Features should be zero for empty graph")

        # Test with non-empty graph but node_id not found
        node_a = L2TNode(id="na", content="Node A", parent_id=None, generation_step=0)
        self.graph.add_node(node_a)
        features_node_not_found = self.processor._featurize_graph_for_gnn(self.graph, "node_not_exist")
        self.assertIsInstance(features_node_not_found, torch.Tensor)
        self.assertEqual(features_node_not_found.shape, (1, self.config.gnn_input_dim))
        self.assertTrue(torch.all(features_node_not_found.eq(0)), "Features should be zero for non-existent node_id")


    def test_single_node_graph(self):
        """Test with a graph containing only a single node."""
        node1 = L2TNode(id="n1", content="Hello world", parent_id=None, generation_step=0)
        self.graph.add_node(node1, is_root=True)

        features = self.processor._featurize_graph_for_gnn(self.graph, "n1")
        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(features.shape, (1, self.config.gnn_input_dim))
        # It's hard to assert non-zero precisely without knowing the embedding,
        # but we can check if its norm is greater than zero.
        self.assertTrue(torch.norm(features).item() > 0, "Features for a single node should not be all zero")

    def test_subgraph_extraction_beta_0(self):
        """Test subgraph extraction with beta = 0 (only center node)."""
        self.processor.config.gnn_subgraph_beta = 0

        node1 = L2TNode(id="n1", content="Root node content", parent_id=None, generation_step=0)
        node2 = L2TNode(id="n2", content="Child node content", parent_id="n1", generation_step=1)
        node3 = L2TNode(id="n3", content="Grandchild node content", parent_id="n2", generation_step=2)
        self.graph.add_node(node1, is_root=True)
        self.graph.add_node(node2)
        self.graph.add_node(node3)

        # Featurize for n2 with beta=0. Only n2's content should be embedded.
        features_n2 = self.processor._featurize_graph_for_gnn(self.graph, "n2")
        self.assertEqual(features_n2.shape, (1, self.config.gnn_input_dim))
        self.assertTrue(torch.norm(features_n2).item() > 0)

        # For comparison, get embedding of "n2" content directly
        direct_embedding_n2 = self.processor.sentence_model.encode("Child node content", convert_to_tensor=True)
        if direct_embedding_n2.ndim == 1: # Ensure it's [1, dim]
            direct_embedding_n2 = direct_embedding_n2.unsqueeze(0)

        self.assertTrue(torch.allclose(features_n2, direct_embedding_n2, atol=1e-6),
                        "Features for beta=0 should be embedding of center node")

    def test_subgraph_extraction_beta_1(self):
        """Test subgraph extraction with beta = 1 (center + immediate neighbors)."""
        self.processor.config.gnn_subgraph_beta = 1

        n_parent = L2TNode(id="np", content="Parent content", parent_id=None, generation_step=0) # Added parent_id=None
        n1_center = L2TNode(id="n1", content="Center content", parent_id="np", generation_step=1)
        n_child0 = L2TNode(id="nc0", content="Child0 content", parent_id="n1", generation_step=2)
        n_child1 = L2TNode(id="nc1", content="Child1 content", parent_id="n1", generation_step=2)
        n_unrelated = L2TNode(id="nu", content="Unrelated content", parent_id=None, generation_step=0) # Added parent_id=None

        self.graph.add_node(n_parent, is_root=True) # np is root for this test branch
        self.graph.add_node(n1_center)
        self.graph.add_node(n_child0)
        self.graph.add_node(n_child1)
        self.graph.add_node(n_unrelated) # Add unrelated node to graph

        features_n1 = self.processor._featurize_graph_for_gnn(self.graph, "n1")
        self.assertEqual(features_n1.shape, (1, self.config.gnn_input_dim))
        self.assertTrue(torch.norm(features_n1).item() > 0)

        # Expected nodes in subgraph for n1 with beta=1: n1, np, nc0, nc1
        contents_in_subgraph = [
            n1_center.content,
            n_parent.content,
            n_child0.content,
            n_child1.content
        ]
        expected_embeddings = self.processor.sentence_model.encode(contents_in_subgraph, convert_to_tensor=True)
        expected_mean_features = torch.mean(expected_embeddings, dim=0, keepdim=True)

        self.assertTrue(torch.allclose(features_n1, expected_mean_features, atol=1e-5),
                        "Features for beta=1 do not match expected mean of neighbor embeddings")

    def test_feature_dimension_mismatch_handling(self):
        """Test padding/truncation if gnn_input_dim mismatches sentence model dim (384)."""
        original_dim = self.processor.config.gnn_input_dim # Store original

        node1 = L2TNode(id="n1", content="Test dimension handling", parent_id=None, generation_step=0)
        self.graph.add_node(node1, is_root=True)

        # Test truncation
        self.processor.config.gnn_input_dim = 200
        features_truncated = self.processor._featurize_graph_for_gnn(self.graph, "n1")
        self.assertEqual(features_truncated.shape, (1, 200))
        self.assertTrue(torch.norm(features_truncated).item() > 0)

        # Test padding
        self.processor.config.gnn_input_dim = 512
        features_padded = self.processor._featurize_graph_for_gnn(self.graph, "n1")
        self.assertEqual(features_padded.shape, (1, 512))
        self.assertTrue(torch.norm(features_padded).item() > 0)
        # Check if the original part is non-zero and padded part is zero
        original_embedding = self.processor.sentence_model.encode("Test dimension handling", convert_to_tensor=True)
        self.assertTrue(torch.allclose(features_padded[:, :384], original_embedding.unsqueeze(0), atol=1e-6))
        self.assertTrue(torch.all(features_padded[:, 384:].eq(0)))


        self.processor.config.gnn_input_dim = original_dim # Restore

if __name__ == '__main__':
    unittest.main()
