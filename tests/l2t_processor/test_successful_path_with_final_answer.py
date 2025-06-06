import unittest
from unittest.mock import patch, MagicMock, call
from typing import cast

from src.l2t.processor import L2TProcessor
from src.l2t.dataclasses import (
    L2TConfig,
    L2TResult,
    L2TGraph,
    L2TNodeCategory,
    L2TNode,
)
from src.aot.dataclasses import LLMCallStats
from src.llm_config import LLMConfig # Added LLMConfig

from src.llm_client import LLMClient
from src.l2t_processor_utils.node_processor import NodeProcessor

import logging
# logging.disable(logging.CRITICAL)

class TestL2TProcessor_SuccessfulPath(unittest.TestCase):
    def setUp(self):
        self.l2t_config = L2TConfig( # Renamed from self.config
            max_steps=5,
            max_total_nodes=10,
            max_time_seconds=60,
            classification_model_names=["mock-classifier"],
            thought_generation_model_names=["mock-generator"],
            initial_prompt_model_names=["mock-initial"],
        )
        # Define LLMConfig objects for L2TProcessor
        self.initial_thought_llm_config = LLMConfig(temperature=0.7)
        self.node_processor_llm_config = LLMConfig(temperature=0.1)

    @patch("src.l2t.processor.L2TResponseParser.parse_l2t_initial_response")
    @patch("src.l2t.processor.LLMClient")
    def test_run_successful_path_with_final_answer(
        self,
        MockL2TProcessorLLMClient,
        mock_parse_initial
    ):
        problem_text = "Test problem: Find the final answer."

        initial_thought_content = "This is the initial thought for the problem."
        generated_thought_content = "This is the generated thought, which is the final answer."

        stats_initial = LLMCallStats(completion_tokens=10, prompt_tokens=5, call_duration_seconds=0.1, model_name="initial_model")
        stats_classify1 = LLMCallStats(completion_tokens=5, prompt_tokens=5, call_duration_seconds=0.05, model_name="classify_model")
        stats_thought_gen = LLMCallStats(completion_tokens=15, prompt_tokens=10, call_duration_seconds=0.1, model_name="gen_model")
        stats_classify2 = LLMCallStats(completion_tokens=5, prompt_tokens=5, call_duration_seconds=0.05, model_name="classify_model")

        MockL2TProcessorLLMClient.return_value.call.return_value = ("Your thought: " + initial_thought_content, stats_initial)
        mock_parse_initial.return_value = initial_thought_content

        processor = L2TProcessor(
            api_key="mock_api_key",
            l2t_config=self.l2t_config, # Use l2t_config
            initial_thought_llm_config=self.initial_thought_llm_config,
            node_processor_llm_config=self.node_processor_llm_config,
        )

        # Use MagicMock for NodeProcessor and its methods for consistency
        mock_node_processor_instance = MagicMock(spec=NodeProcessor)
        processor.node_processor = mock_node_processor_instance

        # Define side effects for the mocked methods
        def process_node_side_effect(node_id, graph, result_obj, step):
            if mock_node_processor_instance.process_node.call_count == 1: # First call (for root node)
                graph.classify_node(node_id, L2TNodeCategory.CONTINUE)
                mock_node_processor_instance._update_result_stats(result_obj, stats_classify1)
                mock_node_processor_instance._update_result_stats(result_obj, stats_thought_gen)
                graph.add_node(L2TNode(id="child1", content=generated_thought_content, parent_id=node_id, generation_step=1))
                graph.move_to_hist(node_id)
                # Simulate adding child1 to v_pres for next iteration
                if node_id in graph.v_pres:
                    graph.v_pres.remove(node_id)
                graph.v_pres.append("child1")
            else: # Second call (for child1 node)
                graph.classify_node(node_id, L2TNodeCategory.FINAL_ANSWER)
                mock_node_processor_instance._update_result_stats(result_obj, stats_classify2)
                setattr(result_obj, 'final_answer', generated_thought_content)
                setattr(result_obj, 'succeeded', True)
                graph.move_to_hist(node_id)
                if node_id in graph.v_pres:
                    graph.v_pres.remove(node_id)
        mock_node_processor_instance.process_node.side_effect = process_node_side_effect

        def update_result_stats_side_effect(result_obj, stats):
            if stats:
                result_obj.total_llm_calls += 1
                result_obj.total_completion_tokens += stats.completion_tokens
                result_obj.total_prompt_tokens += stats.prompt_tokens
                result_obj.total_llm_interaction_time_seconds += stats.call_duration_seconds
        mock_node_processor_instance._update_result_stats.side_effect = update_result_stats_side_effect


        result = processor.run(problem_text)

        self.assertTrue(result.succeeded)
        self.assertEqual(result.final_answer, generated_thought_content)
        self.assertIsNone(result.error_message)
        self.assertEqual(result.total_llm_calls, 4)
        expected_completion_tokens = stats_initial.completion_tokens + stats_classify1.completion_tokens + stats_thought_gen.completion_tokens + stats_classify2.completion_tokens
        expected_prompt_tokens = stats_initial.prompt_tokens + stats_classify1.prompt_tokens + stats_thought_gen.prompt_tokens + stats_classify2.prompt_tokens
        self.assertEqual(result.total_completion_tokens, expected_completion_tokens)
        self.assertEqual(result.total_prompt_tokens, expected_prompt_tokens)
        self.assertIsNotNone(result.reasoning_graph)
        graph = cast(L2TGraph, result.reasoning_graph)
        self.assertIsNotNone(graph.nodes)
        self.assertEqual(len(graph.nodes), 2)
        root_node_id = graph.root_node_id
        self.assertIsNotNone(root_node_id)
        root_node_id = cast(str, root_node_id) # Cast to str
        root_node = graph.get_node(root_node_id)
        self.assertIsNotNone(root_node)
        root_node = cast(L2TNode, root_node) # Cast to L2TNode
        self.assertEqual(root_node.content, initial_thought_content)
        self.assertEqual(root_node.category, L2TNodeCategory.CONTINUE)
        self.assertIsNotNone(root_node.children_ids)
        self.assertEqual(len(root_node.children_ids), 1)
        child_node_id = root_node.children_ids[0]
        self.assertIsNotNone(child_node_id)
        child_node_id = cast(str, child_node_id) # Cast to str
        child_node = graph.get_node(child_node_id)
        self.assertIsNotNone(child_node)
        child_node = cast(L2TNode, child_node) # Cast to L2TNode
        self.assertEqual(child_node.content, generated_thought_content)
        self.assertEqual(child_node.category, L2TNodeCategory.FINAL_ANSWER)
        self.assertEqual(child_node.parent_id, root_node_id)
