import unittest
from unittest.mock import MagicMock, patch, call
import time # For checking time related fields if necessary

# New imports
from src.cot_process import CoTResult 
from src.l2t_processor import L2TProcessor
from src.l2t_dataclasses import (
    L2TConfig,
    L2TResult,
    L2TNode,
    L2TNodeCategory,
    L2TGraph,
)
from src.llm_client import LLMClient # For mocking
from src.aot_dataclasses import LLMCallStats # For mock LLMClient responses
from src.l2t_response_parser import L2TResponseParser # For patching its methods


class TestL2TProcessor(unittest.TestCase):
    def setUp(self):
        self.mock_llm_client = MagicMock(spec=LLMClient)
        self.config = L2TConfig(max_steps=5, max_total_nodes=10, max_time_seconds=30) # Example config
        # Ensure L2TProcessor is initialized with the mocked LLMClient
        self.processor = L2TProcessor(llm_client=self.mock_llm_client, config=self.config)

    def _mock_llm_call(self, response_content: str, completion_tokens: int = 10, prompt_tokens: int = 5, duration: float = 0.5, model_name: str = "test_model"):
        return (response_content, LLMCallStats(completion_tokens=completion_tokens, prompt_tokens=prompt_tokens, call_duration_seconds=duration, model_name=model_name))

    def test_l2t_processor_initialization(self):
        self.assertIsInstance(self.processor, L2TProcessor)
        self.assertEqual(self.processor.l2t_config, self.config)
        self.assertEqual(self.processor.llm_client, self.mock_llm_client)

    @patch.object(L2TResponseParser, 'parse_l2t_initial_response', return_value="Initial thought generated")
    @patch.object(L2TResponseParser, 'parse_l2t_node_classification_response', return_value=L2TNodeCategory.FINAL_ANSWER)
    def test_run_simple_final_answer_path(self, mock_parse_classify, mock_parse_initial):
        problem_text = "What is the capital of France?"
        
        self.mock_llm_client.call.side_effect = [
            self._mock_llm_call("Initial thought response"), # Initial thought generation
            self._mock_llm_call("FINAL_ANSWER")             # Classification of initial thought
        ]

        cot_result = self.processor.run(problem_text)

        # Assertions on CoTResult
        self.assertIsInstance(cot_result, CoTResult)
        self.assertTrue(cot_result.succeeded)
        self.assertEqual(cot_result.final_answer, "Initial thought generated") # Root node content becomes final answer
        self.assertIsNone(cot_result.error_message)
        self.assertGreater(cot_result.total_completion_tokens, 0)
        self.assertGreater(cot_result.total_prompt_tokens, 0)
        self.assertGreater(cot_result.total_llm_interaction_time_seconds, 0)
        self.assertGreater(cot_result.total_process_wall_clock_time_seconds, 0)
        self.assertIsNone(cot_result.reasoning_trace) # L2T uses graph, not linear trace

        # Assertions on L2TResult (process_specific_data)
        self.assertIsInstance(cot_result.process_specific_data, L2TResult)
        l2t_native_result: L2TResult = cot_result.process_specific_data
        
        self.assertTrue(l2t_native_result.succeeded)
        self.assertEqual(l2t_native_result.final_answer, "Initial thought generated")
        self.assertIsNotNone(l2t_native_result.reasoning_graph)
        self.assertEqual(l2t_native_result.total_llm_calls, 2) # Initial + Classification
        
        graph = l2t_native_result.reasoning_graph
        self.assertIsNotNone(graph)
        self.assertEqual(len(graph.nodes), 1) # Only root node
        root_node = graph.get_node(graph.root_node_id) # type: ignore
        self.assertIsNotNone(root_node)
        self.assertEqual(root_node.content, "Initial thought generated") # type: ignore
        self.assertEqual(root_node.category, L2TNodeCategory.FINAL_ANSWER) # type: ignore

        mock_parse_initial.assert_called_once_with("Initial thought response")
        # Classification prompt is complex, check that parser was called with the LLM response
        mock_parse_classify.assert_called_once_with("FINAL_ANSWER")


    @patch.object(L2TResponseParser, 'parse_l2t_initial_response', return_value="Initial thought")
    @patch.object(L2TResponseParser, 'parse_l2t_node_classification_response')
    @patch.object(L2TResponseParser, 'parse_l2t_thought_generation_response', return_value="Generated thought 1")
    def test_run_continue_then_final_answer(self, mock_parse_thought_gen, mock_parse_classify, mock_parse_initial):
        problem_text = "Plan a trip to Mars."

        # Sequence of classifications: CONTINUE for root, then FINAL_ANSWER for child
        mock_parse_classify.side_effect = [L2TNodeCategory.CONTINUE, L2TNodeCategory.FINAL_ANSWER]
        
        self.mock_llm_client.call.side_effect = [
            self._mock_llm_call("Initial thought LLM response"),      # Initial thought
            self._mock_llm_call("Node 1 classification LLM response (CONTINUE)"), # Classify root
            self._mock_llm_call("Generated thought 1 LLM response"), # Generate child from root
            self._mock_llm_call("Node 2 classification LLM response (FINAL_ANSWER)") # Classify child
        ]

        cot_result = self.processor.run(problem_text)

        self.assertTrue(cot_result.succeeded)
        self.assertEqual(cot_result.final_answer, "Generated thought 1")
        self.assertIsInstance(cot_result.process_specific_data, L2TResult)

        l2t_native_result: L2TResult = cot_result.process_specific_data
        self.assertTrue(l2t_native_result.succeeded)
        self.assertEqual(l2t_native_result.final_answer, "Generated thought 1")
        self.assertEqual(l2t_native_result.total_llm_calls, 4)

        graph = l2t_native_result.reasoning_graph
        self.assertIsNotNone(graph)
        self.assertEqual(len(graph.nodes), 2) # Root and one child

        # Check root node classification
        root_node = graph.get_node(graph.root_node_id) # type: ignore
        self.assertEqual(root_node.category, L2TNodeCategory.CONTINUE) # type: ignore

        # Check child node (which should be the final answer)
        child_node_id = root_node.children_ids[0] # type: ignore
        child_node = graph.get_node(child_node_id)
        self.assertIsNotNone(child_node)
        self.assertEqual(child_node.content, "Generated thought 1") # type: ignore
        self.assertEqual(child_node.category, L2TNodeCategory.FINAL_ANSWER) # type: ignore

        # Ensure parsers were called correctly
        mock_parse_initial.assert_called_once_with("Initial thought LLM response")
        self.assertEqual(mock_parse_classify.call_count, 2)
        mock_parse_classify.assert_any_call("Node 1 classification LLM response (CONTINUE)")
        mock_parse_classify.assert_any_call("Node 2 classification LLM response (FINAL_ANSWER)")
        mock_parse_thought_gen.assert_called_once_with("Generated thought 1 LLM response")


    @patch.object(L2TResponseParser, 'parse_l2t_initial_response', return_value="Initial thought")
    @patch.object(L2TResponseParser, 'parse_l2t_node_classification_response', return_value=L2TNodeCategory.TERMINATE_BRANCH)
    def test_run_terminate_branch_no_final_answer(self, mock_parse_classify, mock_parse_initial):
        problem_text = "Explain quantum physics simply."
        self.config.max_steps = 1 # Ensure it stops after one step if no progress

        self.mock_llm_client.call.side_effect = [
            self._mock_llm_call("Initial thought response"), # Initial thought
            self._mock_llm_call("TERMINATE_BRANCH response") # Classification
        ]
        
        cot_result = self.processor.run(problem_text)

        self.assertFalse(cot_result.succeeded)
        self.assertIsNone(cot_result.final_answer)
        self.assertIsNotNone(cot_result.error_message) # Expect an error message
        self.assertIn("L2T process completed without a final answer", cot_result.error_message) # type: ignore

        self.assertIsInstance(cot_result.process_specific_data, L2TResult)
        l2t_native_result: L2TResult = cot_result.process_specific_data
        self.assertFalse(l2t_native_result.succeeded)
        self.assertIsNone(l2t_native_result.final_answer)
        self.assertIsNotNone(l2t_native_result.error_message)
        self.assertIn("L2T process completed without a final answer", l2t_native_result.error_message) # type: ignore

        graph = l2t_native_result.reasoning_graph
        self.assertIsNotNone(graph)
        self.assertEqual(len(graph.nodes), 1)
        root_node = graph.get_node(graph.root_node_id) # type: ignore
        self.assertEqual(root_node.category, L2TNodeCategory.TERMINATE_BRANCH) # type: ignore

    def test_run_initial_thought_failure(self):
        problem_text = "What if the moon was cheese?"
        # Simulate LLM client returning an error string for initial thought
        self.mock_llm_client.call.return_value = self._mock_llm_call("Error: LLM unavailable")
        
        # No need to mock parsers as they won't be reached if initial call fails fundamentally.
        # However, if parse_l2t_initial_response itself returns None for valid LLM output, that's different.
        # Here, we test the direct LLM "Error:" string.
        
        cot_result = self.processor.run(problem_text)

        self.assertFalse(cot_result.succeeded)
        self.assertIsNone(cot_result.final_answer)
        self.assertIsNotNone(cot_result.error_message)
        self.assertIn("Failed during initial thought generation", cot_result.error_message) # type: ignore
        self.assertIn("Error: LLM unavailable", cot_result.error_message) # type: ignore

        self.assertIsInstance(cot_result.process_specific_data, L2TResult)
        l2t_native_result: L2TResult = cot_result.process_specific_data
        self.assertFalse(l2t_native_result.succeeded)
        self.assertIn("Failed during initial thought generation", l2t_native_result.error_message) # type: ignore

    @patch.object(L2TResponseParser, 'parse_l2t_initial_response', return_value=None) # Simulate parser failure
    def test_run_initial_thought_parser_returns_none(self, mock_parse_initial):
        problem_text = "This will cause parser to return None"
        self.mock_llm_client.call.return_value = self._mock_llm_call("Valid LLM response, but parser fails")

        cot_result = self.processor.run(problem_text)
        
        self.assertFalse(cot_result.succeeded)
        self.assertIsNone(cot_result.final_answer)
        self.assertIsNotNone(cot_result.error_message)
        self.assertIn("Failed during initial thought generation", cot_result.error_message) # type: ignore
        
        mock_parse_initial.assert_called_once_with("Valid LLM response, but parser fails")


    def test_get_summary_success_case(self):
        # Construct a mock L2TResult
        mock_graph = L2TGraph()
        mock_root_node = L2TNode(id="root1", content="Root Content", parent_id=None, generation_step=0, category=L2TNodeCategory.CONTINUE)
        mock_child_node = L2TNode(id="child1", content="Final Answer Content", parent_id="root1", generation_step=1, category=L2TNodeCategory.FINAL_ANSWER)
        mock_graph.add_node(mock_root_node, is_root=True)
        mock_graph.add_node(mock_child_node)
        
        mock_l2t_result = L2TResult(
            succeeded=True,
            final_answer="Final Answer Content",
            total_llm_calls=3,
            total_completion_tokens=150,
            total_prompt_tokens=75,
            total_llm_interaction_time_seconds=2.5,
            total_process_wall_clock_time_seconds=3.0,
            reasoning_graph=mock_graph,
            error_message=None
        )
        
        # Construct CoTResult containing the L2TResult
        mock_cot_result = CoTResult(
            succeeded=True,
            final_answer="Final Answer Content",
            error_message=None,
            total_completion_tokens=150, # Should match l2t_native_result
            total_prompt_tokens=75,    # Should match l2t_native_result
            total_llm_interaction_time_seconds=2.5,
            total_process_wall_clock_time_seconds=3.0,
            process_specific_data=mock_l2t_result
        )

        summary_str = self.processor.get_summary(mock_cot_result)

        self.assertIsInstance(summary_str, str)
        self.assertIn("L2T PROCESS SUMMARY", summary_str)
        self.assertIn("Overall Succeeded (from CoTResult): True", summary_str)
        self.assertIn("Total LLM Calls (L2T): 3", summary_str)
        self.assertIn("Total Completion Tokens (L2T): 150", summary_str)
        self.assertIn("Total Prompt Tokens (L2T): 75", summary_str)
        self.assertIn("Grand Total Tokens (L2T Calls): 225", summary_str) # 150 + 75
        self.assertIn("Total L2T LLM Interaction Time: 2.50s", summary_str)
        self.assertIn("Total L2T Process Wall-Clock Time: 3.00s", summary_str)
        self.assertIn("Number of nodes in graph: 2", summary_str)
        self.assertIn("Root node ID: root1", summary_str)
        self.assertIn("Final answer derived from node: child1", summary_str)
        self.assertIn("Final Answer (from CoTResult):\nFinal Answer Content", summary_str)


    def test_get_summary_failure_case(self):
        mock_l2t_result = L2TResult(
            succeeded=False,
            final_answer=None,
            error_message="L2T hit max steps.",
            total_llm_calls=5,
            total_completion_tokens=200,
            total_prompt_tokens=100,
            total_llm_interaction_time_seconds=5.0,
            total_process_wall_clock_time_seconds=5.5,
            reasoning_graph=L2TGraph() # Empty graph
        )
        mock_cot_result = CoTResult(
            succeeded=False,
            final_answer=None, # No final answer from CoT perspective
            error_message="L2T hit max steps.", # Error message propagated
            total_completion_tokens=200,
            total_prompt_tokens=100,
            total_llm_interaction_time_seconds=5.0,
            total_process_wall_clock_time_seconds=5.5,
            process_specific_data=mock_l2t_result
        )

        summary_str = self.processor.get_summary(mock_cot_result)
        
        self.assertIn("Overall Succeeded (from CoTResult): False", summary_str)
        self.assertIn("Overall Error Message (from CoTResult): L2T hit max steps.", summary_str)
        # Also check if L2T's specific error is mentioned if different, but here it's the same
        self.assertIn("Number of nodes in graph: 0", summary_str)
        self.assertIn("Final answer was not successfully obtained.", summary_str)


    def test_get_summary_invalid_process_data(self):
        # Simulate a CoTResult where process_specific_data is not an L2TResult
        mock_cot_result_invalid_data = CoTResult(
            succeeded=False,
            final_answer=None,
            error_message="Some preceding error occurred.",
            process_specific_data="This is not an L2TResult object" # Invalid data
        )

        summary_str = self.processor.get_summary(mock_cot_result_invalid_data)

        self.assertIn("L2T Process Summary (Incomplete Data)", summary_str)
        self.assertIn("CoT Succeeded: False", summary_str)
        self.assertIn("CoT Error Message: Some preceding error occurred.", summary_str)
        self.assertIn("Note: L2T-specific details are unavailable", summary_str)

    @patch.object(L2TResponseParser, 'parse_l2t_initial_response', return_value="Initial thought")
    @patch.object(L2TResponseParser, 'parse_l2t_node_classification_response', side_effect=[L2TNodeCategory.CONTINUE] * 6) # Always continue
    @patch.object(L2TResponseParser, 'parse_l2t_thought_generation_response', return_value="New thought")
    def test_run_max_steps_reached(self, mock_gen, mock_classify, mock_initial):
        self.config.max_steps = 3 # Limit steps
        processor_limited_steps = L2TProcessor(llm_client=self.mock_llm_client, config=self.config)

        # Need enough LLM call mocks: Initial + (Classify + Gen) * max_steps
        # Initial: 1
        # Step 1: Classify (CONTINUE), Gen -> 2 calls
        # Step 2: Classify (CONTINUE), Gen -> 2 calls
        # Step 3: Classify (CONTINUE), Gen -> 2 calls
        # Total: 1 + 2*3 = 7 calls
        self.mock_llm_client.call.side_effect = [self._mock_llm_call("LLM response") for _ in range(7)]
        
        cot_result = processor_limited_steps.run("A long problem...")

        self.assertFalse(cot_result.succeeded)
        self.assertIsNone(cot_result.final_answer)
        self.assertIsNotNone(cot_result.error_message)
        self.assertIn("Max steps reached", cot_result.error_message) # type: ignore

        l2t_native_result: L2TResult = cot_result.process_specific_data # type: ignore
        self.assertFalse(l2t_native_result.succeeded)
        self.assertIn("Max steps reached", l2t_native_result.error_message) # type: ignore
        self.assertIsNotNone(l2t_native_result.reasoning_graph)
        # Number of nodes: Root + 1 per successful generation step
        # Step 0: Root (1 node)
        # Step 1 (CONTINUE): generates 1 child (Total 2 nodes)
        # Step 2 (CONTINUE): generates 1 child (Total 3 nodes)
        # Step 3 (CONTINUE): generates 1 child (Total 4 nodes)
        # Loop terminates as current_process_step (1-indexed) hits max_steps (3)
        self.assertEqual(len(l2t_native_result.reasoning_graph.nodes), self.config.max_steps + 1)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
