import unittest
from unittest.mock import patch, MagicMock, call # Added call for checking multiple calls

import torch

from src.l2t_full.processor import L2TFullProcessor
from src.l2t_full.orchestrator import L2TFullOrchestrator
from src.l2t_full.dataclasses import L2TFullConfig, L2TResult, L2TFullSolution, L2TNode
from src.l2t_full.enums import L2TTriggerMode, L2TNodeCategory, L2TTerminationReason
from src.l2t_full.constants import DEFAULT_GNN_INPUT_NODE_DIM
from src.llm_client import LLMCallStats # To mock return values

# Suppress INFO logs from sentence_transformers during tests
import logging
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


class TestL2TFullProcessorIntegration(unittest.TestCase):
    def setUp(self):
        self.config = L2TFullConfig(
            max_steps=3, # Keep low for testing
            max_total_nodes=10,
            gnn_input_dim=DEFAULT_GNN_INPUT_NODE_DIM, # Ensure it's 384 if using real sentence transformer
            # Use smaller model lists for faster test setup if models were actually loaded
            initial_prompt_model_names=["mock-model"],
            classification_model_names=["mock-model"],
            thought_generation_model_names=["mock-model"],
        )
        # Patch the SentenceTransformer to avoid loading the actual model in tests
        # if it's not already handled by a global test setup.
        # For these integration tests, we assume it loads, or mock it if it causes issues.
        # self.sentence_transformer_patcher = patch('src.l2t_full.processor.SentenceTransformer')
        # self.mock_sentence_transformer_constructor = self.sentence_transformer_patcher.start()
        # self.mock_sentence_transformer_instance = MagicMock()
        # self.mock_sentence_transformer_constructor.return_value = self.mock_sentence_transformer_instance
        # self.mock_sentence_transformer_instance.encode.return_value = torch.randn(1, self.config.gnn_input_dim)

        self.processor = L2TFullProcessor(api_key="test_key_dummy", config=self.config)

        # Mock PPOAgent's update method to check if it's called without full execution
        self.ppo_update_patcher = patch.object(self.processor.ppo_agent, 'update', wraps=self.processor.ppo_agent.update)
        self.mock_ppo_update = self.ppo_update_patcher.start()


    def tearDown(self):
        # self.sentence_transformer_patcher.stop()
        self.ppo_update_patcher.stop()

    @patch('src.l2t_full.processor.LLMClient.call')
    def test_processor_run_simple_path_to_final_answer(self, mock_llm_call):
        # Configure the mock LLMClient.call responses
        mock_llm_call.side_effect = [
            # 1. Initial thought generation
            ("Your thought: Initial thought content", LLMCallStats(model_name="mock", completion_tokens=5, prompt_tokens=5, call_duration_seconds=0.1)),
            # 2. Classification of initial thought (node1) -> CONTINUE
            ("Your classification: CONTINUE", LLMCallStats(model_name="mock", completion_tokens=1, prompt_tokens=5, call_duration_seconds=0.1)),
            # 3. Thought generation from node1 -> node2
            ("Your new thought: Second thought content", LLMCallStats(model_name="mock", completion_tokens=5, prompt_tokens=5, call_duration_seconds=0.1)),
            # 4. Classification of node2 -> FINAL_ANSWER
            ("Your classification: FINAL_ANSWER", LLMCallStats(model_name="mock", completion_tokens=1, prompt_tokens=5, call_duration_seconds=0.1)),
        ]

        result = self.processor.run(problem_text="simple test problem")

        self.assertTrue(result.succeeded, "Run should succeed")
        self.assertIsNotNone(result.final_answer, "Final answer should exist")
        self.assertEqual(result.final_answer, "Second thought content", "Final answer content mismatch")
        self.assertEqual(result.termination_reason, L2TTerminationReason.SOLUTION_FOUND)
        self.assertTrue(len(self.processor.ppo_agent.experience_buffer) > 0, "Experiences should be stored")
        self.mock_ppo_update.assert_called_once() # Check PPO update was called

    @patch('src.l2t_full.processor.LLMClient.call')
    def test_processor_run_reaches_max_steps(self, mock_llm_call):
        self.processor.config.max_steps = 2 # Override for this test

        # Configure mock to always continue
        mock_llm_call.side_effect = [
            # Step 0: Initial thought
            ("Your thought: Step 0 thought", LLMCallStats(model_name="mock", completion_tokens=5, prompt_tokens=5, call_duration_seconds=0.1)),
            # Step 1: Classify node0 -> CONTINUE, Generate thought for node1
            ("Your classification: CONTINUE", LLMCallStats(model_name="mock", completion_tokens=1, prompt_tokens=5, call_duration_seconds=0.1)),
            ("Your new thought: Step 1 thought", LLMCallStats(model_name="mock", completion_tokens=5, prompt_tokens=5, call_duration_seconds=0.1)),
            # Step 2: Classify node1 -> CONTINUE, Generate thought for node2
            ("Your classification: CONTINUE", LLMCallStats(model_name="mock", completion_tokens=1, prompt_tokens=5, call_duration_seconds=0.1)),
            ("Your new thought: Step 2 thought", LLMCallStats(model_name="mock", completion_tokens=5, prompt_tokens=5, call_duration_seconds=0.1)),
            # Step 3 (should not happen if max_steps is 2, processor does current_step > max_steps)
            # Classification for node2 (if it were to happen)
            ("Your classification: FINAL_ANSWER", LLMCallStats(model_name="mock", completion_tokens=1, prompt_tokens=5, call_duration_seconds=0.1)),
        ]

        result = self.processor.run(problem_text="max steps test problem")

        self.assertFalse(result.succeeded, "Run should not succeed if max_steps reached before final answer")
        self.assertEqual(result.termination_reason, L2TTerminationReason.MAX_STEPS_REACHED)
        # Processor runs N steps, storing N experiences.
        # Step 0 (initial) + Step 1 (process node0) + Step 2 (process node1) = 2 nodes processed, 2 experiences
        self.assertEqual(len(self.processor.ppo_agent.experience_buffer), 2, "Should store experiences for each processed node step")
        self.mock_ppo_update.assert_called_once()


class TestL2TFullOrchestratorIntegration(unittest.TestCase):
    def setUp(self):
        self.config = L2TFullConfig(
            max_steps=2, # Keep low for orchestrator tests too
            max_total_nodes=5,
            gnn_input_dim=DEFAULT_GNN_INPUT_NODE_DIM,
            initial_prompt_model_names=["mock-model-orch"],
            classification_model_names=["mock-model-orch"],
            thought_generation_model_names=["mock-model-orch"],
        )
        # Patch the SentenceTransformer for Orchestrator tests too
        self.sentence_transformer_patcher = patch('src.l2t_full.processor.SentenceTransformer')
        self.mock_sentence_transformer_constructor = self.sentence_transformer_patcher.start()
        self.mock_sentence_transformer_instance = MagicMock()
        self.mock_sentence_transformer_constructor.return_value = self.mock_sentence_transformer_instance
        self.mock_sentence_transformer_instance.encode.return_value = torch.randn(1, self.config.gnn_input_dim)

        self.orchestrator = L2TFullOrchestrator(
            trigger_mode=L2TTriggerMode.ALWAYS_L2T,
            l2t_full_config=self.config,
            direct_oneshot_model_names=["mock-fallback-model"],
            direct_oneshot_temperature=0.1,
            assessment_model_names=["mock-assess-model"],
            assessment_temperature=0.1,
            api_key="test_key_dummy"
        )
        # Mock PPOAgent's update for orchestrator's processor instance
        self.orch_ppo_update_patcher = patch.object(self.orchestrator.l2t_full_process_instance.l2t_full_processor.ppo_agent, 'update')
        self.mock_orch_ppo_update = self.orch_ppo_update_patcher.start()


    def tearDown(self):
        self.sentence_transformer_patcher.stop()
        self.orch_ppo_update_patcher.stop()

    @patch('src.llm_client.LLMClient.call') # Patch LLMClient used by Processor and Orchestrator's OneShot
    def test_orchestrator_solve_always_l2t_full_success(self, mock_llm_call):
        # Configure mock LLM calls for a successful L2T-Full run
        mock_llm_call.side_effect = [
            # L2TFullProcessor: Initial thought
            ("Your thought: Initial problem thought", LLMCallStats(model_name="mock", completion_tokens=5, prompt_tokens=5, call_duration_seconds=0.1)),
            # L2TFullProcessor: Classify initial thought -> CONTINUE
            ("Your classification: CONTINUE", LLMCallStats(model_name="mock", completion_tokens=1, prompt_tokens=5, call_duration_seconds=0.1)),
            # L2TFullProcessor: Generate next thought
            ("Your new thought: This is the final answer thought", LLMCallStats(model_name="mock", completion_tokens=5, prompt_tokens=5, call_duration_seconds=0.1)),
            # L2TFullProcessor: Classify second thought -> FINAL_ANSWER
            ("Your classification: FINAL_ANSWER", LLMCallStats(model_name="mock", completion_tokens=1, prompt_tokens=5, call_duration_seconds=0.1)),
        ]

        solution, summary = self.orchestrator.solve(problem_text="orchestrator success test")

        self.assertTrue(solution.succeeded, "Orchestrator solution should succeed")
        self.assertIsNotNone(solution.final_answer, "Final answer should be populated")
        self.assertEqual(solution.final_answer, "This is the final answer thought")
        self.assertIsNotNone(solution.l2t_full_result, "L2T-Full result should be populated")
        self.assertTrue(solution.l2t_full_result.succeeded, "L2T-Full result itself should indicate success")
        self.assertFalse(solution.l2t_failed_and_fell_back, "Should not have fallen back")
        self.assertIsNone(solution.fallback_call_stats, "No fallback stats should exist")
        self.assertTrue(len(summary) > 0, "Summary should be non-empty")
        self.mock_orch_ppo_update.assert_called_once() # PPO update in processor should be called

    @patch('src.llm_client.LLMClient.call')
    def test_orchestrator_solve_l2t_full_fails_then_fallback(self, mock_llm_call):
        self.config.max_steps = 1 # Ensure L2TFull fails quickly

        # L2TFullProcessor fails (e.g. max_steps)
        mock_llm_call.side_effect = [
            # L2TFullProcessor: Initial thought (Overall Call 1)
            ("Your thought: Initial problem thought", LLMCallStats(model_name="mock", completion_tokens=5, prompt_tokens=5, call_duration_seconds=0.1)),
            # L2TFullProcessor: _process_node for initial thought (step 1)
            #   Classification (Overall Call 2)
            ("Your classification: CONTINUE", LLMCallStats(model_name="mock", completion_tokens=1, prompt_tokens=5, call_duration_seconds=0.1)),
            #   Thought generation (Overall Call 3)
            ("Your new thought: Thought from step 1 that will be discarded by max_steps", LLMCallStats(model_name="mock", completion_tokens=5, prompt_tokens=5, call_duration_seconds=0.1)),
            # Fallback OneShotExecutor call (Overall Call 4)
            ("Fallback answer from one-shot", LLMCallStats(model_name="mock-fallback", completion_tokens=5, prompt_tokens=5, call_duration_seconds=0.1)),
        ]

        solution, summary = self.orchestrator.solve(problem_text="orchestrator fallback test")

        self.assertTrue(solution.succeeded, "Overall solution should succeed due to fallback")
        self.assertEqual(solution.final_answer, "Fallback answer from one-shot")
        self.assertIsNotNone(solution.l2t_full_result, "L2T-Full result should still exist")
        self.assertFalse(solution.l2t_full_result.succeeded, "L2T-Full result itself should indicate failure")
        self.assertEqual(solution.l2t_full_result.termination_reason, L2TTerminationReason.MAX_STEPS_REACHED)
        self.assertTrue(solution.l2t_failed_and_fell_back, "Should have fallen back")
        self.assertIsNotNone(solution.fallback_call_stats, "Fallback stats should exist")
        self.assertEqual(solution.fallback_call_stats.model_name, "mock-fallback")
        self.assertTrue(len(summary) > 0)
        self.mock_orch_ppo_update.assert_called_once() # PPO update still called for the L2TFull part

if __name__ == '__main__':
    unittest.main()
