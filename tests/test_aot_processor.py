import unittest
from unittest.mock import MagicMock, patch, call # Ensure call is imported if used for multiple calls
import io
import logging
from typing import Tuple # Added for type hint

# Project imports
from src.aot_processor import AoTProcessor
from src.aot_dataclasses import AoTRunnerConfig, AoTResult, LLMCallStats, ParsedLLMOutput
from src.cot_process import CoTResult # To check return type
from src.llm_client import LLMClient # For mocking
# PromptGenerator might not need mocking if its methods are simple string formatting
# from src.prompt_generator import PromptGenerator 
from src.response_parser import ResponseParser # For mocking if its methods are complex

class TestAoTProcessor(unittest.TestCase):

    def setUp(self):
        self.mock_llm_client = MagicMock(spec=LLMClient)
        # Configure with minimal settings for most tests, can be overridden per test
        self.config = AoTRunnerConfig(
            main_model_names=["test_model"],
            temperature=0.0,
            max_steps=3, # Default for tests unless overridden
            max_reasoning_tokens=500,
            max_time_seconds=30,
            no_progress_limit=2 # Default for tests unless overridden
        )
        self.processor = AoTProcessor(llm_client=self.mock_llm_client, config=self.config)
        
        # Capture logs for tests that check logging output
        self.log_capture_string = io.StringIO()
        self.logger_handler = logging.StreamHandler(self.log_capture_string)
        # Assuming logger used in AoTProcessor is the root logger or one named 'src.aot_processor'
        # For simplicity, let's assume it's easy to get the relevant logger.
        # If AoTProcessor uses `logging.getLogger(__name__)`, then:
        logger_to_capture = logging.getLogger('src.aot_processor') 
        # If it's just logging (root), then logging.getLogger()
        # For this example, let's add handler to root to catch all if not specified.
        # A more targeted approach is better in complex apps.
        logging.getLogger().addHandler(self.logger_handler) 
        logging.getLogger().setLevel(logging.DEBUG) # Ensure logs are captured


    def tearDown(self):
        # Remove the handler to avoid interference between tests or with other logging
        logging.getLogger().removeHandler(self.logger_handler)
        self.log_capture_string.close()

    def _mock_llm_call(self, response_content: str, completion_tokens: int = 10, prompt_tokens: int = 5, duration: float = 0.1, model_name: str = "test_model") -> Tuple[str, LLMCallStats]:
        stats = LLMCallStats(
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            call_duration_seconds=duration,
            model_name=model_name
        )
        return response_content, stats

    # Test case for a successful run that completes within steps
    @patch.object(ResponseParser, 'parse_llm_output')
    def test_run_successful_completion(self, mock_parse_llm_output):
        # Mock LLM responses for steps
        self.mock_llm_client.call.side_effect = [
            self._mock_llm_call("Step 1: Thinking...", 20, 10),
            self._mock_llm_call("Step 2: Current answer is Foo. Final Answer: FooBar END", 30, 10) 
        ]
        # Mock parser outputs
        mock_parse_llm_output.side_effect = [
            ParsedLLMOutput(valid_steps_for_trace=["Step 1: Thinking..."], all_lines_from_model_for_context=["Step 1: Thinking..."], last_current_answer="Thinking..."),
            ParsedLLMOutput(valid_steps_for_trace=["Step 2: Current answer is Foo."], all_lines_from_model_for_context=["Step 2: Current answer is Foo. Final Answer: FooBar END"], last_current_answer="Foo", final_answer_text="FooBar", is_final_answer_marked_done=True)
        ]

        problem_text = "Solve this problem."
        cot_result = self.processor.run(problem_text)

        self.assertIsInstance(cot_result, CoTResult)
        self.assertTrue(cot_result.succeeded)
        self.assertEqual(cot_result.final_answer, "FooBar")
        self.assertIsNotNone(cot_result.process_specific_data)
        self.assertIsInstance(cot_result.process_specific_data, AoTResult)
        
        aot_native_result: AoTResult = cot_result.process_specific_data # type: ignore
        self.assertTrue(aot_native_result.succeeded)
        self.assertEqual(aot_native_result.final_answer, "FooBar")
        self.assertIn("Step 1: Thinking...", cot_result.reasoning_trace) # Check CoTResult's trace
        self.assertIn("Step 2: Current answer is Foo.", cot_result.reasoning_trace)
        self.assertEqual(self.mock_llm_client.call.call_count, 2)


    # Test case for hitting max_steps
    @patch.object(ResponseParser, 'parse_llm_output')
    def test_run_hits_max_steps(self, mock_parse_llm_output):
        self.config.max_steps = 2 # Override for this test
        # Re-initialize processor with the new config for this specific test
        self.processor = AoTProcessor(llm_client=self.mock_llm_client, config=self.config)


        self.mock_llm_client.call.side_effect = [
            self._mock_llm_call("Step 1...", 10, 5),
            self._mock_llm_call("Step 2...", 10, 5),
            self._mock_llm_call("Final Answer: MaxStepAnswer", 10, 5) # Explicit final call
        ]
        mock_parse_llm_output.side_effect = [
            ParsedLLMOutput(valid_steps_for_trace=["Step 1..."], all_lines_from_model_for_context=["Step 1..."], last_current_answer="1"),
            ParsedLLMOutput(valid_steps_for_trace=["Step 2..."], all_lines_from_model_for_context=["Step 2..."], last_current_answer="2"),
            ParsedLLMOutput(final_answer_text="MaxStepAnswer", is_final_answer_marked_done=True) # For explicit final call
        ]

        problem_text = "Solve long problem."
        cot_result = self.processor.run(problem_text)

        self.assertTrue(cot_result.succeeded) # It should succeed if explicit final call gives an answer
        self.assertEqual(cot_result.final_answer, "MaxStepAnswer")
        self.assertEqual(self.mock_llm_client.call.call_count, 3) # 2 steps + 1 final call
        self.assertIsInstance(cot_result.process_specific_data, AoTResult)
        aot_native_result: AoTResult = cot_result.process_specific_data # type: ignore
        self.assertEqual(len(aot_native_result.reasoning_trace), 2)


    # Test for no_progress_limit
    @patch.object(ResponseParser, 'parse_llm_output')
    def test_run_no_progress_limit_reached(self, mock_parse_llm_output):
        self.config.no_progress_limit = 1 # Stop after 1 repetition
        self.processor = AoTProcessor(llm_client=self.mock_llm_client, config=self.config)


        self.mock_llm_client.call.side_effect = [
            self._mock_llm_call("Step 1: Current answer is X", 10, 5),
            self._mock_llm_call("Step 2: Current answer is X", 10, 5), # No progress
            self._mock_llm_call("Final Answer: NoProgressAnswer", 10, 5) # Explicit final call
        ]
        mock_parse_llm_output.side_effect = [
            ParsedLLMOutput(valid_steps_for_trace=["Step 1: Current answer is X"], all_lines_from_model_for_context=["Step 1: Current answer is X"], last_current_answer="X"),
            ParsedLLMOutput(valid_steps_for_trace=["Step 2: Current answer is X"], all_lines_from_model_for_context=["Step 2: Current answer is X"], last_current_answer="X"),
            ParsedLLMOutput(final_answer_text="NoProgressAnswer", is_final_answer_marked_done=True)
        ]
        
        problem_text = "Problem leading to no progress"
        cot_result = self.processor.run(problem_text)

        self.assertTrue(cot_result.succeeded)
        self.assertEqual(cot_result.final_answer, "NoProgressAnswer")
        self.assertEqual(self.mock_llm_client.call.call_count, 3) # 2 steps (step1, step2 which triggers no_progress) + 1 final call
        aot_native_result: AoTResult = cot_result.process_specific_data # type: ignore
        self.assertIn("Step 1: Current answer is X", aot_native_result.reasoning_trace)
        self.assertIn("Step 2: Current answer is X", aot_native_result.reasoning_trace)
        
        # Check log for no progress
        log_contents = self.log_capture_string.getvalue()
        self.assertIn("No progress detected", log_contents)


    # Test get_summary method
    def test_get_summary_success_case(self):
        aot_native_result = AoTResult(
            succeeded=True,
            final_answer="Summary Test Answer",
            reasoning_trace=["Trace 1", "Trace 2"],
            reasoning_completion_tokens=100,
            total_completion_tokens=120, # Includes reasoning + final AoT call tokens
            total_prompt_tokens=50,
            total_llm_interaction_time_seconds=1.5,
            total_process_wall_clock_time_seconds=2.0
        )
        cot_result_input = CoTResult(
            succeeded=True, # This should align with aot_native_result.succeeded
            final_answer="Summary Test Answer", # Align with aot_native_result.final_answer
            error_message=None,
            # These should be the same as in aot_native_result for a direct mapping
            total_completion_tokens=aot_native_result.total_completion_tokens,
            total_prompt_tokens=aot_native_result.total_prompt_tokens,
            total_llm_interaction_time_seconds=aot_native_result.total_llm_interaction_time_seconds,
            total_process_wall_clock_time_seconds=aot_native_result.total_process_wall_clock_time_seconds,
            reasoning_trace=aot_native_result.reasoning_trace, # CoTResult also gets the trace
            process_specific_data=aot_native_result
        )

        summary = self.processor.get_summary(cot_result_input)
        
        self.assertIsInstance(summary, str)
        self.assertIn("AoT Process Summary", summary)
        self.assertIn("AoT Succeeded (from AoTResult): True", summary) # Check specific field from AoTResult
        self.assertIn("Final Answer (from AoTResult): Summary Test Answer", summary)
        self.assertIn("Total reasoning completion tokens: 100", summary)
        self.assertIn("Total completion tokens (AoT phase: reasoning + final AoT call): 120", summary)
        self.assertIn("Total prompt tokens (all AoT calls): 50", summary)
        self.assertIn("Grand total AoT tokens: 170", summary) # 120 + 50
        self.assertIn("Trace 1", summary) # Check if trace elements are in summary
        self.assertIn("Trace 2", summary)


    def test_get_summary_invalid_data(self):
        # Case where process_specific_data is not an AoTResult
        cot_result_input_invalid = CoTResult(
            succeeded=False,
            final_answer=None,
            error_message="Error before AoTResult was populated",
            total_completion_tokens=5,
            total_prompt_tokens=10,
            total_llm_interaction_time_seconds=0.5,
            total_process_wall_clock_time_seconds=0.6,
            process_specific_data="This is not an AoTResult object"
        )
        summary_invalid = self.processor.get_summary(cot_result_input_invalid)
        self.assertIn("AoT Process Summary (Incomplete Data)", summary_invalid)
        self.assertIn("CoT Succeeded: False", summary_invalid)
        self.assertIn("CoT Error Message: Error before AoTResult was populated", summary_invalid)
        self.assertIn("Note: AoT-specific details are unavailable", summary_invalid)

    # Add more tests:
    # - Hitting max_reasoning_tokens
    @patch.object(ResponseParser, 'parse_llm_output')
    def test_run_hits_max_reasoning_tokens(self, mock_parse_llm_output):
        self.config.max_steps = 5 # Allow more steps than tokens
        self.config.max_reasoning_tokens = 25 # Low token limit
        self.processor = AoTProcessor(llm_client=self.mock_llm_client, config=self.config)

        # Step 1 uses 15 tokens, Step 2 uses 15 (predicts 15), total 30 > 25, so it should stop before step 2
        self.mock_llm_client.call.side_effect = [
            self._mock_llm_call("Step 1 response", completion_tokens=15, prompt_tokens=5),
            # This call should ideally not happen for reasoning, but one for final answer might
            self._mock_llm_call("Final Answer: TokenLimitAnswer", completion_tokens=5, prompt_tokens=5) 
        ]
        mock_parse_llm_output.side_effect = [
            ParsedLLMOutput(valid_steps_for_trace=["Step 1 trace"], all_lines_from_model_for_context=["Step 1 context"], last_current_answer="Ans1"),
            ParsedLLMOutput(final_answer_text="TokenLimitAnswer", is_final_answer_marked_done=True) # For explicit final call
        ]

        problem_text = "Problem that consumes tokens."
        cot_result = self.processor.run(problem_text)
        
        self.assertTrue(cot_result.succeeded)
        self.assertEqual(cot_result.final_answer, "TokenLimitAnswer")
        # Should be 1 reasoning step + 1 final call
        self.assertEqual(self.mock_llm_client.call.call_count, 2) 
        
        aot_native_result: AoTResult = cot_result.process_specific_data # type: ignore
        self.assertEqual(aot_native_result.reasoning_completion_tokens, 15) # Only first step's tokens
        self.assertIn("Predicted to exceed token limit", self.log_capture_string.getvalue())


    # - Hitting max_time_seconds (harder to test precisely without actual time.sleep mocks)
    #   This would require patching time.monotonic
    @patch('time.monotonic')
    @patch.object(ResponseParser, 'parse_llm_output')
    def test_run_hits_max_time_seconds(self, mock_parse_llm_output, mock_monotonic):
        self.config.max_time_seconds = 1 # Very short time limit
        self.config.max_steps = 3
        self.processor = AoTProcessor(llm_client=self.mock_llm_client, config=self.config)

        # Simulate time passing: 0s, then 0.5s (step 1), then 1.1s (step 2, exceeds limit)
        mock_monotonic.side_effect = [
            0, # Initial process_start_time
            0, # First check in loop for step 1
            0.5, # After step 1 call
            0.5, # Check for step 2
            1.1, # After step 2 (or during check before it, if duration makes it exceed)
            1.1  # For final answer call time check
        ] 

        self.mock_llm_client.call.side_effect = [
            self._mock_llm_call("Step 1 response", duration=0.5), # Step 1 takes 0.5s
            # Step 2 would take 0.5s, but time limit is hit.
            self._mock_llm_call("Final Answer: TimeLimitAnswer", duration=0.1) # Final call
        ]
        mock_parse_llm_output.side_effect = [
            ParsedLLMOutput(valid_steps_for_trace=["Step 1 trace"], last_current_answer="Ans1"),
            ParsedLLMOutput(final_answer_text="TimeLimitAnswer", is_final_answer_marked_done=True)
        ]
        
        problem_text = "Time sensitive problem."
        cot_result = self.processor.run(problem_text)

        self.assertTrue(cot_result.succeeded)
        self.assertEqual(cot_result.final_answer, "TimeLimitAnswer")
        # 1 reasoning step (step 2 is preempted by time) + 1 final call
        self.assertEqual(self.mock_llm_client.call.call_count, 2) 
        self.assertIn("Maximum process time limit", self.log_capture_string.getvalue())


    # - LLM call fails during steps
    @patch.object(ResponseParser, 'parse_llm_output') # Still need to mock parser for any successful calls
    def test_run_llm_call_fails_during_step(self, mock_parse_llm_output):
        self.mock_llm_client.call.side_effect = [
            self._mock_llm_call("Step 1 response"),
            ("Error: LLM service unavailable", LLMCallStats(model_name="error_model")), # Step 2 fails
            self._mock_llm_call("Final Answer: LLMFailRecovery", 5, 5) # Explicit final call
        ]
        mock_parse_llm_output.side_effect = [
            ParsedLLMOutput(valid_steps_for_trace=["Step 1 trace"], last_current_answer="Ans1"),
            # No parser call for failed LLM step
            ParsedLLMOutput(final_answer_text="LLMFailRecovery", is_final_answer_marked_done=True)
        ]

        problem_text = "Problem with unreliable LLM."
        cot_result = self.processor.run(problem_text)

        self.assertTrue(cot_result.succeeded) # Succeeds due to final answer recovery
        self.assertEqual(cot_result.final_answer, "LLMFailRecovery")
        # 1 successful step + 1 failed step + 1 final call
        self.assertEqual(self.mock_llm_client.call.call_count, 3) 
        self.assertIn("LLM call failed for step 2", self.log_capture_string.getvalue())
        aot_native_result: AoTResult = cot_result.process_specific_data # type: ignore
        self.assertEqual(len(aot_native_result.reasoning_trace), 1) # Only first step's trace


    # - LLM call for final answer fails
    @patch.object(ResponseParser, 'parse_llm_output')
    def test_run_llm_final_call_fails(self, mock_parse_llm_output):
        self.config.max_steps = 1 # Force explicit final call quickly
        self.processor = AoTProcessor(llm_client=self.mock_llm_client, config=self.config)

        self.mock_llm_client.call.side_effect = [
            self._mock_llm_call("Step 1 response"),
            ("Error: Final LLM unavailable", LLMCallStats(model_name="error_model")) # Final call fails
        ]
        mock_parse_llm_output.side_effect = [
            ParsedLLMOutput(valid_steps_for_trace=["Step 1 trace"], last_current_answer="Ans1")
            # No parser call for failed final LLM
        ]
        
        problem_text = "Problem with failing final LLM."
        cot_result = self.processor.run(problem_text)

        self.assertFalse(cot_result.succeeded)
        self.assertTrue(cot_result.final_answer.startswith("Error: AoT process did not yield a final answer")) # type: ignore
        self.assertIsNotNone(cot_result.error_message)
        self.assertIn("LLM call for final answer failed: Error: Final LLM unavailable", self.log_capture_string.getvalue())
        # 1 successful step + 1 failed final call
        self.assertEqual(self.mock_llm_client.call.call_count, 2)


    # - Parser returns no final answer from explicit call (e.g. empty string or not marked done)
    @patch.object(ResponseParser, 'parse_llm_output')
    def test_run_parser_no_final_answer_from_explicit_call(self, mock_parse_llm_output):
        self.config.max_steps = 1
        self.processor = AoTProcessor(llm_client=self.mock_llm_client, config=self.config)

        self.mock_llm_client.call.side_effect = [
            self._mock_llm_call("Step 1 response"),
            self._mock_llm_call("This is not a final answer format.") # Final call LLM response
        ]
        mock_parse_llm_output.side_effect = [
            ParsedLLMOutput(valid_steps_for_trace=["Step 1 trace"], last_current_answer="Ans1"),
            ParsedLLMOutput(final_answer_text=None, is_final_answer_marked_done=False) # Parser finds nothing
        ]

        problem_text = "Problem where final parser fails."
        cot_result = self.processor.run(problem_text)
        
        self.assertFalse(cot_result.succeeded) # Fails because final answer not extracted
        # The final_answer will be the full response as fallback
        self.assertEqual(cot_result.final_answer, "This is not a final answer format.")
        self.assertIsNotNone(cot_result.error_message) # Error message should be set by CoTResult constructor
        self.assertIn("Could not parse final answer from explicit call", self.log_capture_string.getvalue())


    # - Ran_out_of_steps_signal from parser during reasoning
    @patch.object(ResponseParser, 'parse_llm_output')
    def test_run_ran_out_of_steps_signal(self, mock_parse_llm_output):
        self.mock_llm_client.call.side_effect = [
            self._mock_llm_call("Step 1: Thinking..."),
            self._mock_llm_call("Step 2: All done signal!"), # LLM signals completion
            self._mock_llm_call("Final Answer: RanOutAnswer", 10,5) # Explicit final call as no answer in signal
        ]
        mock_parse_llm_output.side_effect = [
            ParsedLLMOutput(valid_steps_for_trace=["Step 1: Thinking..."], last_current_answer="Thinking..."),
            ParsedLLMOutput(valid_steps_for_trace=["Step 2: All done signal!"], ran_out_of_steps_signal=True),
            ParsedLLMOutput(final_answer_text="RanOutAnswer", is_final_answer_marked_done=True)
        ]

        problem_text = "Problem solved quickly by LLM."
        cot_result = self.processor.run(problem_text)

        self.assertTrue(cot_result.succeeded)
        self.assertEqual(cot_result.final_answer, "RanOutAnswer")
        # Step 1, Step 2 (signals end), Explicit final call
        self.assertEqual(self.mock_llm_client.call.call_count, 3) 
        self.assertIn("Model signaled all unique reasoning steps are complete", self.log_capture_string.getvalue())


if __name__ == '__main__':
    # unittest.main() already provides verbosity options.
    # BasicConfig is fine for dev, but for CI, might rely on runner's output.
    # logging.basicConfig(level=logging.DEBUG) # Keep it simple or remove for cleaner test output
    unittest.main(verbosity=2)
