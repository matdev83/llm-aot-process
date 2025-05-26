import time
import logging
import io
from typing import List, Optional, Tuple
from src.aot_dataclasses import LLMCallStats, ParsedLLMOutput, AoTRunnerConfig, AoTResult
from src.llm_client import LLMClient
from src.prompt_generator import PromptGenerator
from src.response_parser import ResponseParser
from src.aot_constants import MIN_PREDICTED_STEP_TOKENS_FALLBACK, MIN_PREDICTED_STEP_DURATION_FALLBACK

class AoTProcessor:
    def __init__(self, llm_client: LLMClient, config: AoTRunnerConfig):
        self.llm_client = llm_client
        self.config = config

    def run(self, problem_text: str) -> Tuple[AoTResult, str]:
        result = AoTResult()
        last_answers_from_steps: List[str] = []
        no_progress_count = 0
        process_start_time = time.monotonic()

        step_completion_tokens_history: List[int] = []
        step_call_duration_history: List[float] = []

        original_configured_max_steps = self.config.max_steps
        current_effective_max_steps = original_configured_max_steps

        logging.info(f"Starting AoT process for problem: '{problem_text[:100].strip()}...'")
        logging.info(f"Main Models: {', '.join(self.config.main_model_names)}, Temperature: {self.config.temperature}")
        if self.config.max_reasoning_tokens: 
            logging.info(f"Reasoning Token Limit: {self.config.max_reasoning_tokens}")
        logging.info(f"Max Time Limit (Overall Process): {self.config.max_time_seconds}s")
        logging.info(f"Original Max Steps Config: {original_configured_max_steps}")
        if self.config.pass_remaining_steps_pct is not None:
             trigger_step_for_advisory = self.config.pass_remaining_steps_pct * original_configured_max_steps
             logging.info(f"Remaining steps advisory will trigger at/after step {trigger_step_for_advisory:.1f} (of original max steps)")
        logging.info(f"No Progress Limit: {self.config.no_progress_limit} steps")

        reasoning_loop_broke_early = False
        current_step_1_indexed = 0 # Initialize to 0 for cases where loop doesn't run
        
        for step_num_for_loop in range(original_configured_max_steps):
            current_step_1_indexed = step_num_for_loop + 1
            elapsed_process_time = time.monotonic() - process_start_time

            if current_step_1_indexed > current_effective_max_steps:
                logging.info(f"Current step {current_step_1_indexed} exceeds dynamically adjusted effective max steps ({current_effective_max_steps}). Stopping reasoning phase.")
                reasoning_loop_broke_early = True; break
            if elapsed_process_time >= self.config.max_time_seconds:
                logging.info(f"Maximum process time limit ({self.config.max_time_seconds}s) reached (overall wall-clock). Stopping reasoning phase.")
                reasoning_loop_broke_early = True; break

            _affordable_total_steps_from_tokens_info = "N/A"
            _affordable_total_steps_from_time_info = "N/A"

            if self.config.max_reasoning_tokens and self.config.max_reasoning_tokens > 0:
                predicted_tokens_for_current_step = MIN_PREDICTED_STEP_TOKENS_FALLBACK
                current_avg_token_delta = 0.0
                if not step_completion_tokens_history:
                    if original_configured_max_steps > 0: predicted_tokens_for_current_step = max(MIN_PREDICTED_STEP_TOKENS_FALLBACK, (self.config.max_reasoning_tokens / original_configured_max_steps))
                else:
                    if len(step_completion_tokens_history) >= 2:
                        deltas = [step_completion_tokens_history[i] - step_completion_tokens_history[i-1] for i in range(1, len(step_completion_tokens_history))]
                        if deltas: current_avg_token_delta = sum(deltas) / len(deltas)
                    predicted_tokens_for_current_step = max(MIN_PREDICTED_STEP_TOKENS_FALLBACK, step_completion_tokens_history[-1] + current_avg_token_delta)
                
                if result.reasoning_completion_tokens + predicted_tokens_for_current_step > self.config.max_reasoning_tokens:
                    logging.info(f"[HARD STOP TOKENS] Predicted to exceed token limit ({self.config.max_reasoning_tokens}) if current step {current_step_1_indexed} is taken. "
                                 f"Current usage: {result.reasoning_completion_tokens}, predicted: {predicted_tokens_for_current_step:.0f}. Stopping.")
                    reasoning_loop_broke_early = True; break
                
                affordable_steps_by_token = 0
                temp_token_budget = self.config.max_reasoning_tokens - result.reasoning_completion_tokens
                cost_of_projected_token_step = predicted_tokens_for_current_step
                max_proj_iter = original_configured_max_steps - current_step_1_indexed + 1
                for _ in range(max_proj_iter):
                    if temp_token_budget >= cost_of_projected_token_step:
                        temp_token_budget -= cost_of_projected_token_step
                        affordable_steps_by_token += 1
                        cost_of_projected_token_step = max(MIN_PREDICTED_STEP_TOKENS_FALLBACK, cost_of_projected_token_step + current_avg_token_delta)
                    else: break
                _affordable_total_steps_from_tokens_info = str(affordable_steps_by_token)
                new_limit_by_token = (current_step_1_indexed - 1) + affordable_steps_by_token
                if new_limit_by_token < current_effective_max_steps:
                    logging.info(f"[TOKEN LIMITING] Token budget suggests at most {affordable_steps_by_token} step(s) from step {current_step_1_indexed}. "
                                 f"Adjusting effective max steps from {current_effective_max_steps} to {new_limit_by_token}.")
                    current_effective_max_steps = new_limit_by_token
                    if current_step_1_indexed > current_effective_max_steps:
                         logging.info(f"Current step {current_step_1_indexed} now exceeds token-adjusted effective max steps ({current_effective_max_steps}). Stopping.")
                         reasoning_loop_broke_early = True; break
            
            if self.config.max_time_seconds > 0:
                remaining_time_budget_for_steps = self.config.max_time_seconds - elapsed_process_time
                predicted_duration_for_current_step = MIN_PREDICTED_STEP_DURATION_FALLBACK
                current_avg_duration_delta = 0.0
                if not step_call_duration_history:
                    steps_remaining_for_initial_pred = original_configured_max_steps - current_step_1_indexed + 1
                    if steps_remaining_for_initial_pred > 0: predicted_duration_for_current_step = max(MIN_PREDICTED_STEP_DURATION_FALLBACK, (remaining_time_budget_for_steps * 0.9) / steps_remaining_for_initial_pred)
                    # else predicted_duration_for_current_step remains MIN_PREDICTED_STEP_DURATION_FALLBACK
                else:
                    if len(step_call_duration_history) >= 2:
                        deltas = [step_call_duration_history[i] - step_call_duration_history[i-1] for i in range(1, len(step_call_duration_history))]
                        if deltas: current_avg_duration_delta = sum(deltas) / len(deltas)
                    predicted_duration_for_current_step = max(MIN_PREDICTED_STEP_TOKENS_FALLBACK, step_call_duration_history[-1] + current_avg_duration_delta)

                if predicted_duration_for_current_step > remaining_time_budget_for_steps and remaining_time_budget_for_steps > 0 :
                    logging.info(f"[HARD STOP TIME] Predicted duration ({predicted_duration_for_current_step:.2f}s) for current step {current_step_1_indexed} "
                                 f"exceeds remaining time budget ({remaining_time_budget_for_steps:.2f}s). Stopping.")
                    reasoning_loop_broke_early = True; break
                
                affordable_steps_by_time = 0
                temp_time_budget_projection = remaining_time_budget_for_steps
                cost_of_projected_time_step = predicted_duration_for_current_step
                max_proj_iter = original_configured_max_steps - current_step_1_indexed + 1
                for _ in range(max_proj_iter):
                    if temp_time_budget_projection >= cost_of_projected_time_step:
                        temp_time_budget_projection -= cost_of_projected_time_step
                        affordable_steps_by_time += 1
                        cost_of_projected_time_step = max(MIN_PREDICTED_STEP_DURATION_FALLBACK, cost_of_projected_time_step + current_avg_duration_delta)
                    else: break
                _affordable_total_steps_from_time_info = str(affordable_steps_by_time)
                new_limit_by_time = (current_step_1_indexed - 1) + affordable_steps_by_time
                if new_limit_by_time < current_effective_max_steps:
                    logging.info(f"[TIME LIMITING] Time budget suggests at most {affordable_steps_by_time} step(s) from step {current_step_1_indexed}. "
                                 f"Adjusting effective max steps from {current_effective_max_steps} to {new_limit_by_time}.")
                    current_effective_max_steps = new_limit_by_time
                    if current_step_1_indexed > current_effective_max_steps:
                         logging.info(f"Current step {current_step_1_indexed} now exceeds time-adjusted effective max steps ({current_effective_max_steps}). Stopping.")
                         reasoning_loop_broke_early = True; break

            step_info_extra = f", AffordTokens: {_affordable_total_steps_from_tokens_info}, AffordTime: {_affordable_total_steps_from_time_info}"
            logging.info(f"--- Step {current_step_1_indexed}/{current_effective_max_steps} (OrigMax: {original_configured_max_steps}{step_info_extra}, ElapsedWallClock: {elapsed_process_time:.2f}s/{self.config.max_time_seconds}s) ---")

            prompt = PromptGenerator.construct_aot_step_prompt(
                problem_text, result.full_history_for_context, current_step_1_indexed,
                current_effective_max_steps, original_configured_max_steps, self.config.pass_remaining_steps_pct
            )
            reply_content, step_stats = self.llm_client.call(
                prompt, models=self.config.main_model_names, temperature=self.config.temperature
            )

            result.total_llm_interaction_time_seconds += step_stats.call_duration_seconds
            result.total_completion_tokens += step_stats.completion_tokens
            result.reasoning_completion_tokens += step_stats.completion_tokens
            result.total_prompt_tokens += step_stats.prompt_tokens
            if step_stats.completion_tokens > 0: step_completion_tokens_history.append(step_stats.completion_tokens)
            if step_stats.call_duration_seconds > 0: step_call_duration_history.append(step_stats.call_duration_seconds)

            logging.debug(f"Response from {step_stats.model_name}:\n{reply_content}")
            logging.info(f"LLM call ({step_stats.model_name}): Duration: {step_stats.call_duration_seconds:.2f}s, Tokens (C:{step_stats.completion_tokens}, P:{step_stats.prompt_tokens})")
            logging.info(f"Cumulative reasoning completion tokens: {result.reasoning_completion_tokens}" +
                         (f" / {self.config.max_reasoning_tokens}" if self.config.max_reasoning_tokens else ""))
            logging.info(f"Cumulative LLM interaction time: {result.total_llm_interaction_time_seconds:.2f}s")

            if reply_content.startswith("Error:"):
                logging.critical(f"LLM call failed for step {current_step_1_indexed}. Aborting reasoning phase. Error: {reply_content}")
                reasoning_loop_broke_early = True; break

            parsed_data = ResponseParser.parse_llm_output(reply_content)
            result.full_history_for_context.extend(parsed_data.all_lines_from_model_for_context)
            result.reasoning_trace.extend(parsed_data.valid_steps_for_trace)

            if parsed_data.final_answer_text and parsed_data.is_final_answer_marked_done:
                logging.info("Final answer found within a reasoning step's response.")
                result.final_answer = parsed_data.final_answer_text
                reasoning_loop_broke_early = True
                if parsed_data.ran_out_of_steps_signal: 
                    logging.info("Model also signaled all unique reasoning steps are complete.")
                break
            if parsed_data.ran_out_of_steps_signal:
                logging.info("Model signaled all unique reasoning steps are complete (no final answer in this response).")
                reasoning_loop_broke_early = True; break

            if parsed_data.last_current_answer is not None:
                if last_answers_from_steps and parsed_data.last_current_answer == last_answers_from_steps[-1]:
                    no_progress_count += 1
                    logging.info(f"No progress: current answer '{parsed_data.last_current_answer}' is same as previous. Count: {no_progress_count}")
                else: no_progress_count = 0
                last_answers_from_steps.append(parsed_data.last_current_answer)
            else:
                no_progress_count += 1
                logging.warning(f"No 'Current answer/state:' found in step output. No-progress count: {no_progress_count}")
            
            if no_progress_count >= self.config.no_progress_limit:
                logging.warning(f"No progress detected for {self.config.no_progress_limit} consecutive steps. Stopping reasoning phase.")
                reasoning_loop_broke_early = True; break
            
            time.sleep(0.2) # Shorter sleep, mainly for API courtesy if calls are rapid

        if not reasoning_loop_broke_early and current_step_1_indexed >= current_effective_max_steps :
            logging.info(f"Max effective steps ({current_effective_max_steps}) reached.")

        if not result.final_answer: 
            logging.info("--- Generating Final Answer (Explicit AoT Call) ---")
            can_make_final_call = True
            if self.config.max_time_seconds > 0 and (self.config.max_time_seconds - (time.monotonic() - process_start_time)) < MIN_PREDICTED_STEP_DURATION_FALLBACK * 1.5: # Adjusted multiplier
                logging.warning("Very low time budget remaining for final AoT call. Skipping.")
                can_make_final_call = False
            if self.config.max_reasoning_tokens and self.config.max_reasoning_tokens > 0 and \
               (self.config.max_reasoning_tokens - result.reasoning_completion_tokens) < MIN_PREDICTED_STEP_TOKENS_FALLBACK * 1.5: # Adjusted multiplier
                logging.warning("Very low token budget remaining for final AoT call (for reasoning tokens). Skipping.")
                can_make_final_call = False

            if can_make_final_call:
                final_prompt = PromptGenerator.construct_aot_final_prompt(problem_text, result.full_history_for_context)
                final_reply_content, final_stats = self.llm_client.call(
                    final_prompt, models=self.config.main_model_names, temperature=self.config.temperature
                )
                result.total_llm_interaction_time_seconds += final_stats.call_duration_seconds
                result.total_completion_tokens += final_stats.completion_tokens # These are not reasoning_completion_tokens
                result.total_prompt_tokens += final_stats.prompt_tokens
                
                logging.debug(f"Final Response from {final_stats.model_name}:\n{final_reply_content}")
                logging.info(f"LLM call ({final_stats.model_name}): Duration: {final_stats.call_duration_seconds:.2f}s, Tokens (C:{final_stats.completion_tokens}, P:{final_stats.prompt_tokens})")
                
                if final_reply_content.startswith("Error:"):
                    logging.warning(f"LLM call for final answer failed: {final_reply_content}")
                else:
                    parsed_final_data = ResponseParser.parse_llm_output(final_reply_content)
                    if parsed_final_data.final_answer_text and parsed_final_data.is_final_answer_marked_done:
                        result.final_answer = parsed_final_data.final_answer_text
                    else:
                        logging.warning("Could not parse final answer from explicit call. Using full response as fallback (if not error).")
                        result.final_answer = final_reply_content # Use raw content if parsing fails
            else:
                 logging.info("Skipped explicit final answer generation due to budget constraints.")
        
        if result.final_answer and not result.final_answer.startswith("Error:"):
            result.succeeded = True
        else:
            result.succeeded = False
            if not result.final_answer:
                 result.final_answer = "Error: AoT process did not yield a final answer due to limits or parsing issues."
                 logging.warning(result.final_answer) # Log this specific failure reason

        result.total_process_wall_clock_time_seconds = time.monotonic() - process_start_time
        aot_summary_output = self._generate_aot_summary(result) # Call the new method
        return result, aot_summary_output # Return both

    def _generate_aot_summary(self, result: AoTResult) -> str: # New method name and return type
        output_buffer = io.StringIO()
        output_buffer.write("\n--- AoT Process Summary ---\n")
        output_buffer.write(f"AoT Succeeded: {result.succeeded}\n")
        output_buffer.write(f"Total reasoning completion tokens: {result.reasoning_completion_tokens}\n")
        output_buffer.write(f"Total completion tokens (AoT phase: reasoning + final AoT call): {result.total_completion_tokens}\n")
        output_buffer.write(f"Total prompt tokens (all AoT calls): {result.total_prompt_tokens}\n")
        output_buffer.write(f"Grand total AoT tokens: {result.total_completion_tokens + result.total_prompt_tokens}\n")
        output_buffer.write(f"Total AoT LLM interaction time: {result.total_llm_interaction_time_seconds:.2f}s\n")
        output_buffer.write(f"Total AoT process wall-clock time: {result.total_process_wall_clock_time_seconds:.2f}s\n")
        return output_buffer.getvalue()
