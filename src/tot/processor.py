import time
import logging
import uuid
import io
from typing import List, Optional, Tuple, Dict, Set

from src.llm_client import LLMClient
from src.llm_config import LLMConfig
from .dataclasses import (
    ToTConfig, ToTNode, ToTThought, ToTResult, LLMCallStats
)
from .enums import ToTSearchStrategy, ToTScoringMethod
from .prompt_generator import ToTPromptGenerator
from .response_parser import ToTResponseParser
from .constants import (
    DEFAULT_TOT_THOUGHT_GENERATION_TEMPERATURE,
    DEFAULT_TOT_EVALUATION_TEMPERATURE
)

class ToTProcessor:
    def __init__(self,
                 llm_client: LLMClient,
                 config: ToTConfig,
                 # Optional specific LLM configs if they differ from defaults in ToTConfig
                 thought_generation_llm_config: Optional[LLMConfig] = None,
                 evaluation_llm_config: Optional[LLMConfig] = None):
        self.llm_client = llm_client
        self.config = config

        self.thought_generation_llm_config = thought_generation_llm_config or LLMConfig(
            temperature=DEFAULT_TOT_THOUGHT_GENERATION_TEMPERATURE
        )
        self.evaluation_llm_config = evaluation_llm_config or LLMConfig(
            temperature=DEFAULT_TOT_EVALUATION_TEMPERATURE
        )
        self.prompt_generator = ToTPromptGenerator()
        self.response_parser = ToTResponseParser()

    def _generate_node_id(self) -> str:
        return str(uuid.uuid4())

    def _call_llm_for_thoughts(self, problem_description: str, current_node: ToTNode) -> List[ToTThought]:
        prompt = self.prompt_generator.generate_thoughts_prompt(problem_description, current_node, self.config)
        response_text, stats = self.llm_client.call(
            prompt,
            models=self.config.thought_generation_model_names,
            config=self.thought_generation_llm_config
        )
        # Update overall stats in the result object from the caller of this method

        if response_text.startswith("Error:"):
            logging.error(f"LLM call for thought generation failed: {response_text}")
            return []

        parsed_thoughts = self.response_parser.parse_generated_thoughts(response_text, self.config.k_thoughts)

        # Attach stats to each thought (or primarily to the first one if stats are per-call)
        if parsed_thoughts:
            for thought in parsed_thoughts:
                thought.raw_generation_response = response_text # Store raw response with each
                thought.generation_stats = stats # Associate call stats
        elif not parsed_thoughts and response_text: # If parsing failed but got a response
             # Create a single fallback thought with the raw response if parsing yields nothing
             # This helps debug or capture malformed LLM output as a single thought
             fallback_thought = ToTThought(text=f"Raw unparsed response: {response_text[:200]}...",
                                           raw_generation_response=response_text,
                                           generation_stats=stats)
             return [fallback_thought]


        return parsed_thoughts

    def _call_llm_for_evaluation(self, problem_description: str, candidate_node: ToTNode) -> Tuple[Optional[float], Optional[str], Optional[LLMCallStats]]:
        if self.config.scoring_method == ToTScoringMethod.HEURISTIC:
            # Placeholder for heuristic evaluation - not implemented in this subtask
            logging.warning("Heuristic scoring method selected but not implemented. Returning default score.")
            return 5.0, "Heuristic evaluation (not implemented - default score).", LLMCallStats()

        prompt = self.prompt_generator.evaluate_state_prompt(problem_description, candidate_node, self.config)
        response_text, stats = self.llm_client.call(
            prompt,
            models=self.config.evaluation_model_names,
            config=self.evaluation_llm_config
        )
        # Update overall stats in the result object from the caller of this method

        if response_text.startswith("Error:"):
            logging.error(f"LLM call for state evaluation failed: {response_text}")
            return None, f"LLM evaluation error: {response_text}", stats

        score, justification = self.response_parser.parse_evaluation_score(response_text)
        return score, justification, stats

    def _call_llm_for_final_answer(self, problem_description: str, best_path_thoughts: List[ToTThought]) -> Tuple[Optional[str], LLMCallStats]:
        best_path_texts = [t.text for t in best_path_thoughts]
        prompt = self.prompt_generator.generate_final_answer_prompt(problem_description, best_path_texts)
        # Using thought_generation config for final answer, could be a separate config
        response_text, stats = self.llm_client.call(
            prompt,
            models=self.config.thought_generation_model_names, # Or a dedicated final answer model
            config=self.thought_generation_llm_config
        )
        if response_text.startswith("Error:"):
            logging.error(f"LLM call for final answer generation failed: {response_text}")
            return f"Error generating final answer: {response_text}", stats

        final_answer = self.response_parser.parse_final_answer(response_text)
        if not final_answer:
            logging.warning("Could not parse a structured final answer. Using the full response.")
            return response_text.strip(), stats # Return raw response if parsing fails
        return final_answer, stats

    def run(self, problem_description: str) -> Tuple[ToTResult, str]:
        result = ToTResult()
        process_start_time = time.monotonic()

        root_id = self._generate_node_id()
        root_node = ToTNode(id=root_id, thoughts_path=[], depth=0)
        result.all_generated_nodes[root_id] = root_node

        frontier: List[ToTNode] = [root_node]
        visited_states: Set[str] = set() # To avoid re-evaluating identical states if paths converge

        current_depth = 0
        final_answer_found_in_thought = False

        while frontier and current_depth < self.config.max_depth:
            if (time.monotonic() - process_start_time) > self.config.max_time_seconds:
                logging.info("ToT process: Max time limit reached.")
                result.error_message = "Max time limit reached."
                break
            if result.total_thoughts_generated >= self.config.max_total_thoughts_generated:
                logging.info("ToT process: Max total thoughts generated limit reached.")
                result.error_message = "Max total thoughts generated limit reached."
                break

            logging.info(f"--- ToT Depth {current_depth + 1} ---")
            next_frontier_candidates: List[ToTNode] = []

            # Sort frontier by score if using beam search and scores are available (for subsequent depths)
            if self.config.search_strategy == ToTSearchStrategy.BEAM and current_depth > 0:
                frontier.sort(key=lambda n: n.score or -float('inf'), reverse=True)
                frontier = frontier[:self.config.b_beam_width] # Prune to beam width before expansion

            for current_node in frontier:
                if (time.monotonic() - process_start_time) > self.config.max_time_seconds: break
                if result.total_thoughts_generated >= self.config.max_total_thoughts_generated: break

                state_key = current_node.current_state_text
                if state_key in visited_states:
                    logging.debug(f"Skipping already visited state: {state_key[:100]}...")
                    continue
                visited_states.add(state_key)

                logging.debug(f"Expanding node {current_node.id} at depth {current_node.depth} with state: {current_node.current_state_text[:100]}...")

                generated_thoughts = self._call_llm_for_thoughts(problem_description, current_node)
                result.total_thoughts_generated += len(generated_thoughts)
                if generated_thoughts and generated_thoughts[0].generation_stats: # Add stats from the call
                    stats = generated_thoughts[0].generation_stats
                    result.total_completion_tokens += stats.completion_tokens
                    result.total_prompt_tokens += stats.prompt_tokens
                    result.total_llm_interaction_time_seconds += stats.call_duration_seconds

                if not generated_thoughts:
                    logging.warning(f"No thoughts generated for node {current_node.id}.")
                    continue

                for thought_obj in generated_thoughts:
                    if len(current_node.thoughts_path) >= self.config.max_thoughts_in_state:
                        logging.warning(f"Node {current_node.id} reached max thoughts in state. Skipping adding new thought.")
                        continue

                    child_id = self._generate_node_id()
                    new_thoughts_path = current_node.thoughts_path + [thought_obj]

                    # Check for early final answer in thought text (basic check)
                    # A more robust check might involve a dedicated LLM call or regex
                    if "final answer:" in thought_obj.text.lower() or "the solution is:" in thought_obj.text.lower():
                        logging.info(f"Potential final answer found in thought for node {child_id}: {thought_obj.text}")
                        # Tentatively set this as the best path, could be overridden by evaluation
                        # This is a simple heuristic, a proper solution might need a separate "is_final" evaluation
                        result.final_answer = thought_obj.text # Or parse more carefully
                        result.best_solution_path = new_thoughts_path
                        result.succeeded = True
                        final_answer_found_in_thought = True
                        # Optionally, we can stop here if config allows finding answer in intermediate thought
                        # For now, we continue evaluation to see if other paths are better

                    child_node = ToTNode(
                        id=child_id,
                        thoughts_path=new_thoughts_path,
                        parent_id=current_node.id,
                        depth=current_node.depth + 1
                    )
                    current_node.children_ids.append(child_id)

                    score, justification, eval_stats = self._call_llm_for_evaluation(problem_description, child_node)
                    result.total_nodes_evaluated += 1
                    if eval_stats:
                        result.total_completion_tokens += eval_stats.completion_tokens
                        result.total_prompt_tokens += eval_stats.prompt_tokens
                        result.total_llm_interaction_time_seconds += eval_stats.call_duration_seconds
                        child_node.evaluation_stats = eval_stats

                    child_node.score = score
                    child_node.evaluation_details = justification
                    result.all_generated_nodes[child_id] = child_node
                    logging.debug(f"  Generated child {child_node.id} (Score: {score:.2f if score else 'N/A'}): {thought_obj.text[:100]}...")
                    next_frontier_candidates.append(child_node)

            if final_answer_found_in_thought and result.succeeded: # If an intermediate thought claimed to be final
                 logging.info("Final answer signal found in an intermediate thought. Ending search early.")
                 break


            if not next_frontier_candidates:
                logging.info("No new candidates generated in this depth. Stopping.")
                break

            # Selection strategy for next frontier
            if self.config.search_strategy == ToTSearchStrategy.BFS:
                frontier = next_frontier_candidates # All evaluated children become next frontier
            elif self.config.search_strategy == ToTSearchStrategy.DFS:
                # For DFS, sort by score and pick the best, then explore its children first.
                # This is a simplified DFS; true DFS would use a stack and expand one child fully.
                # Here, we're still processing level by level but prioritizing high-scoring ones.
                next_frontier_candidates.sort(key=lambda n: n.score or -float('inf'), reverse=True)
                frontier = next_frontier_candidates
            elif self.config.search_strategy == ToTSearchStrategy.BEAM:
                # Beam search selection happens at the START of the loop for subsequent depths.
                # For the very first expansion from root, all children are considered for the beam.
                # For subsequent depths, the frontier was already pruned.
                # Here, we just pass all candidates, and the next iteration's start will prune.
                frontier = next_frontier_candidates
            else: # Default to BFS
                frontier = next_frontier_candidates

            current_depth += 1
            if final_answer_found_in_thought: break # Exit outer loop if answer found

        # --- Solution Extraction ---
        if not result.succeeded: # If no intermediate thought provided a final answer
            best_leaf_node: Optional[ToTNode] = None
            highest_score = -float('inf')

            # Find the best scored terminal node (or any node if max depth reached before terminal)
            for node_id, node_obj in result.all_generated_nodes.items():
                if node_obj.score is not None and node_obj.score > highest_score:
                    # Prefer nodes that are leaves or at max depth
                    is_leaf = not node_obj.children_ids
                    is_max_depth = node_obj.depth == self.config.max_depth -1 # -1 because depth is 0-indexed
                    if is_leaf or is_max_depth or node_obj.depth == current_depth: # Also consider nodes from last fully processed depth
                        highest_score = node_obj.score
                        best_leaf_node = node_obj

            if best_leaf_node:
                result.best_solution_path = best_leaf_node.thoughts_path
                logging.info(f"Best solution path found (Node {best_leaf_node.id}, Score: {best_leaf_node.score:.2f if best_leaf_node.score else 'N/A'}). Generating final answer.")

                final_answer_text, final_answer_stats = self._call_llm_for_final_answer(problem_description, result.best_solution_path)
                result.total_completion_tokens += final_answer_stats.completion_tokens
                result.total_prompt_tokens += final_answer_stats.prompt_tokens
                result.total_llm_interaction_time_seconds += final_answer_stats.call_duration_seconds

                result.final_answer = final_answer_text
                if result.final_answer and not result.final_answer.startswith("Error:"):
                    result.succeeded = True
                else:
                    result.error_message = result.final_answer # Store error if final answer gen failed
            else:
                logging.warning("No suitable best path found to generate a final answer.")
                result.error_message = result.error_message or "No solution path could be determined."

        elif result.succeeded and result.best_solution_path and not result.final_answer:
            # This case handles when an intermediate thought signaled 'final answer' but we didn't extract it directly
            # We use the path that led to that signal to generate a proper final answer
            logging.info(f"Final answer was signaled by an intermediate thought. Using that path to generate a formal final answer.")
            final_answer_text, final_answer_stats = self._call_llm_for_final_answer(problem_description, result.best_solution_path)
            result.total_completion_tokens += final_answer_stats.completion_tokens
            result.total_prompt_tokens += final_answer_stats.prompt_tokens
            result.total_llm_interaction_time_seconds += final_answer_stats.call_duration_seconds
            result.final_answer = final_answer_text
            if not (result.final_answer and not result.final_answer.startswith("Error:")) :
                 result.succeeded = False # Mark as failed if final synthesis fails
                 result.error_message = result.final_answer or "Failed to synthesize final answer from signaled path."


        result.total_process_wall_clock_time_seconds = time.monotonic() - process_start_time
        summary_output = self._generate_tot_summary(result, problem_description)

        if not result.succeeded and not result.error_message:
            result.error_message = "ToT process completed without a successful solution or specific error."

        return result, summary_output

    def _generate_tot_summary(self, result: ToTResult, problem_description: str) -> str:
        output_buffer = io.StringIO()
        output_buffer.write("\n" + "="*20 + " ToT Process Summary " + "="*20 + "\n")
        output_buffer.write(f"Problem: {problem_description[:100]}...\n")
        output_buffer.write(f"Search Strategy: {self.config.search_strategy.value}\n")
        output_buffer.write(f"Scoring Method: {self.config.scoring_method.value}\n")
        output_buffer.write(f"K (thoughts per expansion): {self.config.k_thoughts}\n")
        if self.config.search_strategy == ToTSearchStrategy.BEAM:
            output_buffer.write(f"B (beam width): {self.config.b_beam_width}\n")
        output_buffer.write(f"Max Depth: {self.config.max_depth}\n")

        output_buffer.write(f"ToT Succeeded: {result.succeeded}\n")
        if result.error_message:
            output_buffer.write(f"Error Message: {result.error_message}\n")

        output_buffer.write(f"Total Thoughts Generated (approx): {result.total_thoughts_generated}\n")
        output_buffer.write(f"Total Nodes Evaluated: {result.total_nodes_evaluated}\n")

        output_buffer.write(f"Total Completion Tokens (ToT phase): {result.total_completion_tokens}\n")
        output_buffer.write(f"Total Prompt Tokens (ToT phase): {result.total_prompt_tokens}\n")
        output_buffer.write(f"Grand Total ToT Tokens: {result.grand_total_tokens}\n")
        output_buffer.write(f"Total ToT LLM Interaction Time: {result.total_llm_interaction_time_seconds:.2f}s\n")
        output_buffer.write(f"Total ToT Process Wall-Clock Time: {result.total_process_wall_clock_time_seconds:.2f}s\n")

        if result.succeeded and result.best_solution_path:
            output_buffer.write("\n--- Best Solution Path ---\n")
            for i, thought in enumerate(result.best_solution_path):
                output_buffer.write(f"  Step {i+1}: {thought.text}\n")

        # Optionally, print details of all nodes for debugging (can be very verbose)
        # output_buffer.write("\n--- All Generated Nodes (Details) ---\n")
        # for node_id, node in result.all_generated_nodes.items():
        #     parent_text = f"(Parent: {node.parent_id})" if node.parent_id else "(Root)"
        #     score_text = f"Score: {node.score:.2f}" if node.score is not None else "Score: N/A"
        #     output_buffer.write(f"Node {node.id} {parent_text} Depth: {node.depth} {score_text}\n")
        #     output_buffer.write(f"  State: {node.current_state_text[:150]}...\n")
        #     if node.evaluation_details:
        #         output_buffer.write(f"  Eval Justification: {node.evaluation_details[:100]}...\n")

        output_buffer.write("="*60 + "\n")
        return output_buffer.getvalue()
