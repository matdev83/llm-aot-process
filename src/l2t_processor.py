import time
import logging
import uuid
import io # For summary generation
from typing import Optional 

# Imports for new CoT structure
from src.cot_process import CoTProcess, CoTResult
# Imports for L2T specifics
from src.l2t_dataclasses import (
    L2TConfig, # Already a CoTConfig
    L2TGraph,
    L2TNode,
    L2TNodeCategory,
    L2TResult,
)
from src.llm_client import LLMClient
from src.aot_dataclasses import LLMCallStats # Still needed for _update_result_stats
from src.l2t_prompt_generator import L2TPromptGenerator
from src.l2t_response_parser import L2TResponseParser

logger = logging.getLogger(__name__)

class L2TProcessor(CoTProcess): # Inherits from CoTProcess
    def __init__(self, llm_client: LLMClient, config: Optional[L2TConfig] = None):
        actual_config = config if config else L2TConfig()
        super().__init__(actual_config) # Call super's init
        self.llm_client = llm_client
        self.l2t_config: L2TConfig = actual_config # Store specific L2TConfig
        self.prompt_generator = L2TPromptGenerator(self.l2t_config)
        if not logger.handlers: # Ensure logger is configured
            # Basic config if no handlers are present.
            # Consider moving logger configuration to a higher level application setup.
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


    def _update_result_stats(self, result: L2TResult, stats: LLMCallStats):
        if stats:
            result.total_llm_calls += 1
            result.total_completion_tokens += stats.completion_tokens
            result.total_prompt_tokens += stats.prompt_tokens
            result.total_llm_interaction_time_seconds += stats.call_duration_seconds

    def run(self, problem_text: str) -> CoTResult: # New signature
        l2t_native_result = L2TResult() # This is the existing L2TResult
        graph = L2TGraph()
        process_start_time = time.monotonic()
        current_process_step = 0

        # --- Start of existing run logic (mostly unchanged) ---
        logger.info(f"Starting L2T process for problem: '{problem_text[:100].strip()}...'")
        logger.info(f"Initial Prompt Models: {', '.join(self.l2t_config.initial_prompt_model_names)}, Temp: {self.l2t_config.initial_prompt_temperature}")
        logger.info(f"Thought Gen Models: {', '.join(self.l2t_config.thought_generation_model_names)}, Temp: {self.l2t_config.thought_generation_temperature}")
        logger.info(f"Classification Models: {', '.join(self.l2t_config.classification_model_names)}, Temp: {self.l2t_config.classification_temperature}")
        logger.info(f"Max Steps: {self.l2t_config.max_steps}, Max Total Nodes: {self.l2t_config.max_total_nodes}, Max Time: {self.l2t_config.max_time_seconds}s")


        initial_prompt = self.prompt_generator.construct_l2t_initial_prompt(
            problem_text, self.l2t_config.x_fmt_default, self.l2t_config.x_eva_default
        )
        initial_response_content, initial_stats = self.llm_client.call(
            initial_prompt,
            models=self.l2t_config.initial_prompt_model_names,
            temperature=self.l2t_config.initial_prompt_temperature,
        )
        self._update_result_stats(l2t_native_result, initial_stats)
        logger.info(f"Initial thought LLM call ({initial_stats.model_name if initial_stats else 'N/A'}): Duration: {initial_stats.call_duration_seconds if initial_stats else 0:.2f}s, Tokens (C:{initial_stats.completion_tokens if initial_stats else 0}, P:{initial_stats.prompt_tokens if initial_stats else 0})")


        parsed_initial_thought = (
            L2TResponseParser.parse_l2t_initial_response(initial_response_content)
        )

        if initial_response_content.startswith("Error:") or parsed_initial_thought is None:
            error_msg = f"Failed during initial thought generation. LLM Response: {initial_response_content}"
            logger.error(error_msg)
            l2t_native_result.error_message = error_msg
            l2t_native_result.succeeded = False
            l2t_native_result.total_process_wall_clock_time_seconds = (
                time.monotonic() - process_start_time
            )
            l2t_native_result.reasoning_graph = graph 
            # --- Early exit, map to CoTResult ---
            return CoTResult(
                final_answer=l2t_native_result.final_answer, # Will be None here
                succeeded=l2t_native_result.succeeded,
                error_message=l2t_native_result.error_message,
                total_completion_tokens=l2t_native_result.total_completion_tokens,
                total_prompt_tokens=l2t_native_result.total_prompt_tokens,
                total_llm_interaction_time_seconds=l2t_native_result.total_llm_interaction_time_seconds,
                total_process_wall_clock_time_seconds=l2t_native_result.total_process_wall_clock_time_seconds,
                process_specific_data=l2t_native_result
            )

        root_node_id = str(uuid.uuid4())
        root_node = L2TNode(
            id=root_node_id,
            content=parsed_initial_thought,
            parent_id=None,
            generation_step=0,
        )
        graph.add_node(root_node, is_root=True)
        logger.info(f"Initial thought node {root_node_id[:8]} created: '{parsed_initial_thought[:100].strip().replace(chr(10), ' ')}...'")


        while (
            current_process_step < self.l2t_config.max_steps
            and len(graph.v_pres) > 0
            and (time.monotonic() - process_start_time) < self.l2t_config.max_time_seconds
            and len(graph.nodes) < self.l2t_config.max_total_nodes
            and l2t_native_result.final_answer is None
        ):
            current_process_step += 1
            current_wall_clock = time.monotonic() - process_start_time
            logger.info(
                f"--- L2T Process Step {current_process_step}/{self.l2t_config.max_steps} --- "
                f"V_pres size: {len(graph.v_pres)}, Total nodes: {len(graph.nodes)}/{self.l2t_config.max_total_nodes}, "
                f"Elapsed time: {current_wall_clock:.2f}s/{self.l2t_config.max_time_seconds}s ---"
            )


            nodes_to_process_this_round = list(graph.v_pres) # Iterate over a copy
            for node_id_to_classify in nodes_to_process_this_round:
                if l2t_native_result.final_answer is not None:
                    logger.info("Final answer found, breaking processing round.")
                    break 
                if node_id_to_classify not in graph.v_pres: # Node might have been moved to v_hist by processing a sibling
                    logger.debug(f"Node {node_id_to_classify[:8]} no longer in v_pres, skipping.")
                    continue


                node_to_classify = graph.get_node(node_id_to_classify)
                if not node_to_classify: # Should not happen if logic is correct
                    logger.error(f"Node {node_id_to_classify[:8]} not found in graph during processing round.")
                    graph.move_to_hist(node_id_to_classify) 
                    continue
                # Node might have been classified if it was re-added to v_pres, though current logic doesn't do that.
                if node_to_classify.category is not None: 
                    logger.debug(f"Node {node_id_to_classify[:8]} already classified as {node_to_classify.category}. Moving to hist.")
                    graph.move_to_hist(node_id_to_classify) 
                    continue

                parent_node = graph.get_parent(node_to_classify.id)
                ancestor_path_str = ""
                # Construct a more detailed ancestor path for context
                path_list_for_prompt = []
                temp_node_for_path = node_to_classify
                path_depth = 0
                while temp_node_for_path and path_depth < 4: # Max 3 ancestors + current
                    prefix = "Current thought" if path_depth == 0 else f"Parent thought (-{path_depth})"
                    path_list_for_prompt.append(f"{prefix}: '{temp_node_for_path.content}'")
                    if temp_node_for_path.parent_id:
                        temp_node_for_path = graph.get_node(temp_node_for_path.parent_id)
                    else:
                        temp_node_for_path = None # Reached root or orphaned
                    path_depth += 1
                ancestor_path_str = "\n".join(reversed(path_list_for_prompt))


                graph_context_for_classification = (
                    f"Reasoning path leading to current thought (most recent last):\n{ancestor_path_str}\n\n"
                    f"You are classifying the '{node_to_classify.content[:100].strip().replace(chr(10), ' ')}...' part of this path."
                )
                classification_prompt = self.prompt_generator.construct_l2t_node_classification_prompt(
                    graph_context_for_classification,
                    node_to_classify.content, # Pass the full content for classification
                    self.l2t_config.x_eva_default,
                )
                (
                    classification_response_content,
                    classification_stats,
                ) = self.llm_client.call(
                    classification_prompt,
                    models=self.l2t_config.classification_model_names,
                    temperature=self.l2t_config.classification_temperature,
                )
                self._update_result_stats(l2t_native_result, classification_stats)
                logger.debug(f"Classification LLM call ({classification_stats.model_name if classification_stats else 'N/A'}): Duration: {classification_stats.call_duration_seconds if classification_stats else 0:.2f}s, Tokens (C:{classification_stats.completion_tokens if classification_stats else 0}, P:{classification_stats.prompt_tokens if classification_stats else 0}) for node {node_to_classify.id[:8]}")

                node_category = L2TResponseParser.parse_l2t_node_classification_response(
                    classification_response_content
                )

                if classification_response_content.startswith("Error:") or node_category is None:
                    logger.warning(
                        f"Node classification failed for node {node_to_classify.id[:8]}. "
                        f"Defaulting to TERMINATE_BRANCH. Response: {classification_response_content[:100]}"
                    )
                    node_category = L2TNodeCategory.TERMINATE_BRANCH
                
                graph.classify_node(node_to_classify.id, node_category)
                logger.info(f"Node {node_to_classify.id[:8]} ('{node_to_classify.content[:50].strip().replace(chr(10), ' ')}...') classified as {node_category.name}")


                if node_category == L2TNodeCategory.CONTINUE:
                    if len(graph.nodes) >= self.l2t_config.max_total_nodes:
                        logger.warning(f"Max total nodes ({self.l2t_config.max_total_nodes}) reached. Cannot generate new thought from node {node_to_classify.id[:8]}. Treating as TERMINATE_BRANCH.")
                        graph.classify_node(node_to_classify.id, L2TNodeCategory.TERMINATE_BRANCH) # Re-classify
                    else:
                        # Context for thought generation should ideally be the path leading to this node
                        thought_gen_context_prompt = (
                             f"The current reasoning path (most recent thought last):\n{ancestor_path_str}\n\n"
                             f"Based on this path, generate the next single thought to continue the reasoning process from: '{node_to_classify.content[:100].strip().replace(chr(10), ' ')}...'."
                        )
                        thought_gen_prompt = self.prompt_generator.construct_l2t_thought_generation_prompt(
                            thought_gen_context_prompt, # More detailed context
                            node_to_classify.content, # Current node's content for direct reference
                            self.l2t_config.x_fmt_default,
                            self.l2t_config.x_eva_default,
                        )
                        (
                            new_thought_response_content,
                            new_thought_stats,
                        ) = self.llm_client.call(
                            thought_gen_prompt,
                            models=self.l2t_config.thought_generation_model_names,
                            temperature=self.l2t_config.thought_generation_temperature,
                        )
                        self._update_result_stats(l2t_native_result, new_thought_stats)
                        logger.debug(f"Thought Gen LLM call ({new_thought_stats.model_name if new_thought_stats else 'N/A'}): Duration: {new_thought_stats.call_duration_seconds if new_thought_stats else 0:.2f}s, Tokens (C:{new_thought_stats.completion_tokens if new_thought_stats else 0}, P:{new_thought_stats.prompt_tokens if new_thought_stats else 0}) for parent {node_to_classify.id[:8]}")

                        new_thought_content = L2TResponseParser.parse_l2t_thought_generation_response(
                            new_thought_response_content
                        )

                        if new_thought_response_content.startswith("Error:") or new_thought_content is None or not new_thought_content.strip():
                            logger.warning(
                                f"Thought generation failed for parent node {node_to_classify.id[:8]} or generated empty thought. "
                                f"Response: {new_thought_response_content[:100]}"
                            )
                        else:
                            new_node_id = str(uuid.uuid4())
                            new_node = L2TNode(
                                id=new_node_id,
                                content=new_thought_content,
                                parent_id=node_to_classify.id,
                                generation_step=node_to_classify.generation_step + 1,
                            )
                            graph.add_node(new_node) 
                            logger.info(
                                f"Generated new thought node {new_node_id[:8]} (gen_step {new_node.generation_step}, parent {node_to_classify.id[:8]}): '{new_thought_content[:50].strip().replace(chr(10), ' ')}...'"
                            )
                
                elif node_category == L2TNodeCategory.FINAL_ANSWER:
                    l2t_native_result.final_answer = node_to_classify.content
                    l2t_native_result.succeeded = True # Explicitly set success
                    logger.info(
                        f"Final answer found at node {node_to_classify.id[:8]}: {l2t_native_result.final_answer[:100].strip().replace(chr(10), ' ')}..."
                    )
                    # No need to break here, loop will terminate due to final_answer being set
                
                elif node_category == L2TNodeCategory.TERMINATE_BRANCH:
                    logger.info(
                        f"Terminating branch at node {node_to_classify.id[:8]}: '{node_to_classify.content[:100].strip().replace(chr(10), ' ')}...'"
                    )
                
                elif node_category == L2TNodeCategory.BACKTRACK:
                    logger.info(
                        f"Backtrack requested at node {node_to_classify.id[:8]}. Basic: Treating as TERMINATE_BRANCH for now."
                    )
                    # Future: Implement more sophisticated backtrack logic if needed
                
                graph.move_to_hist(node_id_to_classify) # Move processed node to history

            if not graph.v_pres and l2t_native_result.final_answer is None:
                logger.info("No new thoughts generated in this step (v_pres is empty) and no final answer. Terminating reasoning loop.")
                break
            
            if l2t_native_result.final_answer is not None:
                logger.info("Final answer has been found. Terminating reasoning loop.")
                break
        
        # After loop finishes
        l2t_native_result.reasoning_graph = graph
        if l2t_native_result.final_answer is None: # Check if it was already set by FINAL_ANSWER category
            l2t_native_result.succeeded = False 
            if len(graph.nodes) >= self.l2t_config.max_total_nodes:
                l2t_native_result.error_message = "L2T process completed: Max total nodes reached."
            elif current_process_step >= self.l2t_config.max_steps:
                l2t_native_result.error_message = "L2T process completed: Max steps reached."
            elif (time.monotonic() - process_start_time) >= self.l2t_config.max_time_seconds:
                l2t_native_result.error_message = "L2T process completed: Max time reached."
            elif not graph.v_pres and not any(node.category == L2TNodeCategory.CONTINUE for node in graph.nodes.values() if node.category is not None) : 
                l2t_native_result.error_message = "L2T process completed: No more thoughts to process and no viable continuation paths led to a final answer."
            else: # Default error if no other condition met
                l2t_native_result.error_message = "L2T process completed without a final answer. All explorable paths exhausted or limits met."
            logger.info(f"L2T process finished without a final answer. Reason: {l2t_native_result.error_message}")
        else: # Final answer was found
             if not l2t_native_result.succeeded : # Should be true if final_answer is not None due to FINAL_ANSWER category
                l2t_native_result.succeeded = True # Ensure consistency
                logger.info("L2T process concluded with a final answer.")


        l2t_native_result.total_process_wall_clock_time_seconds = (
            time.monotonic() - process_start_time
        )
        logger.info(
            f"L2T run finished. Success: {l2t_native_result.succeeded}. LLM Calls: {l2t_native_result.total_llm_calls}. "
            f"Total time: {l2t_native_result.total_process_wall_clock_time_seconds:.2f}s. "
            f"Final Answer: {(l2t_native_result.final_answer[:200] + '...' if l2t_native_result.final_answer and len(l2t_native_result.final_answer) > 200 else l2t_native_result.final_answer) if l2t_native_result.final_answer else 'None'}"
        )
        # --- End of existing run logic ---

        # Create and return CoTResult
        cot_result = CoTResult(
            final_answer=l2t_native_result.final_answer,
            succeeded=l2t_native_result.succeeded,
            error_message=l2t_native_result.error_message,
            total_completion_tokens=l2t_native_result.total_completion_tokens,
            total_prompt_tokens=l2t_native_result.total_prompt_tokens,
            total_llm_interaction_time_seconds=l2t_native_result.total_llm_interaction_time_seconds,
            total_process_wall_clock_time_seconds=l2t_native_result.total_process_wall_clock_time_seconds,
            reasoning_trace=None, # L2T uses a graph, not a linear trace
            process_specific_data=l2t_native_result # Store the original L2TResult (contains the graph)
        )
        return cot_result

    def get_summary(self, result: CoTResult) -> str: # New method
        if not result.process_specific_data or not isinstance(result.process_specific_data, L2TResult):
            # Fallback if process_specific_data is missing or not L2TResult
            summary_lines = [
                "--- L2T Process Summary (Incomplete Data) ---",
                f"CoT Succeeded: {result.succeeded}",
                f"CoT Final Answer: {result.final_answer or 'N/A'}",
                f"CoT Error Message: {result.error_message or 'N/A'}",
                f"Total completion tokens (from CoTResult): {result.total_completion_tokens}",
                f"Total prompt tokens (from CoTResult): {result.total_prompt_tokens}",
                f"Grand total tokens (from CoTResult): {result.total_completion_tokens + result.total_prompt_tokens}",
                f"Total LLM interaction time (from CoTResult): {result.total_llm_interaction_time_seconds:.2f}s",
                f"Total process wall-clock time (from CoTResult): {result.total_process_wall_clock_time_seconds:.2f}s",
                "Note: L2T-specific details are unavailable because process_specific_data was not a valid L2TResult."
            ]
            return "\n".join(summary_lines) + "\n"


        l2t_native_result: L2TResult = result.process_specific_data # Type cast for clarity
        
        output_buffer = io.StringIO()
        output_buffer.write("\n" + "="*20 + " L2T PROCESS SUMMARY " + "="*20 + "\n")
        # Use CoTResult fields for primary status, as they are the "official" outcome of the process run
        output_buffer.write(f"Overall Succeeded (from CoTResult): {result.succeeded}\n")
        if result.error_message: # Display error from CoTResult if present
            output_buffer.write(f"Overall Error Message (from CoTResult): {result.error_message}\n")
        elif l2t_native_result.error_message and not result.succeeded: # Fallback to L2T's error if CoT doesn't have one and failed
             output_buffer.write(f"L2T Error Message (from L2TResult): {l2t_native_result.error_message}\n")


        output_buffer.write(f"Total LLM Calls (L2T): {l2t_native_result.total_llm_calls}\n")
        output_buffer.write(f"Total Completion Tokens (L2T): {l2t_native_result.total_completion_tokens}\n") # from CoTResult.total_completion_tokens
        output_buffer.write(f"Total Prompt Tokens (L2T): {l2t_native_result.total_prompt_tokens}\n") # from CoTResult.total_prompt_tokens
        output_buffer.write(f"Grand Total Tokens (L2T Calls): {l2t_native_result.total_completion_tokens + l2t_native_result.total_prompt_tokens}\n")
        output_buffer.write(f"Total L2T LLM Interaction Time: {l2t_native_result.total_llm_interaction_time_seconds:.2f}s\n") # from CoTResult
        output_buffer.write(f"Total L2T Process Wall-Clock Time: {l2t_native_result.total_process_wall_clock_time_seconds:.2f}s\n") # from CoTResult


        if l2t_native_result.reasoning_graph and l2t_native_result.reasoning_graph.nodes:
            output_buffer.write(f"Number of nodes in graph: {len(l2t_native_result.reasoning_graph.nodes)}\n")
            if l2t_native_result.reasoning_graph.root_node_id:
                 root_node_content = l2t_native_result.reasoning_graph.get_node(l2t_native_result.reasoning_graph.root_node_id)
                 output_buffer.write(f"Root node ID: {l2t_native_result.reasoning_graph.root_node_id[:8]}... "
                                     f"Content: '{root_node_content.content[:50].strip().replace(chr(10), ' ')}...' \n" if root_node_content else "Root node content N/A\n")
            
            final_answer_node_id = None
            if l2t_native_result.final_answer:
                for node_id, node in l2t_native_result.reasoning_graph.nodes.items():
                    if node.content == l2t_native_result.final_answer and node.category == L2TNodeCategory.FINAL_ANSWER:
                        final_answer_node_id = node_id
                        break
            if final_answer_node_id:
                 output_buffer.write(f"Final answer derived from node: {final_answer_node_id[:8]}...\n")
            elif l2t_native_result.succeeded and l2t_native_result.final_answer : # Succeeded but not via a FINAL_ANSWER category node (e.g. if logic changes)
                 output_buffer.write(f"Final answer obtained, but not directly from a node marked FINAL_ANSWER.\n")


        if result.final_answer: # Display final answer from CoTResult
            output_buffer.write(f"\nFinal Answer (from CoTResult):\n{result.final_answer}\n")
        elif not result.succeeded:
            output_buffer.write("\nFinal answer was not successfully obtained.\n")

        output_buffer.write("="*59 + "\n")
        return output_buffer.getvalue()
