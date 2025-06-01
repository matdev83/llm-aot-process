import time
import logging
import io
from typing import List, Tuple, Optional, Any

from src.reasoning_process import ReasoningProcess
from src.llm_client import LLMClient
from src.llm_config import LLMConfig
from .dataclasses import ToTConfig, ToTSolution, ToTResult, LLMCallStats
from .processor import ToTProcessor
# Reusing AOT's enums for trigger mode and assessment decision for now,
# can be generalized or made ToT-specific if needed.
from src.aot.enums import AotTriggerMode as ToTTriggerMode, AssessmentDecision # Alias for clarity

# Additional imports for InteractiveToTOrchestrator
from src.complexity_assessor import ComplexityAssessor
from src.heuristic_detector import HeuristicDetector

class ToTProcess(ReasoningProcess):
    """
    Implements the Tree of Thoughts (ToT) reasoning process.
    This class is responsible for executing the ToT chain, including
    running the ToTProcessor and handling fallbacks.
    """
    def __init__(self,
                 llm_client: LLMClient,
                 tot_config: ToTConfig, # Specific ToT configuration
                 # LLMConfigs for various parts of ToT or fallback
                 thought_generation_llm_config: LLMConfig,
                 evaluation_llm_config: LLMConfig,
                 direct_oneshot_llm_config: LLMConfig, # For fallback
                 # Model names for fallback, if different from tot_config
                 direct_oneshot_model_names: Optional[List[str]] = None
                ):

        self.llm_client = llm_client
        self.tot_config = tot_config # Includes model names for thought gen and eval
        self.direct_oneshot_llm_config = direct_oneshot_llm_config

        # Fallback models can be specified, or use ToT's thought generation models by default
        self.direct_oneshot_model_names = direct_oneshot_model_names or self.tot_config.thought_generation_model_names

        self.tot_processor = ToTProcessor(
            llm_client=self.llm_client,
            config=self.tot_config,
            thought_generation_llm_config=thought_generation_llm_config,
            evaluation_llm_config=evaluation_llm_config
        )

        self._solution: Optional[ToTSolution] = None
        self._process_summary: Optional[str] = None # Combined summary

    def _run_direct_oneshot(self, problem_text: str, is_fallback: bool = False) -> Tuple[str, LLMCallStats]:
        mode = "FALLBACK ONESHOT (ToTProcess)" if is_fallback else "ONESHOT (ToTProcess)"
        logging.info(f"--- Proceeding with {mode} Answer ---")
        logging.info(f"Using models: {', '.join(self.direct_oneshot_model_names)}, LLMConfig: {self.direct_oneshot_llm_config}")

        response_content, stats = self.llm_client.call(
            prompt=problem_text,
            models=self.direct_oneshot_model_names,
            config=self.direct_oneshot_llm_config
        )
        logging.debug(f"Direct {mode} response from {stats.model_name}: \
{response_content}")
        logging.info(f"LLM call ({stats.model_name}) for {mode}: Duration: {stats.call_duration_seconds:.2f}s, Tokens (C:{stats.completion_tokens}, P:{stats.prompt_tokens})")
        return response_content, stats

    def execute(self, problem_description: str, model_name: str, *args, **kwargs) -> None:
        # model_name param is part of ReasoningProcess signature, might not be directly used if ToTConfig has models
        overall_start_time = time.monotonic()
        self._solution = ToTSolution() # Initialize solution object
        logging.info(f"ToTProcess executing for problem: {problem_description[:100]}...")

        if not self.tot_processor:
            logging.critical("ToTProcessor not initialized within ToTProcess.")
            self._solution.final_answer = "Error: ToTProcessor not initialized."
            if self._solution.total_wall_clock_time_seconds == 0.0: # Ensure time is set
                 self._solution.total_wall_clock_time_seconds = time.monotonic() - overall_start_time
            self._process_summary = self._generate_process_summary(self._solution, problem_description)
            return

        tot_result_data, tot_processor_summary_str = self.tot_processor.run(problem_description)
        self._solution.tot_result = tot_result_data
        self._solution.tot_summary_output = tot_processor_summary_str

        if tot_result_data.succeeded and tot_result_data.final_answer:
            self._solution.final_answer = tot_result_data.final_answer
            # Potentially add reasoning trace if ToTResult includes it explicitly (like best_solution_path)
        else:
            logging.warning(f"ToT process failed or did not produce a final answer (Reason: {tot_result_data.error_message or 'N/A'}). Falling back to one-shot.")
            self._solution.tot_failed_and_fell_back = True
            fallback_answer, fallback_stats = self._run_direct_oneshot(problem_description, is_fallback=True)
            self._solution.final_answer = fallback_answer
            self._solution.fallback_call_stats = fallback_stats
            # If ToT had a partial path, it could be stored in reasoning_trace even on failure

        if self._solution.total_wall_clock_time_seconds == 0.0: # Ensure time is set
            self._solution.total_wall_clock_time_seconds = time.monotonic() - overall_start_time
        self._process_summary = self._generate_process_summary(self._solution, problem_description)

    def get_result(self) -> Tuple[Optional[ToTSolution], Optional[str]]:
        return self._solution, self._process_summary

    def _generate_process_summary(self, solution: ToTSolution, problem_description: str) -> str:
        output_buffer = io.StringIO()
        output_buffer.write("\n" + "="*20 + " ToTProcess Execution Summary " + "="*20 + "\n")
        output_buffer.write(f"Problem: {problem_description[:100]}...\n")

        if solution.tot_result:
            output_buffer.write(f"ToT Process Attempted: Yes\n")
            if solution.tot_summary_output: # This is the summary from ToTProcessor
                 output_buffer.write(f"--- ToT Processor Internal Summary ---\n{solution.tot_summary_output}\n----------------------------------\n")

            if solution.tot_result.succeeded:
                 output_buffer.write(f"ToT Succeeded (Reported by ToT Processor): Yes\n")
            elif solution.tot_failed_and_fell_back:
                err_msg = solution.tot_result.error_message or "No specific error from ToT Processor."
                output_buffer.write(f"ToT FAILED and Fell Back to One-Shot: Yes (ToT Failure Reason: {err_msg})\n")
                if solution.fallback_call_stats:
                    sfb = solution.fallback_call_stats
                    output_buffer.write(f"  Fallback One-Shot Call ({sfb.model_name}): C={sfb.completion_tokens}, P={sfb.prompt_tokens}, Time={sfb.call_duration_seconds:.2f}s\n")
        else:
            output_buffer.write(f"ToT Process Was Not Fully Attempted (e.g., ToTProcessor initialization error).\n")
            if solution.final_answer and solution.final_answer.startswith("Error:"): # If error was set directly
                 output_buffer.write(f"Status/Error: {solution.final_answer}\n")

        # Overall stats from ToTSolution perspective
        output_buffer.write(f"Total Completion Tokens (ToTProcess Layer): {solution.total_completion_tokens}\n")
        output_buffer.write(f"Total Prompt Tokens (ToTProcess Layer): {solution.total_prompt_tokens}\n")
        output_buffer.write(f"Grand Total Tokens (ToTProcess Layer): {solution.grand_total_tokens}\n")
        output_buffer.write(f"Total LLM Interaction Time (ToTProcess Layer): {solution.total_llm_interaction_time_seconds:.2f}s\n")
        output_buffer.write(f"Total Wall-Clock Time (ToTProcess Execution): {solution.total_wall_clock_time_seconds:.2f}s\n")

        # Reasoning trace (best path from ToT)
        if solution.tot_result and solution.tot_result.best_solution_path:
            output_buffer.write("\n--- Reasoning Trace (Best Path from ToTProcess) ---\n")
            for i, thought in enumerate(solution.tot_result.best_solution_path):
                output_buffer.write(f"  Step {i+1}: {thought.text}\n")
        elif solution.tot_failed_and_fell_back :
             output_buffer.write("\n--- Reasoning Trace (N/A due to ToT failure and fallback) ---\n")


        if solution.final_answer and not solution.final_answer.startswith("Error:"):
            output_buffer.write(f"\nFinal Answer (from ToTProcess Layer):\n{solution.final_answer}\n")
        elif not solution.final_answer:
            output_buffer.write("\nFinal answer not successfully extracted by ToTProcess.\n")
        elif solution.final_answer.startswith("Error:"):
             output_buffer.write(f"\nFinal Answer: Process resulted in an error.\n")


        output_buffer.write("="*60 + "\n")
        return output_buffer.getvalue()


class InteractiveToTOrchestrator:
    """
    Orchestrates the Tree of Thoughts (ToT) process based on trigger modes,
    potentially using a complexity assessment step.
    """
    def __init__(self,
                 llm_client: LLMClient, # Shared LLM client
                 trigger_mode: ToTTriggerMode, # e.g., ALWAYS_TOT, ASSESS_FIRST_TOT, NEVER_TOT
                 tot_config: ToTConfig, # Base ToT configuration
                 # LLMConfigs for different parts of ToT and orchestration
                 thought_generation_llm_config: LLMConfig, # For ToTProcessor thought generation
                 evaluation_llm_config: LLMConfig,         # For ToTProcessor evaluation
                 direct_oneshot_llm_config: LLMConfig,   # For fallback or direct one-shot
                 assessment_llm_config: LLMConfig,       # For complexity assessor
                 # Model names for specific orchestrator actions
                 direct_oneshot_model_names: List[str], # For direct one-shot by orchestrator
                 assessment_model_names: List[str],     # For complexity assessor
                 use_heuristic_shortcut: bool = True,
                 heuristic_detector: Optional[HeuristicDetector] = None
                ):

        self.llm_client = llm_client
        self.trigger_mode = trigger_mode
        self.tot_config = tot_config # Main ToT algo config
        self.use_heuristic_shortcut = use_heuristic_shortcut
        self.heuristic_detector = heuristic_detector

        # Specific LLM configurations to be used by ToTProcess or Assessor
        self.thought_generation_llm_config = thought_generation_llm_config
        self.evaluation_llm_config = evaluation_llm_config
        self.direct_oneshot_llm_config = direct_oneshot_llm_config # Used by ToTProcess fallback & orchestrator direct
        self.assessment_llm_config = assessment_llm_config

        # Model name lists for orchestrator's direct calls or components it initializes
        self.direct_oneshot_model_names = direct_oneshot_model_names
        self.assessment_model_names = assessment_model_names

        self.complexity_assessor: Optional[ComplexityAssessor] = None
        if self.trigger_mode == ToTTriggerMode.ASSESS_FIRST:
            self.complexity_assessor = ComplexityAssessor(
                llm_client=self.llm_client,
                small_model_names=self.assessment_model_names, # Models for assessment
                llm_config=self.assessment_llm_config,         # Config for assessment calls
                use_heuristic_shortcut=self.use_heuristic_shortcut,
                heuristic_detector=self.heuristic_detector
            )

        self.tot_process_instance: Optional[ToTProcess] = None
        # Initialize ToTProcess if a mode that might use it is selected
        # Corrected enum names: ALWAYS_AOT to ALWAYS_TOT
        if self.trigger_mode == ToTTriggerMode.ALWAYS_TOT or self.trigger_mode == ToTTriggerMode.ASSESS_FIRST:
             self.tot_process_instance = ToTProcess(
                llm_client=self.llm_client,
                tot_config=self.tot_config,
                thought_generation_llm_config=self.thought_generation_llm_config,
                evaluation_llm_config=self.evaluation_llm_config,
                direct_oneshot_llm_config=self.direct_oneshot_llm_config, # For ToTProcess's own fallback
                direct_oneshot_model_names=self.direct_oneshot_model_names # Models for ToTProcess's fallback
            )


    def _run_direct_oneshot(self, problem_text: str, is_fallback_orchestrator: bool = False) -> Tuple[str, LLMCallStats]:
        # This is the orchestrator's direct one-shot, NOT ToTProcess's internal one.
        mode = "FALLBACK ONESHOT (Orchestrator)" if is_fallback_orchestrator else "DIRECT ONESHOT (Orchestrator)"
        logging.info(f"--- {mode} ---")
        logging.info(f"Using models: {', '.join(self.direct_oneshot_model_names)}, LLMConfig: {self.direct_oneshot_llm_config}")

        response_content, stats = self.llm_client.call(
            prompt=problem_text,
            models=self.direct_oneshot_model_names,
            config=self.direct_oneshot_llm_config
        )
        logging.debug(f"{mode} response from {stats.model_name}: \
{response_content}")
        logging.info(f"LLM call ({stats.model_name}) for {mode}: Duration: {stats.call_duration_seconds:.2f}s, Tokens (C:{stats.completion_tokens}, P:{stats.prompt_tokens})")
        return response_content, stats

    def solve(self, problem_description: str, model_name_param_for_process: str = "default_tot_model") -> Tuple[Optional[ToTSolution], str]:
        # model_name_param_for_process is for compatibility with a generic execute signature if needed
        overall_start_time = time.monotonic()
        orchestrator_solution = ToTSolution() # Initialize with ToT's solution type
        process_specific_summary: Optional[str] = None # Summary from ToTProcess if executed

        # Corrected enum names: NEVER_AOT to NEVER_TOT, ALWAYS_AOT to ALWAYS_TOT
        if self.trigger_mode == ToTTriggerMode.NEVER_TOT:
            logging.info("Orchestrator trigger mode: NEVER_TOT. Performing direct one-shot call.")
            final_answer, oneshot_stats = self._run_direct_oneshot(problem_description)
            orchestrator_solution.final_answer = final_answer
            orchestrator_solution.fallback_call_stats = oneshot_stats

        elif self.trigger_mode == ToTTriggerMode.ALWAYS_TOT:
            logging.info("Orchestrator trigger mode: ALWAYS_TOT. Delegating to ToTProcess.")
            if not self.tot_process_instance:
                logging.critical("ToTProcess not initialized for ALWAYS_TOT mode.")
                orchestrator_solution.final_answer = "Error: ToTProcess not initialized for ALWAYS_TOT mode."
            else:
                self.tot_process_instance.execute(problem_description, model_name_param_for_process)
                tot_sol_obj, process_specific_summary = self.tot_process_instance.get_result()

                if tot_sol_obj:
                    orchestrator_solution = tot_sol_obj
                else:
                    orchestrator_solution.final_answer = "Error: ToTProcess executed but returned no solution object."
                    logging.error("ToTProcess returned None for solution object in ALWAYS_TOT mode.")

        elif self.trigger_mode == ToTTriggerMode.ASSESS_FIRST:
            logging.info("Orchestrator trigger mode: ASSESS_FIRST_TOT. Performing complexity assessment.")
            if not self.complexity_assessor:
                logging.critical("ComplexityAssessor not initialized for ASSESS_FIRST_TOT mode.")
                orchestrator_solution.final_answer = "Error: ComplexityAssessor not initialized."
            else:
                assessment_decision, assessment_stats = self.complexity_assessor.assess(problem_description)
                orchestrator_solution.assessment_stats = assessment_stats
                orchestrator_solution.assessment_decision = assessment_decision # Field now exists

                if assessment_decision == AssessmentDecision.ONE_SHOT:
                    logging.info("Assessment: ONE_SHOT. Orchestrator performing direct one-shot call.")
                    final_answer, oneshot_stats = self._run_direct_oneshot(problem_description)
                    orchestrator_solution.final_answer = final_answer
                    orchestrator_solution.fallback_call_stats = oneshot_stats
                elif assessment_decision == AssessmentDecision.ADVANCED_REASONING:
                    logging.info("Assessment: ADVANCED_REASONING (ToT). Orchestrator delegating to ToTProcess.")
                    if not self.tot_process_instance:
                        logging.critical("ToTProcess not initialized for ASSESS_FIRST_TOT (ADVANCED_REASONING path).")
                        orchestrator_solution.final_answer = "Error: ToTProcess not initialized for ADVANCED_REASONING path."
                    else:
                        self.tot_process_instance.execute(problem_description, model_name_param_for_process)
                        tot_sol_obj, process_specific_summary = self.tot_process_instance.get_result()
                        if tot_sol_obj:
                            orchestrator_solution = tot_sol_obj
                        else:
                             orchestrator_solution.final_answer = "Error: ToTProcess (post-assessment) returned no solution."
                             logging.error("ToTProcess (ASSESS_FIRST) returned None for solution object.")
                else:
                    logging.error(f"Complexity assessment resulted in '{assessment_decision}'. Orchestrator attempting direct one-shot as last resort.")
                    final_answer, fallback_stats = self._run_direct_oneshot(problem_description, is_fallback_orchestrator=True)
                    orchestrator_solution.final_answer = final_answer
                    orchestrator_solution.fallback_call_stats = fallback_stats

        else:
            logging.critical(f"Unknown ToT trigger mode: {self.trigger_mode}")
            orchestrator_solution.final_answer = f"Error: Unknown trigger mode {self.trigger_mode}"


        if orchestrator_solution.total_wall_clock_time_seconds == 0.0:
             orchestrator_solution.total_wall_clock_time_seconds = time.monotonic() - overall_start_time

        final_summary_output = self._generate_orchestrator_summary(
            orchestrator_solution,
            problem_description,
            process_specific_summary=process_specific_summary
        )
        return orchestrator_solution, final_summary_output

    def _generate_orchestrator_summary(self,
                                       solution: ToTSolution,
                                       problem_description: str,
                                       process_specific_summary: Optional[str] = None) -> str:
        output_buffer = io.StringIO()
        output_buffer.write("\n" + "="*20 + " ToT Orchestrator Overall Summary " + "="*20 + "\n")
        output_buffer.write(f"Problem: {problem_description[:100]}...\n")
        output_buffer.write(f"Orchestrator Trigger Mode: {self.trigger_mode.value}\n")

        if solution.assessment_decision: # Check if field has a value
             output_buffer.write(f"Assessment Decision: {solution.assessment_decision.value}\n")
        if solution.assessment_stats:
            s = solution.assessment_stats
            time_str = f"{s.call_duration_seconds:.2f}s" if s.call_duration_seconds is not None else "N/A"
            output_buffer.write(f"Assessment Phase ({s.model_name if s else 'N/A'}): C={s.completion_tokens if s else 'N/A'}, P={s.prompt_tokens if s else 'N/A'}, Time={time_str}\n")

        if process_specific_summary:
            output_buffer.write("--- Delegated to ToTProcess ---\n")
            output_buffer.write(process_specific_summary)
            output_buffer.write("-------------------------------\n")
            if solution.tot_result and solution.tot_result.succeeded:
                 output_buffer.write(f"ToTProcess Reported Success: Yes\n")
            elif solution.tot_failed_and_fell_back:
                 err_msg = solution.tot_result.error_message if solution.tot_result else "N/A"
                 output_buffer.write(f"ToTProcess Reported Failure and Fallback: Yes (Reason: {err_msg})\n")

        if solution.fallback_call_stats:
            sfb = solution.fallback_call_stats
            if solution.tot_failed_and_fell_back:
                pass # Already covered by ToTProcess summary
            # Corrected enum name: NEVER_AOT to NEVER_TOT
            elif self.trigger_mode == ToTTriggerMode.NEVER_TOT :
                 output_buffer.write(f"Orchestrator Direct One-Shot Call ({sfb.model_name}): C={sfb.completion_tokens}, P={sfb.prompt_tokens}, Time={sfb.call_duration_seconds:.2f}s\n")
            elif self.trigger_mode == ToTTriggerMode.ASSESS_FIRST and solution.assessment_decision == AssessmentDecision.ONE_SHOT:
                 output_buffer.write(f"Orchestrator Post-Assessment One-Shot Call ({sfb.model_name}): C={sfb.completion_tokens}, P={sfb.prompt_tokens}, Time={sfb.call_duration_seconds:.2f}s\n")

        output_buffer.write(f"\n--- Orchestrator Perspective Totals (from ToTSolution) ---\n")
        output_buffer.write(f"Total Completion Tokens (Orchestrator): {solution.total_completion_tokens}\n")
        output_buffer.write(f"Total Prompt Tokens (Orchestrator): {solution.total_prompt_tokens}\n")
        output_buffer.write(f"Grand Total Tokens (Orchestrator): {solution.grand_total_tokens}\n")
        output_buffer.write(f"Total LLM Interaction Time (Orchestrator, sum of calls it's aware of): {solution.total_llm_interaction_time_seconds:.2f}s\n")
        output_buffer.write(f"Total Orchestrator Wall-Clock Time: {solution.total_wall_clock_time_seconds:.2f}s\n")

        if solution.final_answer:
            output_buffer.write(f"\nFinal Answer (Returned by Orchestrator):\n{solution.final_answer}\n")
        else:
            output_buffer.write("\nFinal answer was not successfully extracted or an error occurred (Orchestrator).\n")
        output_buffer.write("="*70 + "\n")
        return output_buffer.getvalue()
