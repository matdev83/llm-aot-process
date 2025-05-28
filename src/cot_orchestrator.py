import time
import logging
import io
from typing import List, Tuple, Optional

from src.cot_process import CoTProcess, CoTResult
from src.orchestrator_solution import OrchestratorSolution
from src.aot_dataclasses import LLMCallStats # For direct one-shot and assessment
from src.llm_client import LLMClient
from src.complexity_assessor import ComplexityAssessor
from src.aot_enums import AssessmentDecision, AotTriggerMode # Re-using AotTriggerMode

# Configure basic logging if no handlers are present for the logger used by orchestrator
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class CoTOrchestrator:
    def __init__(self,
                 process: CoTProcess,
                 trigger_mode: AotTriggerMode, # Conceptually CoTTriggerMode
                 direct_oneshot_model_names: List[str],
                 direct_oneshot_temperature: float,
                 assessment_model_names: List[str],
                 assessment_temperature: float,
                 api_key: str,
                 use_heuristic_shortcut: bool = True):

        self.process = process
        self.trigger_mode = trigger_mode
        self.use_heuristic_shortcut = use_heuristic_shortcut
        self.direct_oneshot_model_names = direct_oneshot_model_names
        self.direct_oneshot_temperature = direct_oneshot_temperature
        
        self.llm_client = LLMClient(api_key=api_key)
        self.complexity_assessor: Optional[ComplexityAssessor] = None
        if self.trigger_mode == AotTriggerMode.ASSESS_FIRST:
            self.complexity_assessor = ComplexityAssessor(
                llm_client=self.llm_client,
                small_model_names=assessment_model_names,
                temperature=assessment_temperature,
                use_heuristic_shortcut=self.use_heuristic_shortcut
            )

    def _run_direct_oneshot(self, problem_text: str, is_fallback:bool = False) -> Tuple[str, LLMCallStats]:
        mode = "FALLBACK ONESHOT" if is_fallback else "ONESHOT"
        logger.info(f"--- Proceeding with {mode} Answer ---")
        logger.info(f"Using models: {', '.join(self.direct_oneshot_model_names)}, Temperature: {self.direct_oneshot_temperature}")
        
        # Construct prompt for direct one-shot. For now, assume problem_text is the full prompt.
        # If a specific prompt template is needed for one-shot, it should be constructed here.
        # Example: prompt = f"Solve the following problem:\n{problem_text}"
        prompt = problem_text 

        response_content, stats = self.llm_client.call(
            prompt=prompt, models=self.direct_oneshot_model_names, temperature=self.direct_oneshot_temperature
        )
        logger.debug(f"Direct {mode} response from {stats.model_name}:\n{response_content}") # Corrected f-string
        logger.info(f"LLM call ({stats.model_name}) for {mode}: Duration: {stats.call_duration_seconds:.2f}s, Tokens (C:{stats.completion_tokens}, P:{stats.prompt_tokens})")
        return response_content, stats

    def solve(self, problem_text: str) -> Tuple[OrchestratorSolution, str]:
        overall_start_time = time.monotonic()
        solution = OrchestratorSolution()
        solution.trigger_mode_used = self.trigger_mode.name # Store enum name

        if self.trigger_mode == AotTriggerMode.NEVER_AOT: # Conceptually NEVER_PROCESS
            logger.info(f"Trigger mode: {self.trigger_mode.name}. Direct one-shot call.")
            final_answer, oneshot_stats = self._run_direct_oneshot(problem_text)
            solution.final_answer = final_answer
            solution.main_oneshot_call_stats = oneshot_stats

        elif self.trigger_mode == AotTriggerMode.ALWAYS_AOT: # Conceptually ALWAYS_PROCESS
            logger.info(f"Trigger mode: {self.trigger_mode.name}. Direct {self.process.__class__.__name__} process.")
            cot_result = self.process.run(problem_text)
            solution.cot_result = cot_result
            # get_summary must be called AFTER cot_result is set, if cot_result is not None
            if solution.cot_result:
                 solution.cot_summary_output = self.process.get_summary(solution.cot_result)


            if cot_result.succeeded:
                solution.final_answer = cot_result.final_answer
            else:
                logger.warning(f"{self.process.__class__.__name__} process ({self.trigger_mode.name} mode) failed (Reason: {cot_result.error_message or 'Unknown'}). Falling back to one-shot.")
                solution.process_failed_and_fell_back = True
                fallback_answer, fallback_stats = self._run_direct_oneshot(problem_text, is_fallback=True)
                solution.final_answer = fallback_answer
                solution.fallback_oneshot_call_stats = fallback_stats
        
        elif self.trigger_mode == AotTriggerMode.ASSESS_FIRST:
            if not self.complexity_assessor: # Should not happen due to __init__ logic
                logger.critical("ComplexityAssessor not initialized for ASSESS_FIRST mode. THIS IS AN UNEXPECTED STATE.")
                # Fallback or raise error - current structure falls back
                solution.process_failed_and_fell_back = True # Treat as a process failure
                fallback_answer, fallback_stats = self._run_direct_oneshot(problem_text, is_fallback=True)
                solution.final_answer = fallback_answer
                solution.fallback_oneshot_call_stats = fallback_stats
                solution.total_wall_clock_time_seconds = time.monotonic() - overall_start_time
                # Ensure summary is generated even in this critical error path
                return solution, self._generate_overall_summary(solution)


            assessment_decision, assessment_stats = self.complexity_assessor.assess(problem_text)
            solution.assessment_stats = assessment_stats
            solution.assessment_decision = assessment_decision.name # Store enum name

            if assessment_decision == AssessmentDecision.ONESHOT:
                logger.info(f"Assessment: {assessment_decision.name}. Direct one-shot call.")
                final_answer, oneshot_stats = self._run_direct_oneshot(problem_text)
                solution.final_answer = final_answer
                solution.main_oneshot_call_stats = oneshot_stats
            elif assessment_decision == AssessmentDecision.AOT: # Conceptually "run the complex process"
                logger.info(f"Assessment: {assessment_decision.name}. Proceeding with {self.process.__class__.__name__} process.")
                cot_result = self.process.run(problem_text)
                solution.cot_result = cot_result
                # get_summary must be called AFTER cot_result is set, if cot_result is not None
                if solution.cot_result:
                    solution.cot_summary_output = self.process.get_summary(solution.cot_result)

                if cot_result.succeeded:
                    solution.final_answer = cot_result.final_answer
                else:
                    logger.warning(f"{self.process.__class__.__name__} process (after {assessment_decision.name}) failed (Reason: {cot_result.error_message or 'Unknown'}). Falling back to one-shot.")
                    solution.process_failed_and_fell_back = True
                    fallback_answer, fallback_stats = self._run_direct_oneshot(problem_text, is_fallback=True)
                    solution.final_answer = fallback_answer
                    solution.fallback_oneshot_call_stats = fallback_stats
            else: # AssessmentDecision.ERROR
                logger.error(f"Complexity assessment failed (Decision: {assessment_decision.name}). Attempting one-shot call as a last resort.")
                solution.process_failed_and_fell_back = True # Mark as a form of fallback due to assessment error
                fallback_answer, fallback_stats = self._run_direct_oneshot(problem_text, is_fallback=True)
                solution.final_answer = fallback_answer
                solution.fallback_oneshot_call_stats = fallback_stats
        
        solution.total_wall_clock_time_seconds = time.monotonic() - overall_start_time
        overall_summary_str = self._generate_overall_summary(solution)
        return solution, overall_summary_str

    def _generate_overall_summary(self, solution: OrchestratorSolution) -> str:
        output_buffer = io.StringIO()
        process_name = self.process.__class__.__name__

        output_buffer.write("\n" + "="*20 + f" OVERALL {process_name} ORCHESTRATOR SUMMARY " + "="*20 + "\n")
        output_buffer.write(f"Trigger Mode Used: {solution.trigger_mode_used}\n")
        output_buffer.write(f"Heuristic Shortcut Enabled for Assessment: {self.use_heuristic_shortcut}\n")

        if solution.assessment_stats:
            s = solution.assessment_stats
            output_buffer.write(f"Assessment ({s.model_name}): Decision={solution.assessment_decision}, Tokens (C:{s.completion_tokens},P:{s.prompt_tokens}), Time:{s.call_duration_seconds:.2f}s\n")
        
        if solution.main_oneshot_call_stats:
            s = solution.main_oneshot_call_stats
            output_buffer.write(f"Main One-Shot Call ({s.model_name}): Tokens (C:{s.completion_tokens},P:{s.prompt_tokens}), Time:{s.call_duration_seconds:.2f}s\n")

        if solution.cot_result: # This implies the CoT process was run
            output_buffer.write(f"{process_name} Process Attempted: Yes\n")
            # The detailed CoT summary is already generated by the process and stored in solution.cot_summary_output
            if solution.cot_summary_output:
                output_buffer.write(solution.cot_summary_output) # Append process-specific summary
            else: # Should not happen if cot_result is present and get_summary was called
                output_buffer.write(f"{process_name} Summary: Not available (This indicates an issue if the process was run).\n")

            
            if solution.cot_result.succeeded:
                output_buffer.write(f"{process_name} Succeeded (as per CoT result): Yes\n")
            # Check if it failed AND fell_back. process_failed_and_fell_back implies cot_result was not successful.
            elif solution.process_failed_and_fell_back: 
                output_buffer.write(f"{process_name} FAILED and Fell Back to One-Shot: Yes (Failure Reason: {solution.cot_result.error_message or 'Unknown'})\n")
            # This case handles if cot_result exists, did not succeed, but did NOT fall back (e.g. NEVER_AOT where only CoT runs)
            # However, current logic for ALWAYS_AOT implies it WILL fall back if CoT fails.
            # This handles a theoretical case or if logic changes.
            elif not solution.cot_result.succeeded:
                 output_buffer.write(f"{process_name} FAILED: Yes (Failure Reason: {solution.cot_result.error_message or 'Unknown'}). Did not fall back.\n")

        elif solution.trigger_mode_used == AotTriggerMode.NEVER_AOT.name: # CoT process was not run by design
             output_buffer.write(f"{process_name} Process Attempted: No (due to {solution.trigger_mode_used} mode)\n")
        elif solution.assessment_decision == AssessmentDecision.ONESHOT.name: # CoT process was not run due to assessment
             output_buffer.write(f"{process_name} Process Attempted: No (due to Assessment Decision: {solution.assessment_decision})\n")


        if solution.fallback_oneshot_call_stats: 
            sfb = solution.fallback_oneshot_call_stats
            # Clarify why fallback happened
            if solution.process_failed_and_fell_back and solution.cot_result and not solution.cot_result.succeeded:
                 output_buffer.write(f"Fallback One-Shot Call (after {process_name} failure) ({sfb.model_name}): Tokens (C:{sfb.completion_tokens},P:{sfb.prompt_tokens}), Time:{sfb.call_duration_seconds:.2f}s\n")
            elif solution.process_failed_and_fell_back and solution.assessment_decision == AssessmentDecision.ERROR.name:
                 output_buffer.write(f"Fallback One-Shot Call (after Assessment Error) ({sfb.model_name}): Tokens (C:{sfb.completion_tokens},P:{sfb.prompt_tokens}), Time:{sfb.call_duration_seconds:.2f}s\n")
            # This case might be redundant if process_failed_and_fell_back is always set with fallback_oneshot_call_stats
            else: 
                 output_buffer.write(f"Fallback One-Shot Call (Reason not specified in cot_result/assessment error) ({sfb.model_name}): Tokens (C:{sfb.completion_tokens},P:{sfb.prompt_tokens}), Time:{sfb.call_duration_seconds:.2f}s\n")


        output_buffer.write(f"\nTotal Completion Tokens (All Calls): {solution.total_completion_tokens}\n")
        output_buffer.write(f"Total Prompt Tokens (All Calls): {solution.total_prompt_tokens}\n")
        output_buffer.write(f"Grand Total Tokens (All Calls): {solution.grand_total_tokens}\n")
        output_buffer.write(f"Total LLM Interaction Time (All Calls): {solution.total_llm_interaction_time_seconds:.2f}s\n")
        output_buffer.write(f"Total Orchestrator Wall-Clock Time: {solution.total_wall_clock_time_seconds:.2f}s\n")

        if solution.final_answer: 
            output_buffer.write(f"\nFinal Answer:\n{solution.final_answer}\n")
        else: 
            output_buffer.write("\nFinal answer was not successfully extracted or an error occurred.\n")
        
        # Adjust length of closing line dynamically based on process_name
        header_footer_len = 20 + len(f" OVERALL {process_name} ORCHESTRATOR SUMMARY ") + 20
        output_buffer.write("="*header_footer_len + "\n") 
        return output_buffer.getvalue()
