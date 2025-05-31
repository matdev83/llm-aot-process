import time
import logging
import io
from typing import List, Tuple, Optional, Any

# Standard library or reasoning_process base class
from src.reasoning_process import ReasoningProcess

# LLM Client
from src.llm_client import LLMClient

# Dataclasses and Enums from other modules (e.g., AoT for fallback comparison)
from src.aot.dataclasses import LLMCallStats as AOTLLMCallStats # Use an alias if names clash
from src.aot.enums import AssessmentDecision as AOTAssessmentDecision # Alias for clarity

# Supporting modules (assessors, detectors, utilities)
from src.complexity_assessor import ComplexityAssessor
from src.heuristic_detector import HeuristicDetector
from src.l2t_orchestrator_utils.oneshot_executor import OneShotExecutor

# L2T-Full specific imports
from .dataclasses import L2TFullConfig, L2TResult, L2TFullSolution
from .enums import L2TTriggerMode
from .processor import L2TFullProcessor

logger = logging.getLogger(__name__)

class L2TFullProcess(ReasoningProcess):
    def __init__(self,
                 l2t_full_config: L2TFullConfig,
                 direct_oneshot_model_names: List[str],
                 direct_oneshot_temperature: float,
                 api_key: str,
                 enable_rate_limiting: bool = True,
                 enable_audit_logging: bool = True):

        self.l2t_full_config = l2t_full_config
        self.direct_oneshot_model_names = direct_oneshot_model_names # Stored for OneShotExecutor
        self.direct_oneshot_temperature = direct_oneshot_temperature # Stored for OneShotExecutor

        self.l2t_full_processor = L2TFullProcessor(
            api_key=api_key,
            config=l2t_full_config,
            enable_rate_limiting=enable_rate_limiting,
            enable_audit_logging=enable_audit_logging
        )

        oneshot_llm_client = LLMClient(api_key=api_key, enable_rate_limiting=enable_rate_limiting, enable_audit_logging=enable_audit_logging)
        self.oneshot_executor = OneShotExecutor(
            llm_client=oneshot_llm_client,
            direct_oneshot_model_names=self.direct_oneshot_model_names, # Pass stored value
            direct_oneshot_temperature=self.direct_oneshot_temperature  # Pass stored value
        )

        self._solution: Optional[L2TFullSolution] = None
        self._process_summary: Optional[str] = None
        logger.info("L2TFullProcess initialized.")

    def execute(self, problem_description: str, model_name: str = "l2t_full_default", *args, **kwargs) -> None:
        overall_start_time = time.monotonic()
        self._solution = L2TFullSolution()
        logger.info(f"L2TFullProcess executing for problem: {problem_description[:100]}...")

        l2t_result: L2TResult = self.l2t_full_processor.run(problem_description)
        self._solution.l2t_full_result = l2t_result

        if l2t_result.succeeded and l2t_result.final_answer is not None:
            self._solution.final_answer = l2t_result.final_answer
            logger.info("L2T-Full process succeeded.")
        else:
            logger.warning(f"L2T-Full process failed or did not produce an answer. Reason: {l2t_result.termination_reason}. Error: {l2t_result.error_message}")
            self._solution.l2t_failed_and_fell_back = True
            logger.info("Falling back to one-shot execution.")
            # Corrected method call and arguments for OneShotExecutor
            fallback_answer, fallback_stats = self.oneshot_executor.run_direct_oneshot(
                problem_text=problem_description,
                is_fallback=True
            )
            self._solution.final_answer = fallback_answer
            if fallback_stats:
                 self._solution.fallback_call_stats = AOTLLMCallStats(
                    model_name=fallback_stats.model_name,
                    prompt_tokens=fallback_stats.prompt_tokens,
                    completion_tokens=fallback_stats.completion_tokens,
                    call_duration_seconds=fallback_stats.call_duration_seconds
                )
            logger.info(f"Fallback one-shot executed. Answer: {fallback_answer[:100]}...")

        self._solution.total_wall_clock_time_seconds = time.monotonic() - overall_start_time
        self._process_summary = self._generate_process_summary(self._solution)
        logger.info(f"L2TFullProcess execution finished in {self._solution.total_wall_clock_time_seconds:.2f}s.")

    def get_result(self) -> Tuple[Optional[L2TFullSolution], Optional[str]]:
        return self._solution, self._process_summary

    def _generate_process_summary(self, solution: L2TFullSolution) -> str:
        output_buffer = io.StringIO()
        output_buffer.write("\n" + "="*20 + " L2TFullProcess Execution Summary " + "="*20 + "\n")
        if solution.l2t_full_result:
            l2t_res = solution.l2t_full_result
            output_buffer.write(f"L2T-Full Attempted: Yes\n")
            output_buffer.write(f"  L2T-Full Succeeded: {l2t_res.succeeded}\n")
            if l2t_res.termination_reason:
                output_buffer.write(f"  Termination Reason: {l2t_res.termination_reason.value if l2t_res.termination_reason else 'N/A'}\n")
            if l2t_res.error_message: output_buffer.write(f"  Error Message: {l2t_res.error_message}\n")
            output_buffer.write(f"  LLM Calls: {l2t_res.total_llm_calls}\n")
            output_buffer.write(f"  Total Processing Time (L2T-Full): {l2t_res.total_processing_time_seconds:.2f}s\n")
        if solution.l2t_failed_and_fell_back:
            output_buffer.write(f"L2T-Full Failed and Fell Back to One-Shot: Yes\n")
            if solution.fallback_call_stats:
                sfb = solution.fallback_call_stats
                output_buffer.write(f"  Fallback One-Shot Call ({sfb.model_name}): C={sfb.completion_tokens}, P={sfb.prompt_tokens}, Time={sfb.call_duration_seconds:.2f}s\n")
        output_buffer.write(f"Final Answer: {solution.final_answer[:200]}...\n" if solution.final_answer else "No final answer.\n")
        output_buffer.write(f"Total L2TFullProcess Wall-Clock Time: {solution.total_wall_clock_time_seconds:.2f}s\n")
        output_buffer.write("="*68 + "\n")
        return output_buffer.getvalue()

class L2TFullOrchestrator:
    def __init__(self,
                 trigger_mode: L2TTriggerMode,
                 l2t_full_config: L2TFullConfig,
                 direct_oneshot_model_names: List[str],
                 direct_oneshot_temperature: float,
                 assessment_model_names: List[str],
                 assessment_temperature: float,
                 api_key: str,
                 use_heuristic_shortcut: bool = True,
                 heuristic_detector: Optional[HeuristicDetector] = None,
                 enable_rate_limiting: bool = True,
                 enable_audit_logging: bool = True):

        self.trigger_mode = trigger_mode
        self.l2t_full_config = l2t_full_config
        self.direct_oneshot_model_names = direct_oneshot_model_names
        self.direct_oneshot_temperature = direct_oneshot_temperature
        self.api_key = api_key
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_audit_logging = enable_audit_logging

        self.llm_client = LLMClient(api_key=api_key, enable_rate_limiting=enable_rate_limiting, enable_audit_logging=enable_audit_logging)
        self.oneshot_executor = OneShotExecutor( # Corrected instantiation
            llm_client=self.llm_client,
            direct_oneshot_model_names=self.direct_oneshot_model_names,
            direct_oneshot_temperature=self.direct_oneshot_temperature
        )

        self.complexity_assessor: Optional[ComplexityAssessor] = None
        if self.trigger_mode == L2TTriggerMode.ASSESS_FIRST:
            self.complexity_assessor = ComplexityAssessor(
                llm_client=self.llm_client,
                small_model_names=assessment_model_names,
                temperature=assessment_temperature,
                use_heuristic_shortcut=use_heuristic_shortcut,
                heuristic_detector=heuristic_detector if heuristic_detector else HeuristicDetector()
            )

        self.l2t_full_process_instance: Optional[L2TFullProcess] = None
        if self.trigger_mode == L2TTriggerMode.ALWAYS_L2T or self.trigger_mode == L2TTriggerMode.ASSESS_FIRST:
            self.l2t_full_process_instance = L2TFullProcess(
                l2t_full_config=self.l2t_full_config,
                direct_oneshot_model_names=self.direct_oneshot_model_names,
                direct_oneshot_temperature=self.direct_oneshot_temperature,
                api_key=self.api_key,
                enable_rate_limiting=self.enable_rate_limiting,
                enable_audit_logging=self.enable_audit_logging
            )
        logger.info(f"L2TFullOrchestrator initialized with trigger mode: {self.trigger_mode.value}")

    def solve(self, problem_text: str, model_name_for_l2t_full: str = "l2t_full_default") -> Tuple[L2TFullSolution, str]:
        overall_start_time = time.monotonic()
        orchestrator_solution = L2TFullSolution()
        l2t_full_process_summary: Optional[str] = None

        if self.trigger_mode == L2TTriggerMode.NEVER_L2T:
            logger.info("Trigger mode: NEVER_L2T. Orchestrator performing direct one-shot call.")
            # Corrected method call for OneShotExecutor
            final_answer, oneshot_stats = self.oneshot_executor.run_direct_oneshot(problem_text=problem_text)
            orchestrator_solution.final_answer = final_answer
            if oneshot_stats:
                 orchestrator_solution.main_call_stats = AOTLLMCallStats(
                    model_name=oneshot_stats.model_name, prompt_tokens=oneshot_stats.prompt_tokens,
                    completion_tokens=oneshot_stats.completion_tokens, call_duration_seconds=oneshot_stats.call_duration_seconds)

        elif self.trigger_mode == L2TTriggerMode.ALWAYS_L2T:
            logger.info("Trigger mode: ALWAYS_L2T. Orchestrator delegating to L2TFullProcess.")
            if not self.l2t_full_process_instance:
                logger.critical("L2TFullProcess not initialized for ALWAYS_L2T mode.")
                orchestrator_solution.final_answer = "Error: L2TFullProcess not initialized."
            else:
                self.l2t_full_process_instance.execute(problem_description=problem_text, model_name=model_name_for_l2t_full)
                l2t_solution_obj, l2t_full_process_summary = self.l2t_full_process_instance.get_result()
                if l2t_solution_obj: orchestrator_solution = l2t_solution_obj
                else:
                    orchestrator_solution.final_answer = "Error: L2TFullProcess executed but returned no solution object."
                    logger.error("L2TFullProcess returned None for solution object in ALWAYS_L2T mode.")

        elif self.trigger_mode == L2TTriggerMode.ASSESS_FIRST:
            logger.info("Trigger mode: ASSESS_FIRST. Orchestrator performing complexity assessment.")
            if not self.complexity_assessor:
                logger.critical("ComplexityAssessor not initialized for ASSESS_FIRST mode.")
                orchestrator_solution.final_answer = "Error: ComplexityAssessor not initialized."
            else:
                assessment_decision, assessment_stats_raw = self.complexity_assessor.assess(problem_text)
                if assessment_stats_raw:
                     orchestrator_solution.assessment_stats = AOTLLMCallStats(
                        model_name=assessment_stats_raw.model_name, prompt_tokens=assessment_stats_raw.prompt_tokens,
                        completion_tokens=assessment_stats_raw.completion_tokens, call_duration_seconds=assessment_stats_raw.call_duration_seconds)
                orchestrator_solution.assessment_decision = AOTAssessmentDecision(assessment_decision.value)

                if assessment_decision == AOTAssessmentDecision.ONE_SHOT:
                    logger.info("Assessment: ONE_SHOT. Orchestrator performing direct one-shot call.")
                    # Corrected method call
                    final_answer, oneshot_stats = self.oneshot_executor.run_direct_oneshot(problem_text=problem_text)
                    orchestrator_solution.final_answer = final_answer
                    if oneshot_stats:
                        orchestrator_solution.main_call_stats = AOTLLMCallStats(
                            model_name=oneshot_stats.model_name, prompt_tokens=oneshot_stats.prompt_tokens,
                            completion_tokens=oneshot_stats.completion_tokens, call_duration_seconds=oneshot_stats.call_duration_seconds)
                elif assessment_decision == AOTAssessmentDecision.ADVANCED_REASONING:
                    logger.info("Assessment: ADVANCED_REASONING. Orchestrator delegating to L2TFullProcess.")
                    if not self.l2t_full_process_instance:
                        logger.critical("L2TFullProcess not initialized for ASSESS_FIRST (ADVANCED_REASONING path).")
                        orchestrator_solution.final_answer = "Error: L2TFullProcess not initialized."
                    else:
                        self.l2t_full_process_instance.execute(problem_description=problem_text, model_name=model_name_for_l2t_full)
                        l2t_solution_obj, l2t_full_process_summary = self.l2t_full_process_instance.get_result()
                        if l2t_solution_obj: orchestrator_solution = l2t_solution_obj
                        else:
                             orchestrator_solution.final_answer = "Error: L2TFullProcess (post-assessment) returned no solution."
                             logger.error("L2TFullProcess returned None for solution in ASSESS_FIRST (ADVANCED_REASONING path).")
                else:
                    logger.error(f"Complexity assessment failed or returned unexpected decision: {assessment_decision}. Orchestrator attempting one-shot call as a last resort.")
                    # Corrected method call
                    fallback_answer, fallback_stats = self.oneshot_executor.run_direct_oneshot(problem_text=problem_text, is_fallback=True)
                    orchestrator_solution.final_answer = fallback_answer
                    if fallback_stats:
                        orchestrator_solution.fallback_call_stats = AOTLLMCallStats(
                            model_name=fallback_stats.model_name, prompt_tokens=fallback_stats.prompt_tokens,
                            completion_tokens=fallback_stats.completion_tokens, call_duration_seconds=fallback_stats.call_duration_seconds)

        if orchestrator_solution.total_wall_clock_time_seconds == 0.0:
             orchestrator_solution.total_wall_clock_time_seconds = time.monotonic() - overall_start_time

        final_summary_output = self._generate_overall_summary(orchestrator_solution,
                                                              l2t_full_process_specific_summary=l2t_full_process_summary)
        return orchestrator_solution, final_summary_output

    def _generate_overall_summary(self, solution: L2TFullSolution, l2t_full_process_specific_summary: Optional[str] = None) -> str:
        output_buffer = io.StringIO()
        output_buffer.write("\n" + "="*20 + " L2TFullOrchestrator Overall Summary " + "="*20 + "\n")
        output_buffer.write(f"Orchestrator Trigger Mode: {self.trigger_mode.value}\n")
        if solution.assessment_stats:
            s = solution.assessment_stats
            decision_val = solution.assessment_decision.value if solution.assessment_decision else 'N/A'
            time_str = f"{s.call_duration_seconds:.2f}s" if s.call_duration_seconds is not None else "N/A"
            output_buffer.write(f"Assessment Phase ({s.model_name if s else 'N/A'}): Decision='{decision_val}', C={s.completion_tokens if s else 'N/A'}, P={s.prompt_tokens if s else 'N/A'}, Time={time_str}\n")
        if solution.main_call_stats:
            s = solution.main_call_stats
            output_buffer.write(f"Orchestrator Main Model Call (Direct ONESHOT path) ({s.model_name}): C={s.completion_tokens}, P={s.prompt_tokens}, Time={s.call_duration_seconds:.2f}s\n")
        if l2t_full_process_specific_summary:
            output_buffer.write("\n--- Delegated to L2TFullProcess ---\n")
            output_buffer.write(l2t_full_process_specific_summary)
            if solution.l2t_full_result and solution.l2t_full_result.succeeded:
                 output_buffer.write(f"L2TFullProcess Reported Success: Yes\n")
            elif solution.l2t_failed_and_fell_back:
                 reason = solution.l2t_full_result.termination_reason.value if solution.l2t_full_result and solution.l2t_full_result.termination_reason else 'N/A'
                 output_buffer.write(f"L2TFullProcess Reported Failure and Fallback: Yes (Reason: {reason})\n")
            output_buffer.write("-----------------------------------\n")
        elif solution.fallback_call_stats and not l2t_full_process_specific_summary:
            sfb = solution.fallback_call_stats
            output_buffer.write(f"Orchestrator Fallback One-Shot Call ({sfb.model_name}): C={sfb.completion_tokens}, P={sfb.prompt_tokens}, Time={sfb.call_duration_seconds:.2f}s\n")
        output_buffer.write(f"\n--- Orchestrator Perspective Totals ---\n")
        output_buffer.write(f"Total Completion Tokens (Orchestrator): {solution.total_completion_tokens}\n")
        output_buffer.write(f"Total Prompt Tokens (Orchestrator): {solution.total_prompt_tokens}\n")
        output_buffer.write(f"Grand Total Tokens (Orchestrator): {solution.grand_total_tokens}\n")
        output_buffer.write(f"Total LLM Interaction Time (Orchestrator): {solution.total_llm_interaction_time_seconds:.2f}s\n")
        output_buffer.write(f"Total Orchestrator Wall-Clock Time: {solution.total_wall_clock_time_seconds:.2f}s\n")
        if solution.final_answer:
            output_buffer.write(f"\nFinal Answer (Returned by Orchestrator):\n{solution.final_answer}\n")
        else:
            output_buffer.write("\nFinal answer was not successfully extracted or an error occurred.\n")
        output_buffer.write("="*75 + "\n")
        return output_buffer.getvalue()

if __name__ == "__main__":
    logger.setLevel(logging.INFO) # Changed to INFO for less verbose default
    # Basic console logging
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


    DUMMY_API_KEY = "dummy_api_key_for_testing"
    l2t_full_config = L2TFullConfig(max_steps=3, max_total_nodes=10,
                                    initial_prompt_model_names=["gpt-3.5-turbo"], # Using common model names
                                    classification_model_names=["gpt-3.5-turbo"],
                                    thought_generation_model_names=["gpt-3.5-turbo"])

    orchestrator = L2TFullOrchestrator(
        trigger_mode=L2TTriggerMode.ALWAYS_L2T,
        l2t_full_config=l2t_full_config,
        direct_oneshot_model_names=["gpt-4"],
        direct_oneshot_temperature=0.1,
        assessment_model_names=["gpt-3.5-turbo"],
        assessment_temperature=0.3,
        api_key=DUMMY_API_KEY,
        use_heuristic_shortcut=True,
        enable_rate_limiting=False
    )
    problem = "Explain the concept of PPO in reinforcement learning to a 5-year-old. Keep it very simple."
    print(f"\n--- Running L2TFullOrchestrator for problem: '{problem}' ---")
    try:
        solution, summary = orchestrator.solve(problem)
        print("\n--- Orchestrator Summary ---")
        print(summary)
        # if solution:
        #     print(f"Final Answer from Solution Object: {solution.final_answer}")
        #     if solution.l2t_full_result:
        #          print(f"L2T-Full process had {solution.l2t_full_result.total_llm_calls} LLM calls.")
    except Exception as e:
        print(f"An error occurred during conceptual orchestrator run: {e}")
        import traceback
        traceback.print_exc()
    print("\n--- Orchestrator Example Finished ---")
