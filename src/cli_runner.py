import sys
import os
import argparse
import logging
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Core components for the new architecture
from src.cot_orchestrator import CoTOrchestrator
from src.aot_processor import AoTProcessor
from src.l2t_processor import L2TProcessor
from src.llm_client import LLMClient # Import LLMClient

# Enums and Configs
from src.aot_enums import AotTriggerMode # Reused for generic trigger mode
from src.aot_dataclasses import AoTRunnerConfig
from src.l2t_dataclasses import L2TConfig

# Constants for default values (consider organizing them if they proliferate)
from src.aot_constants import (
    DEFAULT_MAIN_MODEL_NAMES as DEFAULT_AOT_MAIN_MODEL_NAMES,
    DEFAULT_SMALL_MODEL_NAMES as DEFAULT_AOT_ASSESSMENT_MODEL_NAMES,
    DEFAULT_MAX_STEPS as DEFAULT_AOT_MAX_STEPS,
    DEFAULT_MAX_TIME_SECONDS as DEFAULT_AOT_MAX_TIME_SECONDS,
    DEFAULT_NO_PROGRESS_LIMIT as DEFAULT_AOT_NO_PROGRESS_LIMIT,
    DEFAULT_MAIN_TEMPERATURE as DEFAULT_AOT_MAIN_TEMPERATURE,
    DEFAULT_ASSESSMENT_TEMPERATURE as DEFAULT_AOT_ASSESSMENT_TEMPERATURE
)
from src.l2t_constants import (
    DEFAULT_L2T_CLASSIFICATION_MODEL_NAMES,
    DEFAULT_L2T_THOUGHT_GENERATION_MODEL_NAMES,
    DEFAULT_L2T_INITIAL_PROMPT_MODEL_NAMES,
    DEFAULT_L2T_CLASSIFICATION_TEMPERATURE,
    DEFAULT_L2T_THOUGHT_GENERATION_TEMPERATURE,
    DEFAULT_L2T_INITIAL_PROMPT_TEMPERATURE,
    DEFAULT_L2T_MAX_STEPS,
    DEFAULT_L2T_MAX_TOTAL_NODES,
    DEFAULT_L2T_MAX_TIME_SECONDS,
    DEFAULT_L2T_X_FMT_DEFAULT,
    DEFAULT_L2T_X_EVA_DEFAULT,
)

def main():
    parser = argparse.ArgumentParser(
        description="CLI Runner for Chain-of-Thought (CoT) processes.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    problem_group = parser.add_mutually_exclusive_group(required=True)
    problem_group.add_argument("--problem", "-p", type=str, help="Problem/question to solve.")
    problem_group.add_argument("--problem-filename", type=str, help="File containing the problem.")

    # General Orchestrator and Process Selection Arguments
    parser.add_argument(
        "--process-type", type=str, choices=['aot', 'l2t'], required=True,
        help="Type of CoT process to run (aot or l2t)."
    )
    parser.add_argument(
        "--trigger-mode", type=str, choices=[mode.value for mode in AotTriggerMode],
        default=AotTriggerMode.ASSESS_FIRST.value,
        help=(f"Trigger mode for the CoT process (default: {AotTriggerMode.ASSESS_FIRST.value}).\n"
              f" '{AotTriggerMode.ALWAYS_AOT.value}': Force the selected CoT process.\n" 
              f" '{AotTriggerMode.ASSESS_FIRST.value}': Use LLM to decide between CoT process or ONESHOT.\n"
              f" '{AotTriggerMode.NEVER_AOT.value}': Force ONESHOT (direct answer), bypass CoT process.") 
    )
    # Orchestrator's direct one-shot and assessment config
    parser.add_argument("--oneshot-models", type=str, nargs='+', default=DEFAULT_AOT_MAIN_MODEL_NAMES, 
                           help=f"LLM(s) for direct ONESHOT answers. Default: {' '.join(DEFAULT_AOT_MAIN_MODEL_NAMES)}")
    parser.add_argument("--oneshot-temp", type=float, default=DEFAULT_AOT_MAIN_TEMPERATURE,
                           help=f"Temperature for ONESHOT LLM(s). Default: {DEFAULT_AOT_MAIN_TEMPERATURE}")
    parser.add_argument("--assessment-models", type=str, nargs='+', default=DEFAULT_AOT_ASSESSMENT_MODEL_NAMES,
                           help=f"Small LLM(s) for complexity assessment. Default: {' '.join(DEFAULT_AOT_ASSESSMENT_MODEL_NAMES)}")
    parser.add_argument("--assessment-temp", type=float, default=DEFAULT_AOT_ASSESSMENT_TEMPERATURE,
                           help=f"Temperature for assessment LLM(s). Default: {DEFAULT_AOT_ASSESSMENT_TEMPERATURE}")
    parser.add_argument("--disable-heuristic", action="store_true",
                           help="Disable local heuristic analysis for complexity assessment, always using LLM.")

    # AoT Configuration Group (only relevant if --process-type aot)
    aot_group = parser.add_argument_group('AoT Process Configuration (used if --process-type aot)')
    aot_group.add_argument("--aot-main-models", type=str, nargs='+', default=DEFAULT_AOT_MAIN_MODEL_NAMES,
                           help=f"Main LLM(s) for AoT reasoning steps. Default: {' '.join(DEFAULT_AOT_MAIN_MODEL_NAMES)}")
    aot_group.add_argument("--aot-main-temp", type=float, default=DEFAULT_AOT_MAIN_TEMPERATURE,
                           help=f"Temperature for AoT main LLM(s). Default: {DEFAULT_AOT_MAIN_TEMPERATURE}")
    aot_group.add_argument("--aot-max-steps", type=int, default=DEFAULT_AOT_MAX_STEPS,
                           help=f"Max AoT reasoning steps. Default: {DEFAULT_AOT_MAX_STEPS}.")
    aot_group.add_argument("--aot-max-reasoning-tokens", type=int, default=None,
                           help="Max completion tokens for AoT reasoning phase. Enforced dynamically.")
    aot_group.add_argument("--aot-max-time", type=int, default=DEFAULT_AOT_MAX_TIME_SECONDS,
                           help=f"Overall max time for an AoT run (seconds). Default: {DEFAULT_AOT_MAX_TIME_SECONDS}s")
    aot_group.add_argument("--aot-no-progress-limit", type=int, default=DEFAULT_AOT_NO_PROGRESS_LIMIT,
                           help=f"Stop AoT if no progress for this many steps. Default: {DEFAULT_AOT_NO_PROGRESS_LIMIT}")
    aot_group.add_argument("--aot-pass-remaining-steps-pct", type=int, default=None, metavar="PCT", choices=range(0, 101),
                           help="Percentage (0-100) of original max_steps at which to inform LLM about dynamically remaining steps in AoT. Default: None.")

    # L2T Configuration Group (only relevant if --process-type l2t)
    l2t_group = parser.add_argument_group('L2T Process Configuration (used if --process-type l2t)')
    l2t_group.add_argument("--l2t-classification-models", type=str, nargs='+', default=DEFAULT_L2T_CLASSIFICATION_MODEL_NAMES,
                           help=f"L2T classification model(s). Default: {' '.join(DEFAULT_L2T_CLASSIFICATION_MODEL_NAMES)}")
    l2t_group.add_argument("--l2t-thought-gen-models", type=str, nargs='+', default=DEFAULT_L2T_THOUGHT_GENERATION_MODEL_NAMES,
                           help=f"L2T thought generation model(s). Default: {' '.join(DEFAULT_L2T_THOUGHT_GENERATION_MODEL_NAMES)}")
    l2t_group.add_argument("--l2t-initial-prompt-models", type=str, nargs='+', default=DEFAULT_L2T_INITIAL_PROMPT_MODEL_NAMES,
                           help=f"L2T initial prompt model(s). Default: {' '.join(DEFAULT_L2T_INITIAL_PROMPT_MODEL_NAMES)}")
    l2t_group.add_argument("--l2t-classification-temp", type=float, default=DEFAULT_L2T_CLASSIFICATION_TEMPERATURE,
                           help=f"L2T classification temperature. Default: {DEFAULT_L2T_CLASSIFICATION_TEMPERATURE}")
    l2t_group.add_argument("--l2t-thought-gen-temp", type=float, default=DEFAULT_L2T_THOUGHT_GENERATION_TEMPERATURE,
                           help=f"L2T thought generation temperature. Default: {DEFAULT_L2T_THOUGHT_GENERATION_TEMPERATURE}")
    l2t_group.add_argument("--l2t-initial-prompt-temp", type=float, default=DEFAULT_L2T_INITIAL_PROMPT_TEMPERATURE,
                           help=f"L2T initial prompt temperature. Default: {DEFAULT_L2T_INITIAL_PROMPT_TEMPERATURE}")
    l2t_group.add_argument("--l2t-max-steps", type=int, default=DEFAULT_L2T_MAX_STEPS,
                           help=f"L2T max steps. Default: {DEFAULT_L2T_MAX_STEPS}")
    l2t_group.add_argument("--l2t-max-total-nodes", type=int, default=DEFAULT_L2T_MAX_TOTAL_NODES,
                           help=f"L2T max total nodes. Default: {DEFAULT_L2T_MAX_TOTAL_NODES}")
    l2t_group.add_argument("--l2t-max-time-seconds", type=int, default=DEFAULT_L2T_MAX_TIME_SECONDS,
                           help=f"L2T max time (seconds). Default: {DEFAULT_L2T_MAX_TIME_SECONDS}")
    l2t_group.add_argument("--l2t-x-fmt", dest="l2t_x_fmt_default", type=str, default=DEFAULT_L2T_X_FMT_DEFAULT)
    l2t_group.add_argument("--l2t-x-eva", dest="l2t_x_eva_default", type=str, default=DEFAULT_L2T_X_EVA_DEFAULT)
    
    args = parser.parse_args()

    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level_val = getattr(logging, log_level_str, logging.INFO)
    if not isinstance(log_level_val, int):
        print(f"Warning: Invalid LOG_LEVEL '{log_level_str}'. Defaulting to INFO.", file=sys.stderr)
        log_level_val = logging.INFO
    logging.basicConfig(
        level=log_level_val,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        stream=sys.stderr
    )

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key: 
        logging.critical("OPENROUTER_API_KEY environment variable not set.")
        sys.exit(1)
    
    problem_text: str
    if args.problem_filename:
        try:
            with open(args.problem_filename, 'r', encoding='utf-8') as f: 
                problem_text = f.read()
            logging.info(f"Successfully read problem from file: {args.problem_filename}")
        except Exception as e: 
            logging.critical(f"Error reading problem file '{args.problem_filename}': {e}")
            sys.exit(1)
    else: 
        problem_text = args.problem
        if not problem_text: 
            logging.critical("No problem text provided either directly or via file.")
            sys.exit(1)

    # Instantiate LLMClient once
    llm_client = LLMClient(api_key=api_key)

    # Instantiate selected CoT process
    process_instance = None
    if args.process_type == 'aot':
        logging.info("AoT process type selected.")
        aot_pass_remaining_steps_float: Optional[float] = None
        if args.aot_pass_remaining_steps_pct is not None:
            aot_pass_remaining_steps_float = args.aot_pass_remaining_steps_pct / 100.0
        
        aot_config = AoTRunnerConfig(
            main_model_names=args.aot_main_models,
            temperature=args.aot_main_temp,
            max_steps=args.aot_max_steps,
            max_reasoning_tokens=args.aot_max_reasoning_tokens,
            max_time_seconds=args.aot_max_time,
            no_progress_limit=args.aot_no_progress_limit,
            pass_remaining_steps_pct=aot_pass_remaining_steps_float
        )
        process_instance = AoTProcessor(llm_client=llm_client, config=aot_config)

    elif args.process_type == 'l2t':
        logging.info("L2T process type selected.")
        l2t_config = L2TConfig(
            classification_model_names=args.l2t_classification_models,
            thought_generation_model_names=args.l2t_thought_gen_models,
            initial_prompt_model_names=args.l2t_initial_prompt_models,
            classification_temperature=args.l2t_classification_temp,
            thought_generation_temperature=args.l2t_thought_gen_temp,
            initial_prompt_temperature=args.l2t_initial_prompt_temp,
            max_steps=args.l2t_max_steps,
            max_total_nodes=args.l2t_max_total_nodes,
            max_time_seconds=args.l2t_max_time_seconds,
            x_fmt_default=args.l2t_x_fmt_default,
            x_eva_default=args.l2t_x_eva_default,
        )
        process_instance = L2TProcessor(llm_client=llm_client, config=l2t_config)
    else:
        # This case should ideally be caught by argparse choices, but as a safeguard:
        logging.critical(f"Invalid process type specified: {args.process_type}. Exiting.")
        sys.exit(1)

    # Parse trigger mode
    try:
        trigger_mode_enum_val = AotTriggerMode(args.trigger_mode)
    except ValueError:
        # This case should also ideally be caught by argparse choices
        logging.critical(f"Invalid trigger mode string '{args.trigger_mode}'. Valid values are: {[m.value for m in AotTriggerMode]}. Exiting.")
        sys.exit(1)

    # Instantiate CoTOrchestrator
    # The CoTOrchestrator creates its own LLMClient instance using the api_key.
    # The process_instance (AoTProcessor or L2TProcessor) also has its own LLMClient.
    # This is acceptable as per current design where CoTOrchestrator manages its client for assessment/direct calls,
    # and the process uses its client for its internal operations.
    cot_orchestrator = CoTOrchestrator(
        process=process_instance,
        trigger_mode=trigger_mode_enum_val,
        direct_oneshot_model_names=args.oneshot_models,
        direct_oneshot_temperature=args.oneshot_temp,
        assessment_model_names=args.assessment_models,
        assessment_temperature=args.assessment_temp,
        api_key=api_key, 
        use_heuristic_shortcut=not args.disable_heuristic
    )
    
    # Solve the problem
    solution, overall_summary_str = cot_orchestrator.solve(problem_text)

    # Print results
    print(overall_summary_str) 

    # Determine exit status
    if solution.final_answer and not solution.final_answer.startswith("Error:"):
        logging.info(f"{args.process_type.upper()} orchestration completed. Final answer obtained.")
        # Further check if the CoT process itself (if run) also reported success.
        # This is for more nuanced success reporting. The main check is if a final_answer was produced.
        if solution.cot_result and not solution.cot_result.succeeded:
            logging.warning(
                f"The {args.process_type.upper()} process was run but reported failure "
                f"(Reason: {solution.cot_result.error_message or 'Unknown'}). "
                "However, the orchestrator might have obtained a final answer via fallback."
            )
            # Depending on desired strictness, one might choose to exit with 1 here
            # if the CoT process failing is considered a critical error, even if fallback worked.
            # For now, if a final_answer (non-error) exists, we exit 0.
        sys.exit(0)
    else:
        if solution.final_answer and solution.final_answer.startswith("Error:"):
            logging.error(f"{args.process_type.upper()} orchestration resulted in an error state for the final answer: {solution.final_answer}")
        else: # No final_answer at all
            logging.error(f"{args.process_type.upper()} orchestration did not produce a final answer.")
        
        # Check for specific CoT process errors if it was run and failed, and no fallback answer was generated
        if solution.cot_result and not solution.cot_result.succeeded and not solution.final_answer:
            logging.error(f"The underlying {args.process_type.upper()} process reported failure: {solution.cot_result.error_message or 'Unknown'}")
        elif solution.process_failed_and_fell_back and not solution.final_answer : # Fallback also failed
            logging.error("The CoT process failed, and the subsequent fallback to one-shot also failed to produce an answer.")

        sys.exit(1)

if __name__ == "__main__":
    main()
