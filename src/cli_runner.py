import sys
import os
import argparse
import logging
from typing import Optional, Union

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llm_client import LLMClient
from src.llm_config import LLMConfig
from src.prompt_generator import PromptGenerator

from src.aot.enums import AotTriggerMode
from src.aot.dataclasses import AoTRunnerConfig, Solution as AoTSolution
from src.aot.orchestrator import InteractiveAoTOrchestrator
from src.aot.processor import AoTProcessor
from src.aot.constants import (
    DEFAULT_MAIN_MODEL_NAMES as DEFAULT_AOT_MAIN_MODEL_NAMES, 
    DEFAULT_SMALL_MODEL_NAMES as DEFAULT_AOT_ASSESSMENT_MODEL_NAMES, 
    DEFAULT_MAX_STEPS as DEFAULT_AOT_MAX_STEPS, 
    DEFAULT_MAX_TIME_SECONDS as DEFAULT_AOT_MAX_TIME_SECONDS, 
    DEFAULT_NO_PROGRESS_LIMIT as DEFAULT_AOT_NO_PROGRESS_LIMIT, 
    DEFAULT_MAIN_TEMPERATURE as DEFAULT_AOT_MAIN_TEMPERATURE, 
    DEFAULT_ASSESSMENT_TEMPERATURE as DEFAULT_AOT_ASSESSMENT_TEMPERATURE
)

# ToT Imports
from src.tot.enums import ToTSearchStrategy, ToTScoringMethod
from src.tot.dataclasses import ToTConfig, ToTSolution
from src.tot.orchestrator import InteractiveToTOrchestrator, ToTProcess, ToTTriggerMode # ToTTriggerMode is alias from aot.enums
from src.tot.constants import (
    DEFAULT_TOT_THOUGHT_GENERATION_MODEL_NAMES,
    DEFAULT_TOT_EVALUATION_MODEL_NAMES,
    DEFAULT_TOT_THOUGHT_GENERATION_TEMPERATURE,
    DEFAULT_TOT_EVALUATION_TEMPERATURE,
    DEFAULT_TOT_DIRECT_ONESHOT_TEMPERATURE, # Added in this subtask
    DEFAULT_TOT_ASSESSMENT_TEMPERATURE,     # Added in this subtask
    DEFAULT_K_THOUGHTS, DEFAULT_B_BEAM_WIDTH, DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_TOTAL_THOUGHTS_GENERATED, DEFAULT_MAX_TIME_SECONDS
)

from src.l2t.enums import L2TTriggerMode
from src.l2t.orchestrator import L2TOrchestrator
from src.l2t.processor import L2TProcessor
from src.l2t.dataclasses import L2TConfig, L2TSolution, L2TModelConfigs # Re-added L2TModelConfigs
from src.l2t_orchestrator_utils.summary_generator import L2TSummaryGenerator
from src.l2t.constants import (
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
from src.heuristic_detector import HeuristicDetector

# Hybrid Process Imports
from src.hybrid.enums import HybridTriggerMode
from src.hybrid.orchestrator import HybridOrchestrator, HybridProcess
from src.hybrid.dataclasses import HybridConfig, HybridSolution
from src.hybrid.constants import (
    DEFAULT_HYBRID_REASONING_MODEL_NAMES,
    DEFAULT_HYBRID_RESPONSE_MODEL_NAMES,
    DEFAULT_HYBRID_REASONING_TEMPERATURE,
    DEFAULT_HYBRID_RESPONSE_TEMPERATURE,
    DEFAULT_HYBRID_REASONING_PROMPT_TEMPLATE,
    DEFAULT_HYBRID_REASONING_COMPLETE_TOKEN,
    DEFAULT_HYBRID_RESPONSE_PROMPT_TEMPLATE,
    DEFAULT_HYBRID_MAX_REASONING_TOKENS,
    DEFAULT_HYBRID_MAX_RESPONSE_TOKENS,
    DEFAULT_HYBRID_ASSESSMENT_MODEL_NAMES,
    DEFAULT_HYBRID_ASSESSMENT_TEMPERATURE,
    DEFAULT_HYBRID_ONESHOT_MODEL_NAMES,
    DEFAULT_HYBRID_ONESHOT_TEMPERATURE,
)


# Helper to define new direct processing modes
DIRECT_AOT_MODE = "aot-direct"
DIRECT_L2T_MODE = "l2t-direct"
DIRECT_HYBRID_MODE = "hybrid-direct"
DIRECT_TOT_MODE = "tot-direct" # New ToT direct mode

def main():
    parser = argparse.ArgumentParser(
        description="CLI Runner for Algorithm of Thought (AoT), Learn-to-Think (L2T), and Hybrid reasoning processes.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    problem_group = parser.add_mutually_exclusive_group(required=True)
    problem_group.add_argument("--problem", "-p", type=str, help="Problem/question to solve.")
    problem_group.add_argument("--problem-filename", type=str, help="File containing the problem.")

    # Redefine choices for processing mode to be more explicit
    all_modes = sorted(list(set([
        "aot-always", "aot-assess-first", "aot-never",
        "l2t",
        "hybrid-always", "hybrid-assess-first", "hybrid-never",
        "tot-always", "tot-assess-first", "tot-never", # Orchestrator modes for ToT
        DIRECT_AOT_MODE, DIRECT_L2T_MODE, DIRECT_HYBRID_MODE, DIRECT_TOT_MODE
    ])))

    parser.add_argument(
        "--processing-mode", "--mode", dest="processing_mode", type=str,
        choices=all_modes,
        default="aot-assess-first",
        help=(f"Processing mode (default: aot-assess-first).\n"
              f"Available modes: {', '.join(all_modes)}\n"
              f"  AoT Orchestrator Modes: 'aot-always', 'aot-assess-first', 'aot-never'\n"
              f"  L2T Orchestrator Mode: 'l2t'\n"
              f"  Hybrid Orchestrator Modes: 'hybrid-always', 'hybrid-assess-first', 'hybrid-never'\n"
              f"  ToT Orchestrator Modes: 'tot-always', 'tot-assess-first', 'tot-never'\n"
              f"  Direct Process Modes: '{DIRECT_AOT_MODE}', '{DIRECT_L2T_MODE}', '{DIRECT_HYBRID_MODE}', '{DIRECT_TOT_MODE}'")
    )

    parser.add_argument("--enable-rate-limiting", action="store_true", help="Enable rate limiting for LLM calls.")
    parser.add_argument("--enable-audit-logging", action="store_true", help="Enable audit logging for LLM prompts and responses.")

    # AoT Configuration Group
    aot_group = parser.add_argument_group('AoT Process Configuration')
    aot_group.add_argument("--aot-main-models", type=str, nargs='+', default=DEFAULT_AOT_MAIN_MODEL_NAMES,
                           help=f"Main LLM(s) for AoT/ONESHOT. Default: {' '.join(DEFAULT_AOT_MAIN_MODEL_NAMES)}")
    aot_group.add_argument("--aot-main-temp", type=float, default=DEFAULT_AOT_MAIN_TEMPERATURE,
                           help=f"Temperature for AoT main LLM(s). Default: {DEFAULT_AOT_MAIN_TEMPERATURE}")
    aot_group.add_argument("--aot-assess-models", type=str, nargs='+', default=DEFAULT_AOT_ASSESSMENT_MODEL_NAMES,
                           help=f"Small LLM(s) for AoT assessment. Default: {' '.join(DEFAULT_AOT_ASSESSMENT_MODEL_NAMES)}")
    aot_group.add_argument("--aot-assess-temp", type=float, default=DEFAULT_AOT_ASSESSMENT_TEMPERATURE,
                           help=f"Temperature for AoT assessment LLM(s). Default: {DEFAULT_AOT_ASSESSMENT_TEMPERATURE}")
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
    aot_group.add_argument("--aot-disable-heuristic", action="store_true",
                           help="Disable the local heuristic analysis for AoT complexity assessment, always using the LLM for assessment.")

    # L2T Configuration Group
    l2t_group = parser.add_argument_group('L2T Process Configuration')
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
    l2t_group.add_argument("--l2t-x-fmt", dest="l2t_x_fmt_default", type=str, default=DEFAULT_L2T_X_FMT_DEFAULT,
                           help="L2T default format constraints string.")
    l2t_group.add_argument("--l2t-x-eva", dest="l2t_x_eva_default", type=str, default=DEFAULT_L2T_X_EVA_DEFAULT,
                           help="L2T default evaluation criteria string.")

    # Hybrid Configuration Group
    hybrid_group = parser.add_argument_group('Hybrid Process Configuration')
    hybrid_group.add_argument("--hybrid-reasoning-models", type=str, nargs='+', default=DEFAULT_HYBRID_REASONING_MODEL_NAMES,
                               help=f"Reasoning LLM(s) for Hybrid process. Default: {' '.join(DEFAULT_HYBRID_REASONING_MODEL_NAMES)}")
    hybrid_group.add_argument("--hybrid-reasoning-temp", type=float, default=DEFAULT_HYBRID_REASONING_TEMPERATURE,
                               help=f"Temperature for Hybrid reasoning LLM(s). Default: {DEFAULT_HYBRID_REASONING_TEMPERATURE}")
    hybrid_group.add_argument("--hybrid-response-models", type=str, nargs='+', default=DEFAULT_HYBRID_RESPONSE_MODEL_NAMES,
                               help=f"Response LLM(s) for Hybrid process. Default: {' '.join(DEFAULT_HYBRID_RESPONSE_MODEL_NAMES)}")
    hybrid_group.add_argument("--hybrid-response-temp", type=float, default=DEFAULT_HYBRID_RESPONSE_TEMPERATURE,
                               help=f"Temperature for Hybrid response LLM(s). Default: {DEFAULT_HYBRID_RESPONSE_TEMPERATURE}")
    hybrid_group.add_argument("--hybrid-reasoning-prompt", type=str, default=DEFAULT_HYBRID_REASONING_PROMPT_TEMPLATE,
                               help="Prompt template for Hybrid reasoning stage.")
    hybrid_group.add_argument("--hybrid-reasoning-token", type=str, default=DEFAULT_HYBRID_REASONING_COMPLETE_TOKEN,
                               help="Token indicating completion of reasoning in Hybrid.")
    hybrid_group.add_argument("--hybrid-response-prompt", type=str, default=DEFAULT_HYBRID_RESPONSE_PROMPT_TEMPLATE,
                               help="Prompt template for Hybrid response stage.")
    hybrid_group.add_argument("--hybrid-max-reasoning-tokens", type=int, default=DEFAULT_HYBRID_MAX_REASONING_TOKENS,
                               help=f"Max tokens for Hybrid reasoning stage. Default: {DEFAULT_HYBRID_MAX_REASONING_TOKENS}")
    hybrid_group.add_argument("--hybrid-max-response-tokens", type=int, default=DEFAULT_HYBRID_MAX_RESPONSE_TOKENS,
                               help=f"Max tokens for Hybrid response stage. Default: {DEFAULT_HYBRID_MAX_RESPONSE_TOKENS}")
    hybrid_group.add_argument("--hybrid-assess-models", type=str, nargs='+', default=DEFAULT_HYBRID_ASSESSMENT_MODEL_NAMES,
                               help=f"Assessment LLM(s) for Hybrid (if assess_first). Default: {' '.join(DEFAULT_HYBRID_ASSESSMENT_MODEL_NAMES)}")
    hybrid_group.add_argument("--hybrid-assess-temp", type=float, default=DEFAULT_HYBRID_ASSESSMENT_TEMPERATURE,
                               help=f"Temperature for Hybrid assessment LLM(s). Default: {DEFAULT_HYBRID_ASSESSMENT_TEMPERATURE}")
    hybrid_group.add_argument("--hybrid-oneshot-models", type=str, nargs='+', default=DEFAULT_HYBRID_ONESHOT_MODEL_NAMES,
                               help=f"Fallback one-shot LLM(s) for Hybrid. Default: {' '.join(DEFAULT_HYBRID_ONESHOT_MODEL_NAMES)}")
    hybrid_group.add_argument("--hybrid-oneshot-temp", type=float, default=DEFAULT_HYBRID_ONESHOT_TEMPERATURE,
                               help=f"Temperature for Hybrid fallback one-shot. Default: {DEFAULT_HYBRID_ONESHOT_TEMPERATURE}")
    hybrid_group.add_argument("--hybrid-disable-heuristic", action="store_true",
                           help="Disable the local heuristic analysis for Hybrid complexity assessment, always using the LLM for assessment.")

    # ToT Configuration Group
    tot_group = parser.add_argument_group('ToT Process Configuration')
    tot_group.add_argument("--tot-thought-models", type=str, nargs='+', default=DEFAULT_TOT_THOUGHT_GENERATION_MODEL_NAMES,
                           help=f"LLM(s) for ToT thought generation. Default: {' '.join(DEFAULT_TOT_THOUGHT_GENERATION_MODEL_NAMES)}")
    tot_group.add_argument("--tot-thought-temp", type=float, default=DEFAULT_TOT_THOUGHT_GENERATION_TEMPERATURE,
                           help=f"Temperature for ToT thought generation. Default: {DEFAULT_TOT_THOUGHT_GENERATION_TEMPERATURE}")
    tot_group.add_argument("--tot-eval-models", type=str, nargs='+', default=DEFAULT_TOT_EVALUATION_MODEL_NAMES,
                           help=f"LLM(s) for ToT state evaluation. Default: {' '.join(DEFAULT_TOT_EVALUATION_MODEL_NAMES)}")
    tot_group.add_argument("--tot-eval-temp", type=float, default=DEFAULT_TOT_EVALUATION_TEMPERATURE,
                           help=f"Temperature for ToT state evaluation. Default: {DEFAULT_TOT_EVALUATION_TEMPERATURE}")

    tot_group.add_argument("--tot-direct-oneshot-temp", type=float, default=DEFAULT_TOT_DIRECT_ONESHOT_TEMPERATURE,
                            help=f"Temperature for ToT Orchestrator's direct one-shot or ToTProcess fallback. Default: {DEFAULT_TOT_DIRECT_ONESHOT_TEMPERATURE}")
    tot_group.add_argument("--tot-assess-models", type=str, nargs='+', default=DEFAULT_AOT_ASSESSMENT_MODEL_NAMES,
                            help=f"LLM(s) for ToT complexity assessment. Default: {' '.join(DEFAULT_AOT_ASSESSMENT_MODEL_NAMES)}")
    tot_group.add_argument("--tot-assess-temp", type=float, default=DEFAULT_TOT_ASSESSMENT_TEMPERATURE,
                            help=f"Temperature for ToT complexity assessment. Default: {DEFAULT_TOT_ASSESSMENT_TEMPERATURE}")

    tot_group.add_argument("--tot-k-thoughts", type=int, default=DEFAULT_K_THOUGHTS,
                           help=f"Number of thoughts to generate per expansion (K). Default: {DEFAULT_K_THOUGHTS}")
    tot_group.add_argument("--tot-b-beam-width", type=int, default=DEFAULT_B_BEAM_WIDTH,
                           help=f"Beam width for Beam Search (B). Default: {DEFAULT_B_BEAM_WIDTH}")
    tot_group.add_argument("--tot-max-depth", type=int, default=DEFAULT_MAX_DEPTH,
                           help=f"Maximum depth of the thought tree. Default: {DEFAULT_MAX_DEPTH}")
    tot_group.add_argument("--tot-max-total-thoughts", type=int, default=DEFAULT_MAX_TOTAL_THOUGHTS_GENERATED,
                           help=f"Maximum total thoughts to generate in a run. Default: {DEFAULT_MAX_TOTAL_THOUGHTS_GENERATED}")
    tot_group.add_argument("--tot-max-time", type=int, default=DEFAULT_MAX_TIME_SECONDS,
                           help=f"Overall max time for a ToT run (seconds). Default: {DEFAULT_MAX_TIME_SECONDS}s")
    tot_group.add_argument("--tot-search-strategy", type=str, choices=[s.value for s in ToTSearchStrategy], default=ToTSearchStrategy.BEAM.value,
                           help=f"Search strategy for ToT. Default: {ToTSearchStrategy.BEAM.value}")
    tot_group.add_argument("--tot-scoring-method", type=str, choices=[s.value for s in ToTScoringMethod], default=ToTScoringMethod.LLM.value,
                           help=f"Scoring method for ToT states. Default: {ToTScoringMethod.LLM.value}")
    tot_group.add_argument("--tot-disable-heuristic", action="store_true",
                            help="Disable local heuristic analysis for ToT complexity assessment (if assess-first mode).")

    args = parser.parse_args()

    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level_val = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(level=log_level_val, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s', stream=sys.stderr)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key: 
        logging.critical("OPENROUTER_API_KEY environment variable not set.")
        sys.exit(1)
    
    shared_llm_client = LLMClient(
        api_key=api_key,
        enable_rate_limiting=args.enable_rate_limiting,
        enable_audit_logging=args.enable_audit_logging
    )
    
    problem_text: str
    if args.problem_filename:
        try:
            with open(args.problem_filename, 'r', encoding='utf-8') as f: problem_text = f.read()
            logging.info(f"Successfully read problem from file: {args.problem_filename}")
        except Exception as e: 
            logging.critical(f"Error reading problem file '{args.problem_filename}': {e}")
            sys.exit(1)
    else: 
        problem_text = args.problem
        if not problem_text:
            logging.critical("No problem text provided.")
            sys.exit(1)

    current_processing_mode = args.processing_mode
    solution: Optional[Union[AoTSolution, L2TSolution, HybridSolution, ToTSolution]] = None # Added ToTSolution
    overall_summary_str = ""

    heuristic_detector_instance: Optional[HeuristicDetector] = None
    needs_heuristic_detector = False
    if (current_processing_mode == "aot-assess-first" and not args.aot_disable_heuristic) or \
       (current_processing_mode == "hybrid-assess-first" and not args.hybrid_disable_heuristic) or \
       (current_processing_mode == "tot-assess-first" and not args.tot_disable_heuristic) or \
       (current_processing_mode == "l2t"): # L2T might also use it
        needs_heuristic_detector = True

    if needs_heuristic_detector:
        heuristic_detector_instance = HeuristicDetector()

    aot_pass_remaining_steps_float: Optional[float] = None
    if args.aot_pass_remaining_steps_pct is not None:
        aot_pass_remaining_steps_float = args.aot_pass_remaining_steps_pct / 100.0
    
    aot_runner_config = AoTRunnerConfig(
        main_model_names=args.aot_main_models,
        max_steps=args.aot_max_steps,
        max_reasoning_tokens=args.aot_max_reasoning_tokens,
        max_time_seconds=args.aot_max_time,
        no_progress_limit=args.aot_no_progress_limit,
        pass_remaining_steps_pct=aot_pass_remaining_steps_float
    )
    aot_main_llm_config = LLMConfig(temperature=args.aot_main_temp)
    aot_assessment_llm_config = LLMConfig(temperature=args.aot_assess_temp)

    l2t_config = L2TConfig(
        classification_model_names=args.l2t_classification_models,
        thought_generation_model_names=args.l2t_thought_gen_models,
        initial_prompt_model_names=args.l2t_initial_prompt_models,
        max_steps=args.l2t_max_steps,
        max_total_nodes=args.l2t_max_total_nodes,
        max_time_seconds=args.l2t_max_time_seconds,
        x_fmt_default=args.l2t_x_fmt_default,
        x_eva_default=args.l2t_x_eva_default,
    )
    l2t_model_configs = L2TModelConfigs(
        initial_thought_config=LLMConfig(temperature=args.l2t_initial_prompt_temp),
        node_classification_config=LLMConfig(temperature=args.l2t_classification_temp),
        node_thought_generation_config=LLMConfig(temperature=args.l2t_thought_gen_temp),
        orchestrator_oneshot_config=LLMConfig(temperature=args.l2t_initial_prompt_temp),
        summary_config=LLMConfig(temperature=args.l2t_initial_prompt_temp)
    )

    hybrid_config = HybridConfig(
        reasoning_model_name=args.hybrid_reasoning_models[0],
        reasoning_model_temperature=args.hybrid_reasoning_temp,
        reasoning_prompt_template=args.hybrid_reasoning_prompt,
        reasoning_complete_token=args.hybrid_reasoning_token,
        response_model_name=args.hybrid_response_models[0],
        response_model_temperature=args.hybrid_response_temp,
        response_prompt_template=args.hybrid_response_prompt,
        max_reasoning_tokens=args.hybrid_max_reasoning_tokens,
        max_response_tokens=args.hybrid_max_response_tokens
    )

    # ToT Configs
    tot_base_config = ToTConfig(
        thought_generation_model_names=args.tot_thought_models,
        evaluation_model_names=args.tot_eval_models,
        k_thoughts=args.tot_k_thoughts,
        b_beam_width=args.tot_b_beam_width,
        max_depth=args.tot_max_depth,
        max_total_thoughts_generated=args.tot_max_total_thoughts,
        max_time_seconds=args.tot_max_time,
        search_strategy=ToTSearchStrategy(args.tot_search_strategy),
        scoring_method=ToTScoringMethod(args.tot_scoring_method)
    )

    tot_thought_generation_llm_config = LLMConfig(temperature=args.tot_thought_temp)
    tot_evaluation_llm_config = LLMConfig(temperature=args.tot_eval_temp)
    tot_direct_oneshot_llm_config = LLMConfig(temperature=args.tot_direct_oneshot_temp)
    tot_assessment_llm_config = LLMConfig(temperature=args.tot_assess_temp)
    tot_orchestrator_direct_oneshot_models = args.tot_thought_models

    if current_processing_mode == DIRECT_AOT_MODE:
        logging.info("Direct AoTProcessor mode selected.")
        aot_processor_instance = AoTProcessor(
            llm_client=shared_llm_client,
            runner_config=aot_runner_config,
            llm_config=aot_main_llm_config
        )
        aot_result, overall_summary_str = aot_processor_instance.run(problem_text)
        solution = AoTSolution(aot_result=aot_result, final_answer=aot_result.final_answer, reasoning_trace=aot_result.reasoning_trace, aot_summary_output=overall_summary_str)
        print("\nDirect AoTProcessor Execution Summary:")
        print(overall_summary_str if overall_summary_str else "No summary returned.")
        if not (solution and solution.final_answer):
            error_detail = solution.aot_result.final_answer if solution and solution.aot_result and not solution.aot_result.succeeded else "Unknown error"
            logging.error(f"Direct AoTProcessor did not produce a final answer. Error: {error_detail}")
            sys.exit(1)

    elif current_processing_mode == DIRECT_L2T_MODE:
        logging.info("Direct L2TProcessor mode selected.")
        l2t_processor_instance = L2TProcessor(
            api_key=api_key, # Reverted to api_key
            l2t_config=l2t_config,
            initial_thought_llm_config=l2t_model_configs.initial_thought_config,
            node_processor_llm_config=l2t_model_configs.node_thought_generation_config,
            enable_rate_limiting=args.enable_rate_limiting,
            enable_audit_logging=args.enable_audit_logging
        )
        l2t_result = l2t_processor_instance.run(problem_text)
        summary_generator = L2TSummaryGenerator(trigger_mode=L2TTriggerMode.ALWAYS_L2T, use_heuristic_shortcut=False)
        overall_summary_str = summary_generator.generate_l2t_summary_from_result(l2t_result)
        solution = L2TSolution(l2t_result=l2t_result, final_answer=l2t_result.final_answer)
        print("\nDirect L2TProcessor Execution Summary:")
        print(overall_summary_str if overall_summary_str else "No summary returned.")
        if not (solution and solution.final_answer):
            error_detail = solution.l2t_result.error_message if solution and solution.l2t_result else "Unknown error"
            logging.error(f"Direct L2TProcessor did not produce a final answer. Error: {error_detail}")
            sys.exit(1)

    elif current_processing_mode == DIRECT_HYBRID_MODE:
        logging.info("Direct HybridProcess mode selected.")
        hybrid_direct_process = HybridProcess(
            hybrid_config=hybrid_config,
            direct_oneshot_model_names=args.hybrid_oneshot_models,
            direct_oneshot_temperature=args.hybrid_oneshot_temp,
            api_key=api_key, # Reverted to api_key
            enable_rate_limiting=args.enable_rate_limiting,
            enable_audit_logging=args.enable_audit_logging
        )
        hybrid_direct_process.execute(problem_description=problem_text, model_name="direct_hybrid")
        solution, overall_summary_str = hybrid_direct_process.get_result()
        print("\nDirect HybridProcess Execution Summary:")
        print(overall_summary_str if overall_summary_str else "No summary from HybridProcess.")
        if not (solution and solution.final_answer):
            error_detail = solution.hybrid_result.error_message if solution and solution.hybrid_result else "Unknown error"
            logging.error(f"Direct HybridProcess did not produce a final answer. Error: {error_detail}")
            sys.exit(1)

    elif current_processing_mode == DIRECT_TOT_MODE:
        logging.info("Direct ToTProcessor mode selected.")
        tot_process_direct_fallback_models = args.tot_thought_models
        direct_tot_process_instance = ToTProcess(
            llm_client=shared_llm_client,
            tot_config=tot_base_config,
            thought_generation_llm_config=tot_thought_generation_llm_config,
            evaluation_llm_config=tot_evaluation_llm_config,
            direct_oneshot_llm_config=tot_direct_oneshot_llm_config,
            direct_oneshot_model_names=tot_process_direct_fallback_models
        )
        direct_tot_process_instance.execute(problem_text, model_name="direct_tot")
        solution, overall_summary_str = direct_tot_process_instance.get_result()

        print("\nDirect ToTProcess Execution Summary:")
        print(overall_summary_str if overall_summary_str else "No summary returned by ToTProcess.")
        if not (solution and solution.final_answer and not solution.final_answer.startswith("Error:")):
            error_detail = "Unknown error"
            if solution and solution.tot_result and solution.tot_result.error_message:
                error_detail = solution.tot_result.error_message
            elif solution and solution.final_answer and solution.final_answer.startswith("Error:"):
                error_detail = solution.final_answer
            logging.error(f"Direct ToTProcess did not produce a final answer or failed. Error: {error_detail}")
            sys.exit(1)

    elif current_processing_mode == "l2t":
        logging.info("L2TOrchestrator mode selected.")
        l2t_orchestrator = L2TOrchestrator(
            trigger_mode=L2TTriggerMode.ALWAYS_L2T, # L2T Orchestrator always runs L2T
            l2t_config=l2t_config,
            model_configs=l2t_model_configs,
            api_key=api_key,
            use_heuristic_shortcut=True, 
            heuristic_detector=heuristic_detector_instance,
            enable_rate_limiting=args.enable_rate_limiting,
            enable_audit_logging=args.enable_audit_logging
        )
        solution, overall_summary_str = l2t_orchestrator.solve(problem_text)
        print("\nL2T Orchestrator Process Summary:")
        print(overall_summary_str)
        if not (solution and solution.final_answer):
            error_detail = "Unknown error"
            if solution and solution.l2t_result and solution.l2t_result.error_message: error_detail = solution.l2t_result.error_message
            elif solution and solution.l2t_result and not solution.l2t_result.succeeded: error_detail = "L2T process reported failure without specific message."
            logging.error(f"L2T process (via orchestrator) did not produce a final answer or did not succeed. Error: {error_detail}")
            sys.exit(1)

    elif current_processing_mode.startswith("hybrid-"): # Handle all hybrid orchestrator modes
        try:
            # Map the explicit mode string to the HybridTriggerMode enum
            if current_processing_mode == "hybrid-always":
                hybrid_mode_enum_val = HybridTriggerMode.ALWAYS_HYBRID
            elif current_processing_mode == "hybrid-assess-first":
                hybrid_mode_enum_val = HybridTriggerMode.ASSESS_FIRST_HYBRID
            elif current_processing_mode == "hybrid-never":
                hybrid_mode_enum_val = HybridTriggerMode.NEVER_HYBRID
            else:
                raise ValueError(f"Unknown hybrid mode: {current_processing_mode}")
        except ValueError as e:
            logging.critical(f"Invalid Hybrid mode string '{current_processing_mode}' for Hybrid Orchestrator. Error: {e}. Exiting.")
            sys.exit(1)

        logging.info(f"HybridOrchestrator mode selected: {hybrid_mode_enum_val.value}")
        hybrid_orchestrator = HybridOrchestrator(
            trigger_mode=hybrid_mode_enum_val,
            hybrid_config=hybrid_config,
            direct_oneshot_model_names=args.hybrid_oneshot_models,
            direct_oneshot_temperature=args.hybrid_oneshot_temp,
            assessment_model_names=args.hybrid_assess_models,
            assessment_temperature=args.hybrid_assess_temp,
            api_key=api_key,
            use_heuristic_shortcut=not args.hybrid_disable_heuristic,
            heuristic_detector=heuristic_detector_instance,
            enable_rate_limiting=args.enable_rate_limiting,
            enable_audit_logging=args.enable_audit_logging
        )
        solution, overall_summary_str = hybrid_orchestrator.solve(problem_text)
        print("\nHybrid Orchestrator Summary:")
        print(overall_summary_str)
        if not (solution and solution.final_answer):
            error_detail = "Unknown error"
            if solution and solution.hybrid_result and solution.hybrid_result.error_message: error_detail = solution.hybrid_result.error_message
            logging.error(f"Hybrid Orchestrator process did not produce a final answer. Error: {error_detail}")
            if hybrid_mode_enum_val != HybridTriggerMode.NEVER_HYBRID: sys.exit(1)

    elif current_processing_mode.startswith("aot-"): # Handle all AoT orchestrator modes
        try:
            # Map the explicit mode string to the AotTriggerMode enum
            if current_processing_mode == "aot-always":
                aot_mode_enum_val = AotTriggerMode.ALWAYS_AOT
            elif current_processing_mode == "aot-assess-first":
                aot_mode_enum_val = AotTriggerMode.ASSESS_FIRST
            elif current_processing_mode == "aot-never":
                aot_mode_enum_val = AotTriggerMode.NEVER_AOT
            else:
                raise ValueError(f"Unknown AoT mode: {current_processing_mode}")
        except ValueError as e:
            logging.critical(f"Invalid AoT mode string '{current_processing_mode}' for AoT Orchestrator path. Error: {e}. Exiting.")
            sys.exit(1)

        logging.info(f"InteractiveAoTOrchestrator mode selected: {aot_mode_enum_val.value}")
        aot_orchestrator = InteractiveAoTOrchestrator(
            llm_client=shared_llm_client,
            trigger_mode=aot_mode_enum_val,
            aot_config=aot_runner_config,
            direct_oneshot_llm_config=aot_main_llm_config,
            assessment_llm_config=aot_assessment_llm_config,
            aot_main_llm_config=aot_main_llm_config, # This is for AoTProcess internal to orchestrator
            direct_oneshot_model_names=args.aot_main_models, 
            assessment_model_names=args.aot_assess_models,
            use_heuristic_shortcut=not args.aot_disable_heuristic,
            heuristic_detector=heuristic_detector_instance,
            enable_rate_limiting=args.enable_rate_limiting,
            enable_audit_logging=args.enable_audit_logging
        )
        solution, overall_summary_str = aot_orchestrator.solve(problem_text) 
        print("\nInteractive AoT Orchestrator Summary:")
        print(overall_summary_str) 
        if not (solution and solution.final_answer):
            logging.error("Interactive AoT Orchestrator process did not produce a final answer.")
            if aot_mode_enum_val != AotTriggerMode.NEVER_AOT: sys.exit(1)

    elif current_processing_mode.startswith("tot-"):
        try:
            if current_processing_mode == "tot-always":
                tot_mode_enum_val = ToTTriggerMode.ALWAYS_AOT # Using AOT enum value as per orchestrator alias
            elif current_processing_mode == "tot-assess-first":
                tot_mode_enum_val = ToTTriggerMode.ASSESS_FIRST
            elif current_processing_mode == "tot-never":
                tot_mode_enum_val = ToTTriggerMode.NEVER_AOT
            else:
                raise ValueError(f"Unknown ToT mode: {current_processing_mode}")
        except ValueError as e:
            logging.critical(f"Invalid ToT mode string '{current_processing_mode}'. Error: {e}. Exiting.")
            sys.exit(1)

        logging.info(f"InteractiveToTOrchestrator mode selected: {tot_mode_enum_val.value}")
        tot_orchestrator = InteractiveToTOrchestrator(
            llm_client=shared_llm_client,
            trigger_mode=tot_mode_enum_val,
            tot_config=tot_base_config,
            thought_generation_llm_config=tot_thought_generation_llm_config,
            evaluation_llm_config=tot_evaluation_llm_config,
            direct_oneshot_llm_config=tot_direct_oneshot_llm_config,
            assessment_llm_config=tot_assessment_llm_config,
            direct_oneshot_model_names=tot_orchestrator_direct_oneshot_models,
            assessment_model_names=args.tot_assess_models,
            use_heuristic_shortcut=not args.tot_disable_heuristic,
            heuristic_detector=heuristic_detector_instance
        )
        solution, overall_summary_str = tot_orchestrator.solve(problem_description)
        print("\nInteractive ToT Orchestrator Summary:")
        print(overall_summary_str)
        if not (solution and solution.final_answer and not solution.final_answer.startswith("Error:")):
            error_detail = "Unknown error"
            if solution and solution.tot_result and solution.tot_result.error_message:
                error_detail = solution.tot_result.error_message
            elif solution and solution.final_answer and solution.final_answer.startswith("Error:"):
                error_detail = solution.final_answer
            logging.error(f"Interactive ToT Orchestrator process did not produce a final answer or failed. Error: {error_detail}")
            if tot_mode_enum_val != ToTTriggerMode.NEVER_AOT:
                 sys.exit(1)
    else:
        logging.critical(f"Unknown or unhandled processing mode: {current_processing_mode}. Exiting.")
        sys.exit(1)

    if solution and solution.final_answer:
        logging.info(f"Process '{current_processing_mode}' completed. Final Answer will be printed below.")
        print("\n" + "="*20 + " FINAL ANSWER " + "="*20 + "\n")
        print(solution.final_answer)
        print("="*54 + "\n")
    else:
        # This specific else branch might be redundant if all paths above sys.exit(1) on failure
        logging.error(f"Process '{current_processing_mode}' concluded without a final answer.")

if __name__ == "__main__":
    main()
