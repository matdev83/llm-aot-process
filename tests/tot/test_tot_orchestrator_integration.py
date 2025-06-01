import pytest
from unittest.mock import MagicMock, patch

from src.llm_client import LLMClient
from src.llm_config import LLMConfig
from src.tot.orchestrator import ToTProcess, InteractiveToTOrchestrator, ToTTriggerMode # Using aliased ToTTriggerMode
from src.tot.dataclasses import ToTConfig, ToTSolution, ToTResult, LLMCallStats
from src.tot.processor import ToTProcessor
from src.aot.enums import AssessmentDecision # For orchestrator tests

# --- Fixtures ---
@pytest.fixture
def mock_llm_client_for_orchestrator() -> MagicMock:
    client = MagicMock(spec=LLMClient)
    client.call.return_value = ("Orchestrator Mocked LLM Response", LLMCallStats(completion_tokens=5, prompt_tokens=2, model_name="orch/mock"))
    return client

@pytest.fixture
def default_tot_config_for_orchestrator() -> ToTConfig:
    return ToTConfig(k_thoughts=1, max_depth=1, max_total_thoughts_generated=5) # Simplified config

@pytest.fixture
def base_llm_config() -> LLMConfig:
    return LLMConfig(temperature=0.1)

@pytest.fixture
def mock_tot_processor() -> MagicMock:
    processor = MagicMock(spec=ToTProcessor)
    # Default successful result from ToTProcessor.run()
    processor.run.return_value = (
        ToTResult(succeeded=True, final_answer="Mocked ToT success", total_thoughts_generated=1, total_nodes_evaluated=1),
        "Mocked ToTProcessor summary"
    )
    return processor

# --- Tests for ToTProcess ---
class TestToTProcess:
    @patch('src.tot.processor.ToTProcessor') # Patch the ToTProcessor class where it's imported by ToTProcess
    def test_tot_process_success(self, MockToTProcessor, mock_llm_client_for_orchestrator, default_tot_config_for_orchestrator, base_llm_config):
        mock_processor_instance = MockToTProcessor.return_value # This is the instance used by ToTProcess
        mock_processor_instance.run.return_value = (
            ToTResult(succeeded=True, final_answer="Successful ToT run", best_solution_path=[MagicMock()]),
            "Processor summary for success."
        )

        tot_process = ToTProcess(
            llm_client=mock_llm_client_for_orchestrator,
            tot_config=default_tot_config_for_orchestrator,
            thought_generation_llm_config=base_llm_config,
            evaluation_llm_config=base_llm_config,
            direct_oneshot_llm_config=base_llm_config,
            direct_oneshot_model_names=["fallback/model"]
        )

        problem = "Test problem for ToTProcess success."
        tot_process.execute(problem, "test_model")
        solution, summary = tot_process.get_result()

        assert solution is not None
        assert solution.final_answer == "Successful ToT run"
        assert solution.tot_result.succeeded
        assert not solution.tot_failed_and_fell_back
        mock_processor_instance.run.assert_called_once_with(problem)
        mock_llm_client_for_orchestrator.call.assert_not_called() # Fallback should not be called
        assert "ToT Succeeded (Reported by ToT Processor): Yes" in summary
        assert "Processor summary for success." in summary


    @patch('src.tot.processor.ToTProcessor')
    def test_tot_process_failure_and_fallback(self, MockToTProcessor, mock_llm_client_for_orchestrator, default_tot_config_for_orchestrator, base_llm_config):
        mock_processor_instance = MockToTProcessor.return_value
        mock_processor_instance.run.return_value = (
            ToTResult(succeeded=False, error_message="ToT processor failed"),
            "Processor summary for failure."
        )

        # Mock the LLM client's call method for the fallback
        mock_llm_client_for_orchestrator.call.return_value = ("Fallback answer.", LLMCallStats(model_name="fallback/model", completion_tokens=10, prompt_tokens=10))

        tot_process = ToTProcess(
            llm_client=mock_llm_client_for_orchestrator,
            tot_config=default_tot_config_for_orchestrator,
            thought_generation_llm_config=base_llm_config,
            evaluation_llm_config=base_llm_config,
            direct_oneshot_llm_config=base_llm_config,
            direct_oneshot_model_names=["fallback/model"]
        )

        problem = "Test problem for ToTProcess failure."
        tot_process.execute(problem, "test_model")
        solution, summary = tot_process.get_result()

        assert solution is not None
        assert solution.final_answer == "Fallback answer."
        assert solution.tot_result is not None # ensure tot_result exists
        assert not solution.tot_result.succeeded
        assert solution.tot_failed_and_fell_back
        assert solution.fallback_call_stats is not None
        assert solution.fallback_call_stats.model_name == "fallback/model"
        mock_processor_instance.run.assert_called_once_with(problem)
        mock_llm_client_for_orchestrator.call.assert_called_once() # Fallback was called
        assert "ToT FAILED and Fell Back to One-Shot: Yes" in summary
        assert "Processor summary for failure." in summary

# --- Tests for InteractiveToTOrchestrator ---
class TestInteractiveToTOrchestrator:

    @patch('src.tot.orchestrator.ToTProcess') # Patch ToTProcess where InteractiveToTOrchestrator imports it
    def test_orchestrator_always_tot_mode(self, MockToTProcessClass, mock_llm_client_for_orchestrator, default_tot_config_for_orchestrator, base_llm_config):
        mock_tot_process_instance = MockToTProcessClass.return_value
        mock_tot_process_instance.get_result.return_value = (
            ToTSolution(final_answer="Always ToT success", tot_result=ToTResult(succeeded=True)), # Removed succeeded=True from ToTSolution directly
            "ToTProcess summary for always_tot"
        )

        orchestrator = InteractiveToTOrchestrator(
            llm_client=mock_llm_client_for_orchestrator,
            trigger_mode=ToTTriggerMode.ALWAYS_AOT, # Corresponds to ALWAYS_TOT
            tot_config=default_tot_config_for_orchestrator,
            thought_generation_llm_config=base_llm_config,
            evaluation_llm_config=base_llm_config,
            direct_oneshot_llm_config=base_llm_config,
            assessment_llm_config=base_llm_config,
            direct_oneshot_model_names=["direct/model"],
            assessment_model_names=["assess/model"]
        )

        problem = "Problem for ALWAYS_TOT."
        solution, summary = orchestrator.solve(problem)

        MockToTProcessClass.assert_called_once() # Ensure ToTProcess was instantiated
        mock_tot_process_instance.execute.assert_called_once_with(problem, "default_tot_model")
        assert solution.final_answer == "Always ToT success"
        assert "Orchestrator Trigger Mode: always" in summary # Value from AotTriggerMode
        assert "ToTProcess summary for always_tot" in summary


    def test_orchestrator_never_tot_mode(self, mock_llm_client_for_orchestrator, default_tot_config_for_orchestrator, base_llm_config):
        # Set up LLM client to return a specific response for direct one-shot
        mock_llm_client_for_orchestrator.call.return_value = ("Direct one-shot answer.", LLMCallStats(model_name="direct/model"))

        orchestrator = InteractiveToTOrchestrator(
            llm_client=mock_llm_client_for_orchestrator,
            trigger_mode=ToTTriggerMode.NEVER_AOT, # Corresponds to NEVER_TOT
            tot_config=default_tot_config_for_orchestrator,
            # ... other configs ...
            thought_generation_llm_config=base_llm_config, evaluation_llm_config=base_llm_config,
            direct_oneshot_llm_config=base_llm_config, assessment_llm_config=base_llm_config,
            direct_oneshot_model_names=["direct/model"], assessment_model_names=["assess/model"]
        )

        problem = "Problem for NEVER_TOT."
        solution, summary = orchestrator.solve(problem)

        mock_llm_client_for_orchestrator.call.assert_called_once_with(
            prompt=problem, models=["direct/model"], config=base_llm_config
        )
        assert solution.final_answer == "Direct one-shot answer."
        assert "Orchestrator Trigger Mode: never" in summary
        assert "Orchestrator Direct One-Shot Call" in summary

    @patch('src.complexity_assessor.ComplexityAssessor.assess')
    @patch('src.tot.orchestrator.ToTProcess')
    def test_orchestrator_assess_first_runs_tot(self, MockToTProcessClass, mock_assess_call, mock_llm_client_for_orchestrator, default_tot_config_for_orchestrator, base_llm_config):
        mock_assess_call.return_value = (AssessmentDecision.ADVANCED_REASONING, LLMCallStats(model_name="assess/model"))

        mock_tot_process_instance = MockToTProcessClass.return_value
        mock_tot_process_instance.get_result.return_value = (
            ToTSolution(final_answer="ToT after assessment", tot_result=ToTResult(succeeded=True)), # Removed succeeded=True
            "ToTProcess summary for assess_first_tot"
        )

        orchestrator = InteractiveToTOrchestrator(
            llm_client=mock_llm_client_for_orchestrator,
            trigger_mode=ToTTriggerMode.ASSESS_FIRST,
            tot_config=default_tot_config_for_orchestrator,
            # ... other configs ...
            thought_generation_llm_config=base_llm_config, evaluation_llm_config=base_llm_config,
            direct_oneshot_llm_config=base_llm_config, assessment_llm_config=base_llm_config,
            direct_oneshot_model_names=["direct/model"], assessment_model_names=["assess/model"]
        )

        problem = "Problem for ASSESS_FIRST (run ToT)."
        solution, summary = orchestrator.solve(problem)

        mock_assess_call.assert_called_once_with(problem)
        MockToTProcessClass.assert_called_once()
        mock_tot_process_instance.execute.assert_called_once_with(problem, "default_tot_model")
        assert solution.final_answer == "ToT after assessment"
        assert solution.assessment_decision == AssessmentDecision.ADVANCED_REASONING # Check field directly
        assert "Assessment Decision: ADVANCED_REASONING" in summary
        assert "ToTProcess summary for assess_first_tot" in summary

    @patch('src.complexity_assessor.ComplexityAssessor.assess')
    def test_orchestrator_assess_first_runs_oneshot(self, mock_assess_call, mock_llm_client_for_orchestrator, default_tot_config_for_orchestrator, base_llm_config):
        mock_assess_call.return_value = (AssessmentDecision.ONE_SHOT, LLMCallStats(model_name="assess/model"))
        mock_llm_client_for_orchestrator.call.return_value = ("One-shot after assessment.", LLMCallStats(model_name="direct/model"))

        orchestrator = InteractiveToTOrchestrator(
            llm_client=mock_llm_client_for_orchestrator,
            trigger_mode=ToTTriggerMode.ASSESS_FIRST,
            # ... other configs ...
            tot_config=default_tot_config_for_orchestrator,
            thought_generation_llm_config=base_llm_config, evaluation_llm_config=base_llm_config,
            direct_oneshot_llm_config=base_llm_config, assessment_llm_config=base_llm_config,
            direct_oneshot_model_names=["direct/model"], assessment_model_names=["assess/model"]
        )

        problem = "Problem for ASSESS_FIRST (run one-shot)."
        solution, summary = orchestrator.solve(problem)

        mock_assess_call.assert_called_once_with(problem)
        mock_llm_client_for_orchestrator.call.assert_called_once_with(
            prompt=problem, models=["direct/model"], config=base_llm_config
        )
        assert solution.final_answer == "One-shot after assessment."
        assert solution.assessment_decision == AssessmentDecision.ONE_SHOT # Check field directly
        assert "Assessment Decision: ONE_SHOT" in summary
        assert "Orchestrator Post-Assessment One-Shot Call" in summary
