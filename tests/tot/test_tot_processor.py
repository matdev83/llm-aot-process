import pytest
from unittest.mock import MagicMock, patch, call # Import call for checking call order/args

from src.llm_client import LLMClient
from src.llm_config import LLMConfig
from src.tot.processor import ToTProcessor
from src.tot.dataclasses import ToTConfig, ToTNode, ToTThought, ToTResult, LLMCallStats
from src.tot.enums import ToTSearchStrategy, ToTScoringMethod
from src.tot.prompt_generator import ToTPromptGenerator
from src.tot.response_parser import ToTResponseParser

# Default configs for testing
@pytest.fixture
def default_tot_config() -> ToTConfig:
    return ToTConfig(
        thought_generation_model_names=["test/model-thought"],
        evaluation_model_names=["test/model-eval"],
        k_thoughts=2,
        b_beam_width=2, # Relevant for beam search
        max_depth=3,
        max_total_thoughts_generated=20, # Limit for tests
        max_time_seconds=30, # Short time limit for tests
        search_strategy=ToTSearchStrategy.BFS # Default to BFS for some tests
    )

@pytest.fixture
def mock_llm_client() -> MagicMock:
    client = MagicMock(spec=LLMClient)
    # Default mock behavior for LLM calls
    client.call.return_value = ("Mocked LLM Response", LLMCallStats(completion_tokens=10, prompt_tokens=5, call_duration_seconds=0.1, model_name="test/mocked"))
    return client

@pytest.fixture
def tot_processor(mock_llm_client: MagicMock, default_tot_config: ToTConfig) -> ToTProcessor:
    return ToTProcessor(
        llm_client=mock_llm_client,
        config=default_tot_config,
        thought_generation_llm_config=LLMConfig(temperature=0.1),
        evaluation_llm_config=LLMConfig(temperature=0.1)
    )

class TestToTProcessor:

    def test_initialization(self, tot_processor: ToTProcessor, default_tot_config: ToTConfig):
        assert tot_processor.config == default_tot_config
        assert tot_processor.llm_client is not None
        assert isinstance(tot_processor.prompt_generator, ToTPromptGenerator)
        assert isinstance(tot_processor.response_parser, ToTResponseParser)

    @patch('src.tot.response_parser.ToTResponseParser.parse_generated_thoughts')
    @patch('src.tot.response_parser.ToTResponseParser.parse_evaluation_score')
    @patch('src.tot.response_parser.ToTResponseParser.parse_final_answer')
    def test_run_simple_success_bfs(self, mock_parse_final_answer, mock_parse_eval, mock_parse_thoughts, tot_processor: ToTProcessor, mock_llm_client: MagicMock):
        problem = "What is 2+2?"

        # Mock thought generation
        # Depth 0 (Root expansion)
        mock_parse_thoughts.side_effect = [
            [ToTThought(text="Thought 1.1 (d1)"), ToTThought(text="Thought 1.2 (d1)")], # For root node
            [ToTThought(text="Thought 2.1 (d2)"), ToTThought(text="Thought 2.2 (d2)")], # For child of Thought 1.1
            [ToTThought(text="Thought 3.1 (d2)"), ToTThought(text="Thought 3.2 (d2)")], # For child of Thought 1.2
            # ... further calls if depth was larger
        ]

        # Mock evaluation
        # Scores for children of root: (Thought 1.1), (Thought 1.2)
        # Scores for children of (Thought 1.1): (Thought 2.1), (Thought 2.2)
        # Scores for children of (Thought 1.2): (Thought 3.1), (Thought 3.2)
        mock_parse_eval.side_effect = [
            (7.0, "Eval for T1.1"), (6.0, "Eval for T1.2"),   # Evals for depth 1 nodes
            (8.0, "Eval for T2.1"), (5.0, "Eval for T2.2"),   # Evals for children of T1.1 (depth 2)
            (9.0, "Eval for T3.1 (best)"), (4.0, "Eval for T3.2"), # Evals for children of T1.2 (depth 2)
        ]

        # Mock final answer generation
        mock_parse_final_answer.return_value = "The final answer is 4."

        # Configure for BFS
        tot_processor.config.search_strategy = ToTSearchStrategy.BFS
        tot_processor.config.max_depth = 2 # Root (d0) -> Children (d1) -> Grandchildren (d2)

        result, summary = tot_processor.run(problem)

        assert result.succeeded
        assert result.final_answer == "The final answer is 4."
        assert result.best_solution_path is not None
        # Path should be Root -> Thought 1.2 -> Thought 3.1 (path to highest score 9.0)
        # Note: text might be slightly different if mock_parse_thoughts gives them unique stats/raw_responses
        assert len(result.best_solution_path) == 2 # Two thoughts in the path from root
        assert result.best_solution_path[0].text == "Thought 1.2 (d1)" # Parent of the best node
        assert result.best_solution_path[1].text == "Thought 3.1 (d2)" # The best node

        # Check number of LLM calls (approx)
        # Thought generations: 1 (root) + 2 (d1 children) = 3 calls
        # Evaluations: 2 (d1 children) + 4 (d2 children) = 6 calls
        # Final Answer: 1 call
        # Total = 3 + 6 + 1 = 10 calls
        assert mock_llm_client.call.call_count == 10
        assert result.total_nodes_evaluated == 6 # 2 at d1, 4 at d2
        assert "ToT Succeeded: True" in summary
        assert "Search Strategy: bfs" in summary


    @patch('src.tot.response_parser.ToTResponseParser.parse_generated_thoughts')
    @patch('src.tot.response_parser.ToTResponseParser.parse_evaluation_score')
    @patch('src.tot.response_parser.ToTResponseParser.parse_final_answer')
    def test_run_beam_search_selection(self, mock_parse_final_answer, mock_parse_eval, mock_parse_thoughts, tot_processor: ToTProcessor, mock_llm_client: MagicMock):
        problem = "Solve a puzzle."
        tot_processor.config.search_strategy = ToTSearchStrategy.BEAM
        tot_processor.config.k_thoughts = 3 # Generate 3 thoughts
        tot_processor.config.b_beam_width = 2 # Keep top 2
        tot_processor.config.max_depth = 2

        # Depth 0 (Root) -> 3 thoughts
        # Depth 1 (Beam) -> Top 2 of these 3 are expanded, each generating 3 thoughts

        mock_parse_thoughts.side_effect = [
            # For root (depth 0)
            [ToTThought(text="R_T1"), ToTThought(text="R_T2"), ToTThought(text="R_T3")],
            # For first chosen beam node (depth 1)
            [ToTThought(text="B1_T1"), ToTThought(text="B1_T2"), ToTThought(text="B1_T3")],
            # For second chosen beam node (depth 1)
            [ToTThought(text="B2_T1"), ToTThought(text="B2_T2"), ToTThought(text="B2_T3")],
        ]
        mock_parse_eval.side_effect = [
            # Evaluations for R_T1, R_T2, R_T3 (depth 1 nodes)
            (8.0, "Eval R_T1"), (9.0, "Eval R_T2 (Top)"), (7.0, "Eval R_T3"),
            # R_T2 (9.0) and R_T1 (8.0) should be chosen for the beam.

            # Evaluations for children of R_T2 (B1_T1 to B1_T3)
            (9.5, "Eval B1_T1 (Best overall)"), (8.5, "Eval B1_T2"), (7.5, "Eval B1_T3"),
            # Evaluations for children of R_T1 (B2_T1 to B2_T3)
            (8.2, "Eval B2_T1"), (7.2, "Eval B2_T2"), (6.2, "Eval B2_T3"),
        ]
        mock_parse_final_answer.return_value = "Beam search success."

        result, summary = tot_processor.run(problem)

        assert result.succeeded
        assert result.final_answer == "Beam search success."
        assert len(result.best_solution_path) == 2
        assert result.best_solution_path[0].text == "R_T2" # Parent of best leaf
        assert result.best_solution_path[1].text == "B1_T1" # Best leaf

        # LLM Calls:
        # Thoughts: 1 (root) + 2 (beam width) = 3
        # Evals: 3 (d1) + 2*3 (d2 children of beam) = 3 + 6 = 9
        # Final Answer: 1
        # Total = 3 + 9 + 1 = 13
        assert mock_llm_client.call.call_count == 13
        assert result.total_nodes_evaluated == 3 + (2 * 3) # 3 at d1, 6 at d2 (3 for each of 2 beam nodes)
        assert "Search Strategy: beam" in summary
        assert f"B (beam width): {tot_processor.config.b_beam_width}" in summary


    def test_run_max_depth_reached(self, tot_processor: ToTProcessor, mock_llm_client: MagicMock):
        problem = "Deep thought problem."
        tot_processor.config.max_depth = 1 # Only root and its direct children

        # Mock to generate just one thought to simplify
        # Need two side effects: one for root, one for its child (which won't be expanded)
        mock_llm_client.call.side_effect = [
            # Call 1: Thought gen for root
            ("Thought 1: RootChild1", LLMCallStats(model_name="m1")),
            # Call 2: Eval for RootChild1 (will become a leaf)
            ("Score: 7/10\nJustification: Good enough for depth 1.", LLMCallStats(model_name="m2")),
            # Call 3: Final answer from RootChild1
            ("Final Answer: Max depth answer.", LLMCallStats(model_name="m3")),
        ]
        # Need to refine mock_parse_thoughts and mock_parse_eval if using @patch for them.
        # For simplicity here, assuming direct LLM response parsing happens correctly.
        # To make this test more robust with the current structure, we should patch the parsers.
        with patch('src.tot.response_parser.ToTResponseParser.parse_generated_thoughts', return_value=[ToTThought(text="RootChild1")]) as mock_parse_thoughts, \
             patch('src.tot.response_parser.ToTResponseParser.parse_evaluation_score', return_value=(7.0, "Good enough for depth 1.")) as mock_parse_eval, \
             patch('src.tot.response_parser.ToTResponseParser.parse_final_answer', return_value="Max depth answer.") as mock_parse_final_ans:

            result, summary = tot_processor.run(problem)

            assert result.succeeded
            assert result.final_answer == "Max depth answer."
            assert len(result.best_solution_path) == 1
            assert result.best_solution_path[0].text == "RootChild1"
            assert result.total_nodes_evaluated == 1 # Only RootChild1 evaluated
            assert "Max Depth: 1" in summary
            assert mock_llm_client.call.call_count == 3


    def test_run_max_thoughts_generated_limit(self, tot_processor: ToTProcessor, mock_llm_client: MagicMock):
        problem = "Limit thoughts."
        tot_processor.config.max_total_thoughts_generated = 1 # Allow only 1 thought to be generated
        tot_processor.config.k_thoughts = 1 # Generate 1 per step
        tot_processor.config.max_depth = 5 # High depth, but thoughts limit should hit first

        mock_llm_client.call.side_effect = [
            ("Thought 1: The only thought.", LLMCallStats(model_name="m1")),
            ("Score: 8/10\nJustification: Only one, but good.", LLMCallStats(model_name="m2")),
            ("Final Answer: Limited thought answer.", LLMCallStats(model_name="m3")),
        ]

        with patch('src.tot.response_parser.ToTResponseParser.parse_generated_thoughts', return_value=[ToTThought(text="The only thought.")]) as mock_parse_thoughts, \
             patch('src.tot.response_parser.ToTResponseParser.parse_evaluation_score', return_value=(8.0, "Only one, but good.")) as mock_parse_eval, \
             patch('src.tot.response_parser.ToTResponseParser.parse_final_answer', return_value="Limited thought answer.") as mock_parse_final_ans:

            result, summary = tot_processor.run(problem)

            assert result.succeeded
            assert result.final_answer == "Limited thought answer."
            assert result.total_thoughts_generated == 1
            assert "Max total thoughts generated limit reached." in result.error_message
            assert "Error Message: Max total thoughts generated limit reached." in summary
            assert mock_llm_client.call.call_count == 3


    @patch('src.tot.response_parser.ToTResponseParser.parse_generated_thoughts', MagicMock(return_value=[]))
    def test_run_no_thoughts_generated(self, tot_processor: ToTProcessor, mock_llm_client: MagicMock):
        problem = "No ideas."
        mock_llm_client.call.return_value = ("Thought Gen LLM Call - but parser returns none", LLMCallStats(model_name="m_thought"))

        result, summary = tot_processor.run(problem)

        assert not result.succeeded
        assert result.final_answer is None
        assert "No solution path could be determined." in result.error_message
        assert "ToT Succeeded: False" in summary
        assert mock_llm_client.call.call_count == 1


    @patch('src.tot.response_parser.ToTResponseParser.parse_generated_thoughts')
    @patch('src.tot.response_parser.ToTResponseParser.parse_evaluation_score', MagicMock(return_value=(None, "Could not parse score")))
    @patch('src.tot.response_parser.ToTResponseParser.parse_final_answer', return_value="Default due to eval issues.") # Added mock for final answer
    def test_run_evaluation_parse_failure(self, mock_parse_thoughts, tot_processor: ToTProcessor, mock_llm_client: MagicMock):
        problem = "Cannot evaluate."
        mock_parse_thoughts.return_value = [ToTThought(text="A thought")]

        mock_llm_client.call.side_effect = [
            ("Thought 1: A thought", LLMCallStats(model_name="m_thought")),
            ("Score: ???\nJustification: Bad format", LLMCallStats(model_name="m_eval")),
            ("Final Answer: Default due to eval issues.", LLMCallStats(model_name="m_final")),
        ]

        result, summary = tot_processor.run(problem)

        assert result.succeeded
        assert "Default due to eval issues" in result.final_answer
        assert result.all_generated_nodes
        node_key = list(result.all_generated_nodes.keys())[1]
        assert result.all_generated_nodes[node_key].score is None
        assert "Could not parse score" in result.all_generated_nodes[node_key].evaluation_details
        assert mock_llm_client.call.call_count == 3


    def test_run_early_exit_final_answer_in_thought(self, tot_processor: ToTProcessor, mock_llm_client: MagicMock):
        problem = "Find answer fast."

        early_thought_text = "The final answer is: found early!"
        early_thought = ToTThought(text=early_thought_text)

        mock_llm_client.call.side_effect = [
            ("Thought 1: " + early_thought_text, LLMCallStats(model_name="m_thought_early_ans")),
            ("Score: 10/10\nJustification: Found the answer.", LLMCallStats(model_name="m_eval_early_ans")),
            ("Final Answer: Synthesized from early find: found early!", LLMCallStats(model_name="m_final_early_ans")),
        ]

        with patch('src.tot.response_parser.ToTResponseParser.parse_generated_thoughts', return_value=[early_thought]) as mock_parse_thoughts, \
             patch('src.tot.response_parser.ToTResponseParser.parse_evaluation_score', return_value=(10.0, "Found the answer.")) as mock_parse_eval, \
             patch('src.tot.response_parser.ToTResponseParser.parse_final_answer', return_value="Synthesized from early find: found early!") as mock_parse_final_ans:

            result, summary = tot_processor.run(problem)

            assert result.succeeded
            assert "Synthesized from early find: found early!" in result.final_answer
            assert result.best_solution_path is not None
            assert len(result.best_solution_path) == 1
            assert early_thought_text.lower() in result.best_solution_path[0].text.lower()
            assert mock_llm_client.call.call_count == 3
            # To check for the log message, we'd need to capture logs or have the summary explicitly include it.
            # For now, the functional outcome (early stop leading to fewer calls if depth was > 1) is key.
            # The summary check below is a proxy if the summary string includes details of early exit.
            # Based on current processor, summary doesn't explicitly state "Final answer signal found...".
            # So we rely on call counts and path.
            assert result.total_nodes_evaluated == 1
