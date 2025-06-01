import pytest
from src.tot.prompt_generator import ToTPromptGenerator
from src.tot.dataclasses import ToTNode, ToTConfig, ToTThought
from src.tot.enums import ToTSearchStrategy, ToTScoringMethod

@pytest.fixture
def basic_tot_config(self) -> ToTConfig:
    return ToTConfig(k_thoughts=2, b_beam_width=2, max_depth=3)

@pytest.fixture
def root_node(self) -> ToTNode:
    return ToTNode(id="root", thoughts_path=[], depth=0)

@pytest.fixture
def intermediate_node(self) -> ToTNode:
    return ToTNode(
        id="node1",
        thoughts_path=[ToTThought(text="Initial thought about the problem.")],
        depth=1,
        parent_id="root"
    )

class TestToTPromptGenerator:
    def test_generate_thoughts_prompt_root(self, root_node, basic_tot_config):
        problem = "Solve for X in X+5=10."
        prompt = ToTPromptGenerator.generate_thoughts_prompt(problem, root_node, basic_tot_config)
        assert problem in prompt
        assert "So far, the reasoning steps taken are:\nThis is the beginning of the problem-solving process." in prompt
        assert f"generate exactly {basic_tot_config.k_thoughts} distinct and promising next thoughts" in prompt
        assert "Thought 1:" in prompt

    def test_generate_thoughts_prompt_intermediate(self, intermediate_node, basic_tot_config):
        problem = "Explain photosynthesis."
        prompt = ToTPromptGenerator.generate_thoughts_prompt(problem, intermediate_node, basic_tot_config)
        assert problem in prompt
        assert "Step 1: Initial thought about the problem." in prompt
        assert f"generate exactly {basic_tot_config.k_thoughts} distinct and promising next thoughts" in prompt

    def test_evaluate_state_prompt(self, intermediate_node, basic_tot_config):
        problem = "What is the capital of France?"
        prompt = ToTPromptGenerator.evaluate_state_prompt(problem, intermediate_node, basic_tot_config)
        assert problem in prompt
        assert "Step 1: Initial thought about the problem." in prompt
        assert "On a scale of 1 to 10" in prompt
        assert "Score: [Your score, e.g., 7/10]" in prompt
        assert "Justification: [Your brief justification here]" in prompt

    def test_generate_final_answer_prompt(self):
        problem = "Summarize the process of cellular respiration."
        best_path = ["Glycolysis occurs.", "Krebs cycle follows.", "Electron transport chain produces ATP."]
        prompt = ToTPromptGenerator.generate_final_answer_prompt(problem, best_path)
        assert problem in prompt
        assert "Step 1: Glycolysis occurs." in prompt
        assert "Step 3: Electron transport chain produces ATP." in prompt
        assert "Final Answer:" in prompt
