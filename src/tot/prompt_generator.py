from typing import List, Optional
from .dataclasses import ToTNode, ToTConfig

class ToTPromptGenerator:
    """
    Generates prompts for the Tree of Thoughts (ToT) process.
    """

    @staticmethod
    def _format_current_state(current_thoughts_path: List[str]) -> str:
        if not current_thoughts_path:
            return "This is the beginning of the problem-solving process."
        return "\n".join(f"Step {i+1}: {thought}" for i, thought in enumerate(current_thoughts_path))

    @staticmethod
    def generate_thoughts_prompt(
        problem_description: str,
        current_node: ToTNode,
        config: ToTConfig
    ) -> str:
        """
        Generates a prompt to ask the LLM to produce K candidate next thoughts.

        Args:
            problem_description: The initial problem statement.
            current_node: The current ToTNode representing the current state (path of thoughts).
            config: The ToTConfig object.

        Returns:
            The prompt string.
        """
        current_thoughts_texts = [thought.text for thought in current_node.thoughts_path]
        formatted_state = ToTPromptGenerator._format_current_state(current_thoughts_texts)

        prompt = f"""
You are a helpful AI assistant solving a complex problem.
The problem is: {problem_description}

So far, the reasoning steps taken are:
{formatted_state}

Given the current state, please generate exactly {config.k_thoughts} distinct and promising next thoughts or steps that could lead towards solving the problem.
Each thought should be a concise paragraph.
Present your thoughts clearly, one after another. For example:
Thought 1: [Your first thought here]
Thought 2: [Your second thought here]
...
Thought {config.k_thoughts}: [Your last thought here]

Ensure the thoughts are diverse and explore different aspects or approaches if possible.
Do not repeat previous thoughts from the current reasoning path.
Focus on the next logical step or a creative idea to move forward.
"""
        return prompt

    @staticmethod
    def evaluate_state_prompt(
        problem_description: str,
        candidate_node: ToTNode, # The node (state) to be evaluated
        config: ToTConfig
    ) -> str:
        """
        Generates a prompt to ask the LLM to evaluate a candidate state (partial solution).

        Args:
            problem_description: The initial problem statement.
            candidate_node: The ToTNode representing the candidate state to evaluate.
            config: The ToTConfig object.

        Returns:
            The prompt string.
        """
        candidate_thoughts_texts = [thought.text for thought in candidate_node.thoughts_path]
        formatted_state = ToTPromptGenerator._format_current_state(candidate_thoughts_texts)

        prompt = f"""
You are an expert evaluator assessing the promise of a particular reasoning path for a given problem.
The problem is: {problem_description}

Consider the following partial solution (sequence of thoughts):
{formatted_state}

On a scale of 1 to 10 (where 1 is not promising at all, and 10 is extremely promising), how likely is it that this path will lead to a correct and complete solution?
Please provide your score clearly, like this:
Score: [Your score, e.g., 7/10]

Then, briefly justify your score in one or two sentences.
Justification: [Your brief justification here]

Focus on the coherence, correctness so far, and potential of this reasoning path to fully solve the problem.
"""
        return prompt

    @staticmethod
    def generate_final_answer_prompt(
        problem_description: str,
        best_solution_path: List[str] # List of thought texts
    ) -> str:
        """
        Generates a prompt to synthesize the final answer from the best thought path.

        Args:
            problem_description: The initial problem statement.
            best_solution_path: A list of thought texts forming the solution.

        Returns:
            The prompt string.
        """
        formatted_solution_path = ToTPromptGenerator._format_current_state(best_solution_path)

        prompt = f"""
The problem was: {problem_description}

The following sequence of thoughts was determined to be the best path to a solution:
{formatted_solution_path}

Based on this reasoning path, please provide a comprehensive final answer to the problem.
Present the final answer clearly. If the problem requires a specific format for the answer, please adhere to it.
If the reasoning path is incomplete or flawed, try to synthesize the best possible answer or indicate what's missing.

Final Answer:
"""
        return prompt
