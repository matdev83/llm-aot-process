import os
from typing import Optional, Any # Added Any for node_to_classify and current_node in construct methods

# Changed L2TConfig to L2TFullConfig
from .dataclasses import L2TFullConfig, L2TNode # L2TNode for type hinting

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "..", "conf", "prompts")
_L2T_INITIAL_PROMPT_FILE = os.path.join(_PROMPT_DIR, "l2t_initial.txt")
_L2T_NODE_CLASSIFICATION_PROMPT_FILE = os.path.join(
    _PROMPT_DIR, "l2t_node_classification.txt"
)
_L2T_THOUGHT_GENERATION_PROMPT_FILE = os.path.join(
    _PROMPT_DIR, "l2t_thought_generation.txt"
)


def _read_prompt_template(file_path: str) -> str:
    """Helper function to read prompt template from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        alt_path = os.path.join("conf", "prompts", os.path.basename(file_path))
        try:
            with open(alt_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
             raise FileNotFoundError(f"Prompt template file not found at {file_path} or {alt_path}")

# This class is aliased as L2TFullPromptGenerator in processor.py
# It should eventually be renamed to L2TFullPromptGenerator if it diverges significantly
# from a base L2TPromptGenerator.
class L2TPromptGenerator:
    # Changed L2TConfig to L2TFullConfig
    def __init__(self, l2t_config: Optional[L2TFullConfig] = None):
        self.l2t_config = l2t_config if l2t_config else L2TFullConfig()
        self._initial_prompt_template = _read_prompt_template(
            _L2T_INITIAL_PROMPT_FILE
        )
        self._node_classification_prompt_template = _read_prompt_template(
            _L2T_NODE_CLASSIFICATION_PROMPT_FILE
        )
        self._thought_generation_prompt_template = _read_prompt_template(
            _L2T_THOUGHT_GENERATION_PROMPT_FILE
        )

    def construct_l2t_initial_prompt(
        self, problem_statement: str, x_fmt: Optional[str] = None, x_eva: Optional[str] = None
    ) -> str: # In processor, this is expected to return a list/dict of messages
        """
        Constructs the initial prompt for the L2T process.
        L2TFullProcessor expects a list/dict structure, not a single string.
        This needs adaptation if used directly by L2TFullProcessor's LLMClient call.
        For now, returning string as per original L2T.
        """
        fmt = x_fmt if x_fmt is not None else self.l2t_config.x_fmt_default
        eva = x_eva if x_eva is not None else self.l2t_config.x_eva_default

        # This returns a single string. The L2TFullProcessor's initial call to LLM expects
        # a prompt object (like List[Dict[str,str]]). This will need adjustment either here
        # or in the processor. For now, aligning with the original structure.
        return (
            self._initial_prompt_template.replace("{{problem_text}}", problem_statement)
            .replace("{{x_fmt}}", fmt)
            .replace("{{x_eva}}", eva)
        )

    def construct_l2t_node_classification_prompt(
        self,
        node_to_classify: L2TNode, # Type hint added
        graph_context: str,
        problem_statement: str, # Added, as processor.py uses it
        x_eva: Optional[str] = None, # Kept from original
        remaining_steps_hint: Optional[int] = None,
    ) -> str: # Processor expects list/dict of messages
        """
        Constructs the prompt for classifying a thought node.
        """
        eva = x_eva if x_eva is not None else self.l2t_config.x_eva_default

        # Ensure problem_statement is used if templates expect it (original templates might not)
        # The template might use {{problem_statement}}
        prompt = (
            self._node_classification_prompt_template.replace(
                "{{graph_context}}", graph_context
            )
            .replace("{{node_to_classify_content}}", node_to_classify.content) # Use .content
            .replace("{{x_eva}}", eva)
            .replace("{{problem_statement}}", problem_statement) # Added replacement
        )

        if remaining_steps_hint is not None:
            prompt = prompt.replace(
                "{{remaining_steps_hint}}",
                f"You have approximately {remaining_steps_hint} reasoning steps remaining. Please try to converge to a final answer."
            )
        else:
            prompt = prompt.replace("{{remaining_steps_hint}}", "")

        return prompt

    def construct_l2t_thought_generation_prompt(
        self,
        current_node: L2TNode, # Type hint added, was parent_node_content
        problem_statement: str, # Added
        graph_context: str, # Added
        x_fmt: Optional[str] = None,
        x_eva: Optional[str] = None,
        remaining_steps_hint: Optional[int] = None,
    ) -> str: # Processor expects list/dict of messages
        """
        Constructs the prompt for generating new thoughts.
        """
        fmt = x_fmt if x_fmt is not None else self.l2t_config.x_fmt_default
        eva = x_eva if x_eva is not None else self.l2t_config.x_eva_default

        prompt = (
            self._thought_generation_prompt_template.replace(
                "{{graph_context}}", graph_context # graph_context might be more extensive now
            )
            .replace("{{parent_node_content}}", current_node.content) # Use current_node.content
            .replace("{{x_fmt}}", fmt)
            .replace("{{x_eva}}", eva)
            .replace("{{problem_statement}}", problem_statement) # Added replacement
        )

        if remaining_steps_hint is not None:
            prompt = prompt.replace(
                "{{remaining_steps_hint}}",
                f"You have approximately {remaining_steps_hint} reasoning steps remaining. Please try to converge to a final answer."
            )
        else:
            prompt = prompt.replace("{{remaining_steps_hint}}", "")

        return prompt
