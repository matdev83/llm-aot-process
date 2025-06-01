import re
from typing import List, Tuple, Optional, Dict, Any
from .dataclasses import ToTThought # Assuming ToTThought will be in .dataclasses

class ToTResponseParser:
    """
    Parses LLM responses for the Tree of Thoughts (ToT) process.
    """

    @staticmethod
    def parse_generated_thoughts(
        response_text: str,
        expected_k: int
    ) -> List[ToTThought]:
        """
        Parses the LLM response to extract K generated thoughts.

        Args:
            response_text: The raw text output from the LLM.
            expected_k: The number of thoughts expected (config.k_thoughts).

        Returns:
            A list of ToTThought objects. Returns fewer if parsing fails or fewer are found.
        """
        thoughts: List[ToTThought] = []
        # Regex to find "Thought X: [text]" or "Thought X. [text]"
        # It captures the thought text. It's made flexible to handle minor variations.
        # It looks for "Thought", optional space, a number, a colon or period, optional space, and then the thought text.
        # The thought text is captured until the next "Thought X:" or end of string.
        pattern = re.compile(r"Thought\s*\d+\s*[:.]\s*(.*?)(?=
Thought\s*\d+\s*[:.]|\Z)", re.DOTALL | re.IGNORECASE)

        matches = pattern.findall(response_text)

        for match_text in matches:
            cleaned_text = match_text.strip()
            if cleaned_text: # Ensure we don't add empty thoughts
                thoughts.append(ToTThought(text=cleaned_text, raw_generation_response=response_text))
            if len(thoughts) == expected_k:
                break # Stop if we have found the expected number of thoughts

        # If specific parsing fails, try a more generic split if fewer than K thoughts were found
        if not thoughts or len(thoughts) < expected_k:
            # Fallback: Try to split by "Thought X:" or common markers if regex misses them
            # This is a simpler heuristic
            potential_thoughts = []
            lines = response_text.splitlines()
            current_thought_lines = []
            for line in lines:
                # If a line looks like a new thought marker and we have content for the current thought
                if re.match(r"Thought\s*\d+\s*[:.]", line, re.IGNORECASE) and current_thought_lines:
                    potential_thoughts.append(" ".join(current_thought_lines).strip())
                    current_thought_lines = [line.split(":", 1)[-1].strip() if ":" in line else line.split(".", 1)[-1].strip()]
                elif re.match(r"^\d+\s*[:.]", line): # e.g. "1. Some thought"
                     if current_thought_lines: # save previous thought
                         potential_thoughts.append(" ".join(current_thought_lines).strip())
                     current_thought_lines = [line.split(":", 1)[-1].strip() if ":" in line else line.split(".", 1)[-1].strip()]
                else: # continue current thought
                    # Remove "Thought X:" prefix if it's part of the line but not caught as a new thought
                    cleaned_line = re.sub(r"Thought\s*\d+\s*[:.]\s*", "", line, flags=re.IGNORECASE).strip()
                    if cleaned_line:
                        current_thought_lines.append(cleaned_line)

            if current_thought_lines: # Add any remaining thought
                potential_thoughts.append(" ".join(current_thought_lines).strip())

            # Use these potential thoughts if the primary regex failed to get enough
            if len(thoughts) < expected_k:
                thoughts = [] # Reset if primary regex was insufficient
                for pt_text in potential_thoughts:
                    if pt_text: # Ensure non-empty
                         thoughts.append(ToTThought(text=pt_text, raw_generation_response=response_text))
                    if len(thoughts) == expected_k:
                        break

        # If still no thoughts, and the response is short, treat the whole response as one thought
        if not thoughts and len(response_text.split()) < 100 and expected_k == 1 : # Arbitrary length limit
            cleaned_response = response_text.strip()
            if cleaned_response:
                 thoughts.append(ToTThought(text=cleaned_response, raw_generation_response=response_text))


        return thoughts[:expected_k] # Return at most K thoughts

    @staticmethod
    def parse_evaluation_score(
        response_text: str
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Parses the LLM response to extract the evaluation score and justification.

        Args:
            response_text: The raw text output from the LLM.

        Returns:
            A tuple containing:
            - score (float): The numerical score (e.g., 7.0). None if not found.
            - justification (str): The textual justification. None if not found.
        """
        score: Optional[float] = None
        justification: Optional[str] = None

        # Regex to find "Score: [score_value]/10" or "Score: [score_value]"
        # Captures the score_value. Allows for optional "/10".
        score_match = re.search(r"Score\s*:\s*(\d+(?:\.\d+)?)(?:\s*/\s*10)?", response_text, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
                if not (1.0 <= score <= 10.0): # Validate score range
                    score = None # Invalid score
            except ValueError:
                score = None # Could not convert to float

        # Regex to find "Justification: [text]"
        # Captures the justification text.
        justification_match = re.search(r"Justification\s*:\s*(.+)", response_text, re.IGNORECASE | re.DOTALL)
        if justification_match:
            justification = justification_match.group(1).strip()
        else: # Fallback if "Justification:" is missing but there's text after score
            if score_match:
                remaining_text = response_text[score_match.end():].strip()
                if remaining_text and len(remaining_text.split()) > 2: # Heuristic: justification should be somewhat substantial
                    justification = remaining_text

        # If score is found but justification is not, and there's only one line after score:
        if score is not None and justification is None:
            lines = response_text.splitlines()
            score_line_index = -1
            for i, line in enumerate(lines):
                if score_match and score_match.group(0) in line:
                    score_line_index = i
                    break
            if score_line_index != -1 and score_line_index + 1 < len(lines):
                potential_justification = lines[score_line_index+1].strip()
                if potential_justification and not re.match(r"Score\s*:", potential_justification, re.IGNORECASE):
                    justification = potential_justification


        return score, justification

    @staticmethod
    def parse_final_answer(response_text: str) -> Optional[str]:
        """
        Parses the LLM response to extract the final answer.
        This might be simple if the prompt asks for "Final Answer:" prefix.

        Args:
            response_text: The raw text output from the LLM.

        Returns:
            The extracted final answer string, or None if not clearly found.
        """
        # Look for "Final Answer:" prefix, case-insensitive
        final_answer_match = re.search(r"Final Answer\s*:\s*(.+)", response_text, re.IGNORECASE | re.DOTALL)
        if final_answer_match:
            return final_answer_match.group(1).strip()

        # Fallback: if the response is relatively short and doesn't contain typical markers of ongoing thought,
        # consider the whole response as the answer. This is a heuristic.
        # Avoid this if the response contains common thought/evaluation markers.
        if not re.search(r"Thought\s*\d|Score\s*:|Justification\s*:", response_text, re.IGNORECASE):
            # Further check: if it's a single paragraph or a few lines.
            lines = response_text.strip().splitlines()
            if 0 < len(lines) < 5 : # Arbitrary small number of lines
                return response_text.strip()

        return None # Could not confidently extract a final answer based on markers
