import pytest
from src.tot.response_parser import ToTResponseParser
from src.tot.dataclasses import ToTThought

class TestToTResponseParser:

    @pytest.mark.parametrize("response_text, expected_k, expected_thoughts_texts", [
        ("Thought 1: First idea.\nThought 2: Second idea.", 2, ["First idea.", "Second idea."]),
        ("Thought 1: Only one thought provided.", 1, ["Only one thought provided."]),
        ("Thought 1: Alpha\nThought 2: Beta\nThought 3: Gamma", 2, ["Alpha", "Beta"]), # Expects only k=2
        ("Thought 1: Leading space. \n Thought 2:Trailing space. ", 2, ["Leading space.", "Trailing space."]),
        ("Thought 1: This is thought one.\nThought 2: This is thought two, continuing on new line.", 2, ["This is thought one.", "This is thought two, continuing on new line."]),
        ("No 'Thought X:' prefix, just text.", 1, ["No 'Thought X:' prefix, just text."]), # Fallback for k=1
        ("", 1, []), # Empty response
        ("Thought 1: Correct. Thought 2: Incorrect because it's too much.", 1, ["Correct."]) # k=1
    ])
    def test_parse_generated_thoughts_various_formats(self, response_text, expected_k, expected_thoughts_texts):
        parsed_thoughts = ToTResponseParser.parse_generated_thoughts(response_text, expected_k)
        assert len(parsed_thoughts) == len(expected_thoughts_texts)
        for i, thought in enumerate(parsed_thoughts):
            assert isinstance(thought, ToTThought)
            assert thought.text == expected_thoughts_texts[i]
            assert thought.raw_generation_response == response_text

    def test_parse_generated_thoughts_fallback_split(self):
        response_text = "1. First point\n2. Second point\n3. Third point"
        expected_k = 3
        parsed_thoughts = ToTResponseParser.parse_generated_thoughts(response_text, expected_k)
        assert len(parsed_thoughts) == 3
        assert parsed_thoughts[0].text == "First point"
        assert parsed_thoughts[1].text == "Second point"
        assert parsed_thoughts[2].text == "Third point"

    def test_parse_generated_thoughts_empty_and_malformed(self):
        assert ToTResponseParser.parse_generated_thoughts("", 2) == []
        # Check for cases where regex might fail but fallback should not over-generate
        assert len(ToTResponseParser.parse_generated_thoughts("No thought markers here.", 3)) <= 1


    @pytest.mark.parametrize("response_text, expected_score, expected_justification_contains", [
        ("Score: 8/10\nJustification: Looks promising.", 8.0, "Looks promising."),
        ("Score: 7.5\nJustification: Decent path.", 7.5, "Decent path."),
        ("Score: 9\nJustification: Excellent progress!", 9.0, "Excellent progress!"),
        ("Score: 3/10 Justification: Not very good.", 3.0, "Not very good."),
        ("Score: 5.0 Justification: Average.", 5.0, "Average."),
        ("The score is Score: 6/10 and the Justification: It's okay.", 6.0, "It's okay."),
        ("Score: 10/10\nThis is the justification on a new line.", 10.0, "This is the justification"), # Fallback for justification
        ("Score: 2", 2.0, None), # Score only
        ("Justification: This is a test.", None, "This is a test."), # Justification only
        ("No score or justification here.", None, None),
        ("Score: 11/10\nJustification: Invalid score.", None, "Invalid score."), # Score out of range
        ("Score: 0/10\nJustification: Invalid score.", None, "Invalid score."), # Score out of range
        ("Score: 7/10\nJustification: The path seems coherent and directly addresses a key component of the problem. The steps are logical.", 7.0, "coherent and directly addresses"),
    ])
    def test_parse_evaluation_score(self, response_text, expected_score, expected_justification_contains):
        score, justification = ToTResponseParser.parse_evaluation_score(response_text)
        assert score == expected_score
        if expected_justification_contains:
            assert justification is not None # Added assertion
            assert expected_justification_contains in justification
        else:
            assert justification is None

    @pytest.mark.parametrize("response_text, expected_answer_contains", [
        ("Final Answer: The solution is 42.", "The solution is 42."),
        ("final answer: x = 5", "x = 5"),
        ("Some text before. Final Answer: This is it. Some text after.", "This is it."),
        ("The answer is clearly presented here without the prefix.", "The answer is clearly presented here without the prefix."), # Fallback for short response
        ("Thought 1: step1\nFinal Answer: done", "done"),
        ("No final answer here, just discussion.", None),
        ("Final Answer:", None), # Empty answer
    ])
    def test_parse_final_answer(self, response_text, expected_answer_contains):
        answer = ToTResponseParser.parse_final_answer(response_text)
        if expected_answer_contains:
            assert answer is not None
            assert expected_answer_contains in answer
        else:
            assert answer is None
