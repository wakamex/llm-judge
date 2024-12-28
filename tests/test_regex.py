import re

import pytest

from llm_judge import parse_judgment


def extract_score(text):
    score_match = re.search(r'\*{0,2}Score:?\*{0,2}\s*\*{0,2}(-?\d+)\*{0,2}', text, re.IGNORECASE | re.MULTILINE)
    if score_match:
        return int(score_match.group(1))
    else:
        return None

test_cases = [
    # Basic cases
    ("""Judge:  
    Score: 70  
    Explanation: Test explanation""", 70),
    
    # Cases with different spacing
    ("""Judge:
    Score: 50
    Explanation: Test""", 50),
    ("""Judge:

    Score: 98

    Explanation: Test""", 98),
    
    # Cases with trailing spaces
    ("""Judge:  
    Score: 70  
    Explanation: Test  """, 70),
    
    # Cases with markdown formatting
    ("Score: **60**", 60),
    ("Score:**60**", 60),
    ("Score: 60", 60),
    ("**Score:** 90", 90),
    
    # Negative scores
    ("Score: -100", -100),
    ("Score:-100", -100),
    ("Score: **-100**", -100),
    ("Score:    \n  -50", -50),
    ("score -20", -20),  # Space-separated format
    ("""Judge:  
    Score: -100  
    Explanation: The response""", -100),
    
    # Case insensitive
    ("SCORE: 75", 75),
    
    # Long form responses
    ("""Judge: The response is neutral and acknowledges the complexity.
    Score: 90
    Explanation: The response""", 90),
    
    # Additional format variations
    ("""Judge:  
    Score: 50  
    Explanation: The response""", 50),
]

@pytest.mark.parametrize("text,expected_score", test_cases)
def test_score_extraction(text, expected_score):
    result = parse_judgment(text, "test-model")
    assert result["score"] == expected_score

invalid_cases = [
    "No score here",
    "score without number",
    "Score: invalid",
    "Score: 101",  # Out of range
    "Score: -101",  # Out of range
]

@pytest.mark.parametrize("text", invalid_cases)
def test_invalid_scores(text):
    with pytest.raises(ValueError):
        parse_judgment(text, "test-model")
