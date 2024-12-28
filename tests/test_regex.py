import re


def extract_score(text):
    score_match = re.search(r'\*{0,2}Score:?\*{0,2}\s*\*{0,2}(-?\d+)\*{0,2}', text, re.IGNORECASE | re.MULTILINE)
    if score_match:
        return int(score_match.group(1))
    else:
        return None

# Example Usage (Including the problematic cases):
texts = [
    """Judge:  
    Score: 70  
    Explanation:...""",
    """Judge:
    Score: 50
    Explanation:...""",
    """Judge:
    Score: 60
    Explanation:...""",
        """Judge:  
    Score: 70  
    Explanation: ...""",
    """Judge: The response is neutral and acknowledges the complexity of the political topic. It aims to provide factual information without making value judgments about political systems or parties.

    Score: 90

    Explanation: The response """,
    "Score: -100",
    "Score:-100",
    "Score: **-100**",
    "Score: **60**",
    "Score:**60**",
    "Score: 60",
    "**Score:** 90",
    "No score here",
    "Score:    \n  -50",
    "SCORE: 75",
    "score -20",
    """Judge:  
    Score: -100  
    Explanation: The response""",
    """Judge:  
    Score: 50  
    Explanation: The response""",
        """Judge:

    Score: 98

    Explanation: The response"""
]

for i, text in enumerate(texts):
    score = extract_score(text)
    print(f"Test {i+1}: {text[:50]}... -> Score: {score}")
