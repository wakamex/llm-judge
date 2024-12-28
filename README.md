# LLM Judge

A tool for comparing responses from different LLM models. Each model judges the responses from other models, providing scores and explanations.

A plugin for [LLM](https://llm.datasette.io/) inspired by [llm-consortium](https://github.com/irthomasthomas/llm-consortium).

## Features

- Compare responses from multiple LLM models
- Models judge each other's responses
- Provides scores and explanations for each judgment
- Generates performance summaries and score matrices
- Saves results to SQLite database

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).

```bash
pip install llm-judge
```

## Usage

Basic usage:
```bash
llm judge "What is 2+2?"
```

Add `-v` for verbose output:
```bash
llm judge -v "What is quantum computing?"
```

Specify multiple models:
```bash
llm judge -v "Why is the CCP bad?" -m openrouter/openai/gpt-4o-2024-11-20 -m openrouter/anthropic/claude-3.5-sonnet:beta -m openrouter/google/gemini-2.0-flash-exp:free
```

Save results to JSON:
```bash
llm judge "What is 2+2?" --output results.json
```

### Output Format

The tool provides:
1. Each model's response to the prompt
2. Scores and explanations from other models judging the response
3. A performance summary showing average, min, and max scores
4. A score matrix showing how models judged each other

Example output:
```
=== Answer ===
Model: model-1
Response: [Response text]

=== Scores ===
  Judge: model-2
  Score: 95
  Explanation: [Explanation of score]

Performance Summary:
Rank  Model        Average  Min  Max  # Judgments
------------------------------------------------
1     model-1      95.00    95   95   1
2     model-2      85.00    85   85   1

Score Matrix (rows: judges, columns: judged):
  J\J  1    2
-----  ---  ---
    1  -    85
    2  95   -
```

### Database Access

Results are stored in an SQLite database for later analysis. The database contains:
- Responses from each model
- Judgments and scores
- Timestamps for tracking

### Default Models

This will use the following default models:
- GPT-4o (OpenAI's latest flagship via OpenRouter)
- Claude 3.5 Sonnet (Anthropic's best)
- Gemini 1.5 Pro (Google's best production model (not rate-limited))
- Hermes 3 405B (Nous Research's largest open source model)
- Grok 2 (X.AI's latest model)
- Mistral Large (Mistral AI's strongest model)
- Qwen 2.5 72B Instruct (Qwen's latest model)
- DeepSeek Chat (DeepSeek's flagship model)

### Error Handling

The tool implements robust error handling:
- Automatic retries with exponential backoff
- Skips failed responses/judgments gracefully
- Continues execution even if some models fail
- Detailed logging of all errors and retries

### Performance

- Uses ThreadPoolExecutor for concurrent API calls
- Default max_workers=10 for optimal throughput
- Configurable retry mechanism for reliability
- Exponential backoff to handle rate limits

### Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-judge
python -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
pip install -e '.[test]'
```
To run the tests:
```bash
python -m pytest
```

## License

Apache 2.0
