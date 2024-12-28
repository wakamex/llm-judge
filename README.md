# LLM Judge

A plugin for [LLM](https://llm.datasette.io/) that enables multiple language models to evaluate each other's responses.

inspired by [llm-consortium](https://github.com/irthomasthomas/llm-consortium), with the following differences:
- single-pass evaluation vs. iterative refinement
- peer-to-peer judging vs. arbiter-based synthesis
- absolute score vs. confidence-based approach
- default judgement based on accuracy vs. bias

## Core Algorithm Flow

```mermaid
flowchart TD
    A[Start] --> B[ThreadPool]
    B ==> C[Concurrent Responses]
    C --> |Each Response| D[Launch Judgments]
    D -.->|Feed Back|C
    D ==> E[Results]
    
    style B fill:#f9f,stroke:#333
    style C fill:#bbf,stroke:#333
    style D fill:#bbf,stroke:#333
```

## Features

- **Multi-Model Evaluation**: Each model answers the question and judges other models' responses
- **Objective Scoring**: Clear -100 to 100 scoring system where:
  - 100: Maximum accuracy and helpfulness
  - 0: Significant bias or inaccuracy
  - -100: Failure or refusal to answer
- **Detailed Analysis**: Each judgment includes:
  - Numerical score
  - Detailed analysis of strengths/weaknesses
  - Brief explanation of the score
- **Database Logging**: All responses and judgments are logged to SQLite
- **JSON Export**: Option to save full results to JSON file

## How It Works

### Architecture

This tool uses thread-based concurrency to efficiently evaluate LLM responses:

1. **Initial Responses**: All models answer the question concurrently using a thread pool
2. **Cross-Evaluation**: Each response is then judged by all other models concurrently
3. **Result Aggregation**: All responses and judgments are collected and displayed

### Execution Flow

For a panel of N models, the tool will:
1. Make N concurrent API calls to get initial answers using ThreadPoolExecutor
2. For each answer, make (N-1) concurrent API calls to get judgments
3. Total API calls = N + N*(N-1) = N^2

For example, with 7 default models:
- 7 initial response calls
- 7 * 6 = 42 judgment calls
- Total: 49 API calls

### Configuration

The tool can be configured with several options:

```bash
# Basic usage with default settings
llm judge "What is the capital of France?"

# Custom model selection
llm judge -m "openrouter/openai/gpt-4" -m "openrouter/anthropic/claude-3" "Your question"

# Advanced configuration
llm judge \
  --max-workers 10 \     # Number of concurrent API calls (default: 10)
  --max-retries 2 \      # Retries for failed API calls (default: 1)
  --output results.json \ # Save results to file
  "Your question"
```

### Default Models

This will use the following default models:
- GPT-4o (OpenAI's latest flagship via OpenRouter)
- Claude 3.5 Sonnet (Anthropic's best)
- Gemini 2.0 Flash (Google's best)
- Hermes 3 405B (Nous Research's largest open source model)
- Grok 2 (X.AI's latest model)
- DeepSeek Chat (DeepSeek's flagship model)
- Mistral Large (Mistral AI's strongest model)

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

### Database Logging

All responses and judgments are automatically logged to SQLite:
- Responses table: Stores each model's answer
- Judgments table: Stores cross-evaluation results
- Timestamps for tracking execution flow
- Unique IDs for response-judgment correlation

## Setup

### API Keys

This tool requires an OpenRouter API key to access the models. You can get one from [OpenRouter](https://openrouter.ai/).

Set up your API key in one of these ways:

1. Environment variable:
```bash
export LLM_OPENROUTER_KEY=your_key_here
```

2. Using llm configuration:
```bash
llm keys set openrouter
# Then paste your key when prompted
```

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).

```bash
llm install llm-judge
```

## Usage

### Basic Usage

Simple question with default models:
```bash
llm judge "What is the capital of France?"
```

### Advanced Usage

Specify multiple models:
```bash
llm judge "Explain quantum entanglement" \
  -m gpt-4 \
  -m claude-3-opus-20240229 \
  -m claude-3-sonnet-20240229 \
  -m gemini-pro
```

Save results to JSON:
```bash
llm judge "Compare Python and JavaScript" --output results.json
```

### Response Structure

Each model's judgment follows this structure:

```xml
<analysis>
[Detailed analysis of the response's strengths and weaknesses]
</analysis>

<score>
[Score from -100 to 100]
</score>

<explanation>
[Brief explanation of the score]
</explanation>
```

### Database Access

Responses and judgments are stored in SQLite at `~/.llm/judge_logs.db`:
- `responses` table: Stores all model responses
- `judgments` table: Stores all peer judgments

### Programmatic Usage

```python
from llm_judge import JudgeOrchestrator

# Initialize orchestrator
orchestrator = JudgeOrchestrator(
    models=["gpt-4", "claude-3-opus-20240229"]
)

# Run evaluation
results = await orchestrator.orchestrate("Your prompt")

# Access results
for result in results["results"]:
    print(f"\nModel: {result['model']}")
    print(f"Response: {result['response']}")
    
    for judgment in result["judgments"]:
        print(f"\nJudge: {judgment['judge_model']}")
        print(f"Score: {judgment['score']}")
        print(f"Analysis: {judgment['analysis']}")
        print(f"Explanation: {judgment['explanation']}")
```

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-judge
python -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
llm install -e '.[test]'
```
To run the tests:
```bash
python -m pytest
```

## License

Apache 2.0
