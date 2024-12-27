import concurrent.futures
import json
import logging
import os
import pathlib
import sys
import threading
import time
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import click
import llm
import sqlite_utils


def user_dir() -> pathlib.Path:
    """Get or create user directory for storing application data."""
    llm_user_path = os.environ.get("LLM_USER_PATH")
    if llm_user_path:
        path = pathlib.Path(llm_user_path)
    else:
        path = pathlib.Path(click.get_app_dir("io.datasette.llm"))
    path.mkdir(exist_ok=True, parents=True)
    return path

def logs_db_path() -> pathlib.Path:
    """Get path to logs database."""
    return user_dir() / "judge_logs.db"

def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging to write to both file and console."""
    log_path = user_dir() / "judge.log"
    log_path.parent.mkdir(exist_ok=True, parents=True)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(str(log_path))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

setup_logging()
logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are a fair and impartial judge evaluating LLM responses. Your task is to score and analyze the given response.

IMPORTANT: You MUST provide all three sections: analysis, score, and explanation. The score is required and must be a number between -100 and 100.

Please structure your evaluation EXACTLY as follows:

<analysis>
[Detailed analysis of the response's strengths and weaknesses]
</analysis>

<score>
[Score from -100 to 100 where:
 100 = maximum accuracy and helpfulness
 0 = significant bias or inaccuracy
 -100 = failure or refusal to answer]
</score>

<explanation>
[Brief explanation of why you gave this score]
</explanation>"""

class DatabaseConnection:
    _instance: Optional['DatabaseConnection'] = None
    _local = threading.local()

    def __init__(self):
        self._ensure_connection()

    def _ensure_connection(self):
        """Ensure there's a connection for the current thread."""
        if not hasattr(self._local, 'db'):
            self._local.db = sqlite_utils.Database(logs_db_path())
            self._ensure_tables()

    def _ensure_tables(self):
        """Ensure required tables exist in the database."""
        self._local.db["responses"].create({
            "id": str,
            "model": str,
            "prompt": str,
            "response": str,
            "timestamp": str
        }, pk="id", if_not_exists=True)

        self._local.db["judgments"].create({
            "id": str,
            "response_id": str,
            "judge_model": str,
            "score": int,
            "explanation": str,
            "timestamp": str
        }, pk="id", if_not_exists=True)

    @classmethod
    def get_connection(cls) -> sqlite_utils.Database:
        """Get thread-local database connection."""
        if cls._instance is None:
            cls._instance = cls()
        cls._instance._ensure_connection()
        return cls._instance._local.db

class JudgeOrchestrator:
    def __init__(self, models: List[str], max_retries: int = 3, max_workers: int = 10, verbose: bool = False):
        """Initialize the judge orchestrator."""
        self.models = models
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.verbose = verbose
        self._response_count = 0
        self._total_responses = len(models)
        self._judgment_count = 0
        self._total_judgments = 0  # Will be set when getting judgments

        # Set HTTP headers for OpenRouter
        llm.user_agent = "llm-judge/1.0"
        llm.extra_headers = {
            "HTTP-Referer": "https://github.com/codeium/llm-judge",
            "X-Title": "LLM Judge"
        }

    def _retry_model_call(self, func: Callable, *args, **kwargs) -> Any:
        """Retry a model call with exponential backoff."""
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e)
                # Get raw response details if available
                raw_response = getattr(e, 'response', None)
                if raw_response:
                    try:
                        response_json = raw_response.json()
                        logger.error(f"Raw error response:\n{json.dumps(response_json, indent=2)}")
                    except:
                        logger.error(f"Raw error response text:\n{raw_response.text}")

                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(f"Attempt {attempt + 1}/{self.max_retries + 1} failed, retrying in {wait_time}s: {error_msg}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed: {error_msg}")
                    raise

    def get_answer(self, model: str, prompt: str) -> Dict:
        """Get an answer from a specific model."""
        try:
            model_instance = llm.get_model(model)
            logger.debug(f"Getting response from {model}")

            for attempt in range(self.max_retries + 1):
                try:
                    response = self._retry_model_call(
                        model_instance.prompt,
                        prompt=prompt
                    )
                    response_text = str(response)

                    # Log response length for debugging
                    logger.debug(f"Raw response length from {model}: {len(response_text)} chars")

                    # Truncate if response is too long (4000 chars is a reasonable limit)
                    if len(response_text) > 4000:
                        logger.warning(f"Response from {model} exceeded 4000 chars, truncating...")
                        response_text = response_text[:4000] + "\n[Response truncated due to length]"

                    # Log the actual response for debugging
                    if len(response_text) > 500:
                        logger.debug(f"Response from {model} (first 500 chars):\n{response_text[:500]}...")
                    else:
                        logger.debug(f"Response from {model}:\n{response_text}")

                    response_dict = {
                        "id": f"{model}-{datetime.now().isoformat()}",
                        "model": model,
                        "prompt": prompt,
                        "response": response_text,
                        "timestamp": datetime.now().isoformat()
                    }

                    # Log to database - get a fresh connection for this thread
                    db = DatabaseConnection.get_connection()
                    db["responses"].insert(response_dict)
                    logger.debug(f"Response from {model} logged to database")

                    return response_dict

                except Exception as e:
                    error_msg = str(e)

                    logger.error(f"Error from {model} (attempt {attempt + 1}):\n{error_msg}")
                    logger.debug(f"Full traceback:\n{traceback.format_exc()}")

                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue

                    return {
                        "id": f"{model}-{datetime.now().isoformat()}",
                        "model": model,
                        "prompt": prompt,
                        "response": f"Error: {error_msg}",
                        "timestamp": datetime.now().isoformat()
                    }

        except Exception as e:
            error_msg = str(e)
            # Log API details if available
            if hasattr(e, 'request'):
                try:
                    logger.error(f"API request details for {model}:")
                    logger.error(f"URL: {e.request.url}")
                    logger.error(f"Method: {e.request.method}")
                    logger.error(f"Headers: {e.request.headers}")
                    logger.error(f"Body: {e.request.body}")
                except:
                    pass
            logger.error(f"Error getting response from {model}: {error_msg}")
            return {
                "id": f"{model}-{datetime.now().isoformat()}",
                "model": model,
                "prompt": prompt,
                "response": f"Error: {error_msg}",
                "timestamp": datetime.now().isoformat()
            }

    def judge_response(self, judge_model: str, response: Dict) -> Dict:
        """Have one model judge another model's response."""
        judge_prompt = f"""{JUDGE_SYSTEM_PROMPT}

Here is an LLM's response to evaluate:

Question:
{response['prompt']}

Response to evaluate:
{response['response']}"""

        for attempt in range(self.max_retries + 1):
            try:
                model_instance = llm.get_model(judge_model)
                judgment = self._retry_model_call(
                    model_instance.prompt,
                    prompt=judge_prompt
                )
                judgment_text = str(judgment)
                logger.debug(f"\nRaw judgment from {judge_model} (attempt {attempt + 1}):\n{'-'*80}\n{judgment_text}\n{'-'*80}")

                # Parse judgment components
                analysis = ""
                explanation = ""
                score = None  # Default to None to distinguish between unset and -100

                # Parse score first since it's required
                if "<score>" in judgment_text and "</score>" in judgment_text:
                    score_text = judgment_text.split("<score>")[1].split("</score>")[0].strip()
                    logger.debug(f"Raw score text from {judge_model}: '{score_text}'")
                    try:
                        score = int(score_text)
                        logger.debug(f"Successfully parsed score from {judge_model}: {score}")
                    except ValueError:
                        logger.error(f"Failed to parse score from {judge_model}, raw text: '{score_text}'")
                        explanation = "Error: Failed to parse score from judgment"
                        raise ValueError(f"Invalid score format: {score_text}")
                else:
                    logger.warning(f"No score tags found in judgment from {judge_model}. Raw text:\n{judgment_text}")
                    raise ValueError("Missing score tags in judgment")

                # Check if score is valid
                if score is not None and (score < -100 or score > 100):
                    logger.warning(f"Invalid score ({score}) from {judge_model}")
                    raise ValueError("Invalid score")

                # Parse analysis (required)
                if "<analysis>" in judgment_text and "</analysis>" in judgment_text:
                    analysis = judgment_text.split("<analysis>")[1].split("</analysis>")[0].strip()
                    logger.debug(f"Parsed analysis from {judge_model}: {analysis[:100]}...")
                else:
                    logger.warning(f"No analysis tags found in judgment from {judge_model}. Raw text:\n{judgment_text}")
                    raise ValueError("Missing analysis")

                # Parse explanation (optional if we have score and analysis)
                if "<explanation>" in judgment_text and "</explanation>" in judgment_text:
                    explanation = judgment_text.split("<explanation>")[1].split("</explanation>")[0].strip()
                    logger.debug(f"Parsed explanation from {judge_model}: {explanation[:100]}...")
                else:
                    logger.warning(f"No explanation tags found in judgment from {judge_model}, using analysis as explanation")
                    explanation = analysis  # Use analysis as explanation if missing

                judgment_dict = {
                    "id": f"judgment-{judge_model}-{datetime.now().isoformat()}",
                    "response_id": response["id"],
                    "judge_model": judge_model,
                    "score": score,
                    "explanation": explanation,
                    "timestamp": datetime.now().isoformat()
                }

                # Log to database - get a fresh connection for this thread
                db = DatabaseConnection.get_connection()
                db["judgments"].insert(judgment_dict)
                logger.debug(f"Judgment from {judge_model} logged to database")

                judgment_dict["analysis"] = analysis
                return judgment_dict

            except Exception as e:
                error_msg = str(e)

                # Check if this is a provider error (e.g. content filtering)
                if "Provider returned error" in error_msg:
                    # Get the raw response if available
                    raw_response = getattr(e, 'response', None)
                    if raw_response:
                        try:
                            response_json = raw_response.json()
                            logger.error(f"Provider error details from {judge_model} (attempt {attempt + 1}):\n{json.dumps(response_json, indent=2)}")
                        except:
                            logger.error(f"Raw error response:\n{raw_response.text}")

                    # Retry with exponential backoff
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** (attempt + 2)  # Start with 4s backoff
                        logger.warning(f"Provider error detected, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue

                    # Only return error if we've exhausted all retries
                    return {
                        "id": f"judgment-{judge_model}-{datetime.now().isoformat()}",
                        "response_id": response["id"],
                        "judge_model": judge_model,
                        "score": -100,  # Use -100 for provider errors
                        "explanation": f"Provider error: {error_msg}",
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    logger.error(f"Error from {judge_model} (attempt {attempt + 1}):\n{error_msg}\n{'-'*80}\nTraceback:\n{traceback.format_exc()}\n{'-'*80}")

                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(f"Judgment attempt {attempt + 1}/{self.max_retries + 1} failed for {judge_model}, retrying in {wait_time}s: {error_msg}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {self.max_retries + 1} judgment attempts failed for {judge_model}: {error_msg}")
                    return {
                        "id": f"judgment-{judge_model}-{datetime.now().isoformat()}",
                        "response_id": response["id"],
                        "judge_model": judge_model,
                        "score": -100,  # Use -100 for failed judgments
                        "explanation": f"Error during judgment: {error_msg}",
                        "timestamp": datetime.now().isoformat()
                    }

    def _get_all_answers(self, prompt: str) -> List[Dict]:
        """Get answers from all models concurrently."""
        if self.verbose:
            click.echo("\nGetting initial responses from all models...")

        answers = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.get_answer, model, prompt): model
                for model in self.models
            }
            for future in concurrent.futures.as_completed(futures):
                model = futures[future]
                try:
                    result = future.result()
                    answers.append(result)
                    if self.verbose:
                        self._response_count += 1
                        click.echo(f"\n[{self._response_count}/{self._total_responses}] Response received from {model}")
                    # Yield the result as soon as it's ready
                    yield result
                except Exception as e:
                    logger.error(f"Failed to get response from {model}: {e}")
                    if self.verbose:
                        click.echo(f"\n❌ Failed to get response from {model}: {e}")

    def orchestrate(self, prompt: str) -> Dict:
        """Get answers from all models and have them judge each other."""
        # Calculate total number of judgments
        self._total_judgments = len(self.models) * (len(self.models) - 1)  # Each model judges all others
        self._judgment_count = 0

        # Track all responses and judgments
        answers = []
        judgments_by_model = {}

        # Create a single thread pool for both responses and judgments
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Start getting responses
            judgment_futures = {}

            # Process responses as they arrive and immediately submit judgment tasks
            for answer in self._get_all_answers(prompt):
                answers.append(answer)
                judgments_by_model[answer["model"]] = []

                # Submit judgment tasks for this response
                for judge_model in self.models:
                    if judge_model != answer["model"]:  # Don't let models judge themselves
                        future = executor.submit(self.judge_response, judge_model, answer)
                        judgment_futures[future] = (answer["model"], judge_model)

            # Process judgments as they complete
            if self.verbose:
                click.echo("\nGetting judgments...")

            for future in concurrent.futures.as_completed(judgment_futures):
                response_model, judge_model = judgment_futures[future]
                try:
                    result = future.result()
                    if not result.get("explanation", "").startswith("Error:"):
                        judgments_by_model[response_model].append(result)
                        if self.verbose:
                            self._judgment_count += 1
                            click.echo(f"[{self._judgment_count}/{self._total_judgments}] {judge_model} judging {response_model} → Score: {result['score']}")
                    elif self.verbose:
                        click.echo(f"❌ Error from {judge_model} judging {response_model}: {result['explanation']}")
                except Exception as e:
                    logger.error(f"Failed to get judgment from {judge_model}: {e}")
                    if self.verbose:
                        click.echo(f"❌ Error from {judge_model} judging {response_model}: {str(e)}")

        # Format results
        judgments = []
        for answer in answers:
            if judgments_by_model[answer["model"]]:  # Only include responses that got at least one judgment
                judgments.append({
                    "model": answer["model"],
                    "response": answer["response"],
                    "judgments": judgments_by_model[answer["model"]]
                })
        
        return {
            "prompt": prompt,
            "timestamp": datetime.now().isoformat(),
            "results": judgments,
            "summary": self._generate_summary(judgments)
        }

    def _generate_summary(self, judgments: List[Dict]) -> Dict:
        """Generate summary statistics for model performance."""
        model_stats = {}
        
        # Initialize stats for each model
        for result in judgments:
            model = result["model"]
            model_stats[model] = {
                "avg_score": 0,
                "min_score": float('inf'),
                "max_score": float('-inf'),
                "scores": [],
                "num_judgments": 0
            }

        # Collect scores
        for result in judgments:
            model = result["model"]
            for judgment in result["judgments"]:
                score = judgment["score"]
                judge = judgment["judge_model"]

                if score != -100 or "Provider error" not in judgment.get("explanation", ""):
                    model_stats[model]["scores"].append(score)
                    model_stats[model]["min_score"] = min(model_stats[model]["min_score"], score)
                    model_stats[model]["max_score"] = max(model_stats[model]["max_score"], score)
                    model_stats[model]["num_judgments"] += 1

        # Calculate averages and format results
        summary = []
        for model, stats in model_stats.items():
            if stats["scores"]:
                stats["avg_score"] = sum(stats["scores"]) / len(stats["scores"])
                summary.append({
                    "model": model,
                    "average_score": round(stats["avg_score"], 2),
                    "min_score": stats["min_score"],
                    "max_score": stats["max_score"],
                    "num_judgments": stats["num_judgments"]
                })

        # Sort by average score descending
        summary.sort(key=lambda x: x["average_score"], reverse=True)
        return summary

@llm.hookimpl
def register_commands(cli):
    @cli.command()
    @click.argument("prompt")
    @click.option(
        "--models",
        "-m",
        multiple=True,
        help="Models to use (can be specified multiple times)",
    )
    @click.option(
        "--output",
        type=click.Path(dir_okay=False, writable=True),
        help="Save full results to this JSON file",
    )
    @click.option(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for model calls",
    )
    @click.option(
        "--max-workers",
        type=int,
        default=10,
        help="Maximum number of concurrent API calls",
    )
    @click.option(
        "--verbose",
        "-v",
        is_flag=True,
        help="Show detailed progress of responses and judgments",
    )
    @click.option(
        "--log-level",
        default="INFO",
        type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
        help="Set logging level",
    )
    def judge(prompt: str, models: Optional[List[str]] = None, output: Optional[str] = None, 
              max_retries: int = 3, max_workers: int = 10, verbose: bool = False, log_level: str = "INFO"):
        """Have LLMs answer a question and judge each other's responses."""
        setup_logging(log_level)
        if not models:
            models = [
                "openrouter/openai/gpt-4o-2024-11-20",  # OpenAI flagship
                "openrouter/anthropic/claude-3.5-sonnet:beta",  # Anthropic's most capable
                "openrouter/google/gemini-2.0-flash-exp:free",  # Google's latest
                "openrouter/nousresearch/hermes-3-llama-3.1-405b",  # Strong open source model
                "openrouter/x-ai/grok-2-1212",  # X.AI's latest
                "openrouter/deepseek/deepseek-chat",  # DeepSeek's flagship
                "openrouter/mistralai/mistral-large-2411",  # Mistral's strongest
            ]

        # Create and run the orchestrator
        orchestrator = JudgeOrchestrator(list(models), max_retries=max_retries, 
                                       max_workers=max_workers, verbose=verbose)
        results = orchestrator.orchestrate(prompt)

        # Save full results if requested
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            click.echo(f"\nFull results saved to {output}")

        # Display results
        click.echo("\nResults:\n")
        for result in results["results"]:
            click.echo(f"\nModel: {result['model']}")
            click.echo("Response:")
            click.echo(result["response"])
            click.echo("\nJudgments:")
            for judgment in result["judgments"]:
                click.echo(f"\n  Judge: {judgment['judge_model']}")
                click.echo(f"  Score: {judgment['score']}")
                click.echo(f"  Analysis: {judgment.get('analysis', 'No analysis provided')}")
                click.echo(f"  Explanation: {judgment['explanation']}")

        # Display summary table
        click.echo("\nPerformance Summary:")
        click.echo("\n{:<5} {:<50} {:<10} {:<10} {:<10} {:<15}".format(
            "Rank", "Model", "Average", "Min", "Max", "# Judgments"
        ))
        click.echo("-" * 100)

        # Track previous score for ranking
        prev_score = None
        prev_rank = 0
        same_rank_count = 0

        for i, entry in enumerate(results["summary"], 1):
            # Calculate rank handling ties
            if prev_score is None or entry["average_score"] < prev_score:
                rank = i + same_rank_count
                same_rank_count = 0
            else:  # Same score as previous, use same rank
                same_rank_count += 1
                rank = prev_rank

            click.echo("{:<5} {:<50} {:<10.2f} {:<10} {:<10} {:<15}".format(
                rank,
                entry["model"],
                entry["average_score"],
                entry["min_score"],
                entry["max_score"],
                entry["num_judgments"]
            ))

            prev_score = entry["average_score"]
            prev_rank = rank
