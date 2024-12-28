import concurrent.futures
import json
import logging
import os
import pathlib
import random
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import click
import llm
import sqlite_utils
import sqlite3

# Configuration
CONFIG = {
    'SCORE_RANGE': (-100, 100),
    'DEFAULT_MAX_RETRIES': 3,  # Default number of retry attempts
    'DEFAULT_INITIAL_DELAY': 1.0,  # Default initial delay between retries in seconds
    'DEFAULT_RETRY_MULTIPLIER': 1.5,  # Default exponential backoff factor
    'DEFAULT_MAX_DELAY': 60.0,  # Default maximum delay between retries in seconds
    'DEFAULT_MAX_WORKERS': 10,
    'DEFAULT_MODELS': [
        "openrouter/openai/gpt-4o-2024-11-20",
        "openrouter/anthropic/claude-3.5-sonnet:beta",
        "openrouter/google/gemini-2.0-flash-exp:free",
        "openrouter/nousresearch/hermes-3-llama-3.1-405b",
        "openrouter/x-ai/grok-2-1212",
        "openrouter/deepseek/deepseek-chat",
        "openrouter/mistralai/mistral-large-2411",
    ]
}

def with_retries(max_retries=None, initial_delay=None, max_delay=None):
    """Decorate functions to retry with exponential backoff."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # If inside a class method, get retry params from instance
            if args and hasattr(args[0], 'max_retries'):
                instance = args[0]
                retry_max = instance.max_retries
                retry_initial = instance.initial_delay
                retry_max_delay = instance.max_delay
            else:
                retry_max = max_retries or CONFIG['DEFAULT_MAX_RETRIES']
                retry_initial = initial_delay or CONFIG['DEFAULT_INITIAL_DELAY']
                retry_max_delay = max_delay or CONFIG['DEFAULT_MAX_DELAY']

            retries = 0
            delay = retry_initial
            last_exception = None

            while retries <= retry_max:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    retries += 1

                    if retries > retry_max:
                        logger.error(f"Max retries ({retry_max}) exceeded for {func.__name__}")
                        raise last_exception

                    # Calculate next delay with exponential backoff and jitter
                    # jitter = random.uniform(0, 0.1 * delay)
                    multiplier = CONFIG['DEFAULT_RETRY_MULTIPLIER']
                    multiplier *= random.uniform(0.9, 1.1)  # Add jitter to multiplier up to 10%
                    delay = min(delay * multiplier, retry_max_delay)
                    delay += random.uniform(0, 0.1 * delay)  # Add jitter up to 10%

                    # Log retry attempt
                    logger.warning(
                        f"Error in {func.__name__} (attempt {retries}/{retry_max}): {str(e)}\n"
                        f"Retrying in {delay:.2f} seconds..."
                    )

                    # Always use exponential backoff
                    delay = min(delay * 2, retry_max_delay)
                    logger.warning(f"Failed to get response, increasing delay to {delay:.2f} seconds")

                    time.sleep(delay)

            raise last_exception

        return wrapper
    return decorator

def create_timestamped_dict(prefix: str, **kwargs) -> Dict:
    """Create a dictionary with an ID and timestamp."""
    return {
        "id": f"{prefix}-{datetime.now().isoformat()}",
        "timestamp": datetime.now().isoformat(),
        **kwargs
    }

def parse_judgment(judgment_text: str, judge_model: str) -> Dict:
    """Parse judgment components from raw text."""
    logger.debug(f"\nParsing judgment from {judge_model}:\n{'-'*80}\n{judgment_text}\n{'-'*80}")
    components = {}

    # Parse score (required)
    if "<score>" in judgment_text and "</score>" in judgment_text:
        score_text = judgment_text.split("<score>")[1].split("</score>")[0].strip()
        try:
            score = int(''.join(c for c in score_text if c.isdigit() or c == '-'))
            if not (CONFIG['SCORE_RANGE'][0] <= score <= CONFIG['SCORE_RANGE'][1]):
                raise ValueError(f"Score {score} out of valid range {CONFIG['SCORE_RANGE']}")
            components['score'] = score
            logger.debug(f"Successfully parsed score: {score}")
        except ValueError as e:
            logger.error(f"Failed to parse score from {judge_model}, raw text: '{score_text}'")
            raise ValueError(f"Invalid score format: {score_text}") from e
    else:
        raise ValueError("Missing score tags in judgment")

    # Parse analysis (required)
    if "<analysis>" in judgment_text and "</analysis>" in judgment_text:
        components['analysis'] = judgment_text.split("<analysis>")[1].split("</analysis>")[0].strip()
        logger.debug(f"Successfully parsed analysis: {components['analysis'][:100]}...")
    else:
        raise ValueError("Missing analysis tags in judgment")

    # Parse explanation (optional)
    if "<explanation>" in judgment_text and "</explanation>" in judgment_text:
        components['explanation'] = judgment_text.split("<explanation>")[1].split("</explanation>")[0].strip()
        logger.debug(f"Successfully parsed explanation: {components['explanation'][:100]}...")
    else:
        logger.warning(f"No explanation tags found in judgment from {judge_model}, using analysis as explanation")
        components['explanation'] = components['analysis']

    return components

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

    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers = []

    # Configure formatters
    formatter = logging.Formatter('%(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console handler with simple format
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler with detailed format
    file_handler = logging.FileHandler(str(log_path))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Set root logger level to the lowest level of any handler
    root_logger.setLevel(logging.DEBUG)

setup_logging()
logging.getLogger("httpx").setLevel(logging.WARNING)
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

def ensure_schema(db: sqlite_utils.Database) -> None:
    """Ensure required tables exist in the database with correct schema."""
    schemas = {
        "responses": {
            "id": str,
            "model": str,
            "prompt": str,
            "response": str,
            "timestamp": str
        },
        "judgments": {
            "id": str,
            "response_id": str,
            "judge_model": str,
            "score": int,
            "analysis": str,
            "explanation": str,
            "timestamp": str
        }
    }

    for table_name, schema in schemas.items():
        try:
            if table_name not in db.tables:
                db[table_name].create(schema, pk="id")
                logger.debug(f"Created table {table_name}")
        except sqlite3.OperationalError as e:
            if "already exists" not in str(e):
                raise

class DatabaseConnection:
    """Manage thread-local SQLite database connections."""

    _local = threading.local()

    def __init__(self):
        """Initialize the database connection."""
        if not hasattr(self._local, 'db'):
            db_path = logs_db_path()
            self._local.db = sqlite_utils.Database(db_path)

    @classmethod
    def get_connection(cls) -> sqlite_utils.Database:
        """Get thread-local database connection."""
        if not hasattr(cls._local, 'db'):
            cls()
        return cls._local.db

class JudgeOrchestrator:
    def __init__(self, models: List[str], max_retries: int = CONFIG['DEFAULT_MAX_RETRIES'], 
                 max_workers: int = CONFIG['DEFAULT_MAX_WORKERS'], verbose: bool = False,
                 initial_delay: float = CONFIG['DEFAULT_INITIAL_DELAY'], max_delay: float = CONFIG['DEFAULT_MAX_DELAY']):
        """Initialize the judge orchestrator."""
        self.models = models
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.max_workers = max_workers
        self.verbose = verbose
        self._response_count = 0
        self._total_responses = len(models)
        self._judgment_count = 0
        self._total_judgments = len(models) * (len(models) - 1)  # Each model judges all others
        self.setup_logging()

    def setup_logging(self):
        """Set up logging based on verbosity."""
        if self.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)

    def get_answers(self, prompt: str) -> List[Dict]:
        """Get answers from all models."""
        if self.verbose:
            click.echo("\nGetting responses from models...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(get_answer, model, prompt,
                              self.max_retries, self.initial_delay, self.max_delay): model
                for model in self.models
            }
            answers = []
            for future in concurrent.futures.as_completed(futures):
                model = futures[future]
                try:
                    response = future.result()
                    answers.append(response)
                    if self.verbose:
                        self._response_count += 1
                        click.echo(f"[{self._response_count}/{self._total_responses}] Response received from {model}")
                except Exception as e:
                    logger.error(f"Failed to get answer from {model}: {str(e)}")
                    if self.verbose:
                        click.echo(f"❌ Failed to get response from {model}: {str(e)}")
            return answers

    def get_judgments(self, answers: List[Dict]) -> Dict[str, List[Dict]]:
        """Get judgments for all answers from all models."""
        if self.verbose:
            click.echo("\nGetting judgments...")

        judgments_by_model = defaultdict(list)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for answer in answers:
                for judge_model in self.models:
                    if judge_model != answer["model"]:  # Don't let models judge themselves
                        futures.append((
                            executor.submit(judge_response, judge_model, answer,
                                         self.max_retries, self.initial_delay, self.max_delay),
                            answer["model"],
                            judge_model
                        ))

            for future, model, judge_model in futures:
                try:
                    response = future.result()
                    judgments_by_model[model].append(response)
                    if self.verbose:
                        self._judgment_count += 1
                        click.echo(f"[{self._judgment_count}/{self._total_judgments}] {judge_model} judging {model} → Score: {response['score']}")
                except Exception as e:
                    logger.error(f"Failed to get judgment: {str(e)}")
                    if self.verbose:
                        click.echo(f"❌ Error from {judge_model} judging {model}: {str(e)}")

        return judgments_by_model

    def orchestrate(self, prompt: str) -> Dict:
        """Get answers from all models and have them judge each other."""
        answers = self.get_answers(prompt)
        judgments_by_model = self.get_judgments(answers)

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
            model_stats[model] = {"avg_score": 0,"min_score": float('inf'),"max_score": float('-inf'),"scores": [],"num_judgments": 0}

        # Collect scores
        for result in judgments:
            model = result["model"]
            for judgment in result["judgments"]:
                score = judgment["score"]

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

def get_answer(model: str, prompt: str, max_retries: int,
              initial_delay: float, max_delay: float) -> Dict:
    """Get an answer from a specific model."""
    @with_retries(max_retries=max_retries, initial_delay=initial_delay, max_delay=max_delay)
    def _get_model_response(model: str, prompt: str) -> Dict:
        try:
            model_instance = llm.get_model(model)
            logger.debug(f"Getting response from {model}")
            response = model_instance.prompt(prompt=prompt)
            response_text = str(response)
            logger.debug(f"Raw response length from {model}: {len(response_text)} chars")

            # Create response dictionary
            response_dict = create_timestamped_dict("response",
                model=model,
                prompt=prompt,
                response=response_text
            )

            # Log to database
            db = DatabaseConnection.get_connection()
            db["responses"].insert(response_dict)
            logger.debug(f"Response from {model} logged to database")

            return response_dict
        except Exception as e:
            logger.error(f"API request details for {model}:")
            logger.error(f"URL: {getattr(e, 'url', 'Unknown')}")
            logger.error(f"Method: {getattr(e, 'method', 'Unknown')}")
            logger.error(f"Headers: {getattr(e, 'headers', 'Unknown')}")
            raise

    try:
        return _get_model_response(model, prompt)
    except Exception as e:
        # Create error response
        return create_timestamped_dict("response",
            model=model,
            prompt=prompt,
            response=f"Error: {str(e)}"
        )

def judge_response(judge_model: str, response: Dict, max_retries: int,
                  initial_delay: float, max_delay: float) -> Dict:
    """Have one model judge another model's response."""
    @with_retries(max_retries=max_retries, initial_delay=initial_delay, max_delay=max_delay)
    def _get_judgment(judge_model: str, response: Dict) -> Dict:
        judge_prompt = f"""{JUDGE_SYSTEM_PROMPT}

Here is an LLM's response to evaluate:

Question:
{response['prompt']}

Response to evaluate:
{response['response']}"""

        try:
            model_instance = llm.get_model(judge_model)
            judgment = model_instance.prompt(prompt=judge_prompt)
            judgment_text = str(judgment)

            # Parse judgment components
            components = parse_judgment(judgment_text, judge_model)

            # Create judgment dictionary
            judgment_dict = create_timestamped_dict("judgment",
                response_id=response["id"],
                judge_model=judge_model,
                **components
            )

            # Log to database
            db = DatabaseConnection.get_connection()
            db["judgments"].insert(judgment_dict)
            logger.debug(f"Judgment from {judge_model} logged to database")

            return judgment_dict
        except (ValueError, Exception) as e:
            logger.error(f"Failed to get judgment from {judge_model}. Error: {str(e)}")
            if isinstance(e, ValueError):
                logger.debug(f"Raw judgment text:\n{judgment_text}")
            else:
                logger.error(f"API request details for {judge_model}:")
                logger.error(f"URL: {getattr(e, 'url', 'Unknown')}")
                logger.error(f"Method: {getattr(e, 'method', 'Unknown')}")
                logger.error(f"Headers: {getattr(e, 'headers', 'Unknown')}")
            raise

    try:
        return _get_judgment(judge_model, response)
    except Exception as e:
        # Create error judgment
        return create_timestamped_dict("judgment",
            response_id=response["id"],
            judge_model=judge_model,
            score=CONFIG['SCORE_RANGE'][0],  # Use minimum score for failed judgments
            analysis=f"Error: {str(e)}",
            explanation=f"Failed to get judgment: {str(e)}"
        )

@llm.hookimpl
def register_commands(cli):
    @cli.command()
    @click.argument("prompt")
    @click.option("--models","-m",multiple=True,help="Models to use (can be specified multiple times)")
    @click.option("--output",type=click.Path(dir_okay=False, writable=True),help="Save full results to this JSON file")
    @click.option("--max-retries",type=int,default=CONFIG['DEFAULT_MAX_RETRIES'],help="Maximum number of retries for model calls")
    @click.option("--max-workers",type=int,default=CONFIG['DEFAULT_MAX_WORKERS'],help="Maximum number of concurrent API calls")
    @click.option("--initial-delay",type=float,default=CONFIG['DEFAULT_INITIAL_DELAY'],help="Initial delay between retries in seconds")
    @click.option("--retry-multiplier",type=float,default=CONFIG['DEFAULT_RETRY_MULTIPLIER'],help="Multiplier for exponential backoff")
    @click.option("--max-delay",type=float,default=CONFIG['DEFAULT_MAX_DELAY'],help="Maximum delay between retries in seconds")
    @click.option("--verbose","-v",is_flag=True,help="Show detailed progress of responses and judgments")
    @click.option("--log-level",default="INFO",type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),help="Set logging level")
    def judge(prompt: str, models: Optional[List[str]] = None, output: Optional[str] = None, 
              max_retries: int = CONFIG['DEFAULT_MAX_RETRIES'], 
              max_workers: int = CONFIG['DEFAULT_MAX_WORKERS'], 
              initial_delay: float = CONFIG['DEFAULT_INITIAL_DELAY'],
              retry_multiplier: float = CONFIG['DEFAULT_RETRY_MULTIPLIER'],
              max_delay: float = CONFIG['DEFAULT_MAX_DELAY'],
              verbose: bool = False, log_level: str = "INFO"):
        """Have LLMs answer a question and judge each other's responses."""
        setup_logging(log_level)

        # Ensure schema exists (in main thread)
        db = DatabaseConnection.get_connection()
        ensure_schema(db)

        if models is None:
            models = CONFIG['DEFAULT_MODELS']

        orchestrator = JudgeOrchestrator(
            models=models,
            max_retries=max_retries,
            max_workers=max_workers,
            initial_delay=initial_delay,
            max_delay=max_delay,
            verbose=verbose
        )
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

        for i, entry in enumerate(results["summary"], 1):
            click.echo("{:<5} {:<50} {:<10.2f} {:<10} {:<10} {:<15}".format(
                i, entry["model"], entry["average_score"],
                entry["min_score"], entry["max_score"], entry["num_judgments"]
            ))
