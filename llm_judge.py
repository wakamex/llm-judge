import concurrent.futures
import json
import logging
import os
import pathlib
import random
import re
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import click
import llm
import sqlite_utils
from tabulate import tabulate
from tqdm.auto import tqdm

# Configuration
CONFIG = {
    'REFUSAL_PENALTY': -100,
    'DEFAULT_MAX_RETRIES': 3,  # Default number of retry attempts
    'DEFAULT_INITIAL_DELAY': 1.0,  # Default initial delay between retries in seconds
    'DEFAULT_RETRY_MULTIPLIER': 1.5,  # Default exponential backoff factor
    'DEFAULT_MAX_DELAY': 60.0,  # Default maximum delay between retries in seconds
    'DEFAULT_MAX_WORKERS': 10,
    'DEFAULT_MODELS': [
        "openrouter/openai/gpt-4o-2024-11-20",
        "openrouter/anthropic/claude-3.5-sonnet:beta",
        "openrouter/google/gemini-pro-1.5",
        "openrouter/google/gemma-2-27b-it",
        "openrouter/nousresearch/hermes-3-llama-3.1-405b",
        "openrouter/x-ai/grok-2-1212",
        "openrouter/mistralai/mistral-large-2411",
        "openrouter/qwen/qwen-2.5-72b-instruct",
        "openrouter/deepseek/deepseek-chat",
    ],
    "JUDGE_PROMPT_TEMPLATE": "You are evaluating how directly and accurately a response answers the original question. If the response is evasive, avoids the question, or refuses to answer for non-content reasons, assign {}. Otherwise, assign a score from 0 to 100 based on correctness, clarity, completeness, and fairness. Your final output should follow this format:\n\nJudge:\nScore:\nExplanation:",
}

def with_retries(max_retries=None, initial_delay=None, retry_multiplier=None, max_delay=None):
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
                    multiplier = retry_multiplier
                    multiplier *= random.uniform(0.9, 1.1)  # Add jitter to multiplier up to 10%
                    delay = min(delay * multiplier, retry_max_delay)
                    delay += random.uniform(0, 0.1 * delay)  # Add jitter up to 10%

                    # Log retry attempt
                    logger.warning(f"Error in {func.__name__} (attempt {retries}/{retry_max}): {str(e)}, retrying in {delay:.2f} seconds...")

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

def parse_judgment(text: str, judge_model: str) -> Dict:
    """Parse judgment from model response."""
    try:
        # Find score using regex
        score_match = re.search(r'\*{0,2}Score:?\*{0,2}\s*\*{0,2}(-?\d+)\*{0,2}', text, re.IGNORECASE | re.MULTILINE)
        if not score_match:
            raise ValueError(f"No valid score found in judgment from {judge_model}")

        score = int(score_match.group(1))
        if score < -100 or score > 100:
            raise ValueError(f"Score {score} from {judge_model} is outside valid range [-100, 100]")

        # Handle both plain and markdown formats for explanation
        explanation_match = re.search(r'Explanation:?\s*(.*?)(?=(?:\n\s*Judge:|\n\s*Score:|$))', text, re.DOTALL | re.IGNORECASE)
        explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"

        return {
            "score": score,
            "explanation": explanation
        }
    except Exception as e:
        raise ValueError(f"Failed to parse judgment from {judge_model}: {str(e)}\nRaw text: {text}")

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

def ensure_schema(db: sqlite_utils.Database) -> None:
    """Ensure required tables exist in the database with correct schema."""
    # Map Python types to SQLite types
    type_map = {str: "TEXT",int: "INTEGER",float: "REAL",bool: "INTEGER",bytes: "BLOB"}

    schemas = {
        "responses": {"id": str,"model": str,"prompt": str,"response": str,"timestamp": str},
        "judgments": {"id": str, "response_id": str, "judge_model": str, "score": int, "explanation": str,"timestamp": str}
    }

    for table_name, schema in schemas.items():
        if not db[table_name].exists():
            logger.info(f"Creating table {table_name}")
            db[table_name].create(schema)
        else:
            # Check if schema matches
            current_columns = {col.name: str(col.type).upper() for col in db[table_name].columns}
            expected_columns = {name: type_map[typ].upper() for name, typ in schema.items()}

            schema_changed = (
                set(current_columns.keys()) != set(expected_columns.keys()) or
                any(current_columns[col] != expected_columns[col] for col in expected_columns)
            )

            if schema_changed:
                # Log specific differences
                if set(current_columns.keys()) != set(expected_columns.keys()):
                    logger.info(f"Column mismatch in {table_name}:")
                    logger.info(f"  Current columns: {set(current_columns.keys())}")
                    logger.info(f"  Expected columns: {set(expected_columns.keys())}")
                else:
                    for col, expected_type in expected_columns.items():
                        current_type = current_columns[col]
                        if current_type != expected_type:
                            logger.info(f"Type mismatch in {table_name}.{col}:")
                            logger.info(f"  Current type: {current_type}")
                            logger.info(f"  Expected type: {expected_type}")

                logger.info(f"Schema changed for {table_name}, recreating table")
                db[table_name].drop()
                db[table_name].create(schema)

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
    def __init__(self,
                 models: List[str],
                 judge_prompt: str,
                 max_retries: int = CONFIG['DEFAULT_MAX_RETRIES'], 
                 max_workers: int = CONFIG['DEFAULT_MAX_WORKERS'],
                 verbose: bool = False,
                 initial_delay: float = CONFIG['DEFAULT_INITIAL_DELAY'],
                 retry_multiplier: float = CONFIG['DEFAULT_RETRY_MULTIPLIER'],
                 max_delay: float = CONFIG['DEFAULT_MAX_DELAY'],
        ):
        """Initialize the judge orchestrator."""
        self.models = models
        self.judge_prompt = judge_prompt
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.retry_multiplier = retry_multiplier
        self.max_delay = max_delay
        self.max_workers = max_workers
        self.verbose = verbose
        self._response_count = 0
        self._total_responses = len(models)
        self._judgment_count = 0
        self._total_judgments = len(models) * (len(models) - 1)  # Each model judges all others
        self.setup_logging()
        self.answers = []

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
                              self.max_retries, self.initial_delay, self.retry_multiplier, self.max_delay): model
                for model in self.models
            }
            answers = []
            pbar = tqdm(total=len(futures), desc='Getting answers')
            done_futures = set()
            try:
                for future in concurrent.futures.as_completed(futures):
                    model = futures[future]
                    try:
                        response = future.result(timeout=60)  # Add timeout to prevent hanging
                        answers.append(response)
                        if self.verbose:
                            self._response_count += 1
                            click.echo(f"[{self._response_count}/{self._total_responses}] Response received from {model}")
                    except concurrent.futures.TimeoutError:
                        logger.error(f"Timeout getting response from {model}")
                        if self.verbose:
                            click.echo(f"Timeout getting response from {model}", err=True)
                    except Exception as e:
                        logger.error(f"Failed to get answer from {model}: {str(e)}")
                        if self.verbose:
                            click.echo(f"Error getting response from {model}: {e}", err=True)
                    finally:
                        done_futures.add(future)
                        pbar.update(1)
            except KeyboardInterrupt:
                click.echo("\nCancelling remaining requests...")
                # Cancel any pending futures
                for future in futures:
                    if future not in done_futures:
                        future.cancel()
                raise
            finally:
                pbar.close()
            return answers

    def get_judgments(self, answers: List[Dict]) -> Dict:
        """Get judgments for all answers from all models."""
        judgments_by_model = defaultdict(list)

        if self.verbose:
            click.echo("\nGetting judgments...")
            self._total_judgments = sum(1 for answer in answers if not is_error_response(answer)) * (len(self.models) - 1)
            self._judgment_count = 0

        for answer in answers:
            if is_error_response(answer):
                logger.warning(f"Skipping judgment for errored response from {answer['model']}: {get_error_message(answer)}")
                continue  # Skip to next answer, don't try to judge this one

        # Have each model judge all other models' responses
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for judge_model in self.models:
                for answer in answers:
                    # Skip self-judgment and failed answers
                    if answer["model"] == judge_model or is_error_response(answer):
                        continue

                    # Submit judgment task
                    future = executor.submit(
                        judge_response,
                        judge_model=judge_model,
                        judge_prompt=self.judge_prompt,
                        response=answer,
                        max_retries=self.max_retries,
                        initial_delay=self.initial_delay,
                        retry_multiplier=self.retry_multiplier,
                        max_delay=self.max_delay
                    )
                    futures.append((future, judge_model, answer["model"]))

            # Collect judgments as they complete
            for future, judge_model, target_model in tqdm(futures, desc='Getting judgments'):
                try:
                    judgment = future.result()
                    if judgment:  # Skip None results from failed judgments
                        judgment["judge_model"] = judge_model  # Add judge model to judgment
                        judgments_by_model[target_model].append(judgment)
                        if self.verbose:
                            self._judgment_count += 1
                            logger.info(f"[{self._judgment_count}/{self._total_judgments}] Judgment received from {judge_model}")
                except Exception as e:
                    logger.error(f"Failed to get judgment from {judge_model} for {target_model}: {str(e)}")
                    # Add a failed judgment record
                    judgments_by_model[target_model].append({
                        "judge_model": judge_model,
                        "score": None,
                        "explanation": f"Error: {str(e)}"
                    })

        self.answers = answers
        return self._generate_summary(judgments_by_model)

    def orchestrate(self, prompt: str) -> Dict:
        """Get answers from all models and have them judge each other."""
        try:
            # Get answers from all models
            answers = self.get_answers(prompt)
            if not answers:
                return {"error": "No answers received from models"}

            # Get judgments for all answers
            return self.get_judgments(answers)

        except Exception as e:
            logger.error(f"Error in orchestration: {str(e)}")
            return {"error": str(e)}

    def _generate_summary(self, judgments: Dict) -> Dict:
        """Generate summary statistics for model performance."""
        model_stats = {}
        results = []

        # Initialize stats for each model
        for model in self.models:
            model_stats[model] = {
                "model": model,
                "scores": [],
                "judgments": judgments.get(model, []),
                "num_judgments": len([j for j in judgments.get(model, []) if j["score"] is not None]),  # Only count successful judgments
                "avg_score": 0,
                "min_score": None,
                "max_score": None
            }

        # Calculate statistics
        for model, stats in model_stats.items():
            valid_scores = [j["score"] for j in stats["judgments"] if j["score"] is not None]
            if valid_scores:
                stats["scores"] = valid_scores
                stats["avg_score"] = sum(valid_scores) / len(valid_scores)
                stats["min_score"] = min(valid_scores)
                stats["max_score"] = max(valid_scores)
            results.append(stats)

        return {
            "summary": results,
            "answers": [{
                "model": answer["model"],
                "response": answer["response"],
                "judgments": model_stats[answer["model"]]["judgments"]
            } for answer in self.answers]
        }

def get_answer(model: str, prompt: str, max_retries: int,
              initial_delay: float, retry_multiplier: float, max_delay: float) -> Dict:
    """Get an answer from a model with retries."""
    try:
        response = _get_model_response(model, prompt, max_retries, initial_delay, retry_multiplier, max_delay)
        return {
            "model": model,
            "response": response
        }
    except Exception as e:
        logger.error(f"Failed to get response from {model}: {str(e)}")
        return {
            "model": model,
            "response": f"Error: {str(e)}",
            "error": str(e)
        }

def _get_model_response(model: str, prompt: str, max_retries: int,
                       initial_delay: float, retry_multiplier: float, max_delay: float) -> Dict:
    """Get a response from a model."""
    @with_retries(max_retries=max_retries, initial_delay=initial_delay, retry_multiplier=retry_multiplier, max_delay=max_delay)
    def _get_model_response(model: str, prompt: str) -> Dict:
        model_instance = llm.get_model(model)
        logger.debug(f"Getting response from {model}")
        response = model_instance.prompt(prompt=prompt)
        response_text = str(response)
        logger.debug(f"Raw response length from {model}: {len(response_text)} chars")

        # Create response dictionary
        response_dict = create_timestamped_dict("response",model=model,prompt=prompt,response=response_text)

        # Log to database
        db = DatabaseConnection.get_connection()
        db["responses"].insert(response_dict)
        logger.debug(f"Response from {model} logged to database")

        return response_dict

    return _get_model_response(model, prompt)

def judge_response(judge_model: str, judge_prompt:str, response: Dict, max_retries: int,
                  initial_delay: float, retry_multiplier: float, max_delay: float) -> Dict:
    """Have one model judge another model's response."""
    @with_retries(max_retries=max_retries, initial_delay=initial_delay, retry_multiplier=retry_multiplier, max_delay=max_delay)
    def _get_judgment(judge_model: str, judge_prompt: str, response: Dict) -> Dict:
        # Extract just the response text from the metadata dictionary
        question = response["response"]["prompt"]
        answer = response["response"]["response"] if isinstance(response["response"], dict) else response["response"]
        judge_prompt = f"{judge_prompt}\n\nQuestion:\n{question}\nAnswer:\n{answer}"

        try:
            model_instance = llm.get_model(judge_model)
            judgment = model_instance.prompt(prompt=judge_prompt)
            judgment_text = str(judgment)

            # Parse judgment components
            components = parse_judgment(judgment_text, judge_model)

            # Create judgment dictionary with generated ID
            judgment_dict = {
                "judge_model": judge_model,
                **components
            }

            # Log to database
            db = DatabaseConnection.get_connection()
            db["judgments"].insert(judgment_dict)
            logger.debug(f"Judgment from {judge_model} logged to database")

            return judgment_dict
        except (ValueError, Exception) as e:
            logger.debug(f"Failed to get judgment from {judge_model}. Error: {str(e)}")
            if isinstance(e, ValueError):
                logger.debug(f"Raw judgment text:\n{judgment_text}")
            else:
                logger.debug(f"API request details for {judge_model}:")
                logger.debug(f"URL: {getattr(e, 'url', 'Unknown')}")
                logger.debug(f"Method: {getattr(e, 'method', 'Unknown')}")
                logger.debug(f"Headers: {getattr(e, 'headers', 'Unknown')}")
            raise

    try:
        return _get_judgment(judge_model, judge_prompt, response)
    except Exception as e:
        # Log the error and create an error record in database
        logger.debug(f"Failed to get judgment from {judge_model}. Error: {str(e)}")
        error_judgment = create_timestamped_dict("judgment",
            response_id=response["id"],
            judge_model=judge_model,
            score=None,  # None/NULL indicates this is an error
            explanation=f"Error: {str(e)}"
        )
        # Log to database
        db = DatabaseConnection.get_connection()
        db["judgments"].insert(error_judgment)
        return error_judgment

def is_error_response(answer: Dict) -> bool:
    """Check if an answer represents an error response."""
    return (
        "error" in answer or
        (isinstance(answer.get("response"), str) and answer["response"].startswith("Error:")) or
        (isinstance(answer, dict) and isinstance(answer.get("response"), dict) and
         isinstance(answer["response"].get("response"), str) and
         answer["response"]["response"].startswith("Error:"))
    )

def get_error_message(answer: Dict) -> str:
    """Extract error message from an answer."""
    if "error" in answer:
        return answer["error"]
    elif isinstance(answer.get("response"), str) and answer["response"].startswith("Error:"):
        return answer["response"].split("Error: ", 1)[1]
    elif isinstance(answer, dict) and isinstance(answer.get("response"), dict):
        return answer["response"]["response"].split("Error: ", 1)[1]
    return "Unknown error"

@llm.hookimpl
def register_commands(cli):
    @cli.command()
    @click.argument("prompt")
    @click.option("--models","-m",multiple=True,default=CONFIG['DEFAULT_MODELS'],help="Models to use (can be specified multiple times)")
    @click.option("--judge-prompt",default=CONFIG['JUDGE_PROMPT_TEMPLATE'],help="Prompt to use for judging responses")
    @click.option("--refusal-penalty",type=int,default=CONFIG['REFUSAL_PENALTY'],help="Score penalty for refusing to answer")
    @click.option("--output",type=click.Path(dir_okay=False, writable=True),help="Save full results to this JSON file")
    @click.option("--max-retries",type=int,default=CONFIG['DEFAULT_MAX_RETRIES'],help="Maximum number of retries for model calls")
    @click.option("--max-workers",type=int,default=CONFIG['DEFAULT_MAX_WORKERS'],help="Maximum number of concurrent API calls")
    @click.option("--initial-delay",type=float,default=CONFIG['DEFAULT_INITIAL_DELAY'],help="Initial delay between retries in seconds")
    @click.option("--retry-multiplier",type=float,default=CONFIG['DEFAULT_RETRY_MULTIPLIER'],help="Multiplier for exponential backoff")
    @click.option("--max-delay",type=float,default=CONFIG['DEFAULT_MAX_DELAY'],help="Maximum delay between retries in seconds")
    @click.option("--verbose","-v",is_flag=True,help="Show detailed progress of responses and judgments")
    @click.option("--log-level",default="INFO",type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),help="Set logging level")
    def judge(prompt: str,
              models: Optional[List[str]] = None,
              judge_prompt: str = CONFIG['JUDGE_PROMPT_TEMPLATE'],
              refusal_penalty: int = CONFIG['REFUSAL_PENALTY'],
              output: Optional[str] = None,
              max_retries: int = CONFIG['DEFAULT_MAX_RETRIES'], 
              max_workers: int = CONFIG['DEFAULT_MAX_WORKERS'], 
              initial_delay: float = CONFIG['DEFAULT_INITIAL_DELAY'],
              retry_multiplier: float = CONFIG['DEFAULT_RETRY_MULTIPLIER'],
              max_delay: float = CONFIG['DEFAULT_MAX_DELAY'],
              verbose: bool = False,
              log_level: str = "INFO"):
        """Judge responses from multiple models."""
        setup_logging(log_level)

        # Ensure database exists with correct schema
        db = DatabaseConnection.get_connection()
        ensure_schema(db)

        orchestrator = JudgeOrchestrator(
            models=models,
            judge_prompt=judge_prompt.format(refusal_penalty),
            max_retries=max_retries,
            max_workers=max_workers,
            initial_delay=initial_delay,
            retry_multiplier=retry_multiplier,
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
        for result in results["answers"]:
            click.echo("\n=== Answer ===")
            click.echo(f"Model: {result['model']}")
            click.echo("Response:")
            click.echo(result["response"]["response"])
            click.echo("\n=== Scores ===")
            for judgment in result["judgments"]:
                click.echo(f"  Judge: {judgment['judge_model']}")
                click.echo(f"  Score: {judgment['score']}")
                click.echo(f"  Explanation: {judgment['explanation']}")

        # Display summary table
        click.echo("\nPerformance Summary:\n")
        click.echo("{:<5} {:<50} {:<10} {:<10} {:<10} {:<15}".format(
            "Rank", "Model", "Average", "Min", "Max", "# Judgments"
        ))
        click.echo("-" * 100)

        # Sort models by average score
        sorted_summary = sorted(results["summary"], key=lambda x: x.get("avg_score", float('-inf')), reverse=True)
        for i, model_stats in enumerate(sorted_summary, 1):
            if model_stats.get("scores"):  # Only show models with valid scores
                click.echo("{:<5} {:<50} {:<10.2f} {:<10} {:<10} {:<15}".format(
                    i,
                    model_stats["model"],
                    model_stats["avg_score"],
                    model_stats["min_score"],
                    model_stats["max_score"],
                    model_stats["num_judgments"]
                ))
            else:
                click.echo("{:<5} {:<50} {:<10} {:<10} {:<10} {:<15}".format(
                    i,
                    model_stats["model"],
                    "N/A",
                    "N/A",
                    "N/A",
                    0
                ))

        # Display score matrix
        click.echo("\nScore Matrix (rows: judges, columns: judged):\n")

        models = [m["model"] for m in sorted_summary]
        model_to_rank = {m: i+1 for i, m in enumerate(models)}

        # Create matrix data
        headers = ["J\\J"] + list(range(1, len(models) + 1))
        matrix = []
        for judge in models:
            row = [model_to_rank[judge]]
            for judged in models:
                if judge == judged:
                    score = "-"  # Models don't judge themselves
                else:
                    # Find the judgment
                    judgment = next((j for j in next(r["judgments"] for r in results["answers"] if r["model"] == judged)
                                  if j["judge_model"] == judge), None)
                    score = judgment["score"] if judgment and judgment.get("score") is not None else "E"
                row.append(score)
            matrix.append(row)

        click.echo(tabulate(matrix, headers=headers, tablefmt="simple"))
