import click
import llm
import json
from typing import Dict, List, Optional
import logging
import sys
import os
import pathlib
import sqlite_utils
from datetime import datetime
import time
from typing import Any, Callable

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

def setup_logging() -> None:
    """Configure logging to write to both file and console."""
    log_path = user_dir() / "judge.log"
    log_path.parent.mkdir(exist_ok=True, parents=True)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.ERROR)
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

Please structure your evaluation as follows:

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
[Brief explanation of the score]
</explanation>"""

class DatabaseConnection:
    _instance: Optional['DatabaseConnection'] = None
    
    def __init__(self):
        self.db = sqlite_utils.Database(logs_db_path())
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Ensure required tables exist in the database."""
        self.db["responses"].create({
            "id": str,
            "model": str,
            "prompt": str,
            "response": str,
            "timestamp": str
        }, pk="id", if_not_exists=True)
        
        self.db["judgments"].create({
            "id": str,
            "response_id": str,
            "judge_model": str,
            "score": int,
            "explanation": str,
            "timestamp": str
        }, pk="id", if_not_exists=True)
    
    @classmethod
    def get_connection(cls) -> sqlite_utils.Database:
        """Get singleton database connection."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance.db

class JudgeOrchestrator:
    def __init__(self, models: List[str], max_retries: int = 1):
        self.models = models
        self.db = DatabaseConnection.get_connection()
        self.max_retries = max_retries
    
    def _retry_model_call(self, func: Callable, *args, **kwargs) -> Any:
        """Helper to retry model calls with exponential backoff."""
        last_error = None
        for attempt in range(self.max_retries + 1):  # +1 for initial try
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # exponential backoff
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed: {e}")
        raise last_error

    def get_answer(self, model: str, prompt: str) -> Dict:
        """Get an answer from a specific model."""
        try:
            model_instance = llm.get_model(model)
            system_prompt = "You are a helpful AI assistant. Please provide a clear, accurate answer."
            full_prompt = f"{system_prompt}\n\nQuestion: {prompt}"
            
            response = self._retry_model_call(
                model_instance.prompt,
                prompt=full_prompt
            )
            
            response_dict = {
                "id": f"{model}-{datetime.now().isoformat()}",
                "model": model,
                "prompt": prompt,
                "response": str(response),
                "timestamp": datetime.now().isoformat()
            }
            
            # Log to database
            self.db["responses"].insert(response_dict)
            logger.debug(f"Response from {model} logged to database")
            
            return response_dict
        except Exception as e:
            logger.error(f"Error getting response from {model}: {e}")
            return {
                "id": f"{model}-{datetime.now().isoformat()}",
                "model": model,
                "prompt": prompt,
                "response": f"Error: {str(e)}",
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
        
        try:
            model_instance = llm.get_model(judge_model)
            judgment = self._retry_model_call(
                model_instance.prompt,
                prompt=judge_prompt
            )
            judgment_text = str(judgment)
            
            # Parse judgment components
            analysis = ""
            if "<analysis>" in judgment_text and "</analysis>" in judgment_text:
                analysis = judgment_text.split("<analysis>")[1].split("</analysis>")[0].strip()
            
            score = -100
            if "<score>" in judgment_text and "</score>" in judgment_text:
                try:
                    score = int(judgment_text.split("<score>")[1].split("</score>")[0].strip())
                except ValueError:
                    pass
            
            explanation = "No explanation provided"
            if "<explanation>" in judgment_text and "</explanation>" in judgment_text:
                explanation = judgment_text.split("<explanation>")[1].split("</explanation>")[0].strip()
            
            judgment_dict = {
                "id": f"judgment-{judge_model}-{datetime.now().isoformat()}",
                "response_id": response["id"],
                "judge_model": judge_model,
                "score": score,
                "explanation": explanation,
                "timestamp": datetime.now().isoformat()
            }
            
            # Log to database
            self.db["judgments"].insert(judgment_dict)
            logger.debug(f"Judgment from {judge_model} logged to database")
            
            judgment_dict["analysis"] = analysis
            return judgment_dict
            
        except Exception as e:
            logger.error(f"Error getting judgment from {judge_model}: {e}")
            return {
                "id": f"judgment-{judge_model}-{datetime.now().isoformat()}",
                "response_id": response["id"],
                "judge_model": judge_model,
                "score": -100,
                "explanation": f"Error during judgment: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    def orchestrate(self, prompt: str) -> Dict:
        """Get answers from all models and have them judge each other."""
        # Get answers from all models
        answers = []
        for model in self.models:
            try:
                answer = self.get_answer(model, prompt)
                if not answer["response"].startswith("Error:"):
                    answers.append(answer)
                else:
                    logger.warning(f"Skipping failed response from {model}: {answer['response']}")
            except Exception as e:
                logger.error(f"Failed to get response from {model}: {e}")
                continue
        
        # Have each model judge other models' responses
        judgments = []
        for answer in answers:
            response_judgments = []
            for judge_model in self.models:
                if judge_model != answer["model"]:  # Don't let models judge themselves
                    try:
                        judgment = self.judge_response(judge_model, answer)
                        if not judgment.get("explanation", "").startswith("Error:"):
                            response_judgments.append(judgment)
                        else:
                            logger.warning(f"Skipping failed judgment from {judge_model}: {judgment['explanation']}")
                    except Exception as e:
                        logger.error(f"Failed to get judgment from {judge_model}: {e}")
                        continue
            
            if response_judgments:  # Only include responses that got at least one judgment
                judgments.append({
                    "model": answer["model"],
                    "response": answer["response"],
                    "judgments": response_judgments
                })
        
        return {
            "prompt": prompt,
            "timestamp": datetime.now().isoformat(),
            "results": judgments
        }

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
        default=1,
        help="Maximum number of retries for model calls",
    )
    def judge(prompt: str, models: Optional[List[str]] = None, output: Optional[str] = None, max_retries: int = 1):
        """Have LLMs answer a question and judge each other's responses."""
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
        orchestrator = JudgeOrchestrator(list(models), max_retries=max_retries)
        results = orchestrator.orchestrate(prompt)
        
        # Save full results if requested
        if output:
            with open(output, 'w') as f:
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
