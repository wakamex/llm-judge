#!/usr/bin/env python3

import click
import sqlite_utils
from tabulate import tabulate
from typing import Dict, List, Optional
import json
from datetime import datetime

from llm_judge.llm_judge import logs_db_path

def get_db():
    """Get database connection."""
    return sqlite_utils.Database(logs_db_path())

@click.group()
def cli():
    """Query tool for LLM Judge results"""
    pass

@cli.command()
@click.option('--prompt', help='Filter by prompt text (uses LIKE)')
@click.option('--model', help='Filter by model name')
@click.option('--limit', default=10, help='Limit number of results')
def list_responses(prompt: Optional[str], model: Optional[str], limit: int):
    """List responses from the database"""
    db = get_db()

    query = ["SELECT * FROM responses WHERE 1=1"]
    params = []

    if prompt:
        query.append("AND prompt LIKE ?")
        params.append(f"%{prompt}%")

    if model:
        query.append("AND model LIKE ?")
        params.append(f"%{model}%")

    query.append("ORDER BY timestamp DESC LIMIT ?")
    params.append(limit)

    results = db.query(" ".join(query), params)

    # Format results for display
    formatted_results = []
    for r in results:
        formatted_results.append({
            'timestamp': r['timestamp'],
            'model': r['model'],
            'prompt': r['prompt'][:50] + '...' if len(r['prompt']) > 50 else r['prompt'],
            'response': r['response'][:50] + '...' if len(r['response']) > 50 else r['response']
        })

    print(tabulate(formatted_results, headers='keys', tablefmt='grid'))

@cli.command()
@click.option('--response-id', help='Filter judgments for specific response')
@click.option('--judge-model', help='Filter by judge model')
@click.option('--min-score', type=int, help='Filter by minimum score')
@click.option('--max-score', type=int, help='Filter by maximum score')
@click.option('--limit', default=10, help='Limit number of results')
def list_judgments(response_id: Optional[str], judge_model: Optional[str], 
                  min_score: Optional[int], max_score: Optional[int], limit: int):
    """List judgments from the database"""
    db = get_db()

    query = ["SELECT j.*, r.prompt, r.response, r.model as response_model FROM judgments j"]
    query.append("LEFT JOIN responses r ON j.response_id = r.id WHERE 1=1")
    params = []

    if response_id:
        query.append("AND j.response_id = ?")
        params.append(response_id)

    if judge_model:
        query.append("AND j.judge_model LIKE ?")
        params.append(f"%{judge_model}%")

    if min_score is not None:
        query.append("AND j.score >= ?")
        params.append(min_score)

    if max_score is not None:
        query.append("AND j.score <= ?")
        params.append(max_score)

    query.append("ORDER BY j.timestamp DESC LIMIT ?")
    params.append(limit)

    results = db.query(" ".join(query), params)

    # Format results for display
    formatted_results = []
    for r in results:
        formatted_results.append({
            'timestamp': r['timestamp'],
            'response_model': r['response_model'],
            'judge_model': r['judge_model'],
            'score': r['score'],
            'prompt': (r['prompt'][:50] + '...') if r['prompt'] and len(r['prompt']) > 50 else (r['prompt'] or 'N/A'),
            'explanation': (r['explanation'][:50] + '...') if r['explanation'] and len(r['explanation']) > 50 else (r['explanation'] or 'N/A')
        })

    print(tabulate(formatted_results, headers='keys', tablefmt='grid'))

@cli.command()
@click.option('--prompt', help='Filter by prompt text (uses LIKE)')
def average_scores(prompt: Optional[str]):
    """Show average scores by model"""
    db = get_db()

    query = ["""
        SELECT
            r.model as response_model,
            COUNT(DISTINCT r.id) as total_responses,
            COUNT(DISTINCT CASE
                WHEN j.score IS NOT NULL
                THEN j.id
            END) as total_judgments,
            COALESCE(AVG(NULLIF(j.score, '')), 0) as avg_score,
            MIN(NULLIF(j.score, '')) as min_score,
            MAX(NULLIF(j.score, '')) as max_score
        FROM responses r
        LEFT JOIN judgments j ON r.id = j.response_id
        WHERE 1=1
    """]
    params = []

    if prompt:
        query.append("AND r.prompt LIKE ?")
        params.append(f"%{prompt}%")

    query.append("GROUP BY r.model")
    query.append("HAVING total_judgments > 0")
    query.append("ORDER BY avg_score DESC, total_responses DESC")

    results = db.query(" ".join(query), params)
    print(tabulate(list(results), headers='keys', tablefmt='grid'))

@cli.command()
@click.option('--model', required=True, help='Model to analyze')
def score_distribution(model: str):
    """Show distribution of scores for a specific model"""
    db = get_db()
    query = """
        SELECT
            j.score,
            COUNT(*) as count,
            GROUP_CONCAT(
                SUBSTR(j.explanation, 1, 100)
            ) as sample_explanations
        FROM responses r
        JOIN judgments j ON r.id = j.response_id
        WHERE r.model LIKE ?
        AND j.score IS NOT NULL
        GROUP BY j.score
        ORDER BY j.score DESC
    """
    results = db.query(query, [f"%{model}%"])
    print(tabulate(list(results), headers='keys', tablefmt='grid'))

@cli.command()
def analyze_issues():
    """Analyze models with missing or problematic judgments"""
    db = get_db()

    # Query to find models with null or empty scores
    null_scores_query = """
        SELECT
            r.model,
            COUNT(DISTINCT r.id) as total_responses,
            COUNT(DISTINCT j.id) as total_judgments,
            COUNT(DISTINCT CASE 
                WHEN j.score IS NULL OR j.score = ''
                THEN j.id
            END) as null_scores,
            COUNT(DISTINCT CASE 
                WHEN j.score IS NOT NULL AND j.score != ''
                THEN j.id
            END) as valid_scores,
            GROUP_CONCAT(DISTINCT j.judge_model) as judge_models
        FROM responses r
        LEFT JOIN judgments j ON r.id = j.response_id
        GROUP BY r.model
        HAVING null_scores > 0 OR total_judgments = 0
        ORDER BY null_scores DESC, total_responses DESC
    """

    print("\nModels with missing or invalid scores:")
    results = db.query(null_scores_query)
    print(tabulate(list(results), headers='keys', tablefmt='grid'))

    # Get sample of problematic judgments
    sample_query = """
        SELECT
            r.model as response_model,
            j.judge_model,
            j.score,
            j.explanation,
            r.prompt,
            r.response
        FROM responses r
        JOIN judgments j ON r.id = j.response_id
        WHERE j.score IS NULL OR j.score = ''
        ORDER BY r.timestamp DESC
        LIMIT 5
    """

    print("\nSample problematic judgments:")
    samples = db.query(sample_query)
    print(tabulate(list(samples), headers='keys', tablefmt='grid'))

def get_latest_question_data():
    """Get data for the most recent question."""
    db = get_db()

    # First get the most recent prompt
    latest_prompt_query = """
        SELECT DISTINCT
            prompt,
            MIN(timestamp) as first_asked,
            MAX(timestamp) as last_asked,
            COUNT(DISTINCT id) as total_responses
        FROM responses
        GROUP BY prompt
        ORDER BY first_asked DESC
        LIMIT 1
    """
    latest_prompt = next(db.query(latest_prompt_query))

    # Get responses for this prompt
    responses_query = """
        WITH ranked_responses AS (
            SELECT
                model,
                timestamp,
                response,
                id,
                ROW_NUMBER() OVER (
                    PARTITION BY model
                    ORDER BY timestamp DESC
                ) as rn
            FROM responses
            WHERE prompt = ?
        )
        SELECT
            r.model,
            r.timestamp,
            r.response as response_text,
            r.id as response_id,
            COUNT(j.id) as total_judgments,
            AVG(CASE
                WHEN j.score IS NOT NULL
                THEN j.score
            END) as avg_score,
            MIN(j.score) as min_score,
            MAX(j.score) as max_score,
            COUNT(CASE
                WHEN j.score IS NULL
                THEN 1
            END) as pending_judgments
        FROM ranked_responses r
        LEFT JOIN judgments j ON r.id = j.response_id
        WHERE r.rn = 1  -- Only get latest response for each model
        GROUP BY
            r.model,
            r.timestamp,
            r.response,
            r.id
        ORDER BY r.timestamp DESC
    """

    # Get judgments for this prompt
    judgments_query = """
        SELECT
            r.model as response_model,
            j.judge_model,
            j.score,
            j.timestamp as judged_at,
            j.explanation,
            r.response as response_text
        FROM responses r
        JOIN judgments j ON r.id = j.response_id
        WHERE r.prompt = ?
        ORDER BY j.timestamp DESC
    """

    responses = list(db.query(responses_query, [latest_prompt['prompt']]))
    judgments = list(db.query(judgments_query, [latest_prompt['prompt']]))

    return latest_prompt, responses, judgments

@cli.command()
def show_latest():
    """Show summary results for the most recent question"""
    latest_prompt, responses, judgments = get_latest_question_data()

    print(f"\n=== Results for question: {latest_prompt['prompt']} ===")
    print(f"First asked at: {latest_prompt['first_asked']}")
    print(f"Last asked at: {latest_prompt['last_asked']}")
    print(f"Total responses: {latest_prompt['total_responses']}\n")

    # Group judgments by response model
    judgments_by_model: Dict[str, List[Dict]] = {}
    for j in judgments:
        if j['response_model'] not in judgments_by_model:
            judgments_by_model[j['response_model']] = []
        judgments_by_model[j['response_model']].append(j)

    # Calculate summary statistics for each model
    model_stats = []
    for r in responses:
        model_judgments = judgments_by_model.get(r['model'], [])
        scores = [j['score'] for j in model_judgments if j['score'] is not None]

        stats = {
            'model': r['model'],
            'scores': scores,
            'avg_score': sum(scores) / len(scores) if scores else None,
            'min_score': min(scores) if scores else None,
            'max_score': max(scores) if scores else None,
            'num_judgments': len(scores)
        }
        model_stats.append(stats)

    # Sort models by average score
    sorted_stats = sorted(model_stats, key=lambda x: x.get('avg_score', float('-inf')), reverse=True)

    # Display performance summary table
    print("\nPerformance Summary:\n")
    print("{:<5} {:<50} {:<10} {:<10} {:<10} {:<15}".format(
        "Rank", "Model", "Average", "Min", "Max", "# Judgments"
    ))
    print("-" * 100)

    for i, stats in enumerate(sorted_stats, 1):
        if stats.get('scores'):  # Only show models with valid scores
            print("{:<5} {:<50} {:<10.2f} {:<10} {:<10} {:<15}".format(
                i,
                stats['model'],
                stats['avg_score'],
                stats['min_score'],
                stats['max_score'],
                stats['num_judgments']
            ))
        else:
            print("{:<5} {:<50} {:<10} {:<10} {:<10} {:<15}".format(
                i,
                stats['model'],
                "N/A",
                "N/A",
                "N/A",
                0
            ))

    # Display score matrix
    print("\nScore Matrix (rows: judges, columns: judged):\n")

    models = [s['model'] for s in sorted_stats]
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
                judgment = next((j for j in judgments_by_model.get(judged, []) 
                               if j['judge_model'] == judge), None)
                score = judgment['score'] if judgment and judgment.get('score') is not None else "E"
            row.append(score)
        matrix.append(row)

    print(tabulate(matrix, headers=headers, tablefmt="simple"))
    print("\n" + "="*80 + "\n")

@cli.command()
def show_latest_details():
    """Show detailed results for the most recent question"""
    latest_prompt, responses, judgments = get_latest_question_data()

    print(f"\n=== Detailed Results for question: {latest_prompt['prompt']} ===")

    # Group judgments by response model
    judgments_by_model: Dict[str, List[Dict]] = {}
    for j in judgments:
        if j['response_model'] not in judgments_by_model:
            judgments_by_model[j['response_model']] = []
        judgments_by_model[j['response_model']].append(j)

    # Display each response and its judgments
    for r in responses:
        model_judgments = judgments_by_model.get(r['model'], [])

        # Display response summary
        display_response_summary(r['model'], len(model_judgments))

        # Display response details
        display_response_details(r['response_text'])

        # Display judgments
        if model_judgments:
            display_judgment_summary(model_judgments)
            for j in model_judgments:
                display_judgment_details(j['judge_model'], j['score'], j['explanation'])

        print("\n" + "="*80 + "\n")

def format_model_name(model: str) -> str:
    """Format model name for display."""
    return model.split('/')[-1]

def display_response_summary(model: str, num_judgments: int):
    """Display summary of a model's response."""
    print(f"\n🤖 Response from {format_model_name(model)}")
    print(f"📊 {num_judgments} judgments received\n")

def display_response_details(response_text: str):
    """Display details of a response."""
    try:
        # Try to parse as JSON first
        data = json.loads(response_text)
        text = data.get('response', response_text)
    except:
        text = response_text

    print("Response:")
    print("-" * 40)
    print(text[:1000] + "..." if len(text) > 1000 else text)
    print("-" * 40)

def display_judgment_summary(judgments: List[Dict]):
    """Display summary of judgments."""
    scores = [j['score'] for j in judgments if j['score'] is not None]
    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"\n📈 Average Score: {avg_score:.1f}")
        print(f"📊 Score Range: {min(scores)}-{max(scores)}")
    print(f"✅ {len(judgments)} judgments total\n")

def display_judgment_details(judge_model: str, score: float, explanation: str):
    """Display details of a judgment."""
    print(f"\n🤖 Judge: {format_model_name(judge_model)}")
    print(f"📊 Score: {score}")
    print("\nExplanation:")
    print("-" * 40)
    print(explanation[:500] + "..." if len(explanation) > 500 else explanation)
    print("-" * 40)

if __name__ == '__main__':
    cli()
