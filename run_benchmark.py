#!/usr/bin/env python3
"""
Standalone script to run benchmark performance evaluations.
Run with: python run_benchmark.py
Or in tmux: tmux new -s bench && python run_benchmark.py
"""

import os
import re
import json
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import aiohttp
from tqdm.asyncio import tqdm

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Model to evaluate
SCORER_MODEL = "openai/gpt-5"

# Datasets to evaluate (comment/uncomment as needed)
DATASETS_TO_RUN = [
    "hmmt_feb_2025",
    "hmmt_nov_2025",
    # "olympmath",
    # "aime",
    # "hmmt_feb_2025_rephrased_v3",
    # "hmmt_nov_2025_rephrased_v3",
    # "olympmath_rephrased_v3",
]

MAX_CONCURRENT = 10
REQUEST_TIMEOUT = 300  # 5 minutes per request (GPT-5 can be slow)

# ============================================================================
# System Prompts (dataset-specific)
# ============================================================================

# For AIME-style problems (integer answers 0-999)
AIME_SYSTEM_PROMPT = """You are an expert mathematician. Solve problems step by step, showing your reasoning clearly.

AIME answers are always integers from 0 to 999.

Put your final answer inside \\boxed{} like this: \\boxed{123}"""

# For HMMT/general math problems (can be fractions, radicals, expressions)
HMMT_SYSTEM_PROMPT = """You are an expert mathematician. Solve problems step by step, showing your reasoning clearly.

Put your final answer inside \\boxed{}. Your answer may be an integer, fraction, radical, or mathematical expression.

Examples:
- Integer: \\boxed{42}
- Fraction: \\boxed{\\frac{3}{7}}
- Radical: \\boxed{\\sqrt{5}}
- Expression: \\boxed{2\\sqrt{3} - 1}"""

# For OlympMATH problems (similar to HMMT)
OLYMPMATH_SYSTEM_PROMPT = HMMT_SYSTEM_PROMPT


def get_system_prompt(dataset_name: str) -> str:
    """Get the appropriate system prompt for a dataset."""
    if 'aime' in dataset_name.lower():
        return AIME_SYSTEM_PROMPT
    elif 'hmmt' in dataset_name.lower():
        return HMMT_SYSTEM_PROMPT
    elif 'olymp' in dataset_name.lower():
        return OLYMPMATH_SYSTEM_PROMPT
    else:
        # Default to HMMT-style (more general)
        return HMMT_SYSTEM_PROMPT


def get_user_prompt(problem: str, dataset_name: str) -> str:
    """Format the user prompt based on dataset type."""
    if 'aime' in dataset_name.lower():
        return f"""Solve the following math problem step by step. Put your answer inside \\boxed{{}}.

{problem}

Remember: AIME answers are integers from 0 to 999. Put your answer inside \\boxed{{}}."""
    else:
        return f"""Solve the following math problem step by step. Put your final answer inside \\boxed{{}}.

{problem}

Remember to put your final answer inside \\boxed{{}}."""


# ============================================================================
# Answer Parsing
# ============================================================================

def normalize_latex(s: str) -> str:
    """Normalize a LaTeX string for comparison."""
    if s is None:
        return ""
    s = str(s).strip()
    # Remove outer whitespace and common wrappers
    s = s.replace(" ", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\cdot", "*")
    s = s.replace("\\times", "*")
    s = s.replace("\\div", "/")
    # Normalize fractions: \frac{a}{b} -> a/b
    s = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', s)
    # Normalize sqrt: \sqrt{x} -> sqrt(x)
    s = re.sub(r'\\sqrt\{([^}]*)\}', r'sqrt(\1)', s)
    s = re.sub(r'\\sqrt(\d+)', r'sqrt(\1)', s)  # \sqrt5 -> sqrt(5)
    # Remove backslashes from remaining commands
    s = re.sub(r'\\([a-zA-Z]+)', r'\1', s)
    return s.lower()


def extract_boxed_answer(response: str) -> str | None:
    """Extract the content from \\boxed{...} in the response."""
    if not response:
        return None

    # Try to find \boxed{...} with nested braces handling
    # Start from the end of the response (last \boxed is usually the final answer)
    matches = list(re.finditer(r'\\boxed\s*\{', response))
    if not matches:
        return None

    # Get the last match
    match = matches[-1]
    start = match.end()

    # Find matching closing brace
    depth = 1
    pos = start
    while pos < len(response) and depth > 0:
        if response[pos] == '{':
            depth += 1
        elif response[pos] == '}':
            depth -= 1
        pos += 1

    if depth == 0:
        return response[start:pos-1].strip()

    return None


def parse_answer_aime(response: str) -> str | None:
    """Parse answer for AIME-style problems (integer 0-999)."""
    boxed = extract_boxed_answer(response)
    if boxed:
        # Try to extract integer from boxed content
        match = re.search(r'^-?\d+$', boxed.strip())
        if match:
            return match.group()

    # Fallback: look for patterns at the end
    if response:
        lines = response.strip().split('\n')
        for line in reversed(lines[-5:]):
            match = re.search(r'(?:answer\s*(?:is|:)\s*|=\s*|\*\*)\s*(\d+)', line, re.IGNORECASE)
            if match:
                return match.group(1)

    return None


def parse_answer_general(response: str) -> str | None:
    """Parse answer for general math problems (fractions, radicals, etc.)."""
    return extract_boxed_answer(response)


def answers_match(predicted: str | None, ground_truth: str | None, dataset_name: str) -> bool:
    """Check if predicted answer matches ground truth."""
    if predicted is None or ground_truth is None:
        return False

    # For AIME, do integer comparison
    if 'aime' in dataset_name.lower():
        try:
            return int(predicted) == int(ground_truth)
        except ValueError:
            return False

    # For other datasets, normalize and compare
    pred_norm = normalize_latex(predicted)
    truth_norm = normalize_latex(ground_truth)

    # Direct match after normalization
    if pred_norm == truth_norm:
        return True

    # Try numeric comparison for simple numbers
    try:
        pred_val = float(eval(pred_norm.replace('sqrt', 'import math; math.sqrt')))
        truth_val = float(eval(truth_norm.replace('sqrt', 'import math; math.sqrt')))
        if abs(pred_val - truth_val) < 1e-9:
            return True
    except:
        pass

    # Try sympy for symbolic comparison (if available)
    try:
        from sympy import simplify, sympify, sqrt, pi, E
        from sympy.parsing.latex import parse_latex

        # Try parsing as sympy expressions
        try:
            pred_sym = sympify(pred_norm)
            truth_sym = sympify(truth_norm)
            if simplify(pred_sym - truth_sym) == 0:
                return True
        except:
            pass
    except ImportError:
        pass

    return False


# ============================================================================
# API Calls
# ============================================================================

async def query_model(
    session: aiohttp.ClientSession,
    problem: str,
    dataset_name: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """Query the model to solve a problem."""
    async with semaphore:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

        data = {
            "model": SCORER_MODEL,
            "messages": [
                {"role": "system", "content": get_system_prompt(dataset_name)},
                {"role": "user", "content": get_user_prompt(problem, dataset_name)},
            ],
            "temperature": 0,
        }

        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        async with session.post(OPENROUTER_URL, headers=headers, json=data, timeout=timeout) as response:
            response.raise_for_status()
            result = await response.json()
            return result["choices"][0]["message"]["content"]


async def evaluate_sample(
    session: aiohttp.ClientSession,
    sample: dict,
    dataset_name: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Evaluate a single sample."""
    ground_truth = sample.get('ans')

    try:
        response = await query_model(session, sample['transcript'], dataset_name, semaphore)

        # Parse answer based on dataset type
        if 'aime' in dataset_name.lower():
            predicted = parse_answer_aime(response)
        else:
            predicted = parse_answer_general(response)

        is_correct = answers_match(predicted, ground_truth, dataset_name)

        return {
            "id": sample['id'],
            "source": sample['source'],
            "ground_truth": ground_truth,
            "predicted": predicted,
            "is_correct": is_correct,
            "model_response": response,
        }
    except asyncio.TimeoutError:
        print(f"TIMEOUT on {sample['id']} (>{REQUEST_TIMEOUT}s)")
        return {
            'id': sample['id'],
            'source': sample['source'],
            'ground_truth': ground_truth,
            'predicted': None,
            'is_correct': False,
            'model_response': f'Error: Timeout after {REQUEST_TIMEOUT}s',
            'error': True,
            'error_type': 'timeout',
        }
    except aiohttp.ClientError as e:
        error_type = type(e).__name__
        print(f"CLIENT ERROR on {sample['id']}: {error_type}: {e}")
        return {
            'id': sample['id'],
            'source': sample['source'],
            'ground_truth': ground_truth,
            'predicted': None,
            'is_correct': False,
            'model_response': f'Error: {error_type}: {e}',
            'error': True,
            'error_type': error_type,
        }
    except Exception as e:
        error_type = type(e).__name__
        print(f"ERROR on {sample['id']}: {error_type}: {e}")
        return {
            'id': sample['id'],
            'source': sample['source'],
            'ground_truth': ground_truth,
            'predicted': None,
            'is_correct': False,
            'model_response': f'Error: {error_type}: {e}',
            'error': True,
            'error_type': error_type,
        }


async def evaluate_benchmark(dataset_name: str):
    """
    Evaluate model performance on a benchmark dataset.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {dataset_name}")
    print(f"Model: {SCORER_MODEL}")
    print(f"System prompt: {'AIME' if 'aime' in dataset_name.lower() else 'HMMT/General'}")
    print(f"Timeout: {REQUEST_TIMEOUT}s per request")
    print(f"{'='*60}")

    # Load data
    with open('data/prompt_data.json', 'r') as f:
        data = json.load(f)

    if dataset_name not in data:
        print(f"ERROR: Dataset '{dataset_name}' not found in prompt_data.json")
        print(f"Available datasets: {list(data.keys())}")
        return None

    samples = data[dataset_name]
    print(f"Total samples: {len(samples)}")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async with aiohttp.ClientSession() as session:
        tasks = [
            evaluate_sample(session, sample, dataset_name, semaphore)
            for sample in samples
        ]
        results = await tqdm.gather(*tasks, desc=dataset_name)

    # Calculate metrics
    results = [r for r in results if r is not None]
    correct = sum(1 for r in results if r['is_correct'])
    total = len(results)
    errors = sum(1 for r in results if r.get('error'))
    timeouts = sum(1 for r in results if r.get('error_type') == 'timeout')
    accuracy = correct / total if total > 0 else 0

    # Accuracy excluding errors
    non_error_total = total - errors
    accuracy_excl_errors = correct / non_error_total if non_error_total > 0 else 0

    print(f"\nResults for {dataset_name}:")
    print(f"  Correct: {correct}/{total}")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Errors: {errors} (timeouts: {timeouts})")
    if errors > 0:
        print(f"  Accuracy (excl. errors): {correct}/{non_error_total} = {accuracy_excl_errors:.1%}")

    # Save results
    output = {
        "dataset": dataset_name,
        "model": SCORER_MODEL,
        "timestamp": datetime.now().isoformat(),
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "errors": errors,
        "timeouts": timeouts,
        "accuracy_excl_errors": accuracy_excl_errors,
        "config": {
            "system_prompt_type": "AIME" if 'aime' in dataset_name.lower() else "HMMT/General",
            "timeout_seconds": REQUEST_TIMEOUT,
            "max_concurrent": MAX_CONCURRENT,
        },
        "results": results,
    }

    model_name = SCORER_MODEL.replace("/", "_")
    output_file = f'results/benchmark_{dataset_name}_{model_name}.json'
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved to: {output_file}")

    return output


async def main():
    print("="*60)
    print("BENCHMARK PERFORMANCE EVALUATION")
    print(f"Model: {SCORER_MODEL}")
    print(f"Datasets: {DATASETS_TO_RUN}")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*60)

    all_results = {}

    for dataset_name in DATASETS_TO_RUN:
        result = await evaluate_benchmark(dataset_name)
        if result:
            all_results[dataset_name] = {
                "accuracy": result["accuracy"],
                "correct": result["correct"],
                "total": result["total_samples"],
                "errors": result["errors"],
                "accuracy_excl_errors": result["accuracy_excl_errors"],
            }

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Model: {SCORER_MODEL}")
    for dataset_name, metrics in all_results.items():
        err_str = f" ({metrics['errors']} errors)" if metrics['errors'] > 0 else ""
        print(f"  {dataset_name}: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']}){err_str}")
        if metrics['errors'] > 0:
            print(f"    -> excl. errors: {metrics['accuracy_excl_errors']:.1%}")
    print(f"\nFinished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    asyncio.run(main())
