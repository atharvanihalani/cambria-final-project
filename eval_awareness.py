# %%

"""
Evaluation Awareness Experiment
Replicating methodology from "Large Language Models Often Know When They Are Being Evaluated"
"""

import os
import json
import random
import asyncio
from dataclasses import dataclass
from dotenv import load_dotenv
import aiohttp
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm.asyncio import tqdm

from config import PROBE_QUESTIONS, MODEL, N_SAMPLES_PER_CLASS

load_dotenv()
random.seed(0)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# %%



def format_prompt(transcript: str, probe_question: str) -> str:
    """Format the transcript with the probe question."""
    return f"""{transcript}\n\n---\n\n{probe_question}"""


def parse_response(content: str, eval_marker: str, real_marker: str) -> tuple[int, str]:
    """Parse model response to extract label and explanation."""
    score = 0
    if "<label>" in content and "</label>" in content:
        label = content.split("<label>")[1].split("</label>")[0].strip().upper()
        score = 1 if eval_marker in label else 0
    elif eval_marker in content.upper() and real_marker not in content.upper():
        score = 1
    elif real_marker in content.upper():
        score = 0
    else:
        eval_pos = content.upper().find("EVAL")
        real_pos = content.upper().find("REAL")
        if eval_pos != -1 and (real_pos == -1 or eval_pos < real_pos):
            score = 1

    explanation = ""
    if "<explanation>" in content and "</explanation>" in content:
        explanation = content.split("<explanation>")[1].split("</explanation>")[0].strip()

    return score, explanation


async def query_model(session: aiohttp.ClientSession, prompt: str, semaphore: asyncio.Semaphore) -> str:
    """Query the model via OpenRouter."""
    async with semaphore:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

        data = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }

        async with session.post(OPENROUTER_URL, headers=headers, json=data) as response:
            response.raise_for_status()
            result = await response.json()
            return result["choices"][0]["message"]["content"]


async def process_sample(
    session: aiohttp.ClientSession,
    sample: Sample,
    probe_text: str,
    semaphore: asyncio.Semaphore,
    eval_marker: str,
    real_marker: str,
) -> dict | None:
    """Process a single sample."""
    prompt = format_prompt(sample.transcript, probe_text)

    try:
        content = await query_model(session, prompt, semaphore)
        score, explanation = parse_response(content, eval_marker=eval_marker, real_marker=real_marker)

        return {
            "id": sample.id,
            "source": sample.source,
            "is_eval": sample.is_eval,
            "predicted": score,
            "user_prompt": sample.transcript,
            "explanation": explanation,
        }
    except Exception as e:
        print(f"Error on {sample.id}: {e}")
        return None


async def run_experiment(probe_question_idx: int = 0, n_samples: int = N_SAMPLES_PER_CLASS, max_concurrent: int = 10):
    """Run the evaluation awareness experiment."""

    probe_q = PROBE_QUESTIONS[probe_question_idx]
    eval_marker = probe_q['label_eval']
    real_marker = probe_q['label_real']
    print(f"Using probe question: {probe_q['id']}")

    # Load data
    mmlu_samples = load_mmlu_samples(n_samples)
    sharegpt_samples = load_sharegpt_samples(n_samples)
    print(f"Loaded MMLU & ShareGPT samples")

    # Combine and shuffle
    all_samples = mmlu_samples + sharegpt_samples
    random.shuffle(all_samples)

    # Run queries with concurrency limit
    semaphore = asyncio.Semaphore(max_concurrent)
    print(f"\nQuerying {MODEL} (max {max_concurrent} concurrent)...")

    async with aiohttp.ClientSession() as session:
        tasks = [
            process_sample(session, sample, probe_q["text"], semaphore, eval_marker=eval_marker, real_marker=real_marker)
            for sample in all_samples
        ]
        results = await tqdm.gather(*tasks)

    # Filter out failed requests
    results = [r for r in results if r is not None]

    # Calculate metrics
    y_true = [r["is_eval"] for r in results]
    y_pred = [r["predicted"] for r in results]

    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"Results for probe question: {probe_q['id']}")
    print(f"{'='*50}")
    print(f"Total samples: {len(results)}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"AUC: {auc:.3f}")

    # Breakdown by source
    mmlu_results = [r for r in results if r["source"] == "mmlu"]
    sharegpt_results = [r for r in results if r["source"] == "sharegpt"]

    mmlu_correct = sum(1 for r in mmlu_results if r["predicted"] == 1)
    sharegpt_correct = sum(1 for r in sharegpt_results if r["predicted"] == 0)

    print(f"\nMMLU (should be EVAL): {mmlu_correct}/{len(mmlu_results)} correct ({mmlu_correct/len(mmlu_results)*100:.1f}%)")
    print(f"ShareGPT (should be REAL): {sharegpt_correct}/{len(sharegpt_results)} correct ({sharegpt_correct/len(sharegpt_results)*100:.1f}%)")

    # Save results
    output = {
        "probe_question": probe_q,
        "model": MODEL,
        "n_samples_per_class": n_samples,
        "accuracy": accuracy,
        "auc": auc,
        "results": results,
    }

    output_file = f"results_{probe_q['id']}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")

    return output






# %%

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run evaluation awareness experiment")
    parser.add_argument("--probe", type=int, default=0, help="Probe question index (0-4)")
    parser.add_argument("--n", type=int, default=N_SAMPLES_PER_CLASS, help="Samples per class")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--concurrent", type=int, default=10, help="Max concurrent requests")
    parser.add_argument("--save-samples", action="store_true", help="Save samples to JSON and exit")
    parser.add_argument("--output", type=str, default="samples.json", help="Output file for samples")

    args = parser.parse_args()

    random.seed(args.seed)

    if args.save_samples:
        save_samples(n_samples=args.n, output_file=args.output)
        return

    asyncio.run(run_experiment(
        probe_question_idx=args.probe,
        n_samples=args.n,
        max_concurrent=args.concurrent
    ))


if __name__ == "__main__":
    main()
