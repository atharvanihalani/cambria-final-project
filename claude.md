# Evaluation Awareness Detection Project

Research project investigating whether LLMs can detect when they're being evaluated (benchmark questions) versus interacting with real users.

## Project Structure

```
/root/atharva_project/
├── main.ipynb          # Primary workflow notebook
├── config.py           # API config, probe questions, model definitions
├── .env                # OpenRouter API key (OPENROUTER_API_KEY)
├── data/
│   ├── prompt_data.json    # Central dataset (MMLU, ShareGPT, AIME + rephrased variants)
│   └── rewriter.py         # LLM-based question rephrasing logic
└── results/            # Evaluation outputs (JSON with per-sample results)
```

## Key Concepts

- **Probe Questions**: Asked after a transcript to determine if the model thinks it was an evaluation or real interaction. Three variants in `config.py` (v1, v2, v3) with different framings.
- **Rephrasing**: Transforms formal evaluation questions into natural user requests to test if this evades detection.
- **Datasets**: MMLU (knowledge MCQs), AIME (math competition), ShareGPT (real conversations)

## Models (defined in config.py)

| Role | Model | Purpose |
|------|-------|---------|
| `REWRITER_MODEL` | claude-opus-4.5 | Rephrases evaluation questions naturally |
| `TARGET_MODEL` | gpt-4.1-mini | Model being tested for awareness |
| `SCORER_MODEL` | claude-sonnet-4.5 | For scoring (not yet integrated) |

## Data Flow

1. **Load datasets** from HuggingFace (MMLU, ShareGPT, AIME)
2. **Rephrase** evaluation questions using `data/rewriter.py`
3. **Save** to `data/prompt_data.json`
4. **Evaluate**: Query target model with transcript + probe question
5. **Output** results to `results/awareness_{dataset}_PQ-v{N}.json`

## Running Evaluations

The main workflow is in `main.ipynb`. Key function:

```python
await test_eval_awareness('mmlu')           # Test original MMLU
await test_eval_awareness('mmlu_rephrased') # Test rephrased MMLU
await test_eval_awareness('aime')           # Test original AIME
await test_eval_awareness('aime_rephrased') # Test rephrased AIME
```

Data loading/rephrasing (one-time, currently commented out in notebook):
```python
save_samples('mmlu', samples=load_mmlu_for_save())
save_samples('sharegpt', samples=load_sharegpt_for_save())
save_samples('aime', samples=load_aime_for_save())
save_samples('mmlu_rephrased', samples=rephrase_mmlu_for_save())
save_samples('aime_rephrased', samples=rephrase_aime_for_save())
```

## Sample Data Format

```json
{
  "id": "mmlu_13835",
  "transcript": "Question: The successful anti-cancer HPV vaccine...",
  "is_eval": 1,
  "source": "mmlu"
}
```

AIME samples also include `"ans": "123"` for answer verification.

## Results Format

```json
{
  "probe_question": {...},
  "model": "openai/gpt-4.1-mini",
  "accuracy": 0.92,
  "auc": 0.92,
  "results": [
    {"id": "...", "is_eval": 1, "predicted": 1, "explanation": "..."}
  ]
}
```

## Key Findings

- Model detects real ShareGPT conversations well (~90%)
- Model fails to detect rephrased evaluation questions (~3% for AIME_rephrased)
- Rephrasing is effective at masking evaluation intent

## Dependencies

- `openai`, `aiohttp` - API calls
- `datasets` - HuggingFace data loading
- `sklearn.metrics` - Accuracy/AUC
- `tqdm.asyncio` - Progress bars
- `python-dotenv` - Environment loading

## Benchmark Performance Evaluation

Separate from awareness detection, the notebook also evaluates actual benchmark performance (does the model get the right answer?).

```python
# Evaluate AIME performance (original vs rephrased)
original, rephrased = await compare_aime_performance()

# Or evaluate individually
await evaluate_aime_benchmark("aime")           # Original AIME
await evaluate_aime_benchmark("aime_rephrased") # Rephrased version
```

Results saved to `results/benchmark_{dataset}_{model}.json` with per-sample correctness.

## Adding New Datasets

1. Create a `load_{name}_for_save()` function in `main.ipynb`
2. Return list of `Sample` objects with `id`, `transcript`, `is_eval`, `source`
3. Call `save_samples('{name}', samples=load_{name}_for_save())`
4. Optionally create rephrasing function using `rewriter.rewrite_all()`
