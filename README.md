# qwen3guard-pii-test

Lightweight evaluation harness for testing Qwen3Guard PII detection through a
vLLM OpenAI-compatible API endpoint.

## What is in this repo

- `detect_pii.py`: Runs the evaluation against a JSON dataset and prints metrics.
- `data/pii_test_dataset.json`: Default evaluation dataset.
- `results/`: Output directory for JSON reports.

## Google Colab Quick Start

1. Enable a **T4 GPU** runtime in Colab.
2. Clone this repo and install dependencies:

```bash
!git clone <repo-url>
%cd ollama-qwen3guard-test
!pip install -U transformers accelerate bitsandbytes tabulate tqdm
```

3. Run evaluation (local inference with 4-bit quantization, ~5-6GB VRAM):

```bash
!python detect_pii.py \
  --local --4bit \
  --model Qwen/Qwen3Guard-Gen-4B \
  --output results/results.json \
  --verbose
```

A ready-made notebook is provided at `colab_qwen3guard_gen_4b.ipynb`.

## Local Run (Non-Colab)

Install dependencies:

```bash
pip install -U vllm requests tabulate tqdm
```

Start vLLM:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3Guard-Gen-4B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto
```

Run evaluator:

```bash
python detect_pii.py --api-base http://localhost:8000/v1
```

## CLI options

```text
--model MODEL              Model name (default: Qwen/Qwen3Guard-Gen-4B)
--local                    Run inference locally via transformers (no server needed)
--4bit                     Use 4-bit quantization via bitsandbytes (requires --local)
--api-base API_BASE        OpenAI-compatible API base URL (default: http://localhost:8000/v1)
--api-key API_KEY          Optional API key (or set VLLM_API_KEY)
--timeout TIMEOUT          Per-request timeout in seconds (default: 120)
--dataset DATASET          Path to test dataset JSON
--output OUTPUT            Save full results to JSON file
--verbose                  Show per-query raw model output
```

## Notes

- The evaluator expects model responses in the Qwen3Guard format:
  `Safety: ...`, `Categories: ...`, `Refusal: ...`.
- A case is counted as PII only when:
  `Safety` is `Unsafe` or `Controversial` and categories include `PII`.
- If the server is unreachable, verify vLLM is running and that `--api-base` is correct.
