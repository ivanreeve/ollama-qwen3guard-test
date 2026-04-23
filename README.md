# qwen3guard-pii-test

Lightweight evaluation harness for testing PII detection models.

Currently supported backends:

- `openai/privacy-filter` through OpenAI's local OPF runtime.
- `Qwen/Qwen3Guard-Gen-4B` and similar guard/chat models through a vLLM OpenAI-compatible API endpoint.

## What is in this repo

- `detect_pii.py`: Runs the evaluation against a JSON dataset and prints metrics.
- `data/pii_test_dataset.json`: Default evaluation dataset, including harder multilingual Southeast Asian cases and safe negatives.
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
pip install -r requirements.txt
```

If you want to run the vLLM API path as well, install `vllm` separately:

```bash
pip install -U vllm
```

### Qwen3Guard / chat guard models

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
python detect_pii.py --model Qwen/Qwen3Guard-Gen-4B --api-base http://localhost:8000/v1
```

### OpenAI Privacy Filter

`openai/privacy-filter` is a token-classification model, so it does not use the
OpenAI-compatible chat API path. The evaluator automatically runs it locally via
the `opf` runtime.

```bash
python detect_pii.py \
  --output results/privacy-filter.json \
  --verbose
```

Notes:

- The first run may download the checkpoint into `~/.opf/privacy_filter`.
- Any detected OPF span is treated as a positive PII detection for evaluator metrics.

### OPF decoder calibration

Before doing SFT, it is worth trying a short decoder-calibration sweep.
This is much cheaper than training, but it only changes Viterbi transition
biases. It will not teach the model new semantic patterns.

OPF exposes decoder knobs through `opf eval`:

- `--decode-mode {viterbi,argmax}`
- `--viterbi-calibration-path <path>`

If no calibration path is supplied, OPF auto-discovers
`<checkpoint>/viterbi_calibration.json`; if that file is absent, it falls back
to all-zero transition biases.

`detect_pii.py` now exposes an explicit OPF calibration flag, so you can point
the evaluator at any local calibration artifact directly:

```bash
python detect_pii.py \
  --opf-viterbi-calibration-path calib/recall.json \
  --output results/privacy-filter-calib.json \
  --verbose
```

Example `calib/recall.json`:

```json
{
  "operating_points": {
    "default": {
      "biases": {
        "transition_bias_background_stay": -0.25,
        "transition_bias_background_to_start": 0.25,
        "transition_bias_inside_to_continue": 0.15,
        "transition_bias_inside_to_end": 0.0,
        "transition_bias_end_to_background": 0.0,
        "transition_bias_end_to_start": 0.0
      }
    }
  }
}
```

If you want to score a span-labeled JSONL directly with OPF's own evaluator, it
can consume the same artifact:

```bash
OPF_MOE_TRITON=0 opf eval data/opf_policy_sft_val.jsonl \
  --checkpoint ~/.opf/privacy_filter/original \
  --device cuda \
  --eval-mode untyped \
  --viterbi-calibration-path calib/recall.json \
  --metrics-out results/opf-calib-r1-metrics.json
```

If you prefer embedding calibration into a copied checkpoint directory, that
still works because OPF auto-discovers `<checkpoint>/viterbi_calibration.json`.

Calibration direction:

- For more recall, lower `transition_bias_background_stay` and raise `transition_bias_background_to_start` and `transition_bias_inside_to_continue`.
- For more precision, move those settings in the opposite direction.

## CLI options

```text
--model MODEL              Model name (default: openai/privacy-filter)
--local                    Run inference locally via transformers (no server needed)
--4bit                     Use 4-bit quantization via bitsandbytes (requires --local)
--api-base API_BASE        OpenAI-compatible API base URL (default: http://localhost:8000/v1)
--api-key API_KEY          Optional API key (or set VLLM_API_KEY)
--timeout TIMEOUT          Per-request timeout in seconds (default: 120)
--dataset DATASET          Path to test dataset JSON
--opf-viterbi-calibration-path OPF_VITERBI_CALIBRATION_PATH
                          Local JSON calibration artifact for the OPF Viterbi decoder
--output OUTPUT            Save full results to JSON file
--verbose                  Show per-query raw model output
```

## Notes

- The evaluator expects model responses in the Qwen3Guard format:
  `Safety: ...`, `Categories: ...`, `Refusal: ...`.
- A case is counted as PII only when:
  `Safety` is `Unsafe` or `Controversial` and categories include `PII`.
- `openai/privacy-filter` is handled separately as a span detector instead of a chat model.
- If the server is unreachable, verify vLLM is running and that `--api-base` is correct.
