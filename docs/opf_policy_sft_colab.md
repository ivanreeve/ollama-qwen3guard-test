# OPF policy-targeted SFT on Colab

This branch includes a benchmark-targeted SFT pack for `openai/privacy-filter`.

Files:

- `data/opf_policy_sft_train.jsonl`
- `data/opf_policy_sft_val.jsonl`
- `data/opf_policy_label_space.json`
- `data/opf_policy_sft_manifest.json`
- `notebooks/opf_policy_sft_colab.ipynb`
- `scripts/build_opf_policy_sft_assets.py`

## What this dataset is

This is not a general-purpose privacy redaction dataset.

It is a branch-specific adaptation pack intended to move OPF toward the repo benchmark policy, which mixes:

- literal PII spans
- quasi-identifiers
- indirect requests for personal banking details
- hard negatives with PII-like strings in dummy/spec/policy contexts

The label space is intentionally binary:

```json
{
  "category_version": "benchmark_binary_v1",
  "span_class_names": ["O", "pii"]
}
```

That is deliberate. The repo evaluator only needs "any detected span => positive", so a single custom label is the cleanest fit.

## Colab recommendation

You said you only have a Colab T4 GPU.

That means you should **not** use the stock `opf train` path for this project. It is full-model AdamW and is the wrong fit for a 16 GB T4.

For T4, this repo now includes a dedicated fallback:

- `scripts/train_opf_head_only.py`

This keeps the OPF backbone frozen and trains only the output head. It is a compromise, but it is the first path that is realistically compatible with a Colab T4.

## Try calibration before SFT

You should usually do a short decoder-calibration sweep before training.

Why:

- it is cheap and reversible
- it does not need gradient training
- it can improve precision/recall tradeoffs when the model already has token-level signal

Limit:

- calibration only changes Viterbi transition biases
- it will not teach the model new benchmark semantics
- for this benchmark, many false negatives were total misses, so calibration is a first pass, not the main lever

OPF exposes the relevant knobs through `opf eval`:

- `--decode-mode {viterbi,argmax}`
- `--viterbi-calibration-path <path>`

If no calibration path is supplied, OPF auto-discovers
`<checkpoint>/viterbi_calibration.json`; if the file is absent, it uses all-zero
transition biases.

This repo's `detect_pii.py` now exposes `--opf-viterbi-calibration-path`, so the
practical workflow is:

1. create or edit a local calibration JSON
2. run `detect_pii.py --opf-viterbi-calibration-path <path>`
3. if the sweep is not enough, continue to SFT

Example benchmark run:

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

Or score the span-labeled validation set directly with OPF:

```bash
OPF_MOE_TRITON=0 opf eval data/opf_policy_sft_val.jsonl \
  --checkpoint ~/.opf/privacy_filter/original \
  --device cuda \
  --eval-mode untyped \
  --viterbi-calibration-path calib/recall.json \
  --metrics-out results/opf-calib-r1-metrics.json
```

If you prefer bundling calibration inside a copied checkpoint directory, that
still works because OPF auto-discovers `<checkpoint>/viterbi_calibration.json`.

Bias direction:

- For more recall, lower `transition_bias_background_stay` and raise `transition_bias_background_to_start` and `transition_bias_inside_to_continue`.
- For more precision, move those settings in the opposite direction.

## Minimal training command

From the repo root:

```bash
OPF_MOE_TRITON=0 python scripts/train_opf_head_only.py \
  data/opf_policy_sft_train.jsonl \
  --validation-dataset data/opf_policy_sft_val.jsonl \
  --label-space-json data/opf_policy_label_space.json \
  --device cuda \
  --n-ctx 256 \
  --epochs 25 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --learning-rate 2e-4 \
  --weight-decay 0.0 \
  --max-grad-norm 1.0 \
  --output-dir checkpoints/opf_policy_sft_v1 \
  --overwrite-output
```

Notes:

- `n_ctx=256` is deliberate. These examples are short, and lowering context reduces memory pressure.
- `learning-rate=2e-4` is higher than the full-model recipe because only the output head is trainable.
- `OPF_MOE_TRITON=0` avoids unnecessary Triton-related failures in constrained environments.

## Evaluate the finetuned checkpoint

```bash
python detect_pii.py \
  --model checkpoints/opf_policy_sft_v1 \
  --output results/privacy-filter-sft.json \
  --verbose
```

## Annotation policy

- Literal identifiers: mark the minimal explicit sensitive span.
- Quasi-identifiers: mark the full uniqueness bundle when there is no clean literal token span.
- Indirect requests: mark the request clause or full request when that is the minimal positive unit.
- Hard negatives: leave spans empty, even if the text contains realistic PII-shaped strings.

## Regenerate the assets

If you edit the annotation spec:

```bash
python scripts/build_opf_policy_sft_assets.py
```
