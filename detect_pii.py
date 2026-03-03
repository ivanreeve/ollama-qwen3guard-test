#!/usr/bin/env python3
"""Evaluate Qwen3Guard's PII detection capabilities."""

import argparse
import json
import os
import sys
from pathlib import Path

from tabulate import tabulate
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test Qwen3Guard PII detection via API or local transformers inference"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3Guard-Gen-4B",
        help="Model name (default: Qwen/Qwen3Guard-Gen-4B)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run inference locally via transformers instead of calling an API",
    )
    parser.add_argument(
        "--api-base",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible API base URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("VLLM_API_KEY", ""),
        help="Optional API key for the endpoint (defaults to VLLM_API_KEY env var)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/pii_test_dataset.json"),
        help="Path to test dataset JSON",
    )
    parser.add_argument("--output", type=Path, help="Save full results to JSON file")
    parser.add_argument(
        "--4bit",
        action="store_true",
        dest="quantize_4bit",
        help="Use 4-bit quantization via bitsandbytes (requires --local)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show per-query raw model output"
    )
    return parser.parse_args()


def load_local_model(model_name, quantize_4bit=False):
    """Load model and tokenizer for local transformers inference."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name} locally (4-bit={quantize_4bit})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    load_kwargs = {"device_map": "auto"}
    if quantize_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
        )
    else:
        load_kwargs["torch_dtype"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    print("Model loaded.")
    return model, tokenizer


def query_local_model(model, tokenizer, query):
    """Run inference locally via transformers."""
    messages = [{"role": "user", "content": query}]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=128)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def query_chat_api(api_base, model, query, api_key="", timeout=120):
    """Send a query to an OpenAI-compatible chat endpoint and return response text."""
    import requests

    endpoint = f"{api_base.rstrip('/')}/chat/completions"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.post(
        endpoint,
        headers=headers,
        json={
            "model": model,
            "messages": [{"role": "user", "content": query}],
            "temperature": 0,
            "stream": False,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    choices = payload.get("choices", [])
    if not choices:
        raise ValueError("No choices returned by API")

    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        # Some providers return structured content parts.
        content = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        )
    return str(content)


def parse_guard_output(text):
    """Parse Qwen3Guard output into structured fields.

    Expected format:
        Safety: [Safe|Unsafe|Controversial]
        Categories: [...]
        Refusal: [Yes|No]
    """
    result = {"safety": None, "categories": [], "refusal": None, "raw": text}
    for line in text.strip().splitlines():
        line = line.strip()
        lower = line.lower()
        if lower.startswith("safety:"):
            val = line.split(":", 1)[1].strip().lower()
            if "unsafe" in val:
                result["safety"] = "unsafe"
            elif "controversial" in val:
                result["safety"] = "controversial"
            elif "safe" in val:
                result["safety"] = "safe"
        elif lower.startswith("categories:") or lower.startswith("category:"):
            result["categories"] = line.split(":", 1)[1].strip()
        elif lower.startswith("refusal:"):
            val = line.split(":", 1)[1].strip().lower()
            result["refusal"] = "yes" in val
    return result


def detect_pii(parsed):
    """Determine if the model flagged PII.

    PII detected = (Unsafe or Controversial) AND 'pii' mentioned in categories.
    Parse errors default to False.
    """
    if parsed["safety"] is None:
        return False
    if parsed["safety"] in ("unsafe", "controversial"):
        cats = parsed["categories"]
        if isinstance(cats, str) and "pii" in cats.lower():
            return True
    return False


def compute_metrics(results):
    tp = sum(1 for r in results if r["expected"] and r["predicted"])
    fp = sum(1 for r in results if not r["expected"] and r["predicted"])
    fn = sum(1 for r in results if r["expected"] and not r["predicted"])
    tn = sum(1 for r in results if not r["expected"] and not r["predicted"])

    accuracy = (tp + tn) / len(results) if results else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def print_report(results, metrics, verbose=False):
    if verbose:
        print("\n  Per-Query Results")
        query_rows = []
        for r in results:
            status = "OK" if r["expected"] == r["predicted"] else "MISS"
            query_rows.append([
                r["id"],
                status,
                r["difficulty"],
                r["pii_type"],
                r["query"][:50],
                r["parsed"].get("safety", "N/A"),
                r["parsed"].get("categories", "N/A"),
                "Yes" if r["expected"] else "No",
                "Yes" if r["predicted"] else "No",
            ])
        print(tabulate(
            query_rows,
            headers=["ID", "Result", "Diff", "Type", "Query", "Safety", "Categories", "Exp", "Pred"],
            tablefmt="grid",
        ))

    cm = metrics["confusion_matrix"]
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    # Overall metrics table
    metrics_table = [
        ["Total Samples", len(results)],
        ["Accuracy", f"{metrics['accuracy']:.3f}"],
        ["Precision", f"{metrics['precision']:.3f}"],
        ["Recall", f"{metrics['recall']:.3f}"],
        ["F1 Score", f"{metrics['f1']:.3f}"],
    ]
    print(tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="grid"))

    # Confusion matrix
    print("\n  Confusion Matrix")
    cm_table = [
        ["Actual PII", cm["tp"], cm["fn"]],
        ["Actual Clean", cm["fp"], cm["tn"]],
    ]
    print(tabulate(cm_table, headers=["", "Pred PII", "Pred Clean"], tablefmt="grid"))

    # Breakdown by difficulty
    print("\n  Breakdown by Difficulty")
    diff_rows = []
    for diff in ("easy", "medium", "hard"):
        subset = [r for r in results if r["difficulty"] == diff]
        if not subset:
            continue
        m = compute_metrics(subset)
        correct = sum(1 for r in subset if r["expected"] == r["predicted"])
        diff_rows.append([
            diff,
            f"{correct}/{len(subset)}",
            f"{m['accuracy']:.3f}",
            f"{m['precision']:.3f}",
            f"{m['recall']:.3f}",
            f"{m['f1']:.3f}",
        ])
    print(tabulate(
        diff_rows,
        headers=["Difficulty", "Correct", "Acc", "Prec", "Recall", "F1"],
        tablefmt="grid",
    ))

    # Breakdown by PII type
    print("\n  PII Type Detection (positive cases only)")
    pii_results = [r for r in results if r["expected"]]
    types = sorted(set(r["pii_type"] for r in pii_results))
    type_rows = []
    for ptype in types:
        subset = [r for r in pii_results if r["pii_type"] == ptype]
        detected = sum(1 for r in subset if r["predicted"])
        rate = detected / len(subset) if subset else 0
        type_rows.append([ptype, f"{detected}/{len(subset)}", f"{rate:.0%}"])
    print(tabulate(
        type_rows,
        headers=["PII Type", "Detected", "Rate"],
        tablefmt="grid",
    ))

    # Parse errors
    parse_errors = [r for r in results if r.get("parse_error")]
    if parse_errors:
        print(f"\n  Parse Errors: {len(parse_errors)}")
        err_rows = [[r["id"], r["raw_output"][:60]] for r in parse_errors]
        print(tabulate(err_rows, headers=["ID", "Raw Output"], tablefmt="grid"))

    print("=" * 70)


def main():
    args = parse_args()

    # Load dataset
    if not args.dataset.exists():
        print(f"Error: Dataset not found at {args.dataset}", file=sys.stderr)
        sys.exit(1)

    with open(args.dataset) as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} test entries")

    # Set up inference method
    local_model = None
    local_tokenizer = None
    if args.local:
        local_model, local_tokenizer = load_local_model(
            args.model, quantize_4bit=args.quantize_4bit
        )
        print(f"Using model: {args.model} (local, 4-bit={args.quantize_4bit})")
    else:
        print(f"Using model: {args.model} via {args.api_base}")

    # Run evaluation
    results = []
    for entry in tqdm(dataset, desc="Evaluating"):
        if args.local:
            raw_output = query_local_model(local_model, local_tokenizer, entry["query"])
        else:
            raw_output = query_chat_api(
                api_base=args.api_base,
                model=args.model,
                query=entry["query"],
                api_key=args.api_key,
                timeout=args.timeout,
            )

        parsed = parse_guard_output(raw_output)
        predicted = detect_pii(parsed)
        parse_error = parsed["safety"] is None

        result = {
            "id": entry["id"],
            "query": entry["query"],
            "contains_pii": entry["contains_pii"],
            "difficulty": entry["difficulty"],
            "pii_type": entry.get("pii_type", "none"),
            "expected": entry["contains_pii"],
            "predicted": predicted,
            "raw_output": raw_output,
            "parsed": parsed,
            "parse_error": parse_error,
        }
        results.append(result)

    # Compute and display metrics
    metrics = compute_metrics(results)
    print_report(results, metrics, verbose=args.verbose)

    # Save results
    if args.output:
        output_data = {
            "model": args.model,
            "dataset": str(args.dataset),
            "total_entries": len(results),
            "metrics": metrics,
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
