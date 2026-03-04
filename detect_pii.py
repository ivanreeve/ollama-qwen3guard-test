#!/usr/bin/env python3
"""Evaluate Qwen3Guard's PII detection capabilities."""

import argparse
import base64
import json
import os
import re
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

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--presidio",
        action="store_true",
        help="Use Presidio-only PII detection (no model inference)",
    )
    mode.add_argument(
        "--combined",
        action="store_true",
        help="Use Presidio + Qwen3Guard combined detection (logical OR)",
    )
    return parser.parse_args()


# Entity types that indicate actual PII. We exclude LOCATION, DATE_TIME, NRP,
# URL, and other generic types that cause false positives on banking/finance text.
PII_ENTITY_TYPES = {
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "MEDICAL_LICENSE",
    "US_SSN",
    "US_BANK_NUMBER",
    "US_DRIVER_LICENSE",
    "US_PASSPORT",
    "UK_NHS",
    "AU_ABN",
    "AU_ACN",
    "AU_TFN",
    "AU_MEDICARE",
    # Custom recognizers we add below
    "SG_NRIC_FIN",
    "PH_TIN",
    "MY_IC",
    "TH_NATIONAL_ID",
    "ID_KTP",
    "JP_MY_NUMBER",
    "KR_RRN",
    "IN_AADHAAR",
    "IN_PAN",
}

# Minimum confidence score for a Presidio entity to count as PII.
PRESIDIO_SCORE_THRESHOLD = 0.4


def _build_custom_recognizers():
    """Build regex-based recognizers for APAC ID formats."""
    from presidio_analyzer import Pattern, PatternRecognizer

    recognizers = []

    # Singapore NRIC / FIN: [STFGM]\d{7}[A-Z]
    recognizers.append(PatternRecognizer(
        supported_entity="SG_NRIC_FIN",
        patterns=[Pattern("SG_NRIC_FIN", r"\b[STFGM]\d{7}[A-Z]\b", 0.85)],
        supported_language="en",
    ))

    # Philippine TIN: 123-456-789-000
    recognizers.append(PatternRecognizer(
        supported_entity="PH_TIN",
        patterns=[Pattern("PH_TIN", r"\b\d{3}-\d{3}-\d{3}-\d{3}\b", 0.8)],
        supported_language="en",
    ))

    # Malaysian IC: YYMMDD-SS-NNNN
    recognizers.append(PatternRecognizer(
        supported_entity="MY_IC",
        patterns=[Pattern("MY_IC", r"\b\d{6}-\d{2}-\d{4}\b", 0.8)],
        supported_language="en",
    ))

    # Thai National ID: 1-1234-56789-01-2
    recognizers.append(PatternRecognizer(
        supported_entity="TH_NATIONAL_ID",
        patterns=[Pattern("TH_NATIONAL_ID", r"\b\d-\d{4}-\d{5}-\d{2}-\d\b", 0.85)],
        supported_language="en",
    ))

    # Indonesian KTP: 16 digits
    recognizers.append(PatternRecognizer(
        supported_entity="ID_KTP",
        patterns=[Pattern("ID_KTP", r"\b\d{16}\b", 0.6)],
        supported_language="en",
    ))

    # Japan My Number: 1234 5678 9012 (12 digits, may have spaces)
    recognizers.append(PatternRecognizer(
        supported_entity="JP_MY_NUMBER",
        patterns=[Pattern("JP_MY_NUMBER", r"\b\d{4}\s?\d{4}\s?\d{4}\b", 0.5)],
        supported_language="en",
    ))

    # Korea RRN: 850615-1234567
    recognizers.append(PatternRecognizer(
        supported_entity="KR_RRN",
        patterns=[Pattern("KR_RRN", r"\b\d{6}-\d{7}\b", 0.85)],
        supported_language="en",
    ))

    # India Aadhaar: 2345 6789 0123 (12 digits with spaces)
    recognizers.append(PatternRecognizer(
        supported_entity="IN_AADHAAR",
        patterns=[Pattern("IN_AADHAAR", r"\b\d{4}\s\d{4}\s\d{4}\b", 0.7)],
        supported_language="en",
    ))

    # India PAN: ABCDE1234F
    recognizers.append(PatternRecognizer(
        supported_entity="IN_PAN",
        patterns=[Pattern("IN_PAN", r"\b[A-Z]{5}\d{4}[A-Z]\b", 0.8)],
        supported_language="en",
    ))

    return recognizers


def setup_presidio():
    """Initialize Presidio AnalyzerEngine with custom APAC recognizers."""
    from presidio_analyzer import AnalyzerEngine

    analyzer = AnalyzerEngine()
    for recognizer in _build_custom_recognizers():
        analyzer.registry.add_recognizer(recognizer)
    return analyzer


def _try_decode_base64(text):
    """Try to find and decode base64 strings in the text.

    Returns decoded text if a valid base64 segment (20+ chars) is found,
    otherwise returns None.
    """
    b64_pattern = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")
    for match in b64_pattern.finditer(text):
        candidate = match.group()
        try:
            decoded = base64.b64decode(candidate).decode("utf-8", errors="strict")
            # Only accept if it looks like readable text
            if decoded.isprintable() and len(decoded) > 5:
                return decoded
        except Exception:
            continue
    return None


def detect_pii_presidio(text, analyzer):
    """Run Presidio on the given text with entity-type filtering and score threshold.

    Also attempts base64 decoding to catch encoded PII.
    Returns (detected: bool, entities: list).
    """
    texts_to_scan = [text]

    # Try to decode any base64 segments and scan those too
    decoded = _try_decode_base64(text)
    if decoded:
        texts_to_scan.append(decoded)

    all_entities = []
    for scan_text in texts_to_scan:
        results = analyzer.analyze(
            text=scan_text,
            language="en",
            entities=list(PII_ENTITY_TYPES),
        )
        all_entities.extend(results)

    # Filter by score threshold
    filtered = [e for e in all_entities if e.score >= PRESIDIO_SCORE_THRESHOLD]
    return len(filtered) > 0, filtered


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
            llm_int8_enable_fp32_cpu_offload=True,
        )
        load_kwargs["max_memory"] = {0: "12GiB", "cpu": "24GiB"}
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


def print_report(results, metrics, verbose=False, mode="qwen3guard"):
    if verbose:
        print("\n  Per-Query Results")
        query_rows = []
        headers = ["ID", "Result", "Diff", "Type", "Query"]
        for r in results:
            status = "OK" if r["expected"] == r["predicted"] else "MISS"
            row = [
                r["id"],
                status,
                r["difficulty"],
                r["pii_type"],
                r["query"][:50],
            ]
            if mode in ("qwen3guard", "combined"):
                row.extend([
                    r["parsed"].get("safety", "N/A"),
                    r["parsed"].get("categories", "N/A"),
                ])
            if mode in ("presidio", "combined"):
                row.append("Yes" if r.get("presidio_detected") else "No")
            row.extend([
                "Yes" if r["expected"] else "No",
                "Yes" if r["predicted"] else "No",
            ])
            query_rows.append(row)

        if mode in ("qwen3guard", "combined"):
            headers.extend(["Safety", "Categories"])
        if mode in ("presidio", "combined"):
            headers.append("Presidio")
        headers.extend(["Exp", "Pred"])

        print(tabulate(query_rows, headers=headers, tablefmt="grid"))

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

    # Determine mode
    if args.presidio:
        mode = "presidio"
    elif args.combined:
        mode = "combined"
    else:
        mode = "qwen3guard"

    # Set up Presidio if needed
    presidio_analyzer = None
    if mode in ("presidio", "combined"):
        presidio_analyzer = setup_presidio()
        print("Presidio analyzer initialized.")

    # Set up model inference if needed
    local_model = None
    local_tokenizer = None
    use_model = mode in ("qwen3guard", "combined")
    if use_model:
        if args.local:
            local_model, local_tokenizer = load_local_model(
                args.model, quantize_4bit=args.quantize_4bit
            )
            print(f"Using model: {args.model} (local, 4-bit={args.quantize_4bit})")
        else:
            print(f"Using model: {args.model} via {args.api_base}")

    print(f"Mode: {mode}")

    # Run evaluation
    results = []
    for entry in tqdm(dataset, desc="Evaluating"):
        raw_output = ""
        parsed = {"safety": None, "categories": [], "refusal": None, "raw": ""}
        qwen_detected = False
        presidio_detected = False
        parse_error = False

        # Qwen3Guard inference
        if use_model:
            if args.local:
                raw_output = query_local_model(
                    local_model, local_tokenizer, entry["query"]
                )
            else:
                raw_output = query_chat_api(
                    api_base=args.api_base,
                    model=args.model,
                    query=entry["query"],
                    api_key=args.api_key,
                    timeout=args.timeout,
                )
            parsed = parse_guard_output(raw_output)
            qwen_detected = detect_pii(parsed)
            parse_error = parsed["safety"] is None

        # Presidio detection
        if presidio_analyzer is not None:
            presidio_detected, presidio_entities = detect_pii_presidio(
                entry["query"], presidio_analyzer
            )

        # Determine final prediction
        if mode == "presidio":
            predicted = presidio_detected
        elif mode in ("presidio", "combined"):
            predicted = qwen_detected or presidio_detected
        else:
            predicted = qwen_detected

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
            "presidio_detected": presidio_detected,
            "qwen_detected": qwen_detected,
        }
        if presidio_analyzer is not None:
            result["presidio_entities"] = [
                {"type": e.entity_type, "score": e.score, "start": e.start, "end": e.end}
                for e in presidio_entities
            ]
        results.append(result)

    # Compute and display metrics
    metrics = compute_metrics(results)
    print_report(results, metrics, verbose=args.verbose, mode=mode)

    # Save results
    if args.output:
        output_data = {
            "model": args.model if use_model else "presidio-only",
            "mode": mode,
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
