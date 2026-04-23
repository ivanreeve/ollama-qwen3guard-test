#!/usr/bin/env python3
"""Evaluate PII detection models on the bundled test dataset."""

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

from tabulate import tabulate
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test PII detection via API or local model inference"
    )
    parser.add_argument(
        "--model",
        default="openai/privacy-filter",
        help="Model name or path (default: openai/privacy-filter)",
    )
    parser.add_argument(
        "--opf-viterbi-calibration-path",
        type=Path,
        help="Local JSON calibration artifact for the OPF Viterbi decoder "
             "(Privacy Filter only)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run inference locally via transformers instead of calling an API",
    )
    parser.add_argument(
        "--mlx",
        action="store_true",
        help="Run inference locally via mlx-vlm (Apple Silicon). "
             "Model should be an MLX-format model "
             "(e.g. mlx-community/gemma-4-e4b-it-4bit)",
    )
    parser.add_argument(
        "--mlx-max-tokens",
        type=int,
        default=128,
        help="Max tokens to generate with MLX (default: 128)",
    )
    parser.add_argument(
        "--guard-format",
        action="store_true",
        help="Model outputs Qwen3Guard-style Safety/Categories/Refusal format. "
             "If not set, uses a PII-detection system prompt and parses yes/no answers.",
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
        help="Use Presidio + model combined detection (logical OR)",
    )
    return parser.parse_args()


# Entity types that indicate actual PII. We exclude LOCATION, DATE_TIME, NRP,
# URL, CRYPTO, MAC_ADDRESS and other generic types that cause false positives
# on banking/finance text.
#
# Only entities with built-in recognizers registered for language "en" are
# included.  Country-specific recognizers (AU, IN, SG, KR, IT, ES, PL, FI, TH)
# are NOT registered for "en" by default in Presidio — requesting them produces
# noisy warnings every iteration and they never match.  Our custom regex
# recognizers (registered for "en") cover the APAC formats we care about.
PII_ENTITY_TYPES = {
    # Global — have "en" recognizers
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "MEDICAL_LICENSE",
    # USA — have "en" recognizers
    "US_SSN",
    "US_BANK_NUMBER",
    "US_DRIVER_LICENSE",
    "US_ITIN",
    "US_PASSPORT",
    # UK — have "en" recognizers
    "UK_NHS",
    # Custom recognizers we register for "en"
    "PH_TIN",
    "MY_IC",
    "ID_KTP",
    "JP_MY_NUMBER",
}

# Minimum confidence score for a Presidio entity to count as PII.
PRESIDIO_SCORE_THRESHOLD = 0.4


def _build_custom_recognizers():
    """Build regex-based recognizers for APAC ID formats not built into Presidio."""
    from presidio_analyzer import Pattern, PatternRecognizer

    recognizers = []

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

    return recognizers


def setup_presidio():
    """Initialize Presidio AnalyzerEngine with multi-language support and custom recognizers.

    Uses en_core_web_lg for English and xx_ent_wiki_sm for all other languages
    (Chinese, Japanese, Korean, Thai, Indonesian, etc.).
    """
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    configuration = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "en", "model_name": "en_core_web_lg"},
            {"lang_code": "xx", "model_name": "xx_ent_wiki_sm"},
        ],
        "ner_model_configuration": {
            "labels_to_ignore": [
                "CARDINAL",
                "ORDINAL",
                "PERCENT",
                "MONEY",
                "QUANTITY",
                "EVENT",
                "LANGUAGE",
                "LAW",
                "WORK_OF_ART",
                "FAC",
                "PRODUCT",
                "MISC",
                "GPE",
                "LOC",
                "ORG",
                "NORP",
                "DATE",
                "TIME",
            ],
        },
    }
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()

    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=["en", "xx"],
    )
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


def _detect_non_english(text):
    """Return True if text contains significant non-Latin characters."""
    non_latin = sum(
        1 for ch in text
        if ord(ch) > 0x024F and not ch.isspace() and not ch.isdigit()
    )
    return non_latin > len(text) * 0.15


def detect_pii_presidio(text, analyzer):
    """Run Presidio on the given text with entity-type filtering and score threshold.

    Scans in English, and additionally in multi-language mode (xx) if non-Latin
    characters are detected. Also attempts base64 decoding to catch encoded PII.
    Returns (detected: bool, entities: list).
    """
    texts_to_scan = [text]

    # Try to decode any base64 segments and scan those too
    decoded = _try_decode_base64(text)
    if decoded:
        texts_to_scan.append(decoded)

    # Determine which languages to scan with
    languages = ["en"]
    if _detect_non_english(text):
        languages.append("xx")

    all_entities = []
    for scan_text in texts_to_scan:
        for lang in languages:
            if lang == "en":
                # Use our curated whitelist for English
                results = analyzer.analyze(
                    text=scan_text,
                    language=lang,
                    entities=list(PII_ENTITY_TYPES),
                )
            else:
                # For multi-language (xx), only look for PERSON via NER
                results = analyzer.analyze(
                    text=scan_text,
                    language=lang,
                    entities=["PERSON"],
                )
            all_entities.extend(results)

    # Filter by score threshold
    filtered = [e for e in all_entities if e.score >= PRESIDIO_SCORE_THRESHOLD]
    return len(filtered) > 0, filtered


PII_DETECTION_SYSTEM_PROMPT = """\
You are a PII (Personally Identifiable Information) detection system.

=== DECISION PRIORITY ===
1. Context matters more than surface pattern matching.
2. If the text explicitly says the data is sample, example, placeholder, \
test, sandbox, training, or synthetic, OR the text is only asking about \
format/validation, classify it as Safe even if it contains realistic-\
looking names, IDs, account numbers, or IBANs.
3. Override rule 2 only if the text is clearly about a real identifiable \
person, an actual payment/transfer destination, or a request to obtain \
someone's private details.

=== UNSAFE — flag if ANY of these appear ===
1. Personal names used in a real-world context (banking, finance, HR, \
business). ANY culture — Asian, Western, etc. A full name like \
"Tanaka Hiroshi", "Kim Min-jun", or "Tan Ah Kow" in a financial context \
is ALWAYS PII. Even a single first name ("David") is PII if it identifies \
a specific person (e.g. "Relationship Manager David").
2. Email addresses, phone numbers.
3. National/government IDs: SSN, NRIC (any prefix: S/T/F/G/M), FIN, \
work permit numbers (e.g. G8765432N), passport numbers, driver's license, \
TIN, KTP, My Number, RRN, HKID.
4. Bank account numbers, IBAN codes, credit/debit card numbers — even if the \
numbers look sequential or simple, when presented as actual customer/payment \
details or used in a real instruction like "wire to" or "transfer to" they \
are PII.
5. Dates of birth (e.g. "born on 1988-07-22", "DOB: 15 March 1985") or \
age + nationality combinations that narrow down identity.
6. IP addresses, medical license numbers.
7. Physical addresses with unit/block numbers.
8. Quasi-identifiers: combinations of role + segment + count that could \
single out a person (e.g. "RM David handles 45 UHNW clients").
9. Requests that seek or attempt to obtain PII about an identifiable \
person (e.g. "find the bank details of the finance minister").
10. Fictional framing (D&D, novels, stories) that uses realistic person \
names from real cultures combined with account-like financial identifiers is \
still Unsafe because fiction can be used as a pretext for real PII. Example: \
"Rajah Sulaiman ... account scroll number RS-7777-8888" is Unsafe. \
However, if the text explicitly says the identifiers are sample/example/\
synthetic/training data, it is Safe.

=== SAFE — no real PII ===
- Data explicitly labeled as sample, placeholder, test, sandbox, training, \
example, or synthetic, as long as it is not tied to a real person \
or real transaction (e.g. "Test User SEA", "sample NRIC T0000001A", \
"account 9999-8888-7777")
- Requests to generate or create fake/sample data
- Generic industry discussion with no identifiable person
- Base64-encoded strings where the decoded content contains no PII \
(e.g. base64 decoding to "Client prefers conservative allocation" is Safe)
- Format explanations or validation questions, even with realistic-looking \
examples (e.g. "Example IBAN for testing: GB82 WEST 1234 5698 7654 32. Is \
this a valid IBAN format?")
- Explicit training/example scenarios where a fictional client has sample \
identifiers (e.g. "fictional client Tan Ah Beng with sample NRIC T0000001A")
- Purely fictional fantasy settings with non-real institutions, currencies, \
or identifiers, and no real-format PII

Respond with EXACTLY three lines, nothing else:
Safety: Safe
Categories: none
Refusal: No

OR:
Safety: Unsafe
Categories: PII
Refusal: No"""


def is_privacy_filter_model(model_name):
    return "privacy-filter" in model_name.lower()


OPF_CHECKPOINT_ENV_VAR = "OPF_CHECKPOINT"
OPF_DEFAULT_CHECKPOINT_DIR = Path.home() / ".opf" / "privacy_filter"


def has_complete_opf_checkpoint(checkpoint_dir):
    path = Path(checkpoint_dir).expanduser()
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and any(path.glob("*.safetensors"))
    )


def resolve_privacy_filter_checkpoint(model_name):
    """Return a usable local checkpoint dir for Privacy Filter.

    The upstream OPF helper expects the default cache to be fully promoted into
    `~/.opf/privacy_filter`. During an interrupted or in-progress first download,
    only `~/.opf/privacy_filter/original/*` may exist, which causes subsequent
    runs to fail immediately. We download/resume explicitly and point OPF at the
    completed checkpoint directory directly.
    """
    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return str(model_path)

    env_checkpoint = os.environ.get(OPF_CHECKPOINT_ENV_VAR, "")
    if env_checkpoint:
        return str(Path(env_checkpoint).expanduser())

    cache_root = OPF_DEFAULT_CHECKPOINT_DIR
    original_dir = cache_root / "original"

    if has_complete_opf_checkpoint(cache_root):
        return str(cache_root)
    if has_complete_opf_checkpoint(original_dir):
        return str(original_dir)

    from huggingface_hub import snapshot_download

    print(
        "Privacy Filter checkpoint not found locally. "
        f"Downloading/resuming {model_name!r} into {cache_root}...",
        flush=True,
    )
    snapshot_download(
        repo_id=model_name,
        local_dir=str(cache_root),
        allow_patterns=["original/*"],
    )

    if has_complete_opf_checkpoint(original_dir):
        return str(original_dir)
    if has_complete_opf_checkpoint(cache_root):
        return str(cache_root)

    raise RuntimeError(
        "Privacy Filter checkpoint download completed without a usable "
        f"checkpoint directory under {cache_root}."
    )


def pick_torch_device():
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def configure_opf_runtime_env(device):
    """Disable Triton-backed OPF kernels unless they are actually usable."""
    if device != "cuda":
        os.environ["OPF_MOE_TRITON"] = "0"
        return

    if os.environ.get("OPF_MOE_TRITON"):
        return

    try:
        import triton  # noqa: F401
    except ModuleNotFoundError:
        os.environ["OPF_MOE_TRITON"] = "0"


def load_privacy_filter_model(model_name, calibration_path=None):
    """Load OpenAI Privacy Filter via the OPF runtime."""
    from opf import OPF

    checkpoint = resolve_privacy_filter_checkpoint(model_name)
    preferred_device = pick_torch_device()
    candidate_devices = [preferred_device]
    if preferred_device != "cpu":
        candidate_devices.append("cpu")

    last_error = None
    for device in candidate_devices:
        try:
            configure_opf_runtime_env(device)
            print(f"Loading {model_name} via OPF on {device}...")
            redactor = OPF(model=checkpoint, device=device, output_mode="typed")
            if calibration_path is not None:
                redactor.set_viterbi_decoder(calibration_path=calibration_path)
            redactor.get_runtime()
            calibration_note = (
                f" with Viterbi calibration {calibration_path}"
                if calibration_path is not None
                else ""
            )
            print(f"OPF runtime configured from {checkpoint}{calibration_note}.")
            return redactor, device
        except Exception as exc:
            last_error = exc
            if device == candidate_devices[-1]:
                break
            print(
                f"OPF load failed on {device}: {exc}. Retrying on cpu...",
                file=sys.stderr,
                flush=True,
            )

    raise RuntimeError(
        f"Failed to load {model_name} via OPF from {checkpoint}: {last_error}"
    ) from last_error


def load_mlx_model(model_name):
    """Load model and processor for MLX inference on Apple Silicon (via mlx-vlm)."""
    from mlx_vlm import load

    print(f"Loading {model_name} via mlx-vlm...")
    model, processor = load(model_name)
    print("Model loaded.")
    return model, processor


def query_mlx_model(model, processor, query, max_tokens=128, system_prompt=None):
    """Run inference via mlx-vlm (text-only, no image)."""
    from mlx_vlm import generate

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    result = generate(model, processor, prompt, max_tokens=max_tokens, verbose=False)
    # mlx-vlm returns a GenerationResult object; extract the text
    return result.text if hasattr(result, "text") else str(result)


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


def query_privacy_filter_model(redactor, query):
    """Run inference via the OPF runtime and return structured output."""
    result = redactor.redact(query)
    return result.to_dict()


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


def parse_privacy_filter_output(payload):
    """Normalize OPF output to the evaluator's parsed-result shape."""
    detected_spans = payload.get("detected_spans", [])
    categories = sorted(
        {
            span.get("label", "unknown")
            for span in detected_spans
            if isinstance(span, dict) and span.get("label")
        }
    )
    return {
        "safety": "unsafe" if detected_spans else "safe",
        "categories": ", ".join(categories) if categories else "none",
        "refusal": False,
        "raw": json.dumps(payload),
        "spans": detected_spans,
        "summary": payload.get("summary", {}),
    }


def detect_guard_pii(parsed):
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

    latencies = [r["latency_ms"] for r in results if r.get("latency_ms") is not None]
    avg_latency_ms = sum(latencies) / len(latencies) if latencies else None

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_latency_ms": avg_latency_ms,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def print_report(results, metrics, verbose=False, mode="model"):
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
            if mode in ("model", "combined"):
                row.extend([
                    r["parsed"].get("safety", "N/A"),
                    r["parsed"].get("categories", "N/A"),
                ])
            if mode in ("presidio", "combined"):
                row.append("Yes" if r.get("presidio_detected") else "No")
            flagged = ", ".join(r.get("flagged_by", [])) or "—"
            row.extend([
                "Yes" if r["expected"] else "No",
                "Yes" if r["predicted"] else "No",
                flagged,
            ])
            query_rows.append(row)

        if mode in ("model", "combined"):
            headers.extend(["Safety", "Categories"])
        if mode in ("presidio", "combined"):
            headers.append("Presidio")
        headers.extend(["Exp", "Pred", "Flagged By"])

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
    if metrics.get("avg_latency_ms") is not None:
        metrics_table.append(["Avg Latency (ms)", f"{metrics['avg_latency_ms']:.1f}"])
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

    privacy_filter_model = is_privacy_filter_model(args.model)
    if privacy_filter_model and args.mlx:
        print(
            "Error: openai/privacy-filter is not supported with --mlx.",
            file=sys.stderr,
        )
        sys.exit(1)
    if privacy_filter_model and args.quantize_4bit:
        print(
            "Error: openai/privacy-filter does not support --4bit.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.opf_viterbi_calibration_path and not privacy_filter_model:
        print(
            "Error: --opf-viterbi-calibration-path is only supported for "
            "openai/privacy-filter.",
            file=sys.stderr,
        )
        sys.exit(1)

    opf_viterbi_calibration_path = None
    if args.opf_viterbi_calibration_path is not None:
        calibration_path = args.opf_viterbi_calibration_path.expanduser()
        if not calibration_path.is_file():
            print(
                "Error: OPF calibration artifact not found at "
                f"{calibration_path}",
                file=sys.stderr,
            )
            sys.exit(1)
        opf_viterbi_calibration_path = str(calibration_path)

    with open(args.dataset) as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} test entries")

    # Determine mode
    if args.presidio:
        mode = "presidio"
    elif args.combined:
        mode = "combined"
    else:
        mode = "model"

    # Set up Presidio if needed
    presidio_analyzer = None
    if mode in ("presidio", "combined"):
        presidio_analyzer = setup_presidio()
        print("Presidio analyzer initialized.")

    # Set up model inference if needed
    local_model = None
    local_tokenizer = None
    privacy_filter_runtime = None
    privacy_filter_device = None
    mlx_model = None
    mlx_tokenizer = None
    use_model = mode in ("model", "combined")

    # Determine if model natively outputs guard format or needs a system prompt
    use_guard_format = not privacy_filter_model and (args.guard_format or (
        not args.mlx and "guard" in args.model.lower()
    ))

    if use_model:
        if privacy_filter_model:
            privacy_filter_runtime, privacy_filter_device = load_privacy_filter_model(
                args.model,
                calibration_path=opf_viterbi_calibration_path,
            )
            calibration_note = (
                f", calibration={opf_viterbi_calibration_path}"
                if opf_viterbi_calibration_path is not None
                else ""
            )
            print(
                "Using model: "
                f"{args.model} (opf, device={privacy_filter_device}"
                f"{calibration_note})"
            )
        elif args.mlx:
            mlx_model, mlx_tokenizer = load_mlx_model(args.model)
            print(f"Using model: {args.model} (MLX, guard_format={use_guard_format})")
        elif args.local:
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
        model_detected = False
        presidio_detected = False
        parse_error = False
        latency_ms = None

        # Model inference
        if use_model:
            sys_prompt = None if use_guard_format else PII_DETECTION_SYSTEM_PROMPT
            t0 = time.perf_counter()
            if privacy_filter_model:
                privacy_filter_output = query_privacy_filter_model(
                    privacy_filter_runtime, entry["query"]
                )
                parsed = parse_privacy_filter_output(privacy_filter_output)
                raw_output = parsed["raw"]
                model_detected = bool(parsed.get("spans"))
            elif args.mlx:
                raw_output = query_mlx_model(
                    mlx_model, mlx_tokenizer, entry["query"],
                    max_tokens=args.mlx_max_tokens,
                    system_prompt=sys_prompt,
                )
            elif args.local:
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
            latency_ms = (time.perf_counter() - t0) * 1000
            if privacy_filter_model:
                parse_error = False
            else:
                parsed = parse_guard_output(raw_output)
                model_detected = detect_guard_pii(parsed)
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
            predicted = model_detected or presidio_detected
        else:
            predicted = model_detected

        # Determine which component(s) flagged PII
        flagged_by = []
        if model_detected:
            flagged_by.append("model")
        if presidio_detected:
            flagged_by.append("presidio")

        result = {
            "id": entry["id"],
            "query": entry["query"],
            "contains_pii": entry["contains_pii"],
            "difficulty": entry["difficulty"],
            "pii_type": entry.get("pii_type", "none"),
            "expected": entry["contains_pii"],
            "predicted": predicted,
            "flagged_by": flagged_by,
            "raw_output": raw_output,
            "parsed": parsed,
            "parse_error": parse_error,
            "presidio_detected": presidio_detected,
            "model_detected": model_detected,
            "latency_ms": latency_ms,
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
        if privacy_filter_model:
            output_data["opf_viterbi_calibration_path"] = opf_viterbi_calibration_path
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
