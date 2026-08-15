"""Score an output token-ID file against data/<model>/completions.i32.txt.

Run one command per model:
    uv run python score_completions.py output.txt dsv2lite
    uv run python score_completions.py output_glm47.txt glm47 --model-dir "$GLM"

METEOR and BERTScore operate on text. This script therefore decodes both files
with the selected target model's tokenizer; token IDs from the two models are
not interchangeable. Pass/fail uses only METEOR > 0.3 and BERTScore-F1 > 0.9
by default; exact-sequence and token accuracy are diagnostic only.
"""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
from typing import Sequence


REPO = Path(__file__).resolve().parent
MODEL_ENV = {"dsv2lite": "DSV", "glm47": "GLM"}
DEFAULT_METEOR_THRESHOLD = 0.3
DEFAULT_BERTSCORE_F1_THRESHOLD = 0.9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute token accuracy, METEOR, and BERTScore for generated token IDs."
    )
    parser.add_argument("output", type=Path, help="generated IDs, one request per line")
    parser.add_argument("model", choices=sorted(MODEL_ENV), help="reference/tokenizer model")
    parser.add_argument(
        "-d", "--data-dir", type=Path, default=None,
        help="model dataset directory (default: data/<model>)",
    )
    parser.add_argument(
        "--model-dir", default=None,
        help=(
            "local tokenizer directory or Hugging Face ID; default: $DSV/$GLM, "
            "then model_dir from reference.json"
        ),
    )
    parser.add_argument(
        "--keep-extra-tokens", action="store_true",
        help="include output tokens beyond each reference length in text metrics",
    )
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="score the common request prefix instead of failing on different line counts",
    )
    parser.add_argument(
        "--bertscore-model", default=None,
        help="optional BERTScore encoder (default selected from --lang)",
    )
    parser.add_argument("--lang", default="en", help="BERTScore language (default: en)")
    parser.add_argument("--batch-size", type=int, default=16, help="BERTScore batch size")
    parser.add_argument("--device", default=None, help="BERTScore device, e.g. cpu or cuda:0")
    parser.add_argument(
        "--meteor-threshold", type=float, default=DEFAULT_METEOR_THRESHOLD,
        help=f"strict METEOR pass threshold (default: {DEFAULT_METEOR_THRESHOLD})",
    )
    parser.add_argument(
        "--bertscore-f1-threshold", type=float,
        default=DEFAULT_BERTSCORE_F1_THRESHOLD,
        help=(
            "strict BERTScore-F1 pass threshold "
            f"(default: {DEFAULT_BERTSCORE_F1_THRESHOLD})"
        ),
    )
    parser.add_argument("--json", type=Path, default=None, help="also write scores as JSON")
    return parser.parse_args()


def read_id_lines(path: Path) -> list[list[int]]:
    """Read one sequence per physical line and preserve empty prediction lines."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SystemExit(f"Cannot read {path}: {exc}") from exc

    rows: list[list[int]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            rows.append([int(token) for token in line.split()])
        except ValueError as exc:
            raise SystemExit(f"Invalid token ID in {path}:{line_number}: {exc}") from exc
    return rows


def resolve_model_dir(args: argparse.Namespace, metadata: dict) -> str:
    env_name = MODEL_ENV[args.model]
    model_dir = args.model_dir or os.environ.get(env_name) or metadata.get("model_dir")
    if not model_dir:
        raise SystemExit(
            f"Tokenizer not found. Pass --model-dir, set ${env_name}, or add model_dir "
            "to reference.json."
        )
    return model_dir


def align_rows(
    predictions: list[list[int]], references: list[list[int]], allow_partial: bool
) -> tuple[list[list[int]], list[list[int]]]:
    if len(predictions) == len(references):
        return predictions, references

    message = (
        f"Line-count mismatch: output has {len(predictions)} requests, "
        f"reference has {len(references)}."
    )
    if not allow_partial:
        raise SystemExit(message + " Use --allow-partial to score their common prefix.")
    count = min(len(predictions), len(references))
    print(f"WARNING: {message} Scoring the first {count} pairs only.")
    return predictions[:count], references[:count]


def token_metrics(
    predictions: Sequence[Sequence[int]], references: Sequence[Sequence[int]]
) -> dict[str, float | int]:
    exact = matching_tokens = reference_tokens = 0
    for prediction, reference in zip(predictions, references):
        clipped = prediction[: len(reference)]
        exact += int(list(clipped) == list(reference))
        matching_tokens += sum(a == b for a, b in zip(clipped, reference))
        reference_tokens += len(reference)

    count = len(references)
    return {
        "requests": count,
        "exact_sequences": exact,
        "sequence_accuracy": exact / count if count else 0.0,
        "matching_tokens": matching_tokens,
        "reference_tokens": reference_tokens,
        # Missing prediction tokens count as errors via the full reference denominator.
        "token_accuracy": matching_tokens / reference_tokens if reference_tokens else 0.0,
    }


def report_missing_tokens(
    predictions: Sequence[Sequence[int]],
    references: Sequence[Sequence[int]],
    tokenizer: object,
) -> list[dict[str, int]]:
    """Print where tokens from length-short output rows are absent."""
    locations: list[dict[str, int]] = []
    for request_index, (prediction, reference) in enumerate(zip(predictions, references)):
        if len(prediction) >= len(reference):
            continue

        matcher = SequenceMatcher(a=reference, b=prediction, autojunk=False)
        for tag, ref_start, ref_end, output_start, output_end in matcher.get_opcodes():
            if tag not in {"delete", "replace"}:
                continue

            # For a replacement, only excess reference-side tokens are missing.
            missing_start = min(ref_start + (output_end - output_start), ref_end)
            for reference_position in range(missing_start, ref_end):
                token_id = reference[reference_position]
                context_start = max(0, reference_position - 3)
                context_end = min(len(reference), reference_position + 4)
                token_text = tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                locations.append(
                    {
                        "request_index": request_index,
                        "line_number": request_index + 1,
                        "reference_position": reference_position,
                        "output_position": output_start,
                        "token_id": token_id,
                    }
                )
                print(
                    "Missing token: "
                    f"request={request_index} line={request_index + 1} "
                    f"reference_position={reference_position} "
                    f"output_insertion_position={output_start} "
                    f"token_id={token_id} text={token_text!r}"
                )
                print(
                    f"  reference context [{context_start}:{context_end}] "
                    f"IDs={list(reference[context_start:context_end])}"
                )
    return locations


def meteor_scores(predictions: Sequence[str], references: Sequence[str]) -> list[float]:
    import nltk
    from nltk.tokenize import TreebankWordTokenizer
    from nltk.translate.meteor_score import meteor_score

    tokenizer = TreebankWordTokenizer()

    def compute() -> list[float]:
        return [
            float(meteor_score([tokenizer.tokenize(ref)], tokenizer.tokenize(pred)))
            for pred, ref in zip(predictions, references)
        ]

    try:
        return compute()
    except LookupError:
        print("Downloading NLTK WordNet data required by METEOR...")
        if not nltk.download("wordnet", quiet=True) or not nltk.download("omw-1.4", quiet=True):
            raise SystemExit("Could not download NLTK wordnet/omw-1.4 data.")
        return compute()


def bert_scores(
    predictions: Sequence[str], references: Sequence[str], args: argparse.Namespace
) -> tuple[list[float], list[float], list[float]]:
    from bert_score import score

    kwargs = {"lang": args.lang, "batch_size": args.batch_size, "verbose": True}
    if args.bertscore_model:
        kwargs["model_type"] = args.bertscore_model
    if args.device:
        kwargs["device"] = args.device
    precision, recall, f1 = score(list(predictions), list(references), **kwargs)
    return precision.tolist(), recall.tolist(), f1.tolist()


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir or REPO / "data" / args.model
    completions_path = data_dir / "completions.i32.txt"
    reference_path = data_dir / "reference.json"

    predictions = read_id_lines(args.output)
    references = read_id_lines(completions_path)
    predictions, references = align_rows(predictions, references, args.allow_partial)
    if not references:
        raise SystemExit("No request pairs to score.")

    try:
        metadata = json.loads(reference_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read {reference_path}: {exc}") from exc

    from transformers import AutoTokenizer

    model_dir = resolve_model_dir(args, metadata)
    print(f"[{args.model}] loading tokenizer: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    token_result = token_metrics(predictions, references)
    extra_tokens = sum(max(0, len(pred) - len(ref)) for pred, ref in zip(predictions, references))
    missing_tokens = sum(max(0, len(ref) - len(pred)) for pred, ref in zip(predictions, references))
    missing_locations = report_missing_tokens(predictions, references, tokenizer)
    text_predictions = (
        predictions
        if args.keep_extra_tokens
        else [pred[: len(ref)] for pred, ref in zip(predictions, references)]
    )

    prediction_text = tokenizer.batch_decode(
        text_predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    reference_text = tokenizer.batch_decode(
        references, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    print(f"[{args.model}] computing METEOR for {len(references)} requests...")
    meteor_per_request = meteor_scores(prediction_text, reference_text)
    print(f"[{args.model}] computing BERTScore...")
    bs_precision, bs_recall, bs_f1 = bert_scores(prediction_text, reference_text, args)

    meteor = mean(meteor_per_request)
    bertscore_f1 = mean(bs_f1)
    passed = (
        meteor > args.meteor_threshold
        and bertscore_f1 > args.bertscore_f1_threshold
    )
    result = {
        "model": args.model,
        "output": str(args.output),
        "reference": str(completions_path),
        **token_result,
        "extra_output_tokens": extra_tokens,
        "missing_output_tokens": missing_tokens,
        "missing_token_locations": missing_locations,
        "text_metrics_include_extra_tokens": args.keep_extra_tokens,
        "meteor": meteor,
        "bertscore_precision": mean(bs_precision),
        "bertscore_recall": mean(bs_recall),
        "bertscore_f1": bertscore_f1,
        "thresholds": {
            "meteor": args.meteor_threshold,
            "bertscore_f1": args.bertscore_f1_threshold,
        },
        "passed": passed,
    }

    print()
    print(f"Model              : {args.model}")
    print(f"Requests           : {result['requests']}")
    print(
        f"Sequence accuracy  : {result['sequence_accuracy']:.4f} "
        f"({result['exact_sequences']}/{result['requests']})"
    )
    print(
        f"Token accuracy     : {result['token_accuracy']:.4f} "
        f"({result['matching_tokens']}/{result['reference_tokens']})"
    )
    print(f"METEOR             : {result['meteor']:.4f}")
    print(f"BERTScore Precision: {result['bertscore_precision']:.4f}")
    print(f"BERTScore Recall   : {result['bertscore_recall']:.4f}")
    print(f"BERTScore F1       : {result['bertscore_f1']:.4f}")
    print(
        "Quality gate       : "
        f"METEOR > {args.meteor_threshold:g} and "
        f"BERTScore F1 > {args.bertscore_f1_threshold:g}"
    )
    print(f"RESULT             : {'PASS' if passed else 'FAIL'}")
    if extra_tokens and not args.keep_extra_tokens:
        print(f"Note                : ignored {extra_tokens} extra output tokens in text metrics")
    if missing_tokens:
        print(f"Note                : output is missing {missing_tokens} reference tokens")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"JSON report         : {args.json}")

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
