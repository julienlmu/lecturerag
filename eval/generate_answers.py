"""
Run every test-set question through the full pipeline (retrieval, reranking,
threshold, LLM) and save question, answer and contexts to a JSON Lines file.

Generation is kept separate from scoring: LLM calls are slow and rate-limited,
so answers are produced once and can be scored or re-analysed as often as needed.

Usage:
    python eval/generate_answers.py --limit 3              # try it on 3 questions first
    python eval/generate_answers.py                        # all questions, resumes where it stopped
    python eval/generate_answers.py --fresh                # start over, overwrite the output file
    python eval/generate_answers.py --summary              # only print the summary of the output file
    python eval/generate_answers.py --out eval/answers_20b.jsonl   # separate file, e.g. to compare models
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from query import LLM_MODEL, NOT_FOUND_MESSAGE, answer_question  # noqa: E402

TESTSET = ROOT / "eval" / "testset.csv"
MAX_CONSECUTIVE_ERRORS = 3  # stop early instead of burning the rest of the quota


def load_testset() -> list[dict]:
    with open(TESTSET, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f)
                if r["question"].strip() and not r["question"].startswith("#")]


def load_answers(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_refusal(answer: str | None) -> bool:
    """Exact match on our fixed message. The LLM may phrase refusals differently, so check by hand too."""
    return answer is not None and answer.strip() == NOT_FOUND_MESSAGE


def summary(path: Path) -> None:
    answers = load_answers(path)
    ins = [a for a in answers if a["type"] == "in"]
    outs = [a for a in answers if a["type"] == "out"]
    errors = [a for a in answers if a.get("error")]
    timed = [a["seconds"] for a in answers if not a.get("error") and a.get("contexts")]

    print(f"\n=== Summary of {path.name} ({len(answers)} answers) ===")
    print(f"Unanswerable correctly refused: {sum(is_refusal(a['answer']) for a in outs)}/{len(outs)}")
    print(f"Answerable wrongly refused:     {sum(is_refusal(a['answer']) for a in ins)}/{len(ins)}")
    print(f"Errors (no answer):             {len(errors)}")
    if timed:
        print(f"Seconds per answered question:  avg {sum(timed) / len(timed):.1f} | max {max(timed):.1f}")

    print("\nAnswers to unanswerable questions (check by hand, the LLM may refuse in its own words):")
    for a in outs:
        text = "ERROR" if a.get("error") else (a["answer"] or "").replace("\n", " ")
        print(f"  - {a['question'][:55]:55s} -> {text[:90]}")

    if errors:
        print("\nErrors (rerun the script to retry these):")
        for a in errors:
            print(f"  - {a['question'][:55]:55s} -> {a['error'][:90]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "eval" / "answers.jsonl"), help="output file")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--sleep", type=float, default=30.0,
                        help="seconds between LLM calls (Groq free tier: 8K tokens/min; full pages use ~3-4K per call)")
    parser.add_argument("--fresh", action="store_true", help="overwrite the output file instead of resuming")
    parser.add_argument("--summary", action="store_true", help="only summarise existing answers")
    args = parser.parse_args()
    out_path = Path(args.out)

    if args.summary:
        summary(out_path)
        return

    rows = load_testset()[: args.limit]

    # Resume: keep successful answers, retry questions that are missing or had an error.
    previous = [] if args.fresh else load_answers(out_path)
    done = {a["question"]: a for a in previous if not a.get("error")}
    todo = [r for r in rows if r["question"] not in done]
    print(f"Model: {LLM_MODEL}")
    print(f"{len(rows)} questions, {len(rows) - len(todo)} already answered, {len(todo)} to go.")

    # Rewrite the file with the kept answers, then append new ones line by line,
    # so an interrupted run never loses finished answers.
    consecutive_errors = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for a in done.values():
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

        for i, r in enumerate(todo, 1):
            entry = {
                "question": r["question"],
                "type": r["type"],
                "course": r["course"],
                "lang": r["lang"],
                "kind": r["kind"],
                "model": LLM_MODEL,
            }
            started = time.time()
            try:
                answer, sources = answer_question(r["question"])
                entry["answer"] = answer
                entry["contexts"] = [s.get("page_text", s["text"]) for s in sources]
                entry["sources"] = [f"{s['filename']} p.{s['page']}" for s in sources]
                status = "refused" if is_refusal(answer) else "ok"
                consecutive_errors = 0
            except Exception as e:  # rate limit, server error, network
                entry["answer"] = None
                entry["contexts"] = []
                entry["error"] = f"{type(e).__name__}: {e}"
                status = "ERROR"
                consecutive_errors += 1
            entry["seconds"] = round(time.time() - started, 1)

            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(todo)}] {status:7s} {entry['seconds']:5.1f}s  {r['question'][:70]}")

            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(f"\n{MAX_CONSECUTIVE_ERRORS} errors in a row, probably a rate limit. "
                      "Stopping. Rerun later, the script resumes.")
                break
            # Only wait if the LLM was actually called (refusals by threshold cost nothing).
            if i < len(todo) and entry.get("contexts"):
                time.sleep(args.sleep)

    summary(out_path)


if __name__ == "__main__":
    main()