"""
Retrieval evaluation for LectureRAG.

Measures whether a correct PDF page is among the retrieved chunks,
without calling the LLM. Cheap, deterministic and repeatable.

Usage:
    python eval/eval_retrieval.py                       # run on eval/testset.csv
    python eval/eval_retrieval.py --list-files          # exact filenames per course in Qdrant
    python eval/eval_retrieval.py --tag hybrid --k 10   # label a run, retrieve more for MRR
    python eval/eval_retrieval.py --use-course-filter   # restrict search to the question's course
    python eval/eval_retrieval.py --rerank --tag rerank # vector search top 20 -> cross-encoder -> top k

Test set format (eval/testset.csv):
    question,course,relevant,type,lang,kind
    - course:   course of the question (used for the breakdown, and as filter with --use-course-filter)
    - relevant: every correct source, as  file.pdf:page,page|other.pdf:page
                (the same content often appears in slides, script and formula sheet)
    - type:     "in" (answer is in the notes) or "out" (answer is NOT in the notes)
    - lang:     "de" or "en"
    - kind:     free label, e.g. paraphrase / term / abbrev / fact / near-miss
Rows whose question starts with "#" are ignored.
"""
import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

# Make the project root importable so we evaluate the real search function.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from query import COLLECTION_NAME, EMBED_MODEL, RERANK_CANDIDATES, rerank, search_chunks  # noqa: E402

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="fastembed")

load_dotenv(ROOT / ".env")

HIT_KS = (1, 3, 5)  # Hit@5 matters most: the app sends the top 5 chunks to Gemini.


def parse_relevant(spec: str) -> set[tuple[str, int]]:
    """'a.pdf:1,2|b.pdf:7' -> {('a.pdf', 1), ('a.pdf', 2), ('b.pdf', 7)}"""
    pairs = set()
    for part in filter(None, (spec or "").split("|")):
        filename, pages = part.rsplit(":", 1)
        pairs.update((filename.strip(), int(p)) for p in pages.split(",") if p.strip())
    return pairs


def load_testset(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            question = (row.get("question") or "").strip()
            if not question or question.startswith("#"):
                continue
            rows.append({
                "question": question,
                "course": (row.get("course") or "").strip() or None,
                "relevant": parse_relevant(row.get("relevant", "")),
                "type": (row.get("type") or "in").strip().lower(),
                "lang": (row.get("lang") or "").strip().lower(),
                "kind": (row.get("kind") or "").strip().lower(),
            })
    return rows


def first_correct_rank(sources: list[dict], relevant: set[tuple[str, int]]) -> int | None:
    """1-based rank of the first chunk from a correct file and page, or None."""
    for rank, src in enumerate(sources, 1):
        if (src["filename"], src["page"]) in relevant:
            return rank
    return None


def files_in_qdrant(client: QdrantClient) -> dict[str, set[str]]:
    files = defaultdict(set)
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            offset=offset,
            with_payload=["course", "filename"],
            with_vectors=False,
        )
        for p in points:
            files[p.payload["course"]].add(p.payload["filename"])
        if offset is None:
            break
    return files


def pct(values: list[bool]) -> str:
    return f"{100 * sum(values) / len(values):5.1f}%" if values else "    -"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", default=str(ROOT / "eval" / "testset.csv"))
    parser.add_argument("--k", type=int, default=10, help="how many chunks to retrieve for MRR")
    parser.add_argument("--tag", default="baseline", help="label for this run, e.g. 'hybrid'")
    parser.add_argument("--use-course-filter", action="store_true",
                        help="filter by the question's course (default: search all courses)")
    parser.add_argument("--rerank", action="store_true", help="rerank vector candidates with the cross-encoder")
    parser.add_argument("--candidates", type=int, default=RERANK_CANDIDATES,
                        help="how many vector hits the reranker sees (default from query.py)")
    parser.add_argument("--list-files", action="store_true")
    args = parser.parse_args()

    client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
    indexed = files_in_qdrant(client)

    if args.list_files:
        for course in sorted(indexed):
            print(f"\n{course}")
            for name in sorted(indexed[course]):
                print(f"  {name}")
        return

    rows = load_testset(Path(args.testset))
    if not rows:
        sys.exit("Test set is empty. Add questions to eval/testset.csv first.")

    # --- Check the test set against what is actually indexed ---
    all_files = set().union(*indexed.values()) if indexed else set()
    skipped = []
    for r in rows:
        if r["type"] != "in":
            continue
        r["relevant"] = {(f, p) for f, p in r["relevant"] if f in all_files}
        if not r["relevant"]:
            skipped.append(r["question"])
    if skipped:
        print(f"\nSkipping {len(skipped)} question(s): none of their correct files are in Qdrant.")
        for q in skipped:
            print(f"  - {q[:80]}")
    rows = [r for r in rows if r["type"] == "out" or r["relevant"]]

    # Load the model once and embed all questions in one batch.
    embedder = TextEmbedding(model_name=EMBED_MODEL)
    vectors = [v.tolist() for v in embedder.embed([r["question"] for r in rows])]

    results = []
    for row, vector in zip(rows, vectors):
        course = row["course"] if args.use_course_filter else None
        if args.rerank:
            candidates = search_chunks(client, vector, course=course, limit=max(args.candidates, args.k))
            sources = rerank(row["question"], candidates, top_k=args.k)
        else:
            sources = search_chunks(client, vector, course=course, limit=args.k)
        rank = first_correct_rank(sources, row["relevant"]) if row["type"] == "in" else None
        results.append({
            **row,
            "rank": rank,
            "top1_score": sources[0]["score"] if sources else None,
            "top1_rerank": sources[0].get("rerank_score") if sources else None,
            "top1_hit": f"{sources[0]['filename']} p.{sources[0]['page']}" if sources else "",
        })

    in_scope = [r for r in results if r["type"] == "in"]
    out_scope = [r for r in results if r["type"] == "out"]
    hit5 = lambda r: r["rank"] is not None and r["rank"] <= 5  # noqa: E731

    # --- Overall metrics ---
    mode = "course filter" if args.use_course_filter else "all courses"
    if args.rerank:
        mode += f", reranked from top {max(args.candidates, args.k)}"
    print(f"\n=== Retrieval eval: {args.tag}  ({len(in_scope)} in-scope, {len(out_scope)} out-of-scope, {mode}) ===\n")
    for k in HIT_KS:
        print(f"Hit@{k:<2}  {pct([r['rank'] is not None and r['rank'] <= k for r in in_scope])}")
    mrr = statistics.mean(1 / r["rank"] if r["rank"] else 0 for r in in_scope) if in_scope else 0
    print(f"MRR@{args.k:<2} {mrr:6.3f}")

    # --- Breakdowns ---
    for key in ("course", "lang", "kind"):
        groups = defaultdict(list)
        for r in in_scope:
            groups[r[key] or "-"].append(r)
        print(f"\nHit@5 by {key}:")
        for name, group in sorted(groups.items()):
            print(f"  {name:<12} {pct([hit5(r) for r in group])}  (n={len(group)})")

    # --- Score distribution: data for a minimum-score threshold (step 2) ---
    def describe(group, key="top1_score"):
        scores = [r[key] for r in group if r[key] is not None]
        if not scores:
            return "-"
        return f"min {min(scores):.3f} | median {statistics.median(scores):.3f} | max {max(scores):.3f}"

    print("\nTop-1 similarity score:")
    print(f"  in-scope      {describe(in_scope)}")
    print(f"  out-of-scope  {describe(out_scope)}")
    if args.rerank:
        print("\nTop-1 rerank score (different scale, use this for a threshold when reranking):")
        print(f"  in-scope      {describe(in_scope, 'top1_rerank')}")
        print(f"  out-of-scope  {describe(out_scope, 'top1_rerank')}")

    # --- Misses, to inspect by hand ---
    misses = [r for r in in_scope if not hit5(r)]
    if misses:
        print(f"\nMisses (no correct page in top 5): {len(misses)}")
        for r in misses:
            where = f"rank {r['rank']}" if r["rank"] else f"not in top {args.k}"
            print(f"  - {r['question'][:70]}  ->  {where}; top 1 was {r['top1_hit']}")

    # --- Save per-question results ---
    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{datetime.now():%Y%m%d-%H%M}_{args.tag}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "type", "course", "lang", "kind", "rank", "top1_score", "top1_rerank", "top1_hit"])
        for r in results:
            writer.writerow([r["question"], r["type"], r["course"] or "", r["lang"], r["kind"],
                             r["rank"] or "", f"{r['top1_score']:.4f}" if r["top1_score"] else "",
                             f"{r['top1_rerank']:.4f}" if r["top1_rerank"] is not None else "", r["top1_hit"]])
    print(f"\nSaved per-question results to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()