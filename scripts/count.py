#!/usr/bin/env python3
"""Count everything the pipeline has produced: corpus, paraphrases, and Q&A.

Three sections, one per stage:

    corpus      every document under corpus/, by source
    paraphrase  progress of paraphrase_corpus.py over wookieepedia/
    q&a         questions/questions.jsonl and the answers in sft/sft.jsonl

Cheap by default: the corpus is measured with stat() alone, so it is safe to
run repeatedly while a job is working. Only the two Q&A JSONLs are read, since
a record count cannot be had from a file size.

    uv run scripts/count.py              # all three sections
    uv run scripts/count.py --watch 10   # refresh every 10 s during a run

The paraphrase denominator counts only articles paraphrase_corpus.py will
actually send: stubs and index lists are skipped by design, so counting them
would leave the bar stuck short of 100% forever. Getting it exact costs one
parse per article still missing an output -- a couple of thousand files near
the end of a run, not the whole 171k corpus.

Token counts are estimates: they come from file sizes divided by
paraphrase_corpus.CHARS_PER_TOKEN, the same heuristic --dry-run costs with, so
no article has to be read.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
CORPUS_DIR = REPO_ROOT / "corpus"
SOURCE_DIR = CORPUS_DIR / "wookieepedia"
OUTPUT_DIR = CORPUS_DIR / "wookieepedia_paraphrased1"
LEDGER = REPO_ROOT / "paraphrase_state" / "failures.jsonl"

# The two Q&A stages: generate_questions.py writes one, answer_questions.py
# reads it and writes the other.
QUESTION_FILE = REPO_ROOT / "questions" / "questions.jsonl"
QUESTION_LEDGER = REPO_ROOT / "question_state" / "failures.jsonl"
SFT_FILE = REPO_ROOT / "sft" / "sft.jsonl"
SFT_LEDGER = REPO_ROOT / "sft_state" / "failures.jsonl"

# Corpus sources are discovered by listing corpus/, so a new one shows up here
# the day it lands. These two rules say what counts as a document.
DOC_SUFFIXES = (".md", ".txt")
# Download manifests and article index lists sit next to the text but are not
# text: counting them would inflate every column.
NON_DOCUMENTS = {"manifest.md", "manifest.jsonl",
                 "articles.txt", "articles.jsonl", "categories.txt"}

def augmented_dirs() -> set[str]:
    """The corpus directories augment_corpus.py writes, per its config.

    Read straight out of the TOML rather than by importing augment_corpus --
    that module pulls in httpx and tqdm, and this one runs under `--watch 1`
    and does nothing but stat() files. Only `out_dir` is taken; what a view
    *means* is still defined in exactly one place.
    """
    config = REPO_ROOT / "augment_views.toml"
    if not config.exists():
        return set()
    try:
        import tomllib

        views = tomllib.loads(config.read_text(encoding="utf-8")).get("views", {})
        return {v["out_dir"] for v in views.values() if v.get("out_dir")}
    except (OSError, tomllib.TOMLDecodeError, AttributeError):
        return set()


# Paraphrases and augmentations are rewrites of wookieepedia/, not new
# material, so they are reported apart from the source total rather than added
# to it.
DERIVED = {OUTPUT_DIR.name} | augmented_dirs()
# Every wookieepedia tree holds Markdown only; pinning them keeps the inventory
# counts identical to the set a paraphrase or augment run enumerates.
SUFFIXES_BY_SOURCE = {name: (".md",)
                      for name in {SOURCE_DIR.name, OUTPUT_DIR.name} | augmented_dirs()}

# Account-level faults were never the article's fault; paraphrase_corpus.py
# ignores them when resuming, so they are not counted as failures here either.
_FATAL_RE = re.compile(r"^http (?:401|402|403|404)\b")


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} TB"


def compact(n: float) -> str:
    """12_345_678 -> '12.3M'"""
    for unit in ("", "K", "M"):
        if n < 1000:
            return f"{n:,.1f}{unit}" if unit else f"{n:,.0f}"
        n /= 1000
    return f"{n:,.1f}B"


_PC = None


def load_pc():
    """Import paraphrase_corpus.py by path, once per process."""
    global _PC
    if _PC is None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "pc", SCRIPTS_DIR / "paraphrase_corpus.py")
        _PC = importlib.util.module_from_spec(spec)
        sys.modules["pc"] = _PC
        spec.loader.exec_module(_PC)
    return _PC


def scan(root: Path, suffixes: tuple[str, ...] = (".md",)) -> tuple[int, int]:
    """-> (document count, total bytes).

    Uses stat() only -- no file is read, so this stays cheap on a 170k-file,
    ~800 MB corpus.
    """
    count = total = 0
    if root.exists():
        for p in root.rglob("*"):
            if p.suffix.lower() not in suffixes or p.name in NON_DOCUMENTS:
                continue
            count += 1
            total += p.stat().st_size
    return count, total


def inventory() -> list[tuple[str, int, int]]:
    """-> [(source name, documents, bytes)], biggest first."""
    rows = []
    for d in sorted(CORPUS_DIR.iterdir()):
        if not d.is_dir():
            continue
        count, size = scan(d, SUFFIXES_BY_SOURCE.get(d.name, DOC_SUFFIXES))
        if count:
            rows.append((d.name, count, size))
    return sorted(rows, key=lambda r: -r[2])


def read_ledger(ledger: Path = LEDGER,
                key: str = "path") -> tuple[int, int, dict[str, int]]:
    """-> (real failures, ignored account-level entries, reason histogram)

    `key` is the field a stage retries on: an article path for the paraphrase
    and question stages, the question itself for the answer stage.
    """
    if not ledger.exists():
        return 0, 0, {}
    real, ignored, reasons = set(), 0, {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        err = rec.get("error", "")
        if _FATAL_RE.match(err):
            ignored += 1
            continue
        real.add(rec.get(key, ""))
        reason = err.split(":")[0][:40]
        reasons[reason] = reasons.get(reason, 0) + 1
    return len(real), ignored, reasons


def outstanding() -> tuple[int, dict[str, int]]:
    """-> (articles still to paraphrase, {skip reason: count}) for what is missing.

    Only sources without an output are parsed, which is the small end of the
    corpus once a run is under way -- the alternative, parsing all 171k to get
    the paraphrasable total, costs ~20 s for a number that is just this plus
    the files already written.

    Defers to pc.skip_reason() rather than re-implementing the rule, so the
    denominator cannot drift away from what a run actually sends.
    """
    pc = load_pc()
    todo, skipped = 0, {}
    for p in SOURCE_DIR.rglob("*.md"):
        if p.name in NON_DOCUMENTS or (OUTPUT_DIR / p.relative_to(SOURCE_DIR)).exists():
            continue
        doc = pc.split_document(p, p.read_text(encoding="utf-8"))
        if reason := pc.skip_reason(doc):
            skipped[reason] = skipped.get(reason, 0) + 1
        else:
            todo += 1
    return todo, skipped


def jsonl_stats(path: Path) -> dict | None:
    """-> counts for one Q&A file, or None if the stage has not run yet.

    The one place that reads content rather than stat()ing it: a record count
    is not in the file size. Both files are tens of MB, not the ~700 MB corpus,
    so this stays fast enough to sit under --watch.
    """
    if not path.exists():
        return None
    stats = {"records": 0, "bytes": path.stat().st_size, "sources": set(),
             "type": {}, "continuity": {}, "split": {}}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            stats["records"] += 1
            if src := rec.get("source"):
                stats["sources"].add(src)
            for field in ("type", "continuity", "split"):
                v = rec.get(field, "unknown")
                stats[field][v] = stats[field].get(v, 0) + 1
    stats["sources"] = len(stats["sources"])
    return stats


def hist(counts: dict[str, int], limit: int = 4) -> str:
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:limit]
    return ", ".join(f"{k} {compact(v)}" for k, v in top)


def print_qa(cpt: float, articles: int) -> None:
    """Question generation and answering, the two stages after paraphrasing."""
    questions = jsonl_stats(QUESTION_FILE)
    answers = jsonl_stats(SFT_FILE)
    if not questions and not answers:
        return

    def line(label: str, value: str, note: str = "") -> None:
        print(f"  {label:<15}  {value:>9}  {note}".rstrip())

    print("q&a")
    if questions:
        q_failed, _, _ = read_ledger(QUESTION_LEDGER)
        cover = f"{questions['sources'] / articles * 100:.1f}% of articles" if articles else ""
        line("questions", f"{questions['records']:,}",
             f"~{compact(questions['bytes'] / cpt)} tokens")
        line("from articles", f"{questions['sources']:,}", cover)
        if q_failed:
            line("failed", f"{q_failed:,}", "articles")
        line("types", "", hist(questions["type"]))
        line("continuity", "", hist(questions["continuity"]))
        line("split", "", hist(questions["split"]))
    if answers:
        a_failed, _, _ = read_ledger(SFT_LEDGER, key="q")
        pct = answers["records"] / questions["records"] * 100 if questions else 0.0
        line("answered", f"{answers['records']:,}",
             f"{pct:.1f}% of questions, ~{compact(answers['bytes'] / cpt)} tokens")
        line("from articles", f"{answers['sources']:,}")
        if a_failed:
            line("failed", f"{a_failed:,}", "questions")


def print_inventory(rows: list[tuple[str, int, int]], cpt: float) -> None:
    """Every document in corpus/, grouped by source."""
    width = max((len(n) for n, _, _ in rows), default=6) + 2
    print("corpus")
    print(f"  {'source':<{width}}{'docs':>9}{'size':>12}{'tokens':>10}")
    for name, count, size in rows:
        tag = f"{name} *" if name in DERIVED else name
        print(f"  {tag:<{width}}{count:>9,}{human(size):>12}{'~' + compact(size / cpt):>10}")

    src = [r for r in rows if r[0] not in DERIVED]
    print(f"  {'-' * (width + 31)}")
    print(f"  {'total':<{width}}{sum(r[1] for r in src):>9,}"
          f"{human(sum(r[2] for r in src)):>12}"
          f"{'~' + compact(sum(r[2] for r in src) / cpt):>10}")
    if len(src) != len(rows):
        print(f"  {'':<{width}}* rewrite of a source above, so not in the total")


def report() -> None:
    cpt = load_pc().CHARS_PER_TOKEN
    rows = inventory()
    by_name = {name: (count, size) for name, count, size in rows}
    sources, source_bytes = by_name.get(SOURCE_DIR.name, (0, 0))
    done, written = by_name.get(OUTPUT_DIR.name, (0, 0))
    failed, ignored, reasons = read_ledger()

    print_inventory(rows, cpt)
    print()

    todo, skipped = outstanding()
    denom = done + todo
    pct = done / denom * 100 if denom else 0.0
    # 169,471 of 169,506 rounds to 100.0%, which reads as finished when 35
    # articles are still to go. Only an empty todo prints 100%.
    if todo:
        pct = min(pct, 99.9)

    bar_w = 34
    filled = int(bar_w * pct / 100)
    print("paraphrase")
    print(f"  [{'#' * filled}{'.' * (bar_w - filled)}] {pct:5.1f}%")
    print(f"  paraphrased      {done:>9,}")
    print(f"  paraphrasable    {denom:>9,}  of {sources:,} source articles")
    print(f"  remaining        {todo:>9,}")
    if skipped:
        # One reason is the common case, and "index list 1,934" next to the
        # same 1,934 in the number column reads as two different figures.
        order = sorted(skipped.items(), key=lambda kv: -kv[1])
        why = (order[0][0] if len(order) == 1
               else ", ".join(f"{k} {v:,}" for k, v in order))
        print(f"  {'not sent':<15}  {sum(skipped.values()):>9,}  "
              f"{why} -- skipped by design")
    print(f"  failed           {failed:>9,}")
    if ignored:
        print(f"  ignored (402/401){ignored:>9,}  account faults, will be retried")
    print(f"  output size      {human(written):>9}")
    print(f"  source tokens    {'~' + compact(source_bytes / cpt):>9}")
    print(f"  paraphr. tokens  {'~' + compact(written / cpt):>9}")
    if reasons:
        print(f"  {'top failures':<15}  {'':>9}  {hist(reasons, 3)}")

    print()
    print_qa(cpt, sources)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", type=float, metavar="SECONDS",
                    help="refresh continuously until interrupted")
    args = ap.parse_args()

    if not SOURCE_DIR.exists():
        sys.exit(f"No corpus at {SOURCE_DIR}")

    if not args.watch:
        report()
        return

    prev, prev_t = None, time.time()
    try:
        while True:
            print("\033[2J\033[H", end="")  # clear, home
            report()
            now = len(list(OUTPUT_DIR.rglob("*.md"))) if OUTPUT_DIR.exists() else 0
            t = time.time()
            if prev is not None and t > prev_t:
                rate = (now - prev) / (t - prev_t)
                print(f"  rate             {rate * 60:>9,.1f} art/min")
            prev, prev_t = now, t
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
