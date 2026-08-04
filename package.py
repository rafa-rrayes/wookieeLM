#!/usr/bin/env python3
"""Package the corpus into Parquet shards for distribution.

The pipeline leaves its output as 341,000 loose files under ``corpus/`` plus two
JSONL files. That layout is right for building -- a paraphrase run resumes by
asking whether a path exists -- and wrong for everything downstream: a
dataloader pays a syscall per document, and nobody can ship a directory tree.

This script converts it, once, into the artifact people actually consume::

    dist/
      pretrain/wookieepedia-00000-of-000NN.parquet
      pretrain/wookieepedia_paraphrased1-*.parquet
      pretrain/wikipedia-*.parquet
      sft/{train,eval}-*.parquet
      questions/{train,eval}-*.parquet
      README.md         Hugging Face dataset card
      DATASHEET.md      provenance, build inputs, and what is still missing
      SHA256SUMS

Usage:
    uv run package.py                       # build dist/ from corpus/
    uv run package.py --dry-run             # print the plan, write nothing
    uv run package.py --limit 500           # 500 docs per source, for a smoke test
    uv run package.py --sources wikipedia   # one source
    uv run package.py --include-restricted  # add books/scripts/subtitles (see below)

Restricted sources
------------------
``corpus/books``, ``corpus/scripts`` and ``corpus/subtitles`` are copyrighted
novels, screenplays and subtitle rips. They are excluded by default and are not
redistributable. ``--include-restricted`` writes them to ``dist/restricted/``
for local training only, and stamps the datasheet accordingly; ``upload_hf.py``
refuses to upload a tree that contains it.

Determinism
-----------
Documents are enumerated in sorted path order and sharded by a fixed byte
budget, so the same corpus produces byte-identical shards and a stable
``SHA256SUMS``. Nothing under ``corpus/`` is read more than once or written at
all.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
CORPUS_DIR = REPO_ROOT / "corpus"
QUESTION_FILE = REPO_ROOT / "questions" / "questions.jsonl"
SFT_FILE = REPO_ROOT / "sft" / "sft.jsonl"
DIST_DIR = REPO_ROOT / "dist"

# Files that sit beside the text but are not text. Same list count.py uses --
# counting them would put download manifests into the training data.
NON_DOCUMENTS = {"manifest.md", "manifest.jsonl",
                 "articles.txt", "articles.jsonl", "categories.txt"}

# zstd at this level is ~3.5x on Markdown and still decompresses faster than
# the disk it came off. Level 9+ buys under 3% for several times the CPU.
COMPRESSION = "zstd"
COMPRESSION_LEVEL = 6

# Only appears in the dataset card's load_dataset() snippet, so it is a label,
# not a destination -- upload_hf.py takes the real one on the command line.
DEFAULT_HF_REPO = "rafa-rrayes/wookieelm-corpus"
GITHUB_REPO = "rafa-rrayes/wookieeLM"


# ---- Sources ----------------------------------------------------------------

@dataclass(frozen=True)
class Source:
    """One corpus directory and everything the datasheet has to say about it."""
    name: str
    suffixes: tuple[str, ...]
    license: str
    attribution: str
    restricted: bool = False
    # Set when this source is a rewrite of another, so a consumer can hold the
    # two apart (or pair them up) without parsing paths.
    derived_from: str | None = None

    @property
    def directory(self) -> Path:
        return CORPUS_DIR / self.name


SOURCES = [
    Source("wookieepedia", (".md",), "CC BY-SA 3.0",
           "Wookieepedia contributors (starwars.fandom.com)"),
    Source("wookieepedia_paraphrased1", (".md",), "CC BY-SA 3.0",
           "Wookieepedia contributors (starwars.fandom.com), paraphrased",
           derived_from="wookieepedia"),
    Source("wikipedia", (".md",), "CC BY-SA 4.0",
           "Wikipedia contributors (en.wikipedia.org)"),
    Source("books", (".txt",), "All rights reserved", "various publishers",
           restricted=True),
    Source("scripts", (".txt", ".md"), "All rights reserved",
           "Lucasfilm Ltd. / various", restricted=True),
    Source("subtitles", (".txt",), "All rights reserved",
           "various rights holders", restricted=True),
]
SOURCES_BY_NAME = {s.name: s for s in SOURCES}


# ---- Reused machinery -------------------------------------------------------

_PC = None


def load_pc():
    """Import paraphrase_corpus.py by path, once per process.

    Same trick count.py uses. split_document() is the only correct parser for
    this corpus -- it knows that the infobox ends at the lead paragraph rather
    than the next heading -- and reimplementing it here is how the two would
    drift apart.
    """
    global _PC
    if _PC is None:
        spec = importlib.util.spec_from_file_location(
            "pc", REPO_ROOT / "paraphrase_corpus.py")
        _PC = importlib.util.module_from_spec(spec)
        sys.modules["pc"] = _PC
        spec.loader.exec_module(_PC)
    return _PC


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


# ---- Frontmatter ------------------------------------------------------------

# The frontmatter this corpus writes is a fixed shape -- scalars as
# `key: "value"` and one list form for categories -- so it is parsed directly
# rather than pulling in a YAML dependency for six keys.
_SCALAR_RE = re.compile(r'^([a-z_]+):[ \t]*"?(.*?)"?[ \t]*$')
_ITEM_RE = re.compile(r'^[ \t]+-[ \t]*"?(.*?)"?[ \t]*$')


def parse_frontmatter(block: str) -> dict[str, object]:
    """`---\\ntitle: "X"\\ncategories:\\n  - "Y"\\n---\\n` -> {"title": "X", ...}."""
    meta: dict[str, object] = {}
    key = None
    for line in block.splitlines():
        if line.strip() in ("---", ""):
            continue
        if m := _ITEM_RE.match(line):
            if key:
                meta.setdefault(key, [])
                if isinstance(meta[key], list):
                    meta[key].append(m.group(1))
            continue
        if m := _SCALAR_RE.match(line):
            key, value = m.group(1), m.group(2)
            # `categories:` with nothing after it opens a list block.
            meta[key] = value if value else []
    return meta


def wookieepedia_url(title: str) -> str:
    """CC BY-SA wants a link back, and the title is enough to build one."""
    return "https://starwars.fandom.com/wiki/" + title.replace(" ", "_")


# ---- Document -> row --------------------------------------------------------

PRETRAIN_SCHEMA = pa.schema([
    ("text", pa.string()),
    ("doc_id", pa.string()),
    ("title", pa.string()),
    ("continuity", pa.string()),
    ("source", pa.string()),
    ("url", pa.string()),
    ("categories", pa.list_(pa.string())),
    ("derived_from", pa.string()),
    ("license", pa.string()),
])

CONTINUITY_LABEL = {
    "canon": "Canon",
    "legends": "Legends",
    "non-canon": "Non-canon",
    "real-world": "Real-world",
}


def build_text(title: str, continuity: str | None, source: Source,
               infobox: str, body: str) -> str:
    """Assemble the string a model actually trains on.

    The header is not decoration. Canon and Legends contradict each other
    constantly, and a model can only learn which timeline a fact belongs to if
    the label is in the token stream -- a Parquet column it never sees cannot
    teach it anything. The frontmatter is kept as columns *as well*, for
    filtering.
    """
    parts = [f"# {title}\n"]
    origin = "Wookieepedia" if source.name.startswith("wookieepedia") else \
        "Wikipedia" if source.name == "wikipedia" else source.name
    if continuity and (label := CONTINUITY_LABEL.get(continuity)):
        parts.append(f"\n*{origin} · {label}*\n")
    else:
        parts.append(f"\n*{origin}*\n")
    if infobox:
        parts.append("\n" + infobox.rstrip() + "\n")
    if body:
        parts.append("\n" + body.strip() + "\n")
    return "".join(parts)


def document_row(path: Path, source: Source) -> dict | None:
    """One Markdown or text file -> one pretrain row, or None if it is empty."""
    pc = load_pc()
    raw = path.read_text(encoding="utf-8", errors="replace")
    doc_id = f"{source.name}/{path.relative_to(source.directory).as_posix()}"

    if path.suffix == ".md":
        doc = pc.split_document(path, raw)
        meta = parse_frontmatter(doc.frontmatter)
        title = str(meta.get("title") or path.stem.replace("_", " "))
        infobox, body = doc.infobox, doc.body
    else:
        # Plain text -- books, scripts, subtitles. No frontmatter to mine.
        meta, title, infobox, body = {}, path.stem.replace("_", " "), "", raw

    if not body.strip() and not infobox.strip():
        return None

    continuity = meta.get("continuity")
    categories = meta.get("categories") or []
    url = str(meta.get("url") or "")
    if not url and source.name.startswith("wookieepedia"):
        url = wookieepedia_url(title)

    # The paraphraser records its input path; fall back to the mirrored path,
    # which is how the two trees line up by construction.
    derived = meta.get("paraphrased_from")
    if not derived and source.derived_from:
        derived = f"{source.derived_from}/{path.relative_to(source.directory).as_posix()}"

    return {
        "text": build_text(title, continuity if isinstance(continuity, str) else None,
                           source, infobox, body),
        "doc_id": doc_id,
        "title": title,
        "continuity": continuity if isinstance(continuity, str) else None,
        "source": source.name,
        "url": url or None,
        "categories": [c for c in categories if isinstance(c, str)],
        "derived_from": str(derived) if derived else None,
        "license": source.license,
    }


def enumerate_documents(source: Source, limit: int | None = None) -> list[Path]:
    """Sorted, so shard boundaries -- and therefore checksums -- are stable."""
    if not source.directory.exists():
        return []
    # Do not filter on a leading dot. `.48-caliber_Enforcer_pistol.md` is a real
    # article, and the suffix test already excludes .DS_Store and friends -- a
    # dotfile guard here silently drops training data instead.
    paths = sorted(
        p for p in source.directory.rglob("*")
        if p.is_file() and p.suffix.lower() in source.suffixes
        and p.name not in NON_DOCUMENTS
    )
    return paths[:limit] if limit else paths


# ---- Q&A rows ---------------------------------------------------------------

# wookiee_chat.SYSTEM_PROMPT with the tool scaffolding removed. The teacher's
# tools were how it was kept honest, not part of what is being taught: a
# student trained on this corpus has no search_titles to call, so a prompt that
# told it to would teach it to hallucinate tool calls. What survives is the
# part that describes the answer -- continuity discipline, and refusing to
# invent a detail that was never established.
STUDENT_SYSTEM_PROMPT = """\
You are the definitive Star Wars expert. You have spent a lifetime with this \
material and you simply know it -- every character, world, ship, battle, and \
date, across both continuities.

Every subject is canon (the current timeline), legends (the pre-2014 Expanded \
Universe), non-canon, or real-world (production and publishing). Say which one \
an answer belongs to whenever it could matter, and never blend canon and \
Legends into one account without flagging it -- when both exist and disagree, \
give canon first, then Legends.

Answer in prose, as concisely as the question allows, with the easy authority \
of someone recalling something they have known for years. If a detail was \
never established, say so plainly rather than inventing one.\
"""

SFT_SCHEMA = pa.schema([
    ("messages", pa.list_(pa.struct([("role", pa.string()),
                                     ("content", pa.string())]))),
    ("q", pa.string()),
    ("a", pa.string()),
    ("type", pa.string()),
    ("continuity", pa.string()),
    ("source_article", pa.string()),
    ("read", pa.list_(pa.string())),
    ("tools", pa.int32()),
    ("teacher_model", pa.string()),
    ("unsupported", pa.list_(pa.string())),
])

QUESTION_SCHEMA = pa.schema([
    ("q", pa.string()),
    ("type", pa.string()),
    ("continuity", pa.string()),
    ("source_article", pa.string()),
    ("evidence", pa.string()),
])


def sft_row(rec: dict) -> dict:
    return {
        "messages": [{"role": "system", "content": STUDENT_SYSTEM_PROMPT},
                     {"role": "user", "content": rec["q"]},
                     {"role": "assistant", "content": rec["a"]}],
        "q": rec["q"],
        "a": rec["a"],
        "type": rec.get("type"),
        "continuity": rec.get("continuity"),
        "source_article": rec.get("source"),
        "read": rec.get("read") or [],
        "tools": int(rec.get("tools") or 0),
        "teacher_model": rec.get("model"),
        # Kept, not dropped. The flag marks an answer opening on a bare
        # deictic ("That was...") more often than a hallucination, and 26% of
        # an already-small set is too much to discard sight unseen. One column
        # lets a consumer filter in a line; a dropped row cannot be recovered.
        "unsupported": rec.get("unsupported") or [],
    }


def question_row(rec: dict) -> dict:
    return {
        "q": rec["q"],
        "type": rec.get("type"),
        "continuity": rec.get("continuity"),
        "source_article": rec.get("source"),
        "evidence": rec.get("evidence"),
    }


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


# ---- Shard writing ----------------------------------------------------------

@dataclass
class ShardStats:
    name: str
    rows: int = 0
    files: int = 0
    raw_bytes: int = 0
    disk_bytes: int = 0
    # Rows carrying a non-empty `unsupported` list, for the SFT stage only.
    # Counted here so the datasheet quotes the build it describes rather than a
    # figure someone measured once and pasted in.
    flagged: int = 0


class ShardWriter:
    """Buffers rows to a byte budget, then flushes one Parquet file.

    Sharding on *uncompressed* size rather than row count keeps files
    predictable across sources whose documents differ by an order of magnitude
    (a subtitle transcript against a 60 kB Wookieepedia biography).
    """

    def __init__(self, out_dir: Path, name: str, schema: pa.Schema,
                 shard_bytes: int, dry_run: bool = False):
        self.out_dir = out_dir
        self.name = name
        self.schema = schema
        self.shard_bytes = shard_bytes
        self.dry_run = dry_run
        self.buffer: list[dict] = []
        self.buffered_bytes = 0
        self.paths: list[Path] = []
        self.stats = ShardStats(name)

    def add(self, row: dict) -> None:
        self.buffer.append(row)
        self.buffered_bytes += _row_bytes(row)
        self.stats.rows += 1
        self.stats.raw_bytes += _row_bytes(row)
        if self.buffered_bytes >= self.shard_bytes:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        index = len(self.paths)
        path = self.out_dir / f"{self.name}-{index:05d}.parquet"
        if not self.dry_run:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pylist(self.buffer, schema=self.schema)
            pq.write_table(table, path, compression=COMPRESSION,
                           compression_level=COMPRESSION_LEVEL)
            self.stats.disk_bytes += path.stat().st_size
        self.paths.append(path)
        self.stats.files += 1
        self.buffer.clear()
        self.buffered_bytes = 0

    def finalize(self) -> list[Path]:
        """Flush, then rename into Hugging Face's `-00000-of-00003` convention.

        The total is only known once the last row is in, which is why this is a
        rename rather than the name written the first time.
        """
        self.flush()
        total = len(self.paths)
        renamed = []
        for i, path in enumerate(self.paths):
            final = self.out_dir / f"{self.name}-{i:05d}-of-{total:05d}.parquet"
            if not self.dry_run and path.exists():
                path.replace(final)
            renamed.append(final)
        self.paths = renamed
        return renamed


def _row_bytes(row: dict) -> int:
    """Uncompressed size of the row's text, which dominates every other field."""
    if "text" in row:
        return len(row["text"].encode("utf-8"))
    return len(json.dumps(row, ensure_ascii=False).encode("utf-8"))


# ---- Build stages -----------------------------------------------------------

def build_pretrain(source: Source, out_dir: Path, shard_bytes: int,
                   limit: int | None, dry_run: bool) -> ShardStats:
    paths = enumerate_documents(source, limit)
    writer = ShardWriter(out_dir, source.name, PRETRAIN_SCHEMA, shard_bytes, dry_run)
    skipped = 0
    for path in tqdm(paths, desc=f"  {source.name}", unit="doc",
                     leave=False, disable=not sys.stderr.isatty()):
        row = document_row(path, source)
        if row is None:
            skipped += 1
            continue
        writer.add(row)
    writer.finalize()
    if skipped:
        print(f"    {skipped:,} empty document(s) skipped")
    return writer.stats


def build_jsonl_stage(records: list[dict], out_dir: Path, schema: pa.Schema,
                      to_row, shard_bytes: int, limit: int | None,
                      dry_run: bool) -> dict[str, ShardStats]:
    """Split records by their `split` field, one ShardWriter per split."""
    by_split: dict[str, list[dict]] = {}
    for rec in records:
        by_split.setdefault(rec.get("split") or "train", []).append(rec)

    stats = {}
    for split in sorted(by_split):
        rows = by_split[split][:limit] if limit else by_split[split]
        writer = ShardWriter(out_dir, split, schema, shard_bytes, dry_run)
        for rec in rows:
            row = to_row(rec)
            if row.get("unsupported"):
                writer.stats.flagged += 1
            writer.add(row)
        writer.finalize()
        stats[split] = writer.stats
    return stats


# ---- Reporting --------------------------------------------------------------

def print_table(rows: list[tuple[str, str, str, str]]) -> None:
    headers = ("artifact", "rows", "files", "size")
    widths = [max(len(str(r[i])) for r in [headers, *rows]) for i in range(4)]
    print(f"  {headers[0]:<{widths[0]}}  {headers[1]:>{widths[1]}}  "
          f"{headers[2]:>{widths[2]}}  {headers[3]:>{widths[3]}}")
    print(f"  {'-' * (sum(widths) + 6)}")
    for r in rows:
        print(f"  {r[0]:<{widths[0]}}  {r[1]:>{widths[1]}}  "
              f"{r[2]:>{widths[2]}}  {r[3]:>{widths[3]}}")


def write_checksums(dist: Path) -> Path:
    """SHA256 over every shipped file, in sorted path order."""
    lines = []
    for path in sorted(dist.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(dist).as_posix()}")
    out = dist / "SHA256SUMS"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


# ---- Dataset card + datasheet -----------------------------------------------

def dataset_card(stats: dict, sources: list[Source], restricted: bool,
                 hf_repo: str) -> str:
    configs = []
    for source in sources:
        if source.restricted:
            continue
        configs.append(
            f"  - config_name: {source.name}\n"
            f"    data_files:\n"
            f"      - split: train\n"
            f"        path: pretrain/{source.name}-*.parquet")
    if stats.get("sft"):
        configs.append(
            "  - config_name: sft\n    data_files:\n"
            + "\n".join(f"      - split: {s}\n        path: sft/{s}-*.parquet"
                        for s in sorted(stats["sft"])))
    if stats.get("questions"):
        configs.append(
            "  - config_name: questions\n    data_files:\n"
            + "\n".join(f"      - split: {s}\n        path: questions/{s}-*.parquet"
                        for s in sorted(stats["questions"])))

    total_rows = sum(s.rows for s in stats.get("pretrain", {}).values())

    # Derived, never hardcoded: this card is regenerated on every build and the
    # Q&A stages are still running, so a literal percentage here goes stale the
    # moment answer_questions.py writes another row.
    card_q = sum(s.rows for s in stats.get("questions", {}).values())
    card_a = sum(s.rows for s in stats.get("sft", {}).values())
    answered = (f"Only {card_a / card_q * 100:.1f}% of the generated questions "
                "have answers" if card_q and card_a
                else "Most generated questions have no answer yet")

    warning = ""
    if restricted:
        warning = (
            "\n> **This build contains `restricted/` and MUST NOT be uploaded.**\n"
            "> It was produced with `--include-restricted`, which adds "
            "copyrighted\n> novels, screenplays and subtitles. Rebuild without "
            "that flag to publish.\n")

    return f"""---
license: cc-by-sa-4.0
language:
  - en
pretty_name: WookieeLM Star Wars Corpus
size_categories:
  - 100K<n<1M
task_categories:
  - text-generation
  - question-answering
tags:
  - star-wars
  - wookieepedia
  - synthetic
  - knowledge-distillation
configs:
{chr(10).join(configs)}
---

# WookieeLM Star Wars Corpus
{warning}
A Star Wars training corpus built from Wookieepedia, with a paraphrase pass for
knowledge augmentation and a synthetic QA/SFT set distilled from a tool-using
teacher. Built by [`{GITHUB_REPO}`](https://github.com/{GITHUB_REPO}); every
artifact here is reproducible from that pipeline.

## Configs

| Config | Rows | What it is |
|---|---:|---|
{chr(10).join(f"| `{name}` | {s.rows:,} | pretraining documents |"
              for name, s in sorted(stats.get('pretrain', {}).items()))}
{chr(10).join(f"| `sft` ({split}) | {s.rows:,} | chat-formatted QA pairs |"
              for split, s in sorted(stats.get('sft', {}).items()))}
{chr(10).join(f"| `questions` ({split}) | {s.rows:,} | questions + evidence spans |"
              for split, s in sorted(stats.get('questions', {}).items()))}

```python
from datasets import load_dataset

docs = load_dataset("{hf_repo}", "wookieepedia", split="train", streaming=True)
sft  = load_dataset("{hf_repo}", "sft", split="train")
```

## Fields

**Pretraining** — `text`, `doc_id`, `title`, `continuity`, `source`, `url`,
`categories`, `derived_from`, `license`.

`text` carries a header (`# Title` and `*Wookieepedia · Legends*`) before the
body. That is deliberate: canon and Legends contradict each other throughout,
and a model can only learn which timeline a fact belongs to if the label is in
the token stream. The same values are available as columns for filtering.

**SFT** — `messages` (system/user/assistant), plus `q`, `a`, `type`,
`continuity`, `source_article`, `read`, `tools`, `teacher_model`,
`unsupported`.

**Questions** — `q`, `type`, `continuity`, `source_article`, `evidence`. This is
the whole generated pool, including the minority that `sft` already has answers
for; join on `q` to get the rest. `evidence` is the span of the source article
that answers the question, which is what makes the unanswered ones useful on
their own — as a retrieval or grounding set, or as the input to another
answering run.

## Known limitations

Read `DATASHEET.md` in this repository. In short:

- {answered}.
  The rest ship unanswered, with their evidence spans.
- Answers are model-generated and unaudited by a human.
- Index-list articles have no paraphrase — the paraphraser skips them on
  purpose, because rewording a list of links produces noise, not variety.
- Eval questions come from articles that are themselves in the pretraining
  corpus. Eval measures recall of trained-on material, not generalization.

## Licensing

Wookieepedia text is CC BY-SA 3.0; Wikipedia text is CC BY-SA 4.0. Both require
attribution and share-alike, and **that includes the paraphrases**, which are
derivative works. Each row carries its own `license` and `url` for attribution.
Copyrighted novels, screenplays and subtitles used elsewhere in the pipeline are
not included here.

Generated on {date.today().isoformat()}.
"""


def datasheet(stats: dict, sources: list[Source], restricted: bool,
              args: argparse.Namespace) -> str:
    pre = stats.get("pretrain", {})
    lines = [
        "# Datasheet",
        "",
        f"Generated {date.today().isoformat()} by `package.py` from the "
        f"[`{GITHUB_REPO}`](https://github.com/{GITHUB_REPO}) pipeline.",
        "",
    ]
    if restricted:
        lines += ["> **NOT FOR REDISTRIBUTION.** This build was produced with "
                  "`--include-restricted`\n> and contains copyrighted material "
                  "under `restricted/`.", ""]
    if args.limit:
        lines += [f"> **Partial build.** `--limit {args.limit}` was set, so every "
                  "count below describes\n> a truncated sample, not the corpus.", ""]

    lines += ["## Contents", "",
              "| Artifact | Rows | Files | On disk | License |",
              "|---|---:|---:|---:|---|"]
    for name, s in sorted(pre.items()):
        src = SOURCES_BY_NAME.get(name)
        lines.append(f"| `pretrain/{name}` | {s.rows:,} | {s.files} | "
                     f"{human(s.disk_bytes)} | {src.license if src else '?'} |")
    for stage in ("sft", "questions"):
        for split, s in sorted(stats.get(stage, {}).items()):
            lines.append(f"| `{stage}/{split}` | {s.rows:,} | {s.files} | "
                         f"{human(s.disk_bytes)} | CC BY-SA 3.0 (derived) |")

    lines += [
        "",
        "## How it was built",
        "",
        "1. `download_wookieepedia.py` — Wookieepedia MediaWiki XML dump → "
        "Markdown, one file per article, continuity stamped into frontmatter.",
        "2. `paraphrase_corpus.py` — every article rewritten by DeepSeek V4 "
        "Flash. Infoboxes and tables are preserved verbatim; only prose is "
        "rewritten.",
        "3. `generate_questions.py` — questions mined per article, each with "
        "the evidence span that answers it. Train/eval split by *article*, not "
        "by question.",
        "4. `answer_questions.py` — questions answered by a tool-using teacher "
        "(`wookiee_chat.py`) that may only read the corpus, never its own "
        "memory. The tool trace is discarded; the answer is kept.",
        "5. `package.py` — this file.",
        "",
        "## Known gaps",
        "",
        "These are real and are stated here rather than left for someone to "
        "discover:",
        "",
    ]

    para = pre.get("wookieepedia_paraphrased1")
    base = pre.get("wookieepedia")
    if para and base and (missing := base.rows - para.rows) > 0:
        # Deliberately not phrased as "N% complete". Most of this gap is index
        # lists and table-only articles that paraphrase_corpus.py refuses to
        # send -- rewording a list of links produces noise, not variety -- so a
        # completion percentage here would report a design decision as a
        # shortfall. `uv run count.py` splits the two apart.
        lines.append(
            f"- **{missing:,} of {base.rows:,} articles have no paraphrase.** "
            "Most are index lists and table-only articles that "
            "`paraphrase_corpus.py` skips by design; any remainder is work "
            "still to do. `uv run count.py` separates the two.")

    # The `questions` config is the whole generated pool, answered ones
    # included -- answer_questions.py reads questions.jsonl and never removes
    # from it. So it is the denominator, not a disjoint remainder to add.
    q_rows = sum(s.rows for s in stats.get("questions", {}).values())
    a_rows = sum(s.rows for s in stats.get("sft", {}).values())
    if q_rows and a_rows:
        lines.append(
            f"- **The SFT set is small**: {a_rows:,} of {q_rows:,} generated "
            f"questions have answers ({a_rows / q_rows * 100:.1f}%). The other "
            f"~{compact(q_rows - a_rows)} ship unanswered in the `questions` "
            "config; answering them is the obvious contribution.")
    lines.append(
        "- **Answers are unaudited.** Every answer was written by DeepSeek V4 "
        "Flash reading the corpus through tools, and gated by the automatic "
        "checks in `answer_questions.py` (persona leaks, unsupported spans). No "
        "human read them.")

    flagged = sum(s.flagged for s in stats.get("sft", {}).values())
    if a_rows:
        lines.append(
            f"- **`unsupported` is set on {flagged:,} of {a_rows:,} SFT rows "
            f"({flagged / a_rows * 100:.0f}%).** The check flags an answer "
            "opening on a bare deictic as often as it flags a real fabrication. "
            "Rows are kept so consumers can decide; filter on the column if you "
            "want the strict subset.")
    lines += [
        "- **Eval is not decontaminated, by design.** Train and eval questions "
        "come from disjoint sets of source articles, but every eval article is "
        "itself in the pretraining corpus. Eval therefore measures recall of "
        "material the model was trained on, not generalization to unseen "
        "material. Use it as a knowledge probe, not a held-out benchmark.",
        "- **Wookieepedia is a wiki.** It contains errors, vandalism reverts, "
        "and uneven coverage. Nothing here fact-checks it.",
        "",
        "## Reproducing",
        "",
        "```",
        f"uv run package.py --shard-mb {args.shard_mb}"
        + (f" --limit {args.limit}" if args.limit else "")
        + (" --include-restricted" if restricted else ""),
        "```",
        "",
        "Documents are enumerated in sorted path order, so the same corpus "
        "yields byte-identical shards. Verify with `shasum -c SHA256SUMS`.",
        "",
    ]
    return "\n".join(lines)


# ---- Main -------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DIST_DIR,
                    help="output directory (default: dist/)")
    ap.add_argument("--sources", type=str, default=None,
                    help="comma-separated source names (default: all permitted)")
    ap.add_argument("--shard-mb", type=int, default=256,
                    help="uncompressed MB per shard (default: 256)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap documents/records per source, for smoke tests")
    ap.add_argument("--hf-repo", type=str, default=DEFAULT_HF_REPO,
                    help=f"repo id used in the dataset card's load snippet "
                         f"(default: {DEFAULT_HF_REPO})")
    ap.add_argument("--include-restricted", action="store_true",
                    help="also package copyrighted sources into dist/restricted/ "
                         "(local use only -- not redistributable)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and write nothing")
    args = ap.parse_args()

    selected = SOURCES
    if args.sources:
        wanted = {s.strip() for s in args.sources.split(",") if s.strip()}
        unknown = wanted - set(SOURCES_BY_NAME)
        if unknown:
            sys.exit(f"unknown source(s): {', '.join(sorted(unknown))}\n"
                     f"available: {', '.join(SOURCES_BY_NAME)}")
        selected = [s for s in SOURCES if s.name in wanted]
    if not args.include_restricted:
        selected = [s for s in selected if not s.restricted]

    shard_bytes = args.shard_mb * 1024 * 1024
    dist = args.out
    restricted_used = any(s.restricted for s in selected)

    if args.dry_run:
        print("dry run -- nothing will be written\n")
    print(f"packaging -> {dist}\n")

    stats: dict[str, dict[str, ShardStats]] = {"pretrain": {}}

    for source in selected:
        if not source.directory.exists():
            print(f"  {source.name}: not present, skipped")
            continue
        out_dir = dist / ("restricted" if source.restricted else "pretrain")
        stats["pretrain"][source.name] = build_pretrain(
            source, out_dir, shard_bytes, args.limit, args.dry_run)

    only_named_sources = args.sources is not None
    if not only_named_sources:
        if sft_records := read_jsonl(SFT_FILE):
            stats["sft"] = build_jsonl_stage(
                sft_records, dist / "sft", SFT_SCHEMA, sft_row,
                shard_bytes, args.limit, args.dry_run)
        if q_records := read_jsonl(QUESTION_FILE):
            stats["questions"] = build_jsonl_stage(
                q_records, dist / "questions", QUESTION_SCHEMA, question_row,
                shard_bytes, args.limit, args.dry_run)

    rows = []
    for stage in ("pretrain", "sft", "questions"):
        for name, s in sorted(stats.get(stage, {}).items()):
            label = name if stage == "pretrain" else f"{stage}/{name}"
            size = human(s.disk_bytes) if not args.dry_run else f"~{human(s.raw_bytes)}"
            rows.append((label, f"{s.rows:,}", str(s.files), size))
    if not rows:
        sys.exit("nothing to package -- is corpus/ present?")

    print()
    print_table(rows)

    total_raw = sum(s.raw_bytes for stage in stats.values() for s in stage.values())
    total_disk = sum(s.disk_bytes for stage in stats.values() for s in stage.values())
    pc = load_pc()
    print(f"\n  {sum(int(r[1].replace(',', '')) for r in rows):,} rows, "
          f"~{compact(total_raw / pc.CHARS_PER_TOKEN)} tokens, "
          f"{human(total_raw)} raw"
          + (f" -> {human(total_disk)} on disk "
             f"({total_raw / total_disk:.1f}x)" if total_disk else ""))

    if args.dry_run:
        print("\ndry run complete -- rerun without --dry-run to write.")
        return

    (dist / "README.md").write_text(
        dataset_card(stats, selected, restricted_used, args.hf_repo),
        encoding="utf-8")
    (dist / "DATASHEET.md").write_text(
        datasheet(stats, selected, restricted_used, args), encoding="utf-8")
    sums = write_checksums(dist)
    print(f"\n  wrote README.md, DATASHEET.md, "
          f"{sums.name} ({len(sums.read_text().splitlines())} files)")

    if restricted_used:
        print("\n  ** dist/restricted/ contains copyrighted material. "
              "Do not upload this build. **")


if __name__ == "__main__":
    main()
