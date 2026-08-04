#!/usr/bin/env python3
"""Paraphrase the Wookieepedia Markdown corpus with DeepSeek.

Reads every ``corpus/wookieepedia/<L>/<Title>.md`` produced by
``download_wookieepedia.py`` and writes a reworded twin to
``corpus/wookieepedia_paraphrased1/<L>/<Title>.md``.

What is and isn't rewritten:

    frontmatter   preserved verbatim, plus provenance fields recording the
                  source path and the model used (Wookieepedia is CC-BY-SA,
                  so attribution has to survive into derived text)
    ## Infobox    the key/value bullet list only, preserved verbatim by default
                  -- rewording it produces noise, not variety
                  (--paraphrase-infobox to override)
    tables        preserved verbatim wherever they appear, for the same reason
                  as the infobox: a filmography or release grid is proper nouns
                  and dates with no sentence structure to vary. Recorded as
                  paraphrase_verbatim_table_chars in the output's frontmatter
    body          everything else, including the lead paragraph that sits
                  between the infobox and the first heading -- paraphrased,
                  preserving every fact, name, date and number

Thinking is disabled by default. DeepSeek's V4 models reason before answering
unless told not to, and those tokens bill as output; on this workload they were
~82% of output tokens and ~70% of total spend, for a task whose constraints are
already fully specified in the system prompt. Pass --thinking to re-enable.

Long articles are split on ``## `` headings (then on paragraphs if a section is
still oversized), paraphrased chunk by chunk, and rejoined. A chunk that no
attempt can reword degrades to its source text rather than discarding the whole
article, and the output's frontmatter records that it did:

    paraphrase_degraded_chunks: 1
    paraphrase_degraded_reason: "truncated"

so partially-verbatim files stay greppable. An article degrading past
MAX_DEGRADED_FRACTION of its body is failed outright instead -- at that point
the file would be mostly a copy, which is worse than not having it.

Requirements:
    DEEPSEEK_API_KEY in the environment

Usage:
    uv run paraphrase_corpus.py --dry-run          # cost/token estimate only
    uv run paraphrase_corpus.py --limit 50         # quick smoke test
    uv run paraphrase_corpus.py                    # full run, many hours
    uv run paraphrase_corpus.py --force            # redo existing outputs

Resume: rerun the same command. Articles whose output file already exists are
skipped, so an interrupted run picks up where it stopped. Articles that failed
every retry are appended to paraphrase_state/failures.jsonl and skipped on
later runs unless --retry-failed is passed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
CORPUS_DIR = REPO_ROOT / "corpus"
SOURCE_DIR = CORPUS_DIR / "wookieepedia"
OUTPUT_DIR = CORPUS_DIR / "wookieepedia_paraphrased1"
STATE_DIR = REPO_ROOT / "paraphrase_state"

API_BASE = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEFAULT_MODEL = "deepseek-v4-flash"

# $/1M tokens for deepseek-v4-flash, used by --dry-run and the run summary.
# Override with --price-in/--price-out for another model or after a rate change.
# PRICE_IN is the cache-*miss* rate; the shared system-prompt prefix bills at
# PRICE_CACHED, roughly 50x less.
DEFAULT_PRICE_IN = 0.14
DEFAULT_PRICE_OUT = 0.28
DEFAULT_PRICE_CACHED = 0.0028

# Chunking. DeepSeek handles far more than this per call, but smaller chunks
# paraphrase more faithfully and fail more cheaply.
CHUNK_CHARS = 6000
MIN_BODY_CHARS = 120        # below this a "paraphrase" is not worth a request
INDEX_MIN_ITEMS = 20        # bullet/table-row counts that mark an article as
INDEX_LINE_FRACTION = 0.5   # data, not prose -- see is_index_list()
CHARS_PER_TOKEN = 3.6       # empirical for English Markdown w/ many proper nouns

# A paraphrase this far off the source length means truncation, a refusal, or
# the model summarising instead of rewording.
LEN_RATIO_MIN = 0.55
LEN_RATIO_MAX = 2.0

MAX_ATTEMPTS = 5
BACKOFF_BASE = 2.0
BACKOFF_CAP = 60.0

# A chunk that survives no attempt degrades to its source text rather than
# taking the whole article down with it -- but only while it stays a minority of
# the article. Past this share of body characters the "paraphrase" would be
# mostly a copy, which is worse than having no file: it looks finished, and the
# verbatim text would be indistinguishable from a real paraphrase downstream.
MAX_DEGRADED_FRACTION = 0.5

SYSTEM_PROMPT = """\
You reword Star Wars encyclopedia articles for a text corpus.

Rewrite the passage the user gives you so the wording and sentence structure \
differ substantially from the original, while the meaning stays identical.

Hard requirements:
- Preserve every fact. Do not add, remove, speculate about, or embellish any detail.
- Reproduce all proper nouns exactly as written: characters, species, planets, \
ships, organisations, titles. Never translate, shorten, or invent a name.
- Reproduce all numbers, dates and era abbreviations (BBY, ABY) exactly.
- Keep the Markdown structure: same heading levels, lists stay lists, tables \
stay tables, emphasis is preserved where it marks a title.
- Match the source length closely. This is a rewording, not a summary or an \
expansion.
- Keep the neutral, past-tense encyclopedic register of the original.

Output only the rewritten passage. No preamble, no commentary, no code fences."""

# Models sometimes bolt on a lead-in despite the instruction above.
_PREAMBLE_RE = re.compile(
    r"^\s*(?:here(?:'s| is)|sure[,!]?|certainly[,!]?)[^\n]*(?:paraphras|rewrit|reword)[^\n]*\n+",
    re.IGNORECASE,
)
_FENCE_RE = re.compile(r"^\s*```(?:markdown|md)?\s*\n(.*)\n```\s*$", re.DOTALL)
_REFUSAL_RE = re.compile(
    r"^\s*(?:i (?:can'?t|cannot|won'?t|am unable)|as an ai|i'm sorry)", re.IGNORECASE
)


# ---- Document splitting -----------------------------------------------------

@dataclass
class Document:
    """A source article decomposed into the parts we treat differently."""
    path: Path
    frontmatter: str        # including the --- fences, or "" if absent
    infobox: str            # the "## Infobox" section, or ""
    body: str               # everything we actually paraphrase

    @property
    def title(self) -> str:
        m = re.search(r'^title:\s*"(.*)"\s*$', self.frontmatter, re.MULTILINE)
        return m.group(1) if m else self.path.stem.replace("_", " ")


_FRONTMATTER_RE = re.compile(r"\A(---\n.*?\n---\n)", re.DOTALL)

# The infobox is the bullet list render_infobox() emits: top-level "- **Key:** v"
# lines plus indented continuation lines for multi-valued keys. It ends at the
# first line that is neither, which is the article's lead paragraph.
#
# Do NOT extend this to the next "## " heading. Wookieepedia puts the lead
# section between the infobox and the first real heading, so a heading-anchored
# match swallows the lead -- the single highest-value paragraph in the article --
# and copies it into the output verbatim instead of paraphrasing it. Worse, on
# the ~28% of articles whose prose is *only* a lead, that leaves a body below
# MIN_BODY_CHARS and the article is dropped from the run entirely.
_INFOBOX_RE = re.compile(
    r"\A(## Infobox\n"
    r"(?:-[^\n]*\n|[ \t]+[^\n]*\n|[ \t]*\n)*)"
)
_HEADING_RE = re.compile(r"^(#{2,6}) ")
# Every heading in a chunk, not just the one it opens on -- see paraphrase_chunk.
_HEADING_LINE_RE = re.compile(r"^#{2,6} .*$", re.MULTILINE)


def strip_empty_sections(body: str) -> str:
    """Drop headings that have no content under them.

    ~3% of the corpus ends on a bare heading: sections such as "Cover gallery"
    held only images, which the wikitext->Markdown conversion in
    download_wookieepedia.py strips, orphaning the heading. Paraphrasing them
    just launders noise into the training data.

    Runs to a fixed point so a parent heading whose only child was an empty
    subsection is removed too.
    """
    lines = body.split("\n")
    while True:
        drop: set[int] = set()
        for i, line in enumerate(lines):
            m = _HEADING_RE.match(line)
            if not m:
                continue
            level = len(m.group(1))
            # Find the next line with content, skipping blanks and drops.
            nxt = next((j for j in range(i + 1, len(lines))
                        if lines[j].strip() and j not in drop), None)
            if nxt is None:
                drop.add(i)                      # trailing heading, nothing after
            elif (m2 := _HEADING_RE.match(lines[nxt])) and len(m2.group(1)) <= level:
                drop.add(i)                      # next heading is a sibling/parent
        if not drop:
            break
        lines = [l for i, l in enumerate(lines) if i not in drop]
    return "\n".join(lines).strip()


def split_document(path: Path, text: str) -> Document:
    frontmatter = ""
    if m := _FRONTMATTER_RE.match(text):
        frontmatter = m.group(1)
        text = text[m.end():]

    text = text.lstrip("\n")
    infobox = ""
    if m := _INFOBOX_RE.match(text):
        infobox = m.group(1).rstrip() + "\n"
        text = text[m.end():]

    return Document(path=path, frontmatter=frontmatter, infobox=infobox,
                    body=strip_empty_sections(text.strip()))


# A table row, in either of the two forms the corpus uses: pipe Markdown, and
# the raw HTML <table> that pandoc leaves behind for merged or nested cells.
_TABLE_LINE_RE = re.compile(
    r"^\s*(?:\||</?(?:table|thead|tbody|tfoot|tr|th|td|caption)\b)", re.IGNORECASE)


def is_table_block(text: str) -> bool:
    """True for a passage that is table markup rather than prose.

    Tables are preserved verbatim for the reason the infobox is: a filmography
    or release table is a grid of proper nouns, dates and ISBNs with no
    sentence structure to vary, so rewording it can only corrupt it.

    They also cannot be chunked. A table carries no blank line, so the
    paragraph split in chunk_body() cannot break one up, and a 117 KB
    Episode_I_Snapshot table went out as a single request that came back
    truncated -- which is how the ten largest data pages in the corpus each
    failed a whole run.
    """
    lines = [l for l in text.split("\n") if l.strip()]
    return bool(lines) and all(_TABLE_LINE_RE.match(l) for l in lines)


def split_segments(body: str) -> list[str]:
    """Split a body into alternating prose and table passages, in order.

    Each returned segment is either entirely table markup (is_table_block) or
    entirely free of it, so a caller can route the two differently without
    re-splitting.
    """
    segments: list[str] = []
    current: list[str] = []
    in_table = False
    for line in body.split("\n"):
        # A blank line inside a run of table markup belongs to whatever follows
        # it, so a trailing blank does not end the table one line early.
        if not line.strip():
            current.append(line)
            continue
        if bool(_TABLE_LINE_RE.match(line)) != in_table:
            if (text := "\n".join(current).strip("\n")):
                segments.append(text)
            current, in_table = [], not in_table
        current.append(line)
    if (text := "\n".join(current).strip("\n")):
        segments.append(text)
    return segments


def prose_chars(body: str) -> int:
    """Characters of body that are not table markup -- the part worth sending."""
    return sum(len(s) for s in split_segments(body) if not is_table_block(s))


def is_index_list(body: str) -> bool:
    """True for an article that is an index rather than prose.

    Appearance lists, release-year pages and "List of ..." articles are bullets
    of canonical names with no connecting text. Paraphrasing one rewrites the
    names themselves -- measured on Star_Wars_Galaxies_Appearances: "Deathsting"
    -> "Deathstings", "Baz nitch" -> "Baz nitches", "Asteroid field" ->
    "Asteroid belt" -- which corrupts 337 entries and buys nothing, because an
    index has no sentence structure to vary in the first place.

    Bullets inside a prose article are fine and stay: the threshold is on the
    body being *mostly* bullets. ~1% of the corpus.

    Tables are not judged here. They are line-dense -- pandoc's HTML puts each
    cell on its own line -- so a line fraction would call A New Hope an index
    on the strength of one appearances table while 77% of its characters are
    prose. They are preserved passage by passage instead; see is_table_block().
    """
    lines = [l for l in body.split("\n") if l.strip()]
    items = [l for l in lines if l.lstrip().startswith("- ")]
    return len(items) >= INDEX_MIN_ITEMS and len(items) > INDEX_LINE_FRACTION * len(lines)


def skip_reason(doc: Document) -> str | None:
    """Why this article should not be sent, or None to paraphrase it.

    One predicate shared by the worker, --dry-run and count.py, so the three
    cannot drift into disagreeing about what the denominator is.
    """
    if len(doc.body) < MIN_BODY_CHARS:
        return "too short"
    if is_index_list(doc.body):
        return "index list"
    # Tables survive verbatim, so an article that is nothing but tables has
    # nothing left to send: Kevin_Kiner is a filmography with 445 characters of
    # biography, Panini a release grid with 653.
    if prose_chars(doc.body) < MIN_BODY_CHARS:
        return "table only"
    return None


def chunk_body(body: str, limit: int = CHUNK_CHARS) -> list[str]:
    """Split a body into paraphrase-sized pieces on the cleanest boundary.

    Table passages come back as chunks of their own, however big, so a caller
    can pass them through verbatim; every other chunk is prose within `limit`.
    Use is_table_block() to tell the two apart.
    """
    # Tables are cut out first: they bound no paragraph and obey no limit, so
    # leaving them in would let one wedge itself into a prose chunk and blow
    # past every size the splitting below is trying to respect.
    segments = split_segments(body)
    if len(segments) > 1 or (segments and is_table_block(segments[0])):
        chunks: list[str] = []
        for segment in segments:
            if is_table_block(segment):
                chunks.append(segment)
            else:
                chunks.extend(chunk_body(segment, limit))
        return chunks

    if len(body) <= limit:
        return [body] if body.strip() else []

    # Prefer splitting between top-level sections, keeping the heading attached.
    sections = re.split(r"\n(?=## )", body)
    chunks: list[str] = []
    for section in sections:
        if len(section) <= limit:
            if section.strip():
                chunks.append(section)
            continue
        # Still too big: pack paragraphs up to the limit.
        current: list[str] = []
        size = 0
        for para in section.split("\n\n"):
            if size + len(para) > limit and current:
                chunks.append("\n\n".join(current))
                current, size = [], 0
            current.append(para)
            size += len(para) + 2
        if current:
            chunks.append("\n\n".join(current))
    return [c for c in chunks if c.strip()]


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def boundary_context(prev_chunk: str, max_chars: int = 400) -> str:
    """The tail of the preceding *source* chunk, to anchor dangling references.

    A chunk split out of the middle of a long section can open on "This
    alerted..." or "The Hunter then...", whose antecedent lives in the previous
    chunk. Without it the model may resolve the reference by inventing a
    specific antecedent -- a factual error that reads perfectly well.

    Deliberately taken from the previous chunk's SOURCE, never its output, so
    every chunk of an article remains independently dispatchable.
    """
    # Prose only: a heading leaking into the reference block invites the model
    # to echo it back as part of the rewrite.
    prose = "\n".join(l for l in prev_chunk.strip().split("\n")
                      if not l.lstrip().startswith("#"))
    sentences = [s for s in _SENTENCE_RE.split(prose.strip()) if s]
    ctx = ""
    for s in reversed(sentences):
        if ctx and len(ctx) + len(s) + 1 > max_chars:
            break
        ctx = f"{s} {ctx}".strip()
    return ctx[-max_chars:]


def plan_context(chunks: list[str]) -> list[str]:
    """Per-chunk preceding context; empty where the chunk restarts on a heading."""
    contexts = [""]
    for i, chunk in enumerate(chunks[1:], start=1):
        if chunk.lstrip().startswith("#") or is_table_block(chunks[i - 1]):
            # A heading restarts context on its own, and so does a table: the
            # tail of one is grid cells, which anchor no dangling reference and
            # only invite the model to echo the markup back.
            contexts.append("")
        else:
            contexts.append(boundary_context(chunks[i - 1]))
    return contexts


def clean_response(text: str) -> str:
    text = text.strip()
    if m := _FENCE_RE.match(text):
        text = m.group(1).strip()
    text = _PREAMBLE_RE.sub("", text)
    return text.strip()


def restore_heading(source_chunk: str, out: str) -> str:
    """Put back a leading heading the model dropped.

    On ~10% of articles the model answers a chunk that opens on "## Works" or
    "### Filmography" with the list alone, silently demoting a section into the
    one above it. A dropped "## " heading also makes the output re-parse wrong
    on the next pass, since _INFOBOX_RE keys off the section structure.

    The heading is a verbatim-preserved token, not something to reword, so
    restoring it deterministically beats spending another request on a retry.
    """
    src_head = source_chunk.lstrip().split("\n", 1)[0].strip()
    if not _HEADING_RE.match(src_head):
        return out
    first = out.lstrip().split("\n", 1)[0].strip()
    if first == src_head:
        return out
    # A reworded heading at the same level is still a heading -- replace it so
    # the text matches the source exactly; otherwise prepend the missing one.
    if (m := _HEADING_RE.match(first)) and m.group(1) == _HEADING_RE.match(src_head).group(1):
        body = out.lstrip().split("\n", 1)
        return f"{src_head}\n{body[1]}" if len(body) > 1 else src_head
    return f"{src_head}\n\n{out}"


def build_output(doc: Document, body: str, model: str,
                 degraded: list[str] | None = None) -> str:
    """Reassemble a paraphrased article, threading provenance into frontmatter.

    ``paraphrase_degraded_chunks`` marks a file that contains some verbatim
    source text (see paraphrase_document). It is written only when non-zero, so
    a plain grep separates the fully-reworded corpus from the partial ones
    rather than leaving them silently mixed together.

    ``paraphrase_verbatim_table_chars`` does the same for tables, which are
    kept as-is on purpose. The two are separate fields because they mean
    different things -- one is a failure, the other is the design -- but both
    have to be greppable: on Kevin_Kiner the preserved filmography is 41 KB
    against 445 characters of reworded prose, and nothing downstream should
    have to guess that.
    """
    rel = doc.path.relative_to(CORPUS_DIR).as_posix()
    provenance = (
        f'paraphrased_from: "{rel}"\n'
        f'paraphrase_model: "{model}"\n'
    )
    if verbatim := sum(len(s) for s in split_segments(body) if is_table_block(s)):
        provenance += f'paraphrase_verbatim_table_chars: {verbatim}\n'
    if degraded:
        reasons = ", ".join(sorted(set(degraded)))
        provenance += (
            f'paraphrase_degraded_chunks: {len(degraded)}\n'
            f'paraphrase_degraded_reason: "{reasons}"\n'
        )
    if doc.frontmatter:
        frontmatter = doc.frontmatter[: -len("---\n")] + provenance + "---\n"
    else:
        frontmatter = f'---\ntitle: "{doc.title}"\n{provenance}---\n'

    parts = [frontmatter, "\n"]
    if doc.infobox:
        parts += [doc.infobox, "\n"]
    parts.append(body.rstrip() + "\n")
    return "".join(parts)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


# ---- API --------------------------------------------------------------------

class ParaphraseError(RuntimeError):
    """Raised when a chunk could not be paraphrased acceptably."""


class FatalAPIError(RuntimeError):
    """An account- or config-level fault that every article would hit.

    Never recorded per-article: a bad key, an empty balance or an unknown model
    says nothing about the article being processed, and logging it to the
    failure ledger would silently exclude those articles from every later run.
    """


# Statuses that mean "stop the whole run", not "this article is bad".
FATAL_STATUSES = {401, 402, 403, 404}


@dataclass
class Stats:
    done: int = 0
    skipped: int = 0
    failed: int = 0
    requests: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    # Broken out because the totals alone hide the two things that actually
    # drive the bill: reasoning tokens you pay for and throw away, and the
    # shared system-prompt prefix you pay full price for on every cache miss.
    tokens_cached: int = 0
    tokens_reasoning: int = 0
    retries: int = 0
    # Articles written with some verbatim source text in them, and how many
    # chunks that was. Surfaced in the run summary because it is corpus quality
    # debt, not a neutral detail: those passages are not paraphrases.
    degraded_articles: int = 0
    degraded_chunks: int = 0
    fatal: str | None = None
    errors: dict[str, int] = field(default_factory=dict)

    def note_error(self, kind: str) -> None:
        self.errors[kind] = self.errors.get(kind, 0) + 1

    def note_usage(self, usage: dict) -> None:
        self.tokens_in += usage.get("prompt_tokens", 0)
        self.tokens_out += usage.get("completion_tokens", 0)
        self.tokens_cached += usage.get("prompt_cache_hit_tokens", 0)
        details = usage.get("completion_tokens_details") or {}
        self.tokens_reasoning += details.get("reasoning_tokens", 0)


async def paraphrase_chunk(client: httpx.AsyncClient, chunk: str, title: str,
                           model: str, stats: Stats, req_sem: asyncio.Semaphore,
                           context: str = "", thinking: bool = False) -> str:
    """One chunk -> one paraphrase, with retries on transport and quality faults."""
    user = f"Article: {title}\n\n"
    if context:
        user += ("Preceding text, for reference only. Do NOT rewrite it and do "
                 "NOT include it in your output -- it is here only so pronouns "
                 f"and references in the passage have an antecedent:\n\n{context}\n\n")
    user += f"Passage to reword:\n\n{chunk}"

    payload = {
        "model": model,
        "temperature": 1.1,      # variety is the point; facts are pinned by the prompt
        # DeepSeek's V4 models think by default, and reasoning tokens bill as
        # output. Rewording needs no deliberation -- the constraints are all in
        # the system prompt -- so the chain of thought is generated, charged for
        # and discarded. Measured over 2,162 articles it was 82% of output
        # tokens and 71% of total spend. Off unless --thinking is passed.
        "thinking": {"type": "enabled" if thinking else "disabled"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
    }

    last_error = "unknown"
    fallback = None     # a reword that was sound except for a dropped heading
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt > 1:
            stats.retries += 1
        try:
            # Bound actual in-flight requests, not articles: one article may now
            # dispatch all of its chunks at once.
            async with req_sem:
                resp = await client.post("/chat/completions", json=payload)
        except (httpx.TransportError, httpx.TimeoutException) as e:
            last_error = f"transport: {type(e).__name__}"
            await _backoff(attempt)
            continue

        if resp.status_code == 429 or resp.status_code >= 500:
            last_error = f"http {resp.status_code}"
            await _backoff(attempt, resp.headers.get("Retry-After"))
            continue
        if resp.status_code in FATAL_STATUSES:
            # Account/config fault: every other article would fail identically.
            raise FatalAPIError(f"http {resp.status_code}: {resp.text[:200]}")
        if resp.status_code >= 400:
            raise ParaphraseError(f"http {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        stats.requests += 1
        if usage := data.get("usage"):
            stats.note_usage(usage)

        choice = data["choices"][0]
        out = clean_response(choice["message"]["content"] or "")
        if out:
            out = restore_heading(chunk, out)

        if not out:
            last_error = "empty response"
        elif _REFUSAL_RE.match(out):
            last_error = "refusal"
        elif choice.get("finish_reason") == "length":
            last_error = "truncated"
        else:
            ratio = len(out) / max(len(chunk), 1)
            if not (LEN_RATIO_MIN <= ratio <= LEN_RATIO_MAX):
                last_error = f"length ratio {ratio:.2f}"
            elif len(_HEADING_LINE_RE.findall(out)) < len(_HEADING_LINE_RE.findall(chunk)):
                # restore_heading() above covers the heading a chunk opens on,
                # and measured over the 60 largest articles it loses none of
                # them. The ones it cannot see are the headings *inside* a
                # chunk, 5.2% of which come back missing, silently merging a
                # section into the one above it. Re-asking is cheaper than
                # aligning a reworded heading back into paraphrased prose.
                last_error = "dropped heading"
                fallback = out
            else:
                return out

        # Quality fault: nudge temperature down and try again.
        payload["temperature"] = max(0.6, payload["temperature"] - 0.2)
        await _backoff(attempt)

    stats.note_error(last_error)
    if fallback is not None:
        # Every attempt dropped a heading. Keep the text: a section boundary
        # lost is worth far less than the whole article, which is what raising
        # here would cost -- the chunk's siblings are cancelled with it.
        return fallback
    raise ParaphraseError(last_error)


async def _backoff(attempt: int, retry_after: str | None = None) -> None:
    if retry_after:
        try:
            await asyncio.sleep(min(float(retry_after), BACKOFF_CAP))
            return
        except ValueError:
            pass
    delay = min(BACKOFF_BASE ** attempt, BACKOFF_CAP)
    await asyncio.sleep(delay * (0.5 + random.random()))


async def _verbatim(chunk: str) -> str:
    """A passage that is kept as-is, shaped like a paraphrase_chunk() result."""
    return chunk


async def paraphrase_document(client: httpx.AsyncClient, doc: Document,
                              model: str, args, stats: Stats,
                              req_sem: asyncio.Semaphore) -> tuple[str, list[str]]:
    """-> (paraphrased body, reason per chunk that degraded to source text).

    A chunk that exhausts every attempt no longer cancels its siblings. On a
    600 KB article that is ~97 chunks discarded because one of them came back
    truncated, and it is why the largest articles failed at ~8% against 0.02%
    for the corpus overall. The chunk degrades to its source text instead --
    the same trade already made for a dropped heading in paraphrase_chunk().

    Degrading is recorded, never silent: the caller stamps the count into the
    output's frontmatter so verbatim passages stay identifiable downstream.

    Table passages are kept verbatim by design rather than by failure, so they
    are never sent and never counted as degraded; see is_table_block().
    """
    chunks = chunk_body(doc.body, args.chunk_chars)
    sent = [not is_table_block(c) for c in chunks]
    if not any(sent):
        raise ParaphraseError("no paraphrasable body")

    # Chunks share no state, so they all dispatch at once. A 15-chunk article
    # costs one request's latency instead of fifteen; req_sem keeps the total
    # in-flight count at the configured concurrency.
    contexts = plan_context(chunks)
    results = await asyncio.gather(*(
        paraphrase_chunk(client, c, doc.title, model, stats, req_sem, ctx,
                         args.thinking) if send else _verbatim(c)
        for c, ctx, send in zip(chunks, contexts, sent)
    ), return_exceptions=True)

    out_parts: list[str] = []
    degraded: list[str] = []
    degraded_chars = 0
    for chunk, res in zip(chunks, results):
        if isinstance(res, ParaphraseError):
            # Keep the source text for this passage; the rest of the article is
            # still a good paraphrase and is worth far more than this chunk.
            out_parts.append(chunk)
            degraded.append(str(res))
            degraded_chars += len(chunk)
        elif isinstance(res, BaseException):
            # FatalAPIError (account/config) and cancellation must still stop
            # the run -- neither says anything about this article.
            raise res
        else:
            out_parts.append(res)

    if degraded:
        # Measured against what was actually sent, not the whole body: on an
        # article that is half filmography, counting the verbatim-by-design
        # table as denominator would let half the *prose* fail unnoticed.
        share = degraded_chars / max(sum(len(c) for c, s in zip(chunks, sent) if s), 1)
        if share > MAX_DEGRADED_FRACTION:
            raise ParaphraseError(
                f"{len(degraded)}/{sum(sent)} chunks unparaphrasable "
                f"({share:.0%} of prose): {degraded[0]}")

    body = "\n\n".join(out_parts)

    if args.paraphrase_infobox and doc.infobox:
        doc.infobox = await paraphrase_chunk(client, doc.infobox, doc.title,
                                             model, stats, req_sem, "",
                                             args.thinking)
    return body, degraded


# ---- Orchestration ----------------------------------------------------------

def iter_sources(limit: int | None, sample: int | None, seed: int) -> list[Path]:
    """Collect source articles.

    --limit takes the head of the sorted list, which is a poor smoke test: the
    corpus starts with numeric-titled droid stubs that carry an infobox and no
    prose, so a small --limit exercises nothing but the skip path. --sample
    draws evenly across the whole corpus instead, seeded so a run is repeatable.
    """
    paths = sorted(SOURCE_DIR.rglob("*.md"))
    if sample and sample < len(paths):
        return sorted(random.Random(seed).sample(paths, sample))
    return paths[:limit] if limit else paths


_FATAL_LEDGER_RE = re.compile(r"^http (?:%s)\b" % "|".join(map(str, FATAL_STATUSES)))


def load_failures() -> set[str]:
    """Paths to skip because they genuinely failed.

    Entries recording an account-level fault (402 Insufficient Balance, 401,
    403, 404) are ignored rather than skipped. Earlier versions logged those
    per-article, which would have quietly excluded a whole run's worth of
    articles once the underlying problem was fixed.
    """
    ledger = STATE_DIR / "failures.jsonl"
    if not ledger.exists():
        return set()
    failed = set()
    for line in ledger.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
            if _FATAL_LEDGER_RE.match(rec.get("error", "")):
                continue
            failed.add(rec["path"])
        except (json.JSONDecodeError, KeyError):
            continue
    return failed


def output_path_for(src: Path) -> Path:
    return OUTPUT_DIR / src.relative_to(SOURCE_DIR)


async def worker(src: Path, client: httpx.AsyncClient, sem: asyncio.Semaphore,
                 model: str, args, stats: Stats, ledger, bar: tqdm,
                 abort: asyncio.Event, req_sem: asyncio.Semaphore) -> None:
    async with sem:
        if abort.is_set():
            bar.update(1)
            return
        try:
            text = src.read_text(encoding="utf-8")
            doc = split_document(src, text)

            if skip_reason(doc):
                stats.skipped += 1
                return

            body, degraded = await paraphrase_document(client, doc, model, args,
                                                       stats, req_sem)
            atomic_write(output_path_for(src),
                         build_output(doc, body, model, degraded))
            stats.done += 1
            if degraded:
                stats.degraded_articles += 1
                stats.degraded_chunks += len(degraded)
        except FatalAPIError as e:
            # Stop everything; do not blame this article in the ledger.
            if not abort.is_set():
                stats.fatal = str(e)
                abort.set()
        except ParaphraseError as e:
            stats.failed += 1
            ledger.write(json.dumps({
                "path": src.relative_to(SOURCE_DIR).as_posix(),
                "error": str(e),
            }) + "\n")
            ledger.flush()
        except Exception as e:  # noqa: BLE001 - one bad page must not stop the run
            stats.failed += 1
            stats.note_error(type(e).__name__)
            ledger.write(json.dumps({
                "path": src.relative_to(SOURCE_DIR).as_posix(),
                "error": f"{type(e).__name__}: {e}",
            }) + "\n")
            ledger.flush()
        finally:
            bar.update(1)
            bar.set_postfix(ok=stats.done, fail=stats.failed, skip=stats.skipped,
                            refresh=False)


def estimate(paths: list[Path], args) -> None:
    """--dry-run: size the job without spending anything."""
    body_chars = 0
    chunks = 0
    skipped: dict[str, int] = {}
    for p in tqdm(paths, desc="measuring", unit="file"):
        doc = split_document(p, p.read_text(encoding="utf-8"))
        if reason := skip_reason(doc):
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        # Table passages are kept verbatim, so they cost neither a request nor
        # a token and must not be sized as though they did.
        sent = [c for c in chunk_body(doc.body, args.chunk_chars)
                if not is_table_block(c)]
        body_chars += sum(len(c) for c in sent)
        chunks += len(sent)

    body_tokens = body_chars / CHARS_PER_TOKEN
    # The system prompt is an identical prefix on every request, so after the
    # first few it bills at the cache-hit rate rather than the miss rate.
    cached = chunks * len(SYSTEM_PROMPT) / CHARS_PER_TOKEN
    tok_in = body_tokens + cached + chunks * 20
    tok_out = body_tokens          # a faithful reword is ~1:1
    if args.thinking:
        # Measured over 2,162 articles: reasoning ran ~4.6x the answer itself.
        tok_out *= 5.6
    cost = ((tok_in - cached) / 1e6 * args.price_in
            + cached / 1e6 * args.price_cached
            + tok_out / 1e6 * args.price_out)

    print(f"\narticles           {len(paths):,}")
    print(f"  paraphrasable    {len(paths) - sum(skipped.values()):,}")
    for reason, n in sorted(skipped.items()):
        print(f"  - {reason:<15}{n:,}")
    print(f"requests (chunks)  {chunks:,}")
    print(f"input tokens  ~    {tok_in/1e6:.1f}M  (of which {cached/1e6:.1f}M "
          f"cacheable system prompt)")
    print(f"output tokens ~    {tok_out/1e6:.1f}M"
          f"{'  (thinking ON -- ~82% of this is discarded)' if args.thinking else ''}")
    print(f"est. cost     ~    ${cost:,.2f}"
          f"   (at ${args.price_in}/${args.price_out} per 1M in/out, "
          f"${args.price_cached} cached in)")
    eta = chunks / max(args.concurrency, 1) * 3.0 / 3600
    print(f"est. wall clock ~  {eta:.1f} h at {args.concurrency} concurrent, 3 s/request")
    print("\nPrices are the deepseek-v4-flash rate card as of 2026-08. Pass "
          "--price-in/--price-out/--price-cached if it has moved.")


async def run(paths: list[Path], args, stats: Stats) -> None:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        sys.exit("DEEPSEEK_API_KEY is not set. export it and rerun.")

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    # --concurrency budgets in-flight *requests*, which is what the API sees and
    # what costs money. Articles are allowed to run at twice that so a slot is
    # never idled by an article that turns out to need no request at all.
    # (Measured after the _INFOBOX_RE fix: 0 articles fall under MIN_BODY_CHARS
    # and ~1.1% are index lists, so the headroom is smaller than it once was.)
    req_sem = asyncio.Semaphore(args.concurrency)
    sem = asyncio.Semaphore(args.concurrency * 2)
    abort = asyncio.Event()
    limits = httpx.Limits(max_connections=args.concurrency + 4,
                          max_keepalive_connections=args.concurrency)

    async with httpx.AsyncClient(
        base_url=API_BASE,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
        timeout=httpx.Timeout(args.timeout, connect=15.0),
        limits=limits,
    ) as client:
        with (STATE_DIR / "failures.jsonl").open("a", encoding="utf-8") as ledger:
            with tqdm(total=len(paths), desc="paraphrasing", unit="art") as bar:
                await asyncio.gather(*(
                    worker(p, client, sem, args.model, args, stats, ledger, bar,
                           abort, req_sem)
                    for p in paths
                ))


# ---- Main -------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"DeepSeek model id (default: {DEFAULT_MODEL})")
    ap.add_argument("--concurrency", type=int, default=16,
                    help="in-flight requests (default: 16)")
    ap.add_argument("--limit", type=int,
                    help="process the first N articles (sorted; mostly stubs)")
    ap.add_argument("--sample", type=int,
                    help="process N articles drawn from across the whole "
                         "corpus -- prefer this for smoke tests")
    ap.add_argument("--seed", type=int, default=1977,
                    help="seed for --sample (default: 1977)")
    ap.add_argument("--chunk-chars", type=int, default=CHUNK_CHARS,
                    help=f"max chars per request (default: {CHUNK_CHARS})")
    ap.add_argument("--timeout", type=float, default=180.0, help="per-request timeout")
    ap.add_argument("--force", action="store_true",
                    help="re-paraphrase articles that already have output")
    ap.add_argument("--retry-failed", action="store_true",
                    help="also retry articles recorded in failures.jsonl")
    ap.add_argument("--paraphrase-infobox", action="store_true",
                    help="also reword the ## Infobox key/value block")
    ap.add_argument("--thinking", action="store_true",
                    help="let the model reason before answering. Off by "
                         "default: rewording does not need it, and the "
                         "reasoning tokens bill as output (~70%% of spend)")
    ap.add_argument("--dry-run", action="store_true",
                    help="estimate tokens, cost and runtime, then exit")
    ap.add_argument("--price-in", type=float, default=DEFAULT_PRICE_IN,
                    help=f"$/1M cache-miss input (default: {DEFAULT_PRICE_IN})")
    ap.add_argument("--price-out", type=float, default=DEFAULT_PRICE_OUT,
                    help=f"$/1M output (default: {DEFAULT_PRICE_OUT})")
    ap.add_argument("--price-cached", type=float, default=DEFAULT_PRICE_CACHED,
                    help=f"$/1M cache-hit input (default: {DEFAULT_PRICE_CACHED})")
    args = ap.parse_args()

    if not SOURCE_DIR.exists():
        sys.exit(f"No corpus at {SOURCE_DIR}. Run download_wookieepedia.py first.")

    paths = iter_sources(args.limit, args.sample, args.seed)
    if not paths:
        sys.exit(f"No .md files under {SOURCE_DIR}.")

    if args.dry_run:
        estimate(paths, args)
        return

    total = len(paths)
    if not args.force:
        paths = [p for p in paths if not output_path_for(p).exists()]
    if not args.retry_failed:
        failed = load_failures()
        paths = [p for p in paths
                 if p.relative_to(SOURCE_DIR).as_posix() not in failed]

    print(f"{total:,} articles found, {len(paths):,} to do "
          f"({total - len(paths):,} already done or previously failed)")
    if not paths:
        return

    stats = Stats()
    started = time.time()
    try:
        asyncio.run(run(paths, args, stats))
    except KeyboardInterrupt:
        print("\ninterrupted -- rerun the same command to resume")

    elapsed = time.time() - started
    if stats.fatal:
        print(f"\nRUN ABORTED -- {stats.fatal}")
        print("This is an account/config fault, not a problem with any article.")
        print("Nothing was written to the failure ledger. Fix it and rerun:")
        print("  402 Insufficient Balance -> top up at platform.deepseek.com")
        print("  401/403                  -> check DEEPSEEK_API_KEY")
        print("  404                      -> wrong --model id")
        print(f"\n{stats.done:,} articles completed before the abort.")
        sys.exit(1)

    miss = stats.tokens_in - stats.tokens_cached
    cost = (miss / 1e6 * args.price_in
            + stats.tokens_cached / 1e6 * args.price_cached
            + stats.tokens_out / 1e6 * args.price_out)
    print(f"\nparaphrased {stats.done:,} | skipped {stats.skipped:,} "
          f"| failed {stats.failed:,}")
    if stats.degraded_articles:
        print(f"  {stats.degraded_articles:,} of those kept verbatim source for "
              f"{stats.degraded_chunks:,} chunk(s) that no attempt could reword "
              f"-- find them with:\n"
              f"    grep -rl paraphrase_degraded_chunks {OUTPUT_DIR}")
    print(f"requests {stats.requests:,} (retries {stats.retries:,}) | "
          f"tokens {stats.tokens_in:,} in / {stats.tokens_out:,} out "
          f"| ~${cost:,.2f}")
    if stats.tokens_in:
        print(f"  input cache hits {stats.tokens_cached/stats.tokens_in:.0%} "
              f"({stats.tokens_cached:,} of {stats.tokens_in:,})")
    if stats.tokens_reasoning:
        # Loud on purpose: this is billed output you do not keep.
        share = stats.tokens_reasoning / max(stats.tokens_out, 1)
        print(f"  reasoning tokens {stats.tokens_reasoning:,} ({share:.0%} of "
              f"output, ~${stats.tokens_reasoning/1e6*args.price_out:,.2f} "
              f"discarded) -- drop --thinking to save it")
    print(f"elapsed {elapsed/60:.1f} min -> {OUTPUT_DIR}")
    if stats.errors:
        top = sorted(stats.errors.items(), key=lambda kv: -kv[1])[:5]
        print("top failure reasons: " + ", ".join(f"{k} x{v}" for k, v in top))
    if stats.failed:
        print(f"failures logged to {STATE_DIR / 'failures.jsonl'} "
              f"(rerun with --retry-failed)")


if __name__ == "__main__":
    main()
