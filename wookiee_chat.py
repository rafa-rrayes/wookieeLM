#!/usr/bin/env python3
"""A Star Wars expert chatbot that answers from the Wookieepedia corpus.

DeepSeek V4 Flash drives an agentic loop over ``corpus/wookieepedia`` -- the
171,440 Markdown articles written by ``download_wookieepedia.py``. The model
never answers from memory: it is given three tools and told to search and read
before it speaks, so every claim traces back to a named article.

    search_titles   ranked substring match over all 171k article titles,
                    served from an in-memory index -- instant, and the right
                    first move for any question about a named thing
    search_text     a scan of the full text of every article, for questions
                    no single title answers ("which Jedi survived Order 66")
                    -- the corpus is concatenated into one file under
                    ``.index/`` and mmapped, so the first run pays ~9s to
                    build it, later ones start instantly, and a query is
                    ~0.3s (or ~0.3/N with a batch script's scan pool)
    read_article    the article itself, by title, whole or one ``## `` section

Answers cite article titles and their continuity (canon / legends / non-canon /
real-world), because Wookieepedia carries both timelines and conflating them is
the main way a Star Wars answer goes wrong.

Thinking is disabled by default: it bills as output, and on this workload the
tool results carry the reasoning. Pass --thinking for questions that need the
model to plan several searches ahead.

Requirements:
    DEEPSEEK_API_KEY in the environment (or in a .env file beside this script)
    ~400 MB of disk under .index/ for the full-text index, built on the first
    search_text call and shared, read-only, by every later process

Usage:
    uv run wookiee_chat.py                          # interactive chat
    uv run wookiee_chat.py --ask "Who was Ahsoka Tano's master?"
    uv run wookiee_chat.py --thinking               # let it plan its searches
    uv run wookiee_chat.py --show-tools             # print every tool result
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import re
import sys
import time
from array import array
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

import corpus_scan as cs

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = REPO_ROOT / "corpus" / "wookieepedia"
INDEX_DIR = REPO_ROOT / ".index"

API_BASE = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEFAULT_MODEL = "deepseek-v4-flash"

MAX_TOOL_ROUNDS = 14        # tool calls per question before we force an answer
MAX_TOOL_CHARS = 7000       # cap on one tool result, to keep the context small
MAX_ARTICLE_CHARS = 14000   # read_article default; sections are read separately
SEPARATOR = b"\n\x00\n"    # article boundary inside the full-text buffer
MAX_ATTEMPTS = 4            # API retries on transport / 429 / 5xx

SYSTEM_PROMPT = """\
You are the definitive Star Wars expert. You have spent a lifetime with this \
material and you simply know it -- every character, world, ship, battle, and \
date, across both continuities.

Your tools are your memory. Use them constantly and silently. They are how you \
recall a fact, not a library you consult; the person you are talking to has no \
idea they exist and must never learn otherwise.

How you recall:

1. Check before you speak. Even when you feel certain, confirm it -- unaided \
recall is not enough, and a confident wrong date is the worst answer you can \
give.
2. Start with search_titles when the question names a thing (a person, planet, \
ship, battle, species, film, episode). Reach for search_text when no single \
entry would hold the answer, or when a title search comes up empty.
3. Read what you found. A search result line is a pointer, not evidence; \
reason only from text you have actually read.
4. Keep going until you can answer. If the first search misses, try the other \
tool, a different spelling, a shorter query, or a related subject that would \
mention the thing. Several tool calls per question is normal and expected.
5. Know your continuities. Every subject is `canon` (the current timeline), \
`legends` (the pre-2014 Expanded Universe), `non-canon`, or `real-world` \
(production and publishing). Say which one an answer belongs to whenever it \
could matter, and never blend canon and Legends into one account without \
flagging it -- when both exist and disagree, give canon first, then Legends.
6. If the fact does not exist, say so plainly. Do not invent a plausible \
detail. "That was never established" is a correct answer.

How you speak:

- Never mention Wookieepedia, articles, entries, sources, databases, corpora, \
searching, reading, looking something up, or your tools. Never say "the \
article says", "according to", "based on what I found", "the sources note", \
or anything that frames the answer as retrieved rather than known.
- No "Sources:" line. No citations. You are the source.
- When a fact is genuinely unrecorded, say it about the universe, not about \
your material: "That detail was never established", not "the entry doesn't \
say".
- Answer in prose, as concisely as the question allows, with the easy \
authority of someone recalling something they have known for years. No \
preamble about what you are about to do -- just the answer.\
"""


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------

_NORM_RE = re.compile(r"[^a-z0-9]+")


def normalize(title: str) -> str:
    """Article title -> lookup key.

    ``download_wookieepedia.py`` writes ``<L>/<Title>.md`` with spaces and the
    filesystem-unsafe set ``\\/:*?"<>|`` all mapped to underscores, so a title
    cannot be recovered from a filename exactly ("Star Wars: Ahsoka" and
    "Star Wars  Ahsoka" share a stem). Both sides normalise to the same key,
    which makes the ambiguity harmless for lookup.
    """
    return _NORM_RE.sub(" ", title.replace("_", " ").lower()).strip()


class Corpus:
    """The article set, the title index, and the full-text index."""

    def __init__(self, root: Path):
        self.root = root
        if not root.is_dir():
            sys.exit(f"corpus not found: {root}\n"
                     f"run download_wookieepedia.py first, or pass --corpus")

        self._text = None            # mmap of every article, lowercased
        self._starts = None          # where each article begins in it
        self._key_blob: str | None = None    # every title key, newline-joined
        self._key_at: list[int] = []         # where each key begins in it
        # Set by attach_scanner() to run search_text across processes; None
        # means scan in this one, which is what the interactive chat does.
        self.scanner: cs.SearchPool | None = None

        self.paths: list[Path] = []
        self.keys: list[str] = []
        self.by_key: dict[str, Path] = {}
        for shard in sorted(root.iterdir()):
            if not shard.is_dir():
                continue
            for path in shard.glob("*.md"):
                key = normalize(path.stem)
                self.paths.append(path)
                self.keys.append(key)
                # First writer wins; collisions are near-duplicate punctuation
                # variants of the same title, so either file answers the query.
                self.by_key.setdefault(key, path)

        if not self.paths:
            sys.exit(f"no .md articles under {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def title_of(self, path: Path) -> str:
        return path.stem.replace("_", " ")

    def resolve(self, title: str) -> Path | None:
        return self.by_key.get(normalize(title))

    # ---- titles ---------------------------------------------------------
    #
    # This used to be a linear scan of 171,440 keys with up to four substring
    # tests each, and the docstring called it ~50 ms. Measured on this
    # machine it is 79 ms -- of pure-Python, GIL-held loop, per call, on the
    # same critical path as the full-text scan and for the same reason
    # (search_titles is also read_article's fallback when a title misses).
    #
    # The keys never change, so they are joined into one newline-delimited
    # string and searched with `str.find`, which is C. `normalize` maps every
    # non-alphanumeric byte to a space, so no key contains a newline and no
    # match can straddle two of them. The ranking below is unchanged, and
    # tests/ checks it against the old implementation on random queries.

    def _build_key_blob(self) -> None:
        blob, at = [], []
        pos = 1                         # leading "\n", so key starts and
        for key in self.keys:           # `startswith` share one test
            at.append(pos)
            blob.append(key)
            pos += len(key) + 1
        self._key_blob = "\n" + "\n".join(blob) + "\n"
        self._key_at = at

    def _keys_containing(self, needle: str) -> list[int]:
        """Indices of every title key with ``needle`` in it, in corpus order."""
        if self._key_blob is None:
            self._build_key_blob()
        blob, at = self._key_blob, self._key_at
        found, pos, i = [], blob.find(needle), 0
        while pos >= 0:
            # Matches arrive in order, so the owning key walks forward.
            i = bisect.bisect_right(at, pos, lo=i) - 1
            found.append(i)
            pos = blob.find(needle, pos + 1)
        return found

    def search_titles(self, query: str, limit: int = 20) -> list[Path]:
        """Rank titles by how tightly they match, exact first."""
        q = normalize(query)
        if not q:
            return []
        keys = self.keys
        tokens = q.split()

        scored: list[tuple[int, int, int]] = []   # (-score, len, index)
        seen: set[int] = set()
        for i in self._keys_containing(q):
            if i in seen:               # a key can contain the query twice
                continue
            seen.add(i)
            key = keys[i]
            if key == q:
                score = 100
            elif key.startswith(q + " "):
                score = 80
            else:
                score = 60              # `q in key`, guaranteed by the scan
            scored.append((-score, len(key), i))

        # The scattered-tokens tier only ever loses to the three above, so it
        # is skipped entirely once they have filled the result set -- which
        # they do for any ordinary "who was X" title lookup.
        if len(tokens) > 1 and len(scored) < limit:
            hits = None
            for token in sorted(set(tokens), key=len, reverse=True):
                found = set(self._keys_containing(token))
                hits = found if hits is None else (hits & found)
                if not hits:
                    break
            for i in sorted(hits or ()):
                if i not in seen:
                    scored.append((-40, len(keys[i]), i))

        scored.sort()
        return [self.paths[i] for _, _, i in scored[:limit]]

    # ---- full text ------------------------------------------------------
    #
    # search_text used to shell out to ripgrep, which meant a fresh scan of
    # 828 MB across 171,440 files for every query -- measured at 7.5-9.9s
    # each. That is tolerable in a chat and ruinous in a batch: it pinned
    # answer_questions.py to the throughput of its search semaphore, 2.9
    # questions/second against 9.8 without search.
    #
    # The corpus is 828 MB and it does not change during a run, so it is read
    # once into a single lowercased buffer and every query is a memmem over
    # it. Build costs ~20s; a query costs ~0.3s and touches no disk.
    #
    # The buffer is bytes rather than str deliberately: str would store this
    # text at 2 or 4 bytes per character (any one non-ASCII character in
    # 828 MB decides it for the whole string), so bytes is both 2x smaller
    # and free of the decode. The cost is that case folding is ASCII-only --
    # a query for "Cafe" finds "cafe" and "CAFE", but "É" does not fold to
    # "é". Every case difference in a Star Wars name is an ASCII letter.

    def index_built(self) -> bool:
        return self._text is not None

    # ---- the on-disk index ----------------------------------------------
    #
    # The buffer lives in a file rather than a bytearray so that other
    # *processes* can share it. That is the whole point: `bytes.find` holds
    # the GIL for its entire 275 ms, so a batch run that searches is pinned to
    # one core no matter how much concurrency it is given (measured: 89-99% of
    # one core on a ten-core machine, at --concurrency 192). Every process
    # mmaps the same file, the page cache holds exactly one copy, and each
    # worker scans a shard under its own GIL. See corpus_scan.py.
    #
    # Keeping it costs 365 MB of disk and saves the ~9 s rebuild on every
    # later run, which for a chat session is the whole start-up.

    def _stamp(self) -> dict:
        """What the index was built from, cheaply enough to check every run.

        Article count plus the mtime of each shard directory. A directory's
        mtime moves when an article is added or removed, which is how the
        corpus actually changes -- download_wookieepedia.py writes whole files.
        An edit *in place* to an existing article is not caught; pass
        ``rebuild=True`` (``--rebuild-index``) after one.
        """
        shards = sorted(p for p in self.root.iterdir() if p.is_dir())
        return {"version": 1,
                "root": str(self.root),
                "n_articles": len(self.paths),
                "shards": {p.name: p.stat().st_mtime_ns for p in shards}}

    def _index_paths(self) -> tuple[Path, Path, Path]:
        # Keyed by the resolved corpus path, so --corpus on a second corpus
        # does not silently reuse the first one's index.
        tag = hashlib.sha1(str(self.root.resolve()).encode()).hexdigest()[:8]
        d = INDEX_DIR / f"{self.root.name}-{tag}"
        return d / "text.bin", d / "starts.bin", d / "meta.json"

    def build_text_index(self, workers: int = 8, rebuild: bool = False) -> int:
        """Make the search buffer available. -> bytes indexed.

        Reuses the file from a previous run when the corpus has not moved.
        """
        if self._text is not None:
            return len(self._text)

        text_p, starts_p, meta_p = self._index_paths()
        if not rebuild and meta_p.exists():
            try:
                if json.loads(meta_p.read_text()) == self._stamp():
                    self._text, self._starts = cs.open_index(text_p, starts_p)
                    return len(self._text)
            except (OSError, ValueError):
                pass                      # unreadable or stale: build it again

        text_p.parent.mkdir(parents=True, exist_ok=True)
        starts = array(cs.OFFSET_TYPECODE)

        def read_bytes(path: Path) -> bytes:
            try:
                return path.read_bytes()
            except OSError:
                return b""

        # Threads: the read releases the GIL, and the join has to stay in
        # corpus order so a match offset maps back to the right article.
        # Written to a temporary name and renamed, so a run interrupted here
        # leaves no half-index for the next one to trust.
        tmp = text_p.with_name(text_p.name + ".partial")
        total = 0
        with tmp.open("wb", buffering=1 << 20) as fh, \
                ThreadPoolExecutor(max_workers=workers) as pool:
            for blob in pool.map(read_bytes, self.paths, chunksize=256):
                starts.append(total)
                total += fh.write(blob.lower())
                # A NUL cannot appear in a query, so no match can straddle
                # the boundary between two articles.
                total += fh.write(SEPARATOR)
        tmp.replace(text_p)
        starts_p.write_bytes(starts.tobytes())
        # Written last: the stamp is what makes the other two trustworthy.
        meta_p.write_text(json.dumps(self._stamp()))

        self._text, self._starts = cs.open_index(text_p, starts_p)
        return len(self._text)

    def attach_scanner(self, workers: int) -> "cs.SearchPool":
        """Fan search_text out across processes. -> the pool, for shutdown.

        Everything downstream -- run_tool, tool_search_text, count_matches --
        goes through the scanner without knowing it exists.
        """
        self.build_text_index()
        text_p, starts_p, _ = self._index_paths()
        self.scanner = cs.SearchPool(text_p, starts_p, len(self.paths), workers)
        return self.scanner

    def count_matches_idx(self, query: str, regex: bool = False,
                          cap: int = 500_000) -> tuple[dict[int, int], bool]:
        """Matches per article *index*. -> ({index: count}, hit the cap?).

        Indices rather than paths because the caller wants the normalised
        title too, and `self.keys[i]` already holds it -- recomputing
        `normalize(path.stem)` for every one of 50,000 matching articles is
        50 ms of the parent's GIL for a value that was computed at start-up.
        """
        self.build_text_index()
        if regex:                     # validated here so a bad pattern from
            try:                      # the model is an error the tool can
                re.compile(query.encode(), re.IGNORECASE)   # report, rather
            except re.error as exc:                         # than a worker
                raise ValueError(str(exc)) from exc         # traceback
        if self.scanner is not None:
            return self.scanner.scan(query, regex, cap)
        return cs.scan_range(self._text, self._starts, 0, len(self._starts),
                             query, regex, cap)

    def count_matches(self, query: str, regex: bool = False,
                      cap: int = 500_000) -> tuple[list[tuple[Path, int]], bool]:
        """Matches per article, as ripgrep --count-matches gave them.

        -> ([(path, count), ...], whether the scan stopped at `cap`).
        """
        counts, capped = self.count_matches_idx(query, regex, cap)
        return [(self.paths[i], n) for i, n in counts.items()], capped

    def read(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="replace")

    def continuity_of(self, text: str) -> str:
        m = re.search(r"^continuity:\s*(\S+)", text, re.M)
        return m.group(1) if m else "unknown"

    def sections_of(self, text: str) -> list[str]:
        return re.findall(r"^## (.+)$", text, re.M)


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------

def tool_search_titles(corpus: Corpus, query: str, limit: int = 20) -> str:
    hits = corpus.search_titles(query, limit=min(limit, 40))
    if not hits:
        return (f"Nothing you know is named {query!r}. Try fewer words, another "
                f"spelling, or search_text for subjects that mention it.")
    lines = [f"{len(hits)} name match(es) for {query!r}:"]
    lines += [f"- {corpus.title_of(p)}" for p in hits]
    lines.append("\nUse read_article on the one you want.")
    return "\n".join(lines)


def tool_search_text(corpus: Corpus, query: str, limit: int = 8,
                     regex: bool = False) -> str:
    """Full-text scan, ranked, with a snippet from each winning article.

    The scan counts matches per article in one pass over the in-memory buffer;
    ranking and snippet extraction then touch only the handful of articles that
    won, so the cost is the scan and nothing more.
    """
    if not query.strip():
        return "search_text needs something to look for."
    try:
        matches, capped = corpus.count_matches_idx(query, regex=regex)
    except ValueError as exc:            # a regex the model wrote by hand
        return f"search_text could not read that pattern: {exc}"

    tokens = normalize(query).split()
    scored = []
    # `corpus.keys[i]` is the normalised stem, computed once at start-up.
    # Recomputing it here cost a regex substitution per matching article --
    # 50 ms of the parent's GIL on a query that hits 50,000 of them.
    for i, count in matches.items():
        key = corpus.keys[i]
        # A title that carries the query words is far likelier to be the
        # article about the thing than one that merely mentions it in passing.
        bonus = 30 * sum(t in key for t in tokens)
        # `key` before `i` in the tiebreak so the ranking does not depend on
        # the order the filesystem handed the articles back at start-up.
        scored.append((-(count + bonus), len(key), key, i))
    if not scored:
        return (f"Nothing you know mentions {query!r}. Try a shorter phrase, a "
                f"different spelling, or search_titles.")
    scored.sort()

    total = len(scored)
    out = [f"{'over ' if capped else ''}{total} subject(s) mention {query!r}. "
           f"Top {min(limit, total)} by relevance:"]
    needle = None if regex else query.lower()
    for *_, i in scored[:min(limit, 20)]:
        path = corpus.paths[i]
        text = corpus.read(path)
        snippet = _first_hit_line(text, needle, query, regex)
        out.append(f"\n### {corpus.title_of(path)}  [{corpus.continuity_of(text)}]"
                   f"\n{snippet}")
    out.append("\nUse read_article on any of these for the full text.")
    return "\n".join(out)


def _first_hit_line(text: str, needle: str | None, query: str,
                    regex: bool) -> str:
    """The first line of the article that matched, trimmed for context."""
    pattern = re.compile(query, re.I) if regex else None
    for line in text.splitlines():
        hit = pattern.search(line) if pattern else (needle in line.lower())
        if hit:
            line = line.strip()
            return line[:400] + ("..." if len(line) > 400 else "")
    return "(matched, but the line could not be isolated)"


def tool_read_article(corpus: Corpus, title: str, section: str | None = None,
                      max_chars: int = MAX_ARTICLE_CHARS) -> str:
    path = corpus.resolve(title)
    if path is None:
        near = corpus.search_titles(title, limit=8)
        if not near:
            return (f"You know of no {title!r}. Use search_titles or "
                    f"search_text to find the right name first.")
        names = "\n".join(f"- {corpus.title_of(p)}" for p in near)
        return f"Nothing named exactly {title!r}. Closest names:\n{names}"

    text = corpus.read(path)
    name = corpus.title_of(path)
    sections = corpus.sections_of(text)

    if section:
        body = _extract_section(text, section)
        if body is None:
            have = ", ".join(sections) or "(none)"
            return (f"{name} has no section {section!r}. Sections: {have}")
        header = f"# {name} -- section '{section}'\n\n"
        return header + _truncate(body, max_chars, sections)

    return f"# {name}\n\n" + _truncate(text, max_chars, sections)


def _extract_section(text: str, section: str) -> str | None:
    """One ``## `` block, from its heading to the next heading of any level."""
    want = normalize(section)
    matches = list(re.finditer(r"^(#{2,})\s*(.+)$", text, re.M))
    for i, m in enumerate(matches):
        if normalize(m.group(2)) != want:
            continue
        depth = len(m.group(1))
        end = len(text)
        for nxt in matches[i + 1:]:
            if len(nxt.group(1)) <= depth:
                end = nxt.start()
                break
        return text[m.start():end].strip()
    return None


def _truncate(text: str, max_chars: int, sections: list[str]) -> str:
    if len(text) <= max_chars:
        return text
    have = ", ".join(sections) or "(none)"
    return (text[:max_chars] +
            f"\n\n[truncated at {max_chars} of {len(text)} characters. "
            f"Sections you can recall: {have}. Call read_article again with "
            f"`section` to read one of them in full.]")


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_titles",
            "description":
                "Find subjects by name. Instant. Use this first whenever the "
                "question names a person, planet, ship, species, battle, film, "
                "episode, book or organisation. Matches on substrings, so a "
                "partial title works.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "Title or part of one, e.g. "
                                             "'Ahsoka Tano' or 'Battle of Endor'."},
                    "limit": {"type": "integer",
                              "description": "Max titles to return (default 20)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description":
                "Search everything you know, in full. Takes a few seconds. Use "
                "it when no single subject answers the question (cross-cutting "
                "or 'which X did Y' questions), or when search_titles found "
                "nothing. Returns the most relevant subjects with a matching "
                "line from each.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "Exact phrase to find. Short and "
                                             "distinctive beats long."},
                    "limit": {"type": "integer",
                              "description": "Max articles to return (default 8, max 20)."},
                    "regex": {"type": "boolean",
                              "description": "Treat query as a regular "
                                             "expression (default false)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_article",
            "description":
                "Recall everything you know about a subject, by its exact "
                "title. Returns frontmatter (including the continuity field), "
                "the infobox and the body. Long entries are truncated with "
                "their section list "
                "-- call again with `section` to read one in full.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string",
                              "description": "Exact article title, as returned "
                                             "by a search tool."},
                    "section": {"type": "string",
                                "description": "Optional '## ' section to read "
                                               "instead of the whole article, "
                                               "e.g. 'Biography'."},
                },
                "required": ["title"],
            },
        },
    },
]

DISPATCH = {
    "search_titles": tool_search_titles,
    "search_text": tool_search_text,
    "read_article": tool_read_article,
}


def run_tool(corpus: Corpus, name: str, raw_args: str) -> str:
    """Execute one tool call. Never raises -- the model gets the error instead."""
    fn = DISPATCH.get(name)
    if fn is None:
        return f"No such tool: {name}. Available: {', '.join(DISPATCH)}"
    try:
        args = json.loads(raw_args or "{}")
    except json.JSONDecodeError as e:
        return f"Could not parse your arguments as JSON: {e}"
    try:
        result = fn(corpus, **args)
    except TypeError as e:
        return f"Bad arguments for {name}: {e}"
    except Exception as e:                            # noqa: BLE001
        return f"{name} failed: {type(e).__name__}: {e}"
    if len(result) > MAX_TOOL_CHARS:
        result = result[:MAX_TOOL_CHARS] + "\n[tool output truncated]"
    return result


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

class ChatError(RuntimeError):
    pass


def call_model(client: httpx.Client, messages: list[dict], model: str,
               thinking: bool) -> dict:
    """One /chat/completions round, retried on transport and transient faults."""
    payload = {
        "model": model,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        # Facts, not variety: the corpus is the source and the answer should not
        # drift between runs.
        "temperature": 0.2,
        # DeepSeek's V4 models reason before answering unless told not to, and
        # those tokens bill as output. Here the tool results carry the evidence,
        # so the chain of thought is mostly restating them -- see tasks/lessons.md.
        "thinking": {"type": "enabled" if thinking else "disabled"},
    }

    last = "unknown"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = client.post("/chat/completions", json=payload)
        except (httpx.TransportError, httpx.TimeoutException) as e:
            last = f"transport: {type(e).__name__}"
        else:
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]
            if resp.status_code == 401 or resp.status_code == 403:
                raise ChatError("401/403 -- check DEEPSEEK_API_KEY")
            if resp.status_code == 402:
                raise ChatError("402 Insufficient Balance -- top up at "
                                "platform.deepseek.com")
            if resp.status_code == 404:
                raise ChatError(f"404 -- no such model {model!r}, check --model")
            if resp.status_code not in (429,) and resp.status_code < 500:
                raise ChatError(f"http {resp.status_code}: {resp.text[:200]}")
            last = f"http {resp.status_code}"
        if attempt < MAX_ATTEMPTS:
            time.sleep(min(2 ** attempt, 15))
    raise ChatError(f"gave up after {MAX_ATTEMPTS} attempts ({last})")


def answer(client: httpx.Client, corpus: Corpus, messages: list[dict],
           model: str, thinking: bool, show_tools: bool) -> str:
    """Drive the tool loop until the model stops asking for tools."""
    for round_no in range(MAX_TOOL_ROUNDS):
        msg = call_model(client, messages, model, thinking)
        messages.append(msg)

        calls = msg.get("tool_calls") or []
        if not calls:
            return (msg.get("content") or "").strip() or "(empty response)"

        for call in calls:
            name = call["function"]["name"]
            raw_args = call["function"].get("arguments", "")
            _echo_call(name, raw_args)
            result = run_tool(corpus, name, raw_args)
            if show_tools:
                print(_dim(_indent(result)))
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": result})

    # Out of rounds: make it answer from what it has rather than dropping the turn.
    messages.append({"role": "user", "content":
                     "You have used your search budget for this question. "
                     "Answer now from what you have read, and say plainly what "
                     "you could not confirm."})
    msg = call_model(client, messages, model, thinking)
    messages.append(msg)
    return (msg.get("content") or "").strip() or "(empty response)"


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------

def _dim(s: str) -> str:
    return f"\033[2m{s}\033[0m" if sys.stdout.isatty() else s


def _indent(s: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in s.splitlines())


def _echo_call(name: str, raw_args: str) -> None:
    try:
        args = json.loads(raw_args or "{}")
        shown = ", ".join(f"{k}={v!r}" for k, v in args.items())
    except json.JSONDecodeError:
        shown = raw_args[:120]
    print(_dim(f"  · {name}({shown})"), flush=True)


def load_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    env_file = REPO_ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, _, value = line.strip().partition("=")
            if name.strip() == "DEEPSEEK_API_KEY":
                return value.strip().strip("'\"")
    sys.exit("DEEPSEEK_API_KEY is not set. export it, or put it in a .env file "
             "beside this script, and rerun.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ask", metavar="QUESTION",
                    help="answer one question and exit, instead of chatting")
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS,
                    help=f"corpus directory (default: {DEFAULT_CORPUS})")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"DeepSeek model id (default: {DEFAULT_MODEL})")
    ap.add_argument("--thinking", action="store_true",
                    help="let the model reason before answering. Off by "
                         "default: reasoning tokens bill as output")
    ap.add_argument("--show-tools", action="store_true",
                    help="print what each tool returned, not just the call")
    args = ap.parse_args()

    key = load_api_key()
    corpus = Corpus(args.corpus)

    client = httpx.Client(
        base_url=API_BASE,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
        timeout=httpx.Timeout(180.0, connect=15.0),
    )

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def ask(question: str) -> None:
        messages.append({"role": "user", "content": question})
        try:
            reply = answer(client, corpus, messages, args.model, args.thinking,
                           args.show_tools)
        except ChatError as e:
            print(f"\nerror: {e}\n", file=sys.stderr)
            return
        print(f"\n{reply}\n")

    if args.ask:
        with client:
            ask(args.ask)
        return

    print(f"Wookieepedia expert -- {len(corpus):,} articles, {args.model}")
    print("Ask anything about Star Wars. Ctrl-D or /exit to quit.\n")
    with client:
        while True:
            try:
                question = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not question:
                continue
            if question in ("/exit", "/quit"):
                return
            ask(question)


if __name__ == "__main__":
    main()
