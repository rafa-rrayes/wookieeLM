#!/usr/bin/env python3
"""Rewrite every Wookieepedia article into several other forms.

One article, one fact, one phrasing is one chance for a model to learn it.
`paraphrase_corpus.py` already gives each article a second phrasing; this gives
it a timeline, an in-universe archive entry, a dialogue, a quiz, a fact list, a
summary, an explainer, and -- the one that is not just variety -- a rewrite in
which some *other* entity is the subject.

That last one exists for the reversal curse. A model trained on "Luke Skywalker
was the son of Anakin Skywalker" routinely cannot answer "who was Anakin
Skywalker's son?", because the fact was only ever presented in one direction.
The standard mitigation is to put the reverse direction in the training data,
so `entityflip` walks the entities an article mentions and restates each
relationship with that entity as the grammatical subject.

Every form is a *view*: a name, an output directory, a system prompt and a
handful of thresholds, all declared in `augment_views.toml`. The engine here
knows nothing about any particular form, so adding one is editing that file --
no code, and `package.py` picks the new directory up from the same file.

    corpus/wookieepedia/L/Luke_Skywalker.md
        -> corpus/wookieepedia_timeline1/L/Luke_Skywalker.md
        -> corpus/wookieepedia_quiz1/L/Luke_Skywalker.md
        -> corpus/wookieepedia_entityflip1/L/Luke_Skywalker.md
        -> ... one mirrored tree per view

Output is a Markdown document per (article, view), carrying the source
frontmatter forward -- title, continuity, categories -- plus provenance:

    augment_view: "timeline"
    augmented_from: "wookieepedia/L/Luke_Skywalker.md"
    augment_model: "deepseek-v4-flash"

so a view can be filtered, weighted or dropped downstream, and CC BY-SA
attribution survives into derived text.

What the engine enforces, per view and configurable:
    eligibility     minimum body, a required regex (a timeline needs dates),
                    a continuity allowlist (an actor has no in-universe entry)
    packing         tables dropped, long articles condensed to lead + section
                    openings rather than truncated blind
    validation      length bounds, the subject actually appearing, no meta-talk
                    about "the passage", and every BBY/ABY date in the output
                    having been in the input -- the cheapest hallucination
                    check there is, since inventing a date is the failure this
                    task actually has

Thinking is off by default for the reason it is off in paraphrase_corpus.py:
DeepSeek's V4 models reason before answering unless told not to, those tokens
bill as output, and the constraints here are already fully specified in the
prompt.

Requirements:
    DEEPSEEK_API_KEY in the environment

Usage:
    uv run scripts/augment_corpus.py --list-views                 what is configured
    uv run scripts/augment_corpus.py --dry-run                    cost of the full run
    uv run scripts/augment_corpus.py --views timeline,quiz --sample 20
    uv run scripts/augment_corpus.py --views entityflip           one form, whole corpus
    uv run scripts/augment_corpus.py                              every enabled view

Resume: rerun the same command. A (article, view) whose output file exists is
skipped, so an interrupted run picks up where it stopped. Pairs that failed
every attempt are appended to augment_state/<view>/failures.jsonl and skipped
later unless --retry-failed is passed.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.util
import json
import os
import random
import re
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from tqdm import tqdm

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
CORPUS_DIR = REPO_ROOT / "corpus"
SOURCE_DIR = CORPUS_DIR / "wookieepedia"
CONFIG_FILE = REPO_ROOT / "augment_views.toml"
STATE_DIR = REPO_ROOT / "augment_state"

API_BASE = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# Same rate card as paraphrase_corpus.py; --price-* overrides it.
DEFAULT_PRICE_IN = 0.14
DEFAULT_PRICE_OUT = 0.28
DEFAULT_PRICE_CACHED = 0.0028

CHARS_PER_TOKEN = 3.6       # empirical for English Markdown w/ many proper nouns

BACKOFF_BASE = 2.0
BACKOFF_CAP = 60.0

CONTINUITY_LABEL = {
    "canon": "Canon",
    "legends": "Legends",
    "non-canon": "Non-canon",
    "real-world": "Real-world (out-of-universe)",
}


# ---- Reuse of paraphrase_corpus.py -------------------------------------------

def load_pc():
    """Import paraphrase_corpus.py by path.

    Same trick count.py, package.py and generate_questions.py use. The
    splitter, the skip rule and the table detector are already written and
    audited there, and this stage has to agree with them about what a body is
    and which articles are worth a request -- reimplementing any of it here is
    how the two would drift.

    `pc` is the repo-wide name for this module, and package.py loads it under
    the same one. Reusing an already-loaded copy keeps a process that imports
    both from holding two, whose classes would not compare equal.
    """
    if (loaded := sys.modules.get("pc")) is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(
        "pc", SCRIPTS_DIR / "paraphrase_corpus.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["pc"] = module
    spec.loader.exec_module(module)
    return module


pc = load_pc()


# ---- Views -------------------------------------------------------------------

class ConfigError(RuntimeError):
    """augment_views.toml says something the engine cannot act on."""


# Every knob a view has, with its default and therefore its type. A key in the
# file that is not in here is a typo, and typos in a config file are silent
# unless something rejects them -- a misspelled `min_body_chars` would quietly
# augment the whole corpus at the default threshold.
_SCHEMA: dict[str, object] = {
    "enabled": True,
    "out_dir": "",              # required
    "system": "",               # required
    "system_prefix": "",
    "system_suffix": "",
    "user_template": "",
    "title_suffix": "",
    "model": "",                # "" -> --model
    "temperature": 1.0,
    "thinking": False,
    "max_attempts": 4,
    "max_input_chars": 12000,
    "infobox_chars": 1500,
    "include_infobox": True,
    "drop_tables": True,
    "min_body_chars": 400,
    "require_pattern": "",
    "continuity": [],
    "min_out_chars": 200,
    "min_out_ratio": 0.0,
    "max_out_ratio": 1.5,
    "require_title": True,
    "check_dates": True,
    "est_out_ratio": 0.6,
    "forbid": [],
}

_REQUIRED = ("out_dir", "system", "user_template")


@dataclass(frozen=True)
class View:
    """One augmentation form, fully specified by augment_views.toml."""
    name: str
    out_dir: str
    system: str
    user_template: str
    title_suffix: str
    enabled: bool
    model: str
    temperature: float
    thinking: bool
    max_attempts: int
    max_input_chars: int
    infobox_chars: int
    include_infobox: bool
    drop_tables: bool
    min_body_chars: int
    require_pattern: re.Pattern | None
    continuity: tuple[str, ...]
    min_out_chars: int
    min_out_ratio: float
    max_out_ratio: float
    require_title: bool
    check_dates: bool
    est_out_ratio: float
    forbid: tuple[re.Pattern, ...]

    @property
    def directory(self) -> Path:
        return CORPUS_DIR / self.out_dir

    @property
    def state_dir(self) -> Path:
        return STATE_DIR / self.name


def _coerce(name: str, key: str, value, default):
    """Reject a value whose type cannot be what the engine will do with it."""
    if isinstance(default, bool) and not isinstance(value, bool):
        raise ConfigError(f"views.{name}.{key}: expected true/false")
    if isinstance(default, (int, float)) and not isinstance(default, bool):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"views.{name}.{key}: expected a number")
        return type(default)(value)
    if isinstance(default, str) and not isinstance(value, str):
        raise ConfigError(f"views.{name}.{key}: expected a string")
    if isinstance(default, list):
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ConfigError(f"views.{name}.{key}: expected a list of strings")
    return value


def _compile(name: str, key: str, pattern: str) -> re.Pattern:
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        raise ConfigError(f"views.{name}.{key}: bad regex {pattern!r} -- {e}") from e


def parse_views(path: Path = CONFIG_FILE) -> list[View]:
    """Parse augment_views.toml into Views, in file order, enabled or not."""
    if not path.exists():
        raise ConfigError(f"no view config at {path}")
    try:
        config = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path.name}: {e}") from e

    defaults = config.get("defaults", {})
    if unknown := set(defaults) - set(_SCHEMA):
        raise ConfigError(f"[defaults]: unknown key(s) {', '.join(sorted(unknown))}")
    if not (tables := config.get("views")):
        raise ConfigError(f"{path.name} declares no [views.*]")

    views: list[View] = []
    for name, table in tables.items():
        if unknown := set(table) - set(_SCHEMA):
            raise ConfigError(
                f"views.{name}: unknown key(s) {', '.join(sorted(unknown))}. "
                f"Known keys: {', '.join(sorted(_SCHEMA))}")
        merged = {k: table.get(k, defaults.get(k, d)) for k, d in _SCHEMA.items()}
        merged = {k: _coerce(name, k, v, _SCHEMA[k]) for k, v in merged.items()}
        for key in _REQUIRED:
            if not merged[key]:
                raise ConfigError(f"views.{name}: {key} is required")

        system = merged.pop("system_prefix") + merged["system"] + merged.pop("system_suffix")
        # A placeholder in the system prompt would make it per-article, and the
        # shared prefix is the whole reason input bills at the cache-hit rate.
        if stray := re.findall(r"\{(\w+)\}", system):
            raise ConfigError(
                f"views.{name}.system: contains placeholder(s) "
                f"{{{'}, {'.join(sorted(set(stray)))}}}. The system prompt must be "
                f"identical on every request -- put per-article text in "
                f"user_template.")
        merged["system"] = system

        if missing := {"title", "continuity", "passage"} - set(
                re.findall(r"\{(\w+)\}", merged["user_template"])):
            raise ConfigError(
                f"views.{name}.user_template: missing placeholder(s) "
                f"{', '.join('{%s}' % m for m in sorted(missing))}")

        merged["require_pattern"] = (
            _compile(name, "require_pattern", merged["require_pattern"])
            if merged["require_pattern"] else None)
        merged["forbid"] = tuple(_compile(name, "forbid", p) for p in merged["forbid"])
        merged["continuity"] = tuple(merged["continuity"])
        views.append(View(name=name, **merged))
    return views


def load_views(path: Path = CONFIG_FILE, names: list[str] | None = None) -> list[View]:
    """The views to run, in file order.

    `names` selects a subset and overrides `enabled`: naming a view explicitly
    is a decision to run it. Without `names`, only the enabled ones run.
    Selection never changes what a view *is*, only whether it runs.
    """
    views = parse_views(path)
    if names is None:
        return [v for v in views if v.enabled]
    known = {v.name for v in views}
    if unknown := set(names) - known:
        raise ConfigError(f"unknown view(s): {', '.join(sorted(unknown))}. "
                          f"available: {', '.join(sorted(known))}")
    return [v for v in views if v.name in set(names)]


def view_output_dirs(path: Path = CONFIG_FILE) -> dict[str, str]:
    """{view name -> corpus directory}, for package.py and count.py.

    Read from the config rather than hardcoded downstream, so adding a view is
    still a one-file change even though three scripts care about the result.
    Returns {} rather than raising if the file is missing or malformed: a
    packaging run must not fail because an optional stage is unconfigured.
    """
    try:
        return {v.name: v.out_dir for v in parse_views(path)}
    except (ConfigError, OSError):
        return {}


# ---- What gets sent ----------------------------------------------------------

_CONTINUITY_RE = re.compile(r'^continuity:\s*"?([\w-]+)"?\s*$', re.MULTILINE)
_TITLE_LINE_RE = re.compile(r'^title:.*$', re.MULTILINE)
_URL_LINE_RE = re.compile(r'^url:', re.MULTILINE)


def continuity_of(doc) -> str:
    m = _CONTINUITY_RE.search(doc.frontmatter)
    return m.group(1) if m else ""


def strip_tables(body: str) -> str:
    """Drop table passages from the text that gets sent.

    A filmography grid or a release table is proper nouns and ISBNs laid out in
    markup. None of these forms can use it -- a timeline cannot date it, a
    dialogue cannot speak it -- and on the data-heavy articles it is most of
    the file, so leaving it in would spend the whole input budget on the one
    part of the article no view wants.
    """
    kept = [s for s in pc.split_segments(body) if not pc.is_table_block(s)]
    return "\n\n".join(s.strip() for s in kept if s.strip())


def _cut(text: str, limit: int) -> str:
    """Trim to `limit` on a sentence or paragraph boundary where there is one."""
    if len(text) <= limit:
        return text
    if limit <= 0:
        return ""
    head = text[:limit]
    stop = max(head.rfind(". "), head.rfind(".\n"), head.rfind("\n\n"))
    return (head[:stop + 1] if stop > limit * 0.5 else head).rstrip()


def condense(body: str, limit: int) -> str:
    """Fit a long article into `limit` chars, losing depth before breadth.

    Blind truncation would hand every view the first third of a long article
    and silently drop the rest, so a timeline of Palpatine would end at the
    Naboo blockade. Each section keeps its heading and its opening instead, so
    the whole span of the article is represented and what is lost is detail
    inside sections rather than sections themselves.
    """
    if len(body) <= limit:
        return body

    sections = re.split(r"\n(?=## )", body)
    lead = "" if sections[0].startswith("## ") else sections[0].strip()
    rest = [s.strip() for s in (sections[1:] if lead else sections) if s.strip()]

    parts: list[str] = []
    budget = limit
    if lead:
        # The lead is the article's own summary and is worth more per character
        # than anything under a heading.
        take = _cut(lead, max(int(limit * 0.4), 600))
        parts.append(take)
        budget -= len(take)
    if rest:
        # No floor under `per`. A floor looks generous and is the opposite: on
        # an article with 40 sections it hands the first ten their full share
        # and lets the final trim delete the other thirty, which is the blind
        # truncation this function exists to avoid. A section too small to hold
        # prose contributes its heading, which still says the section is there.
        per = budget // len(rest)
        for section in rest:
            head, _, tail = section.partition("\n")
            if take := _cut(tail.strip(), per - len(head) - 2):
                parts.append(f"{head}\n\n{take}")
            else:
                parts.append(head)
    return _cut("\n\n".join(parts), limit)


def pack_input(doc, view: View) -> str:
    """The article text one request carries: infobox, then body."""
    body = strip_tables(doc.body) if view.drop_tables else doc.body
    limit = view.max_input_chars
    parts: list[str] = []
    if view.include_infobox and doc.infobox:
        infobox = _cut(doc.infobox.strip(), view.infobox_chars)
        parts.append(infobox)
        limit -= len(infobox)
    parts.append(condense(body, max(limit, 500)))
    return "\n\n".join(p for p in parts if p)


def eligible(doc, view: View, body: str | None = None) -> str | None:
    """Why this article should not be sent to this view, or None to send it.

    One predicate shared by the worker and --dry-run, so the estimate and the
    run cannot disagree about the denominator.
    """
    if reason := pc.skip_reason(doc):
        return reason
    body = strip_tables(doc.body) if body is None else body
    if len(body) < view.min_body_chars:
        return "too short for view"
    if view.continuity and continuity_of(doc) not in view.continuity:
        return "continuity excluded"
    if view.require_pattern and not view.require_pattern.search(body):
        return "pattern not found"
    return None


# ---- What comes back ---------------------------------------------------------

# "25,797 BBY", "19 ABY" -- the only numbers in this corpus a model reliably
# invents, and the only ones worth a deterministic check.
_DATE_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(BBY|ABY)\b")
_PAREN_RE = re.compile(r"\s*\([^)]*\)\s*$")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def dates(text: str) -> set[str]:
    """{'25797bby', ...} -- comma- and spacing-insensitive era dates."""
    return {f"{n.replace(',', '')}{era}".lower() for n, era in _DATE_RE.findall(text)}


def title_present(title: str, out: str) -> bool:
    """Whether the rewrite is actually about the subject.

    Exact containment first, then a majority of the title's significant words,
    because an article titled "Battle of Yavin (Legends)" or "Death Star I" is
    legitimately written about under a slightly different name -- and rejecting
    those costs a whole extra request each.
    """
    base = _PAREN_RE.sub("", title).strip()
    body = _norm(out)
    if _norm(base) and _norm(base) in body:
        return True
    words = [w for w in _WORD_RE.findall(base) if len(w) > 3]
    if not words:
        return True     # nothing distinctive to look for: "A to Z", "3 ABY"
    hits = sum(1 for w in words if w.lower() in body)
    return hits >= 0.6 * len(words)


def validate(out: str, view: View, doc, packed: str,
             finish_reason: str | None) -> str | None:
    """Why this response is unusable, or None to keep it."""
    if not out:
        return "empty response"
    if pc._REFUSAL_RE.match(out):
        return "refusal"
    if finish_reason == "length":
        return "truncated"
    if len(out) < view.min_out_chars:
        return f"too short ({len(out)} chars)"

    ratio = len(out) / max(len(packed), 1)
    if view.min_out_ratio and ratio < view.min_out_ratio:
        return f"length ratio {ratio:.2f}"
    if view.max_out_ratio and ratio > view.max_out_ratio:
        return f"length ratio {ratio:.2f}"

    if view.require_title and not title_present(doc.title, out):
        return "subject missing"
    for pattern in view.forbid:
        if m := pattern.search(out):
            return f"meta-talk: {m.group(0)!r}"
    if view.check_dates and (invented := dates(out) - dates(packed)):
        # A date the input never contained was invented, computed, or converted
        # between eras. All three are the same kind of wrong.
        return f"date not in source: {sorted(invented)[0]}"
    return None


# ---- Output ------------------------------------------------------------------

def wookieepedia_url(title: str) -> str:
    """CC BY-SA wants a link back. Same rule package.py uses."""
    return "https://starwars.fandom.com/wiki/" + title.replace(" ", "_")


def build_output(view: View, doc, text: str, model: str) -> str:
    """Assemble the document, carrying the source frontmatter forward.

    Continuity and categories are copied rather than re-derived: Canon and
    Legends contradict each other constantly, and a rewrite that loses its
    continuity label is worse than not having it -- it looks authoritative and
    belongs to no timeline.

    `title_suffix` renames the document ("Luke Skywalker (Timeline)") so the
    form is visible in the text a model trains on, which is the only place it
    can condition on it. The unsuffixed title still determines the attribution
    URL, written out explicitly here because the suffixed one would not resolve.
    """
    frontmatter = doc.frontmatter or f'---\ntitle: "{doc.title}"\n---\n'
    head = frontmatter[: -len("---\n")]

    if view.title_suffix:
        title = f"{doc.title}{view.title_suffix}"
        # A function replacement, not a string: a backslash in a title would
        # otherwise be read as a group reference.
        head, n = _TITLE_LINE_RE.subn(lambda _: f'title: "{title}"', head, count=1)
        if not n:
            head += f'title: "{title}"\n'
    if not _URL_LINE_RE.search(head):
        head += f'url: "{wookieepedia_url(doc.title)}"\n'

    head += (f'augment_view: "{view.name}"\n'
             f'augmented_from: "{_corpus_rel(doc.path)}"\n'
             f'augment_model: "{model}"\n')
    return f"{head}---\n\n{text.rstrip()}\n"


def _corpus_rel(path: Path) -> str:
    """Provenance path, relative to corpus/ when it can be.

    Falls back to the last three components rather than raising: an unexpected
    source location is a reason to write a slightly uglier provenance line, not
    to lose the article.
    """
    try:
        return path.relative_to(CORPUS_DIR).as_posix()
    except ValueError:
        return "/".join(path.parts[-3:])


def output_path_for(view: View, src: Path) -> Path:
    return view.directory / src.relative_to(SOURCE_DIR)


# ---- API ---------------------------------------------------------------------

class AugmentError(RuntimeError):
    """This (article, view) produced nothing usable."""


@dataclass
class ViewStats:
    done: int = 0
    skipped: int = 0
    failed: int = 0
    out_chars: int = 0


@dataclass
class Stats:
    requests: int = 0
    retries: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cached: int = 0
    tokens_reasoning: int = 0
    fatal: str | None = None
    views: dict[str, ViewStats] = field(default_factory=dict)
    errors: dict[str, int] = field(default_factory=dict)

    def view(self, name: str) -> ViewStats:
        return self.views.setdefault(name, ViewStats())

    def note_error(self, view: str, kind: str) -> None:
        key = f"{view}: {kind}"
        self.errors[key] = self.errors.get(key, 0) + 1

    def note_usage(self, usage: dict) -> None:
        self.tokens_in += usage.get("prompt_tokens", 0)
        self.tokens_out += usage.get("completion_tokens", 0)
        self.tokens_cached += usage.get("prompt_cache_hit_tokens", 0)
        details = usage.get("completion_tokens_details") or {}
        self.tokens_reasoning += details.get("reasoning_tokens", 0)

    @property
    def done(self) -> int:
        return sum(v.done for v in self.views.values())

    @property
    def failed(self) -> int:
        return sum(v.failed for v in self.views.values())

    @property
    def skipped(self) -> int:
        return sum(v.skipped for v in self.views.values())


async def generate(client: httpx.AsyncClient, view: View, doc, packed: str,
                   model: str, stats: Stats, req_sem: asyncio.Semaphore) -> str:
    """One (article, view) -> one rewrite, retrying transport and quality faults.

    Unlike paraphrase_corpus.py there is no degrade-to-source fallback: a
    timeline that could not be written is simply absent, and the source article
    copied into corpus/wookieepedia_timeline1/ would be a lie about what that
    directory contains.
    """
    user = view.user_template.format(
        title=doc.title,
        continuity=CONTINUITY_LABEL.get(continuity_of(doc), "Unspecified"),
        passage=packed,
    )
    payload = {
        "model": model,
        "temperature": view.temperature,
        "thinking": {"type": "enabled" if view.thinking else "disabled"},
        "messages": [
            {"role": "system", "content": view.system},
            {"role": "user", "content": user},
        ],
    }

    last_error = "unknown"
    for attempt in range(1, view.max_attempts + 1):
        if attempt > 1:
            stats.retries += 1
        try:
            async with req_sem:
                resp = await client.post("/chat/completions", json=payload)
        except (httpx.TransportError, httpx.TimeoutException) as e:
            last_error = f"transport: {type(e).__name__}"
            await pc._backoff(attempt)
            continue

        if resp.status_code == 429 or resp.status_code >= 500:
            last_error = f"http {resp.status_code}"
            await pc._backoff(attempt, resp.headers.get("Retry-After"))
            continue
        if resp.status_code in pc.FATAL_STATUSES:
            raise pc.FatalAPIError(f"http {resp.status_code}: {resp.text[:200]}")
        if resp.status_code >= 400:
            raise AugmentError(f"http {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        stats.requests += 1
        if usage := data.get("usage"):
            stats.note_usage(usage)

        choice = data["choices"][0]
        out = pc.clean_response(choice["message"]["content"] or "")
        if (fault := validate(out, view, doc, packed,
                              choice.get("finish_reason"))) is None:
            return out

        last_error = fault
        # Quality fault: nudge temperature down and try again.
        payload["temperature"] = max(0.6, payload["temperature"] - 0.2)
        await pc._backoff(attempt)

    stats.note_error(view.name, last_error)
    raise AugmentError(last_error)


# ---- Orchestration -----------------------------------------------------------

def iter_sources(limit: int | None, sample: int | None, seed: int) -> list[Path]:
    """Collect source articles.

    --limit takes the head of the sorted list, which is a poor smoke test: the
    corpus starts with numeric-titled droid stubs. --sample draws from across
    the whole corpus instead, seeded so a run is repeatable.
    """
    paths = sorted(SOURCE_DIR.rglob("*.md"))
    if sample and sample < len(paths):
        return sorted(random.Random(seed).sample(paths, sample))
    return paths[:limit] if limit else paths


def load_failures(view: View) -> set[str]:
    """Article paths this view already failed on.

    Entries recording an account-level fault are ignored rather than skipped,
    for the reason paraphrase_corpus.py ignores them: a 402 says nothing about
    the article, and skipping it would quietly exclude a whole run's worth of
    work once the balance is topped up.
    """
    ledger = view.state_dir / "failures.jsonl"
    if not ledger.exists():
        return set()
    failed: set[str] = set()
    for line in ledger.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
            if pc._FATAL_LEDGER_RE.match(rec.get("error", "")):
                continue
            failed.add(rec["path"])
        except (json.JSONDecodeError, KeyError):
            continue
    return failed


async def augment_article(src: Path, views: list[View], client: httpx.AsyncClient,
                          sem: asyncio.Semaphore, req_sem: asyncio.Semaphore,
                          model: str, stats: Stats, ledgers: dict[str, object],
                          bar: tqdm, abort: asyncio.Event) -> None:
    """Every view of one article, dispatched together.

    The views share no state, so they all go out at once: an eight-view article
    costs one request's latency rather than eight. req_sem holds the total
    in-flight count at --concurrency regardless.
    """
    rel = src.relative_to(SOURCE_DIR).as_posix()

    def fail(view: View, error: str) -> None:
        stats.view(view.name).failed += 1
        ledger = ledgers[view.name]
        ledger.write(json.dumps({"path": rel, "error": error}) + "\n")
        ledger.flush()

    async with sem:
        if abort.is_set():
            bar.update(1)
            return
        try:
            try:
                doc = pc.split_document(src, src.read_text(encoding="utf-8"))
            except OSError as e:
                # Unreadable source: every view of it fails, and each has to say
                # so in its own ledger or a rerun would retry it forever.
                stats.note_error("*", type(e).__name__)
                for view in views:
                    fail(view, f"{type(e).__name__}: {e}")
                return

            jobs: list[tuple[View, str]] = []
            for view in views:
                if eligible(doc, view):
                    stats.view(view.name).skipped += 1
                    continue
                jobs.append((view, pack_input(doc, view)))

            results = await asyncio.gather(*(
                generate(client, view, doc, packed, view.model or model, stats,
                         req_sem)
                for view, packed in jobs
            ), return_exceptions=True)

            for (view, _), result in zip(jobs, results):
                if isinstance(result, pc.FatalAPIError):
                    # Account/config fault: stop the run, blame no article.
                    if not abort.is_set():
                        stats.fatal = str(result)
                        abort.set()
                elif isinstance(result, AugmentError):
                    fail(view, str(result))
                elif isinstance(result, BaseException):
                    # Cancellation and anything unforeseen: this view only.
                    stats.note_error(view.name, type(result).__name__)
                    fail(view, f"{type(result).__name__}: {result}")
                else:
                    try:
                        pc.atomic_write(
                            output_path_for(view, src),
                            build_output(view, doc, result, view.model or model))
                    except OSError as e:
                        stats.note_error(view.name, type(e).__name__)
                        fail(view, f"{type(e).__name__}: {e}")
                        continue
                    vs = stats.view(view.name)
                    vs.done += 1
                    vs.out_chars += len(result)
        finally:
            bar.update(1)
            bar.set_postfix(docs=stats.done, fail=stats.failed, refresh=False)


async def run(work: list[tuple[Path, list[View]]], views: list[View], args,
              stats: Stats) -> None:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        sys.exit("DEEPSEEK_API_KEY is not set. export it and rerun.")

    req_sem = asyncio.Semaphore(args.concurrency)
    # Articles may run ahead of requests: one whose views are all ineligible
    # costs no request at all and must not idle a slot.
    sem = asyncio.Semaphore(args.concurrency * 2)
    abort = asyncio.Event()
    limits = httpx.Limits(max_connections=args.concurrency + 4,
                          max_keepalive_connections=args.concurrency)

    with contextlib.ExitStack() as stack:
        ledgers = {}
        for view in views:
            view.state_dir.mkdir(parents=True, exist_ok=True)
            ledgers[view.name] = stack.enter_context(
                (view.state_dir / "failures.jsonl").open("a", encoding="utf-8"))

        async with httpx.AsyncClient(
            base_url=API_BASE,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            timeout=httpx.Timeout(args.timeout, connect=15.0),
            limits=limits,
        ) as client:
            with tqdm(total=len(work), desc="augmenting", unit="art") as bar:
                await asyncio.gather(*(
                    augment_article(src, todo, client, sem, req_sem, args.model,
                                    stats, ledgers, bar, abort)
                    for src, todo in work
                ))


# ---- Estimate ----------------------------------------------------------------

def estimate(paths: list[Path], views: list[View], args) -> None:
    """--dry-run: size the job without spending anything."""
    counts = {v.name: 0 for v in views}
    in_chars = {v.name: 0 for v in views}
    skips: dict[str, dict[str, int]] = {v.name: {} for v in views}

    for path in tqdm(paths, desc="measuring", unit="file"):
        doc = pc.split_document(path, path.read_text(encoding="utf-8"))
        body = strip_tables(doc.body)
        for view in views:
            if reason := eligible(doc, view, body):
                skips[view.name][reason] = skips[view.name].get(reason, 0) + 1
                continue
            counts[view.name] += 1
            in_chars[view.name] += len(pack_input(doc, view))

    rows = []
    total_in = total_out = total_cached = total_cost = 0.0
    for view in views:
        n = counts[view.name]
        body_tokens = in_chars[view.name] / CHARS_PER_TOKEN
        # The system prompt is an identical prefix within a view, so after the
        # first few requests it bills at the cache-hit rate.
        cached = n * len(view.system) / CHARS_PER_TOKEN
        tok_in = body_tokens + cached + n * 20
        tok_out = in_chars[view.name] * view.est_out_ratio / CHARS_PER_TOKEN
        if view.thinking:
            tok_out *= 5.6      # measured in paraphrase_corpus.py
        cost = ((tok_in - cached) / 1e6 * args.price_in
                + cached / 1e6 * args.price_cached
                + tok_out / 1e6 * args.price_out)
        rows.append((view.name, n, tok_in, tok_out, cost))
        total_in += tok_in
        total_out += tok_out
        total_cached += cached
        total_cost += cost

    width = max(len(r[0]) for r in rows)
    print(f"\narticles scanned   {len(paths):,}\n")
    print(f"{'view':<{width}}  {'eligible':>9}  {'tok in':>9}  {'tok out':>9}  {'cost':>9}")
    for name, n, tin, tout, cost in rows:
        print(f"{name:<{width}}  {n:>9,}  {tin/1e6:>8.1f}M  {tout/1e6:>8.1f}M  "
              f"${cost:>8,.2f}")
    print(f"{'TOTAL':<{width}}  {sum(r[1] for r in rows):>9,}  {total_in/1e6:>8.1f}M  "
          f"{total_out/1e6:>8.1f}M  ${total_cost:>8,.2f}")
    print(f"\n(of the input, {total_cached/1e6:.1f}M is cacheable system prompt)")

    for view in views:
        if reasons := skips[view.name]:
            detail = ", ".join(f"{r} {n:,}" for r, n in sorted(reasons.items()))
            print(f"  {view.name} skipped: {detail}")

    requests = sum(r[1] for r in rows)
    eta = requests / max(args.concurrency, 1) * 6.0 / 3600
    print(f"\nrequests           {requests:,}")
    print(f"est. wall clock ~  {eta:,.1f} h at {args.concurrency} concurrent, "
          f"6 s/request")
    print("\nPrices are the deepseek-v4-flash rate card as of 2026-08. Pass "
          "--price-in/--price-out/--price-cached if it has moved.")


# ---- Main --------------------------------------------------------------------

def list_views(views: list[View]) -> None:
    width = max(len(v.name) for v in views)
    print(f"{'view':<{width}}  {'enabled':>7}  output directory")
    for v in views:
        done = len(list(v.directory.rglob("*.md"))) if v.directory.exists() else 0
        print(f"{v.name:<{width}}  {'yes' if v.enabled else 'no':>7}  "
              f"corpus/{v.out_dir}  ({done:,} written)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=CONFIG_FILE,
                    help=f"view definitions (default: {CONFIG_FILE.name})")
    ap.add_argument("--views", help="comma-separated view names "
                                    "(default: every enabled view)")
    ap.add_argument("--list-views", action="store_true",
                    help="show configured views and how many docs each has")
    ap.add_argument("--model", default=pc.DEFAULT_MODEL,
                    help=f"DeepSeek model id (default: {pc.DEFAULT_MODEL}); a "
                         f"view may override it")
    ap.add_argument("--concurrency", type=int, default=16,
                    help="in-flight requests (default: 16)")
    ap.add_argument("--limit", type=int,
                    help="process the first N articles (sorted; mostly stubs)")
    ap.add_argument("--sample", type=int,
                    help="process N articles drawn from across the whole "
                         "corpus -- prefer this for smoke tests")
    ap.add_argument("--seed", type=int, default=1977,
                    help="seed for --sample (default: 1977)")
    ap.add_argument("--timeout", type=float, default=240.0,
                    help="per-request timeout")
    ap.add_argument("--force", action="store_true",
                    help="rewrite (article, view) pairs that already exist")
    ap.add_argument("--retry-failed", action="store_true",
                    help="also retry pairs recorded in the failure ledgers")
    ap.add_argument("--dry-run", action="store_true",
                    help="estimate tokens, cost and runtime, then exit")
    ap.add_argument("--price-in", type=float, default=DEFAULT_PRICE_IN,
                    help=f"$/1M cache-miss input (default: {DEFAULT_PRICE_IN})")
    ap.add_argument("--price-out", type=float, default=DEFAULT_PRICE_OUT,
                    help=f"$/1M output (default: {DEFAULT_PRICE_OUT})")
    ap.add_argument("--price-cached", type=float, default=DEFAULT_PRICE_CACHED,
                    help=f"$/1M cache-hit input (default: {DEFAULT_PRICE_CACHED})")
    args = ap.parse_args()

    names = [n.strip() for n in args.views.split(",") if n.strip()] if args.views else None
    try:
        views = load_views(args.config, names)
    except ConfigError as e:
        sys.exit(f"{args.config.name}: {e}")

    if args.list_views:
        list_views(parse_views(args.config))
        return
    if not views:
        sys.exit("no views selected (all disabled?). See --list-views.")
    if not SOURCE_DIR.exists():
        sys.exit(f"No corpus at {SOURCE_DIR}. Run download_wookieepedia.py first.")

    paths = iter_sources(args.limit, args.sample, args.seed)
    if not paths:
        sys.exit(f"No .md files under {SOURCE_DIR}.")

    if args.dry_run:
        estimate(paths, views, args)
        return

    # Per (article, view): an article keeps only the views it still owes. This
    # is also the resume path -- a rerun rebuilds it and finds most pairs done.
    failures = {v.name: (set() if args.retry_failed else load_failures(v))
                for v in views}
    work: list[tuple[Path, list[View]]] = []
    pairs = 0
    for src in paths:
        rel = src.relative_to(SOURCE_DIR).as_posix()
        todo = [v for v in views
                if (args.force or not output_path_for(v, src).exists())
                and rel not in failures[v.name]]
        if todo:
            work.append((src, todo))
            pairs += len(todo)

    total = len(paths) * len(views)
    print(f"{len(paths):,} articles x {len(views)} view(s) = {total:,} pairs; "
          f"{pairs:,} to do ({total - pairs:,} already written or previously "
          f"failed)")
    if not work:
        return

    stats = Stats()
    started = time.time()
    try:
        asyncio.run(run(work, views, args, stats))
    except KeyboardInterrupt:
        print("\ninterrupted -- rerun the same command to resume")

    elapsed = time.time() - started
    if stats.fatal:
        print(f"\nRUN ABORTED -- {stats.fatal}")
        print("This is an account/config fault, not a problem with any article.")
        print("Nothing was written to the failure ledgers. Fix it and rerun:")
        print("  402 Insufficient Balance -> top up at platform.deepseek.com")
        print("  401/403                  -> check DEEPSEEK_API_KEY")
        print("  404                      -> wrong --model id")
        print(f"\n{stats.done:,} documents written before the abort.")
        sys.exit(1)

    miss = stats.tokens_in - stats.tokens_cached
    cost = (miss / 1e6 * args.price_in
            + stats.tokens_cached / 1e6 * args.price_cached
            + stats.tokens_out / 1e6 * args.price_out)

    print()
    width = max(len(v.name) for v in views)
    print(f"{'view':<{width}}  {'written':>8}  {'skipped':>8}  {'failed':>7}  "
          f"{'avg chars':>9}")
    for view in views:
        vs = stats.view(view.name)
        avg = vs.out_chars / vs.done if vs.done else 0
        print(f"{view.name:<{width}}  {vs.done:>8,}  {vs.skipped:>8,}  "
              f"{vs.failed:>7,}  {avg:>9,.0f}")
    print(f"{'TOTAL':<{width}}  {stats.done:>8,}  {stats.skipped:>8,}  "
          f"{stats.failed:>7,}")

    print(f"\nrequests {stats.requests:,} (retries {stats.retries:,}) | "
          f"tokens {stats.tokens_in:,} in / {stats.tokens_out:,} out "
          f"| ~${cost:,.2f}")
    if stats.tokens_in:
        print(f"  input cache hits {stats.tokens_cached/stats.tokens_in:.0%} "
              f"({stats.tokens_cached:,} of {stats.tokens_in:,})")
    if stats.tokens_reasoning:
        share = stats.tokens_reasoning / max(stats.tokens_out, 1)
        print(f"  reasoning tokens {stats.tokens_reasoning:,} ({share:.0%} of "
              f"output, ~${stats.tokens_reasoning/1e6*args.price_out:,.2f} "
              f"discarded) -- set thinking = false in {args.config.name}")
    print(f"elapsed {elapsed/60:.1f} min -> corpus/")
    if stats.errors:
        top = sorted(stats.errors.items(), key=lambda kv: -kv[1])[:8]
        print("top failure reasons: " + ", ".join(f"{k} x{v}" for k, v in top))
    if stats.failed:
        print(f"failures logged under {STATE_DIR}/<view>/failures.jsonl "
              f"(rerun with --retry-failed)")


if __name__ == "__main__":
    main()
