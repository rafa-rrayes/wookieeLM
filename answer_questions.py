#!/usr/bin/env python3
"""Answer the generated questions with the tool-using chatbot -> SFT corpus.

Reads ``questions/questions.jsonl`` (written by ``generate_questions.py``) and
routes every question through ``wookiee_chat.py``'s agentic loop, which answers
only from ``corpus/wookieepedia``. One JSON object per answered question lands
in ``sft/sft.jsonl``:

    {"q": "How tall was Ahsoka Tano as an adult?",
     "a": "1.7 metres.",
     "type": "numeric",
     "source": "wookieepedia/A/Ahsoka_Tano.md",
     "continuity": "canon",
     "split": "train",
     "read": ["Ahsoka Tano"],
     "tools": 1,
     "model": "deepseek-v4-flash"}

The training pair is ``(q, a)``. Everything else is provenance: which articles
the answer was actually built from, how much work it took, and which model
said it. ``--keep-trace`` additionally stores the whole message list, for
training tool use rather than recall.

Why the teacher uses tools and the student will not
---------------------------------------------------
This is a distillation setup. The teacher is told its tools *are* its memory
and that the reader must never learn they exist, so its answers read as recall
and carry no citations -- which is exactly the shape a student trained on the
corpus should produce. The tools are how the teacher is kept honest, not part
of what is being taught.

Seeding the source article
--------------------------
Every question records the article it was drawn from, and (since the evidence
rewrite) a span of that article that answers it. Making the model rediscover
that article by search costs three to six tool rounds per question and
sometimes fails. So the conversation is seeded with a ``read_article`` call the
model did not make, whose result is the real text of the recorded source. The
model starts from the article that provably answers the question and spends its
remaining budget only on what that article does not cover. ``--no-seed``
restores blind search.

Nothing is fabricated by this: the seeded tool result is corpus text, produced
by the same code path the real tool uses. What is fabricated is the *decision*
to read that article -- which is why, in a kept trace, it reads as a model that
searched well.

Quality gates
-------------
An answer is rejected and re-asked when it:

    leaks the persona    "the article says", "according to my search", a
                         trailing "Sources:" block. A leaked reference in the
                         training data teaches the student to cite sources it
                         will not have.
    used no tools        zero tool calls means the answer came from the
                         teacher's own weights, unverified. That is the one
                         thing the whole design exists to prevent.
    came back empty
    declines the fact    "that was never established" is a correct answer in
                         general, but not here: the question was written from
                         a verified span of a named article, so a refusal is a
                         retrieval failure, and training on it teaches the
                         student to decline things it should know.

A fifth check -- every proper noun and number in the answer must occur in text
the model actually read -- is *recorded but not enforced*, because
``tasks/lessons.md`` says a new validator gets measured against real output
before it is allowed to delete anything. Run a sample, read the
``unsupported`` field, then turn on ``--reject-unsupported``.

Requirements:
    DEEPSEEK_API_KEY in the environment (or a .env file beside this script)
    ~400 MB of disk under .index/ for the full-text index that backs
    search_text -- built once, then mmapped by the scan pool (corpus_scan.py)

Usage:
    uv run answer_questions.py --sample 200 --dry-run   # cost estimate only
    uv run answer_questions.py --sample 200             # smoke test
    uv run answer_questions.py --continuity canon,legends,non-canon
    uv run answer_questions.py                          # everything

Resume: rerun the same command. Questions already present in the output are
skipped, as are ones recorded in ``sft_state/failures.jsonl``.
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
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
CORPUS_DIR = REPO_ROOT / "corpus"
QUESTION_FILE = REPO_ROOT / "questions" / "questions.jsonl"
OUTPUT_DIR = REPO_ROOT / "sft"
OUTPUT_FILE = OUTPUT_DIR / "sft.jsonl"
STATE_DIR = REPO_ROOT / "sft_state"

API_BASE = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEFAULT_MODEL = "deepseek-v4-flash"

MAX_ROUNDS = 6              # tool rounds per question; the seed pays for itself
MAX_ATTEMPTS = 3            # transport/429/5xx retries, and re-asks after a gate
CHARS_PER_TOKEN = 3.6       # matches paraphrase_corpus.py, for --dry-run
ANSWER_CHARS = 600          # --dry-run only: measured answers run 300-900
EXTRA_ROUND_FACTOR = 0.6    # --dry-run only: rounds beyond the seeded one, as a
                            # fraction of questions. A guess until a sample run
                            # reports the real number -- which it does.

MIN_ANSWER_CHARS = 2
MAX_ANSWER_CHARS = 4000     # a "recall" answer this long is reciting the article

# Leave the event loop and the tool threads a core each; everything else scans.
DEFAULT_SEARCH_WORKERS = max(1, (os.cpu_count() or 4) - 2)


# ---- Reuse of wookiee_chat.py and generate_questions.py ----------------------

def load_module(name: str, filename: str):
    """Import a sibling script by path, as count.py and generate_questions.py do.

    The corpus index, the three tools, the system prompt and the evidence
    checker are all written and audited elsewhere in this repo. Re-implementing
    any of them here would let this script and the chatbot drift into
    disagreeing about what the chatbot does.
    """
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


wc = load_module("wc", "wookiee_chat.py")       # Corpus, TOOLS, run_tool, prompt
gq = load_module("gq", "generate_questions.py")  # phrase checker, split hashing
pc = gq.pc                                       # backoff, prices, fatal statuses


# Appended to the system prompt when a first answer broke character. Cheaper
# than a correcting turn, and it keeps a kept trace clean: the conversation is
# started over, so the transcript never contains the scolding.
HARDENED = """

You have just broken character. Say nothing about articles, entries, sources, \
searching or reading. Do not append a sources or citations line. State the \
fact as something you have always known.\
"""


# ---- Answer gates -----------------------------------------------------------

# Written narrowly on purpose. The failure being caught is the model framing an
# answer as retrieved; the failure to avoid causing is rejecting in-universe
# prose, where "the Jedi archives", "no record survives" and "according to Sith
# tradition" are all ordinary things to write. So each pattern names the
# machinery ("this article", "my search") rather than a bare word.
#
# Every pattern below was replayed over 40,231 real sentences of corpus prose
# and the hits were read one at a time, because tasks/lessons.md is explicit
# about what an unmeasured validator does to good work. Three were narrowed as
# a result and are marked; the surviving false-positive rate is ~0.2%, almost
# all of it Wookieepedia's own editorial asides ("This article assumes..."),
# which an answer should never be echoing anyway.
_LEAK_PATTERNS = [
    r"\bwookieepedia\b",
    r"\bwikipedia\b",
    # (measured) The negative lookahead spares a real citation of a published
    # piece -- `the article "Star Systems of the Galaxy"` on StarWars.com is a
    # legitimate answer to a production question, where `the article says` is
    # the model narrating its own retrieval.
    r"\b(?:the|this|that|our|its)\s+(?:article|entry|excerpt|passage|corpus|"
    r"database|dataset|wiki)\b(?!\s*[\"“*'])",
    # No in-universe reading at all -- "infobox" is a wiki structure the
    # model can only know by having looked at one. 4 of the first 988
    # answers said it ("Based on the infobox, the Imperial forces...").
    r"\binfobox\b",
    # "record", "text" and "archive" are deliberately absent from this noun
    # list: "Imperial records list it as decommissioned", "the ancient Sith
    # texts describe", "the Jedi archives note" are all ordinary in-universe
    # prose, and rejecting them would delete good answers to catch a phrasing
    # the nouns below already cover.
    r"\b(?:article|entry|passage|excerpt)s?\s+"
    r"(?:says?|states?|notes?|mentions?|indicates?|reads?|describes?|lists?|"
    r"do(?:es)?\s+not|don't|doesn't)\b",
    # A bare "Imperial sources say" is in-universe; "the sources say" is the
    # model narrating its retrieval, which the system prompt bans by name.
    r"\b(?:the|my|our)\s+sources?\s+"
    r"(?:says?|states?|notes?|mentions?|indicates?|do(?:es)?\s+not|don't)\b",
    r"\baccording to (?:the |my |our )?(?:article|entry|passage|text|source|"
    r"excerpt|record|available|what)\b",
    r"\bbased on (?:the |my |what )?(?:article|entry|passage|text|search|"
    r"source|information|data|I (?:found|read))\b",
    r"\bI (?:searched|looked (?:it |this )?up|could not find|couldn't find|"
    r"was unable to find|have no (?:record|information|data))\b",
    # (measured) "the search" and "the tool" are ordinary in-universe nouns --
    # the search for a missing Jedi, a mechanic's tool. Only the first-person
    # possessive names the machinery.
    r"\bmy (?:search|tools?|memory banks?|training data|knowledge base)\b",
    r"\bthe available (?:information|data|sources?)\b",
    # (measured) "search for" is ordinary English -- "snuck off to search for
    # new creatures" -- and was the single largest source of false positives.
    r"\bsearch(?:ing)? (?:results?|the corpus|turned up)\b",
    r"\bas an AI\b",
    # (measured) Caught by the grounding check on the first 1,000 answers and
    # missed here: "That was an accidental search on my part -- let me address
    # your actual question." Both halves are the model narrating its own
    # process, and the second is the preamble the system prompt bans outright.
    r"\bon my part\b",
    r"\blet me (?:address|answer|correct|clarify|rephrase|start over|try again)\b",
    r"^\s*sources?\s*:",
    r"\[\d+\]",                       # footnote markers copied out of the wiki
]
_LEAK_RE = re.compile("|".join(_LEAK_PATTERNS), re.IGNORECASE | re.MULTILINE)

# A trailing citations block is a verbatim, deterministically removable token --
# tasks/lessons.md: prefer a deterministic repair to a retry. Only ever stripped
# from the end, so a mid-answer reference still fails the gate above.
_TRAILING_CITE_RE = re.compile(
    r"\n+\s*(?:\*\*)?sources?(?:\*\*)?\s*:.*\Z", re.IGNORECASE | re.DOTALL)

# The teacher declining a fact. Every question here was written from a verified
# span of a named article, so this is a retrieval failure wearing the costume of
# a correct answer -- and the costume is why it has to be caught by name.
#
# Only applied to a *short* answer (DECLINE_MAX_CHARS), because the corpus
# itself says "whether all of these participated is not known" and "his fate
# remains unknown" -- an answer that delivers the fact and notes an unrecorded
# corner of it is a good answer, not a refusal. Measured at 0.035% of corpus
# sentences, so the phrase alone is nearly safe; the length condition makes it
# safe in the direction that matters, which is keeping good work.
DECLINE_MAX_CHARS = 240

_DECLINE_RE = re.compile(
    r"\b(?:never (?:been )?(?:established|recorded|specified|revealed|stated|"
    r"detailed|named)"
    r"|(?:was|is|has) not (?:been )?(?:established|recorded|specified|revealed|"
    r"documented)"
    r"|no (?:such )?(?:detail|record|account|information) (?:was|is|has|exists)"
    r"|remains? unknown|isn't (?:known|recorded)|is not known)\b",
    re.IGNORECASE,
)


# The tool loop leaking out of the front of the answer: "I have what I need.",
# "Let me give you the answer.", "That was an accidental search on my part.".
# 40 of the first 988 answers (4.0%) opened on one of these. They are pure
# process narration -- the fact starts in the next sentence -- so this is the
# same deterministic repair as the citations block rather than a gate: rejecting
# the answer would throw away a good one over its first eight words.
#
# Every alternative below has to consume a whole leading sentence, and the
# result is only kept if a real answer is left behind (MIN_KEPT_CHARS).
_PREAMBLE_RE = re.compile(
    r"\A\s*(?:"
    r"(?:okay|alright|all right|right|good|great|perfect)[,.!—-]+\s*"
    r"|(?:you(?:'re| are) right|that was an accidental|apolog\w+)"
    r"[^.!?\n]*[.!—]\s*"
    r"|I(?:'ve| have| now have| can confirm| see| understand)"
    r"[^.!?\n]*[.!—]\s*"
    r"|now I have[^.!?\n]*[.!—]\s*"
    r"|let me [^.!?\n]*[.!:—]\s*"
    r")+",
    re.IGNORECASE,
)
MIN_KEPT_CHARS = 80


def strip_preamble(answer: str) -> tuple[str, bool]:
    """-> (answer without a leading tool-loop preamble, whether one was cut)."""
    kept = _PREAMBLE_RE.sub("", answer).lstrip()
    if kept != answer and len(kept) >= MIN_KEPT_CHARS:
        return kept, True
    return answer, False


def strip_citations(answer: str) -> tuple[str, bool]:
    """-> (answer without a trailing sources block, whether one was removed)."""
    repaired = _TRAILING_CITE_RE.sub("", answer).strip()
    return (repaired, True) if repaired != answer.strip() else (answer.strip(), False)


def gate(answer: str, tool_calls: int, args) -> str | None:
    """Why this answer must not enter the corpus, or None to keep it."""
    if not answer or len(answer) < MIN_ANSWER_CHARS:
        return "empty answer"
    if len(answer) > args.max_answer_chars:
        return "answer too long"
    if tool_calls == 0 and not args.allow_untooled:
        return "no tools used"
    if m := _LEAK_RE.search(answer):
        # The matched phrase rides along so the run summary can show which
        # pattern is doing the work, and whether it is doing too much.
        return f"persona leak: {m.group(0).strip().lower()[:40]}"
    if len(answer) <= args.decline_max_chars and _DECLINE_RE.search(answer):
        return "declined a recorded fact"
    return None


# Words that open a sentence rather than name anything. A question is one
# sentence, so generate_questions.py handles this by skipping a single leading
# character; an answer is five or six sentences, and every one of them starts
# on a capital. Measured on the first 988 answers, this was the entire top of
# the flag list -- "This" alone accounted for 141 of 846 flags -- and the
# greedy phrase matcher also glued openers onto the name beside them ("When
# Crosshair", "After Order", "But the Empire").
_GRAMMAR_WORDS = frozenset("""
this that these those it its he she they them his her their there here who
what when where why how which while after before during since because but and
or so yes no not both either neither however although though instead meanwhile
afterward afterwards later then also still first second third finally
notably crucially interestingly specifically importantly essentially
ultimately eventually initially originally later once given based beyond
across between within without under over into onto upon about
i'm i've i'd i'll we we're they're it's he's she's there's that's you your
rather indeed likewise besides moreover furthermore nonetheless regardless
let by in on at to for with as from of the a an
""".split())

# Vocabulary the system prompt requires the model to produce and the article
# cannot supply: an article's frontmatter says `continuity: legends`, never
# "the Legends continuity", but instruction 5 asks for exactly that phrasing.
# Flagging the model for obeying its instructions is the checker's fault.
_LICENSED = ("legends legends-era legends-continuity canon canon-continuity "
             "non-canon noncanon real-world expanded universe star wars "
             "eu continuity timeline")


def _trim_grammar(phrase: str, extra: frozenset[str]) -> str:
    """Strip grammar and bare rank words from both ends of a captured phrase.

    The same move generate_questions.py makes with the article title, and made
    for the same reason: `_PHRASE_RE` is deliberately greedy so that "Relay
    Station Epsilon One" is tested as one unit, and the price is that it also
    swallows whatever capitalised word happens to sit next to a name. Trimming
    only at the ends, and only words that cannot themselves be a name, leaves a
    corrupted middle visible.
    """
    words = phrase.split()
    drop = _GRAMMAR_WORDS | extra
    while words and words[0].casefold() in drop:
        words.pop(0)
    while words and words[-1].casefold() in drop:
        words.pop()
    return " ".join(words)


def unsupported_terms(answer: str, question: str, title: str,
                      read_text: str) -> list[str]:
    """Names and numbers asserted by the answer that occur in nothing it read.

    Reuses generate_questions.py's phrase checker with the window and the
    passage set to the same text: everything the model actually retrieved, plus
    the question itself, since a name the question supplies was already
    evidence-checked when the question was written.

    Recorded, not enforced (see --reject-unsupported). It is a new validator on
    unseen output, and tasks/lessons.md is unambiguous about what those do to
    good work before they have been measured.
    """
    haystack = gq.normalize_for_match(read_text + "\n" + question + "\n" + _LICENSED)
    # significant_terms() skips the first character, because a question always
    # opens on a capital that is grammar rather than a name. An answer does
    # not -- "Thrawn destroyed it" opens on the very name most worth checking
    # -- so the leading space takes the skip instead.
    phrases, numbers, title_words = gq.significant_terms(" " + answer, title)
    bad = []
    for phrase in phrases:
        # "Hutt Wallanooga" for the article's "Wallanooga the Hutt": a rank or
        # species used as a title is not the name, and the two orders are both
        # ordinary English.
        trimmed = _trim_grammar(phrase, gq._ROLE_WORDS)
        if not trimmed:
            continue                    # nothing but grammar and rank words
        if gq.check_phrase(trimmed, haystack, haystack, title_words):
            bad.append(trimmed)
    for num in numbers:
        if not (gq.occurs_in(num, haystack)
                or gq.occurs_in(num.replace(",", ""), haystack)):
            bad.append(num)
    # Deduplicated, order preserved: an answer that repeats a bad name is one
    # fault, not three.
    return list(dict.fromkeys(bad))


# ---- Tools ------------------------------------------------------------------

class ToolRunner:
    """The chatbot's tools, made safe to call from an event loop.

    Two things the interactive script does not have to worry about:

    - every tool blocks. ``search_text`` scans 365 MB of article text; both it
      and ``search_titles`` used to hold the GIL for the whole scan, which is
      what pinned this script to a single core. The scan now runs in a process
      pool (``corpus_scan.SearchPool``), so the thread this waits in is
      blocked on IPC and the GIL is free. The semaphore bounds how many scans
      are in flight, and so how deep the pool's queue can grow.
    - the same call arrives over and over. Questions are processed grouped by
      source article, so neighbouring workers ask for the same article and run
      the same searches. Results are memoised, and an in-flight call is awaited
      rather than duplicated.
    """

    def __init__(self, corpus, search_concurrency: int, cache_size: int):
        self.corpus = corpus
        self.search_sem = asyncio.Semaphore(max(1, search_concurrency))
        # LRU. The previous dict simply stopped inserting once it reached
        # `cache_size` and never evicted, so on a 179k-question run it froze
        # holding the first few hundred articles' searches and every question
        # after that missed -- which put the full cost of the scan back on
        # every call, for the whole rest of the run.
        self.cache: OrderedDict[str, str] = OrderedDict()
        self.inflight: dict[str, asyncio.Future] = {}
        self.cache_size = max(1, cache_size)
        self.hits = 0
        self.calls = 0

    async def run(self, name: str, raw_args: str) -> str:
        self.calls += 1
        try:                       # canonicalised so key order cannot miss
            key = f"{name}:{json.dumps(json.loads(raw_args or '{}'), sort_keys=True)}"
        except json.JSONDecodeError:
            key = f"{name}:{raw_args}"

        if (cached := self.cache.get(key)) is not None:
            self.cache.move_to_end(key)
            self.hits += 1
            return cached
        if (pending := self.inflight.get(key)) is not None:
            self.hits += 1
            return await asyncio.shield(pending)

        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self.inflight[key] = fut
        # Only the full-corpus scan is throttled; a title lookup or an article
        # read is a millisecond of work that should not queue behind it.
        guard = self.search_sem if name == "search_text" else contextlib.nullcontext()
        try:
            async with guard:
                result = await asyncio.to_thread(
                    wc.run_tool, self.corpus, name, raw_args)
        except BaseException as e:                              # noqa: BLE001
            self.inflight.pop(key, None)
            if not fut.done():
                fut.set_exception(e)
            fut.exception()          # mark retrieved; nothing else awaits it
            raise
        self.inflight.pop(key, None)
        if not fut.done():
            fut.set_result(result)
        self.cache[key] = result
        self.cache.move_to_end(key)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return result


def seed_messages(corpus, path: Path) -> tuple[list[dict], str, str] | None:
    """The read_article call the model did not make, and its real result.

    Formatted by the same helper the live tool uses, so a kept trace is
    indistinguishable from one where the model asked for the article itself.
    """
    try:
        text = corpus.read(path)
    except OSError:
        return None
    title = corpus.title_of(path)
    result = f"# {title}\n\n" + wc._truncate(text, wc.MAX_ARTICLE_CHARS,
                                             corpus.sections_of(text))
    call_id = "call_seed"
    messages = [
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": call_id, "type": "function",
                         "function": {"name": "read_article",
                                      "arguments": json.dumps({"title": title})}}]},
        {"role": "tool", "tool_call_id": call_id, "content": result},
    ]
    return messages, title, result


async def get_seed(corpus, source: str, seeds: dict) -> tuple | None:
    """``seed_messages`` for one article, built once however many ask for it.

    Off the event loop, because it reads the article off disk and runs two
    regexes over it -- once per article, on the loop every other question's
    response parsing has to share. The future goes into the map *before* the
    first await, so the questions from one article (they are processed
    together, and there are hundreds of workers) queue behind one build
    instead of stampeding it.
    """
    if (pending := seeds.get(source)) is not None:
        return await asyncio.shield(pending)
    fut = asyncio.get_running_loop().create_future()
    seeds[source] = fut
    try:
        fut.set_result(await asyncio.to_thread(
            seed_messages, corpus, CORPUS_DIR / source))
    except BaseException as e:                              # noqa: BLE001
        seeds.pop(source, None)      # let the next question try again
        fut.set_exception(e)
        fut.exception()              # marked retrieved; the raise below carries it
        raise
    return fut.result()


# ---- API --------------------------------------------------------------------

class AnswerError(RuntimeError):
    """Raised when a question could not be answered acceptably."""


@dataclass
class Answer:
    text: str
    read: list[str]        # article titles, in the order they were read
    tool_calls: int
    trace: list[dict]
    repaired: bool = False
    preamble: bool = False
    forced: bool = False


@dataclass
class Stats:
    answered: int = 0
    failed: int = 0
    skipped: int = 0
    requests: int = 0
    retries: int = 0
    reasks: int = 0
    forced: int = 0
    repaired: int = 0
    preambles: int = 0
    unsupported: int = 0
    tool_calls: int = 0
    tool_hits: int = 0
    extra_rounds: int = 0
    answer_chars: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cached: int = 0
    tokens_reasoning: int = 0
    fatal: str | None = None
    rejects: dict[str, int] = field(default_factory=dict)
    errors: dict[str, int] = field(default_factory=dict)
    by_type: dict[str, int] = field(default_factory=dict)

    def note(self, bucket: dict[str, int], key: str) -> None:
        bucket[key] = bucket.get(key, 0) + 1

    def note_usage(self, usage: dict) -> None:
        self.tokens_in += usage.get("prompt_tokens", 0)
        self.tokens_out += usage.get("completion_tokens", 0)
        self.tokens_cached += usage.get("prompt_cache_hit_tokens", 0)
        details = usage.get("completion_tokens_details") or {}
        self.tokens_reasoning += details.get("reasoning_tokens", 0)


async def call_model(client: httpx.AsyncClient, messages: list[dict], args,
                     stats: Stats, req_sem: asyncio.Semaphore) -> dict:
    """One /chat/completions round, retried on transport and transient faults."""
    payload = {
        "model": args.model,
        "messages": messages,
        "tools": wc.TOOLS,
        "tool_choice": "auto",
        # Facts, not variety. The corpus is the source; two runs over the same
        # question should not disagree.
        "temperature": args.temperature,
        "thinking": {"type": "enabled" if args.thinking else "disabled"},
    }

    last = "unknown"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt > 1:
            stats.retries += 1
        try:
            async with req_sem:
                resp = await client.post("/chat/completions", json=payload)
        except (httpx.TransportError, httpx.TimeoutException) as e:
            last = f"transport: {type(e).__name__}"
            await pc._backoff(attempt)
            continue

        if resp.status_code == 429 or resp.status_code >= 500:
            last = f"http {resp.status_code}"
            await pc._backoff(attempt, resp.headers.get("Retry-After"))
            continue
        if resp.status_code in pc.FATAL_STATUSES:
            raise pc.FatalAPIError(f"http {resp.status_code}: {resp.text[:200]}")
        if resp.status_code >= 400:
            raise AnswerError(f"http {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        stats.requests += 1
        if usage := data.get("usage"):
            stats.note_usage(usage)
        return data["choices"][0]["message"]

    raise AnswerError(last)


async def one_attempt(client: httpx.AsyncClient, tools: ToolRunner, question: str,
                      seed: tuple[list[dict], str, str] | None, system: str,
                      args, stats: Stats, req_sem: asyncio.Semaphore) -> Answer:
    """Drive the tool loop for one question until the model stops asking."""
    messages: list[dict] = [{"role": "system", "content": system},
                            {"role": "user", "content": question}]
    read: list[str] = []
    read_text: list[str] = []
    tool_calls = 0

    if seed is not None:
        seeded, title, result = seed
        messages += [dict(m) for m in seeded]
        read.append(title)
        read_text.append(result)
        tool_calls += 1

    for _ in range(args.max_rounds):
        msg = await call_model(client, messages, args, stats, req_sem)
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            answer = (msg.get("content") or "").strip()
            return Answer(answer, read, tool_calls, messages)

        for call in calls:
            name = call["function"]["name"]
            raw_args = call["function"].get("arguments", "")
            result = await tools.run(name, raw_args)
            tool_calls += 1
            read_text.append(result)
            if name == "read_article":
                try:
                    if title := json.loads(raw_args or "{}").get("title"):
                        read.append(title)
                except json.JSONDecodeError:
                    pass
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": result})
        stats.extra_rounds += 1

    # Out of rounds. Make it answer from what it has rather than dropping a
    # question we have already paid for -- and record that it happened, because
    # a run where this is common wants a larger --max-rounds, not more spend.
    messages.append({"role": "user", "content":
                     "Answer now, from what you already recall. State plainly "
                     "anything you cannot confirm."})
    msg = await call_model(client, messages, args, stats, req_sem)
    messages.append(msg)
    answer = (msg.get("content") or "").strip()
    return Answer(answer, read, tool_calls, messages, forced=True)


async def answer_question(client: httpx.AsyncClient, tools: ToolRunner,
                          rec: dict, seed: tuple | None, args, stats: Stats,
                          req_sem: asyncio.Semaphore) -> tuple[Answer, list[str]]:
    """-> (accepted answer, unsupported terms). Re-asks once per failed gate."""
    question = rec["q"]
    system = wc.SYSTEM_PROMPT
    last_reason = "unknown"

    for attempt in range(1, args.max_reasks + 2):
        if attempt > 1:
            stats.reasks += 1
        got = await one_attempt(client, tools, question, seed, system, args,
                                stats, req_sem)
        got.text, got.preamble = strip_preamble(got.text)
        got.text, got.repaired = strip_citations(got.text)
        reason = gate(got.text, got.tool_calls, args)
        if reason is None:
            read_text = "\n".join(m["content"] for m in got.trace
                                  if m.get("role") == "tool")
            title = tools.corpus.title_of(Path(rec["source"]))
            bad = (unsupported_terms(got.text, question, title, read_text)
                   if args.check_grounding else [])
            if bad and args.reject_unsupported:
                last_reason = f"unsupported: {bad[0][:40]}"
                stats.note(stats.rejects, last_reason)
                system = wc.SYSTEM_PROMPT + HARDENED
                continue
            if got.repaired:
                stats.repaired += 1
            if got.preamble:
                stats.preambles += 1
            if got.forced:
                stats.forced += 1
            return got, bad

        stats.note(stats.rejects, reason)
        last_reason = reason
        # Only a persona break is worth re-asking with a hardened prompt; the
        # others are re-asked plain, since a sterner prompt does not make the
        # model reach for a tool it decided it did not need.
        if reason.startswith("persona leak"):
            system = wc.SYSTEM_PROMPT + HARDENED

    raise AnswerError(last_reason)


# ---- Orchestration ----------------------------------------------------------

def load_questions(args) -> list[dict]:
    """Read questions.jsonl, filtered, grouped by source article.

    Grouping is what makes the tool cache pay: every question from one article
    reads the same file, and the neighbouring searches repeat too. Selection
    (--sample, --continuity, --types) happens before grouping so a sample is
    drawn from the whole corpus rather than the first few articles.
    """
    if not QUESTION_FILE.exists():
        sys.exit(f"No questions at {QUESTION_FILE}. Run generate_questions.py "
                 f"first.")

    want_types = {t.strip() for t in args.types.split(",") if t.strip()}
    want_cont = {c.strip() for c in args.continuity.split(",") if c.strip()}
    want_split = {s.strip() for s in args.split.split(",") if s.strip()}

    records: list[dict] = []
    with QUESTION_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # generate_questions.py may still be appending; a torn final
                # line is not an error, it is a file being written.
                continue
            if not rec.get("q") or not rec.get("source"):
                continue
            # The first 1,360 questions predate the train/eval split. Filling
            # it in with the same path hash keeps one article on one side of
            # the boundary, which is the whole point of hashing the path.
            rec.setdefault("split", gq.assign_split(rec["source"],
                                                    args.eval_fraction))
            if want_types and rec.get("type") not in want_types:
                continue
            if want_cont and rec.get("continuity") not in want_cont:
                continue
            if want_split and rec["split"] not in want_split:
                continue
            records.append(rec)

    if args.sample and args.sample < len(records):
        records = random.Random(args.seed).sample(records, args.sample)
    elif args.limit:
        records = records[:args.limit]
    records.sort(key=lambda r: (r["source"], r["q"]))
    return records


def load_done() -> set[str]:
    """Questions already answered. Keyed by the question, which is unique."""
    if not OUTPUT_FILE.exists():
        return set()
    done = set()
    with OUTPUT_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if q := rec.get("q"):
                done.add(gq.dedup_key(q))
    return done


def load_failures() -> set[str]:
    """Questions that genuinely failed. Account-level faults are not the
    question's fault, so they are ignored here exactly as the other scripts
    ignore them."""
    ledger = STATE_DIR / "failures.jsonl"
    if not ledger.exists():
        return set()
    failed = set()
    for line in ledger.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if pc._FATAL_LEDGER_RE.match(rec.get("error", "")):
            continue
        if q := rec.get("q"):
            failed.add(gq.dedup_key(q))
    return failed


async def worker(rec: dict, client: httpx.AsyncClient, tools: ToolRunner,
                 seeds: dict, sem: asyncio.Semaphore, args, stats: Stats,
                 out, ledger, bar: tqdm, abort: asyncio.Event,
                 req_sem: asyncio.Semaphore) -> None:
    async with sem:
        if abort.is_set():
            bar.update(1)
            return
        try:
            seed = None
            if not args.no_seed:
                seed = await get_seed(tools.corpus, rec["source"], seeds)
                if seed is None and args.require_source:
                    stats.skipped += 1
                    return

            got, unsupported = await answer_question(client, tools, rec, seed,
                                                     args, stats, req_sem)

            record = {
                "q": rec["q"],
                "a": got.text,
                "type": rec.get("type", "other"),
                "source": rec["source"],
                "continuity": rec.get("continuity", "unknown"),
                "split": rec["split"],
                "read": got.read,
                "tools": got.tool_calls,
                "model": args.model,
            }
            if unsupported:
                record["unsupported"] = unsupported
                stats.unsupported += 1
            if args.keep_trace:
                record["trace"] = got.trace
            # One complete line per write, with no await in between, so
            # concurrent workers cannot interleave inside a record.
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            stats.answered += 1
            stats.tool_calls += got.tool_calls
            stats.answer_chars += len(got.text)
            stats.note(stats.by_type, record["type"])
        except pc.FatalAPIError as e:
            if not abort.is_set():
                stats.fatal = str(e)
                abort.set()
        except AnswerError as e:
            stats.failed += 1
            stats.note(stats.errors, str(e).split(":")[0])
            ledger.write(json.dumps({"q": rec["q"], "source": rec["source"],
                                     "error": str(e)}, ensure_ascii=False) + "\n")
            ledger.flush()
        except Exception as e:  # noqa: BLE001 - one bad question must not stop a run
            stats.failed += 1
            stats.note(stats.errors, type(e).__name__)
            ledger.write(json.dumps({"q": rec["q"], "source": rec["source"],
                                     "error": f"{type(e).__name__}: {e}"},
                                    ensure_ascii=False) + "\n")
            ledger.flush()
        finally:
            bar.update(1)
            bar.set_postfix(ok=stats.answered, fail=stats.failed,
                            reask=stats.reasks, refresh=False)


def estimate(records: list[dict], args) -> None:
    """--dry-run: size the job without spending anything."""
    system_chars = len(wc.SYSTEM_PROMPT) + sum(len(json.dumps(t)) for t in wc.TOOLS)
    seed_chars = 0
    missing = 0
    sizes: dict[str, int] = {}
    for rec in tqdm(records, desc="measuring", unit="q"):
        src = rec["source"]
        if src not in sizes:
            path = CORPUS_DIR / src
            try:
                sizes[src] = min(path.stat().st_size, wc.MAX_ARTICLE_CHARS)
            except OSError:
                sizes[src] = 0
        if not sizes[src]:
            missing += 1
        seed_chars += 0 if args.no_seed else sizes[src]

    n = len(records)
    q_chars = sum(len(r["q"]) for r in records)
    base_in = (n * system_chars + seed_chars + q_chars) / CHARS_PER_TOKEN
    # A round beyond the seeded one re-sends the whole conversation and adds a
    # tool result, so it costs about a whole request again plus the result.
    extra = EXTRA_ROUND_FACTOR * (base_in + n * wc.MAX_TOOL_CHARS / CHARS_PER_TOKEN)
    tok_in = base_in + extra
    cached = n * system_chars / CHARS_PER_TOKEN     # the shared prefix, at best
    tok_out = n * (1 + EXTRA_ROUND_FACTOR) * ANSWER_CHARS / CHARS_PER_TOKEN
    if args.thinking:
        tok_out *= 5.6          # measured on the paraphrase run
    cost = ((tok_in - cached) / 1e6 * args.price_in
            + cached / 1e6 * args.price_cached
            + tok_out / 1e6 * args.price_out)

    print(f"\nquestions          {n:,}")
    print(f"  source articles  {len(sizes):,}"
          f"{f'  ({missing:,} questions cite a missing file)' if missing else ''}")
    print(f"seeded article     {'off (blind search)' if args.no_seed else 'on'}"
          f"  -- {seed_chars/1e6:.1f}M chars of corpus text sent")
    print(f"input tokens  ~    {tok_in/1e6:.1f}M  (of which {cached/1e6:.1f}M "
          f"cacheable system prompt)")
    print(f"output tokens ~    {tok_out/1e6:.1f}M"
          f"{'  THINKING ON -- ~82% discarded' if args.thinking else ''}")
    print(f"est. cost     ~    ${cost:,.2f}   (at ${args.price_in}/${args.price_out} "
          f"per 1M in/out, ${args.price_cached} cached in)")
    eta = n * (1 + EXTRA_ROUND_FACTOR) / max(args.concurrency, 1) * 4.0 / 3600
    print(f"est. wall clock ~  {eta:.1f} h at {args.concurrency} concurrent, "
          f"4 s/request")
    print(f"\nassumes {EXTRA_ROUND_FACTOR:.0%} of questions need one tool round "
          f"beyond the seed.\nRun --sample 200 first: the summary reports the "
          f"real figure.")


async def run(records: list[dict], corpus, key: str, args, stats: Stats) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    req_sem = asyncio.Semaphore(args.concurrency)
    sem = asyncio.Semaphore(args.concurrency * 2)
    abort = asyncio.Event()
    tools = ToolRunner(corpus, args.search_concurrency, args.cache_size)
    seeds: dict[str, tuple | None] = {}
    limits = httpx.Limits(max_connections=args.concurrency + 4,
                          max_keepalive_connections=args.concurrency)

    async with httpx.AsyncClient(
        base_url=API_BASE,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
        timeout=httpx.Timeout(args.timeout, connect=15.0),
        limits=limits,
    ) as client:
        mode = "a" if not args.overwrite else "w"
        with OUTPUT_FILE.open(mode, encoding="utf-8") as out, \
             (STATE_DIR / "failures.jsonl").open("a", encoding="utf-8") as ledger:
            with tqdm(total=len(records), desc="answering", unit="q") as bar:
                await asyncio.gather(*(
                    worker(rec, client, tools, seeds, sem, args, stats, out,
                           ledger, bar, abort, req_sem)
                    for rec in records
                ))
    stats.tool_hits = tools.hits


# ---- Main -------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=wc.DEFAULT_CORPUS,
                    help=f"corpus directory (default: {wc.DEFAULT_CORPUS})")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"DeepSeek model id (default: {DEFAULT_MODEL})")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="in-flight requests (default: 8)")
    ap.add_argument("--search-workers", type=int, default=DEFAULT_SEARCH_WORKERS,
                    help=f"processes that scan the corpus for search_text "
                         f"(default: {DEFAULT_SEARCH_WORKERS}, from this "
                         f"machine's core count). The scan holds the GIL, so "
                         f"this is where the parallelism actually comes from; "
                         f"0 scans in-process, as the chat does")
    ap.add_argument("--search-concurrency", type=int, default=8,
                    help="scans allowed in flight at once (default: 8). Bounds "
                         "how deep the queue behind the search workers grows")
    ap.add_argument("--rebuild-index", action="store_true",
                    help="rebuild the cached full-text index instead of reusing "
                         "the one under .index/ (needed only after editing "
                         "articles in place)")
    ap.add_argument("--limit", type=int,
                    help="answer the first N questions (grouped by article)")
    ap.add_argument("--sample", type=int,
                    help="answer N questions drawn from across the whole set -- "
                         "prefer this for smoke tests")
    ap.add_argument("--seed", type=int, default=1977,
                    help="seed for --sample (default: 1977)")
    ap.add_argument("--types", default="",
                    help="comma-separated question types to answer "
                         "(default: all)")
    ap.add_argument("--continuity", default="",
                    help="comma-separated continuities to answer, e.g. "
                         "'canon,legends,non-canon' to drop the real-world "
                         "production trivia (default: all)")
    ap.add_argument("--split", default="",
                    help="comma-separated splits to answer, e.g. 'train' "
                         "(default: all)")
    ap.add_argument("--eval-fraction", type=float,
                    default=gq.DEFAULT_EVAL_FRACTION,
                    help="only used to fill in the split on the older questions "
                         f"that predate it (default: {gq.DEFAULT_EVAL_FRACTION})")
    ap.add_argument("--max-rounds", type=int, default=MAX_ROUNDS,
                    help=f"tool rounds before the model is made to answer "
                         f"(default: {MAX_ROUNDS})")
    ap.add_argument("--max-reasks", type=int, default=1,
                    help="re-asks after a failed quality gate (default: 1)")
    ap.add_argument("--no-seed", action="store_true",
                    help="do not seed the source article; make the model find "
                         "it by search. Truer to a real session, several times "
                         "the cost")
    ap.add_argument("--require-source", action="store_true",
                    help="skip questions whose source article is missing, "
                         "instead of answering them by search")
    ap.add_argument("--allow-untooled", action="store_true",
                    help="keep answers the model produced without calling any "
                         "tool. Off by default: they are unverified recall")
    ap.add_argument("--no-check-grounding", dest="check_grounding",
                    action="store_false",
                    help="skip the names-and-numbers check on the answer")
    ap.add_argument("--reject-unsupported", action="store_true",
                    help="re-ask, then drop, answers carrying a name or number "
                         "that occurs in nothing the model read. Off until you "
                         "have read a sample's `unsupported` fields")
    ap.add_argument("--max-answer-chars", type=int, default=MAX_ANSWER_CHARS,
                    help=f"reject answers longer than this "
                         f"(default: {MAX_ANSWER_CHARS})")
    ap.add_argument("--decline-max-chars", type=int, default=DECLINE_MAX_CHARS,
                    help=f"an answer this short that says the fact was never "
                         f"recorded is treated as a retrieval failure; a longer "
                         f"one is allowed to note an unrecorded corner "
                         f"(default: {DECLINE_MAX_CHARS})")
    ap.add_argument("--keep-trace", action="store_true",
                    help="store the full message list in each record, for "
                         "training tool use rather than recall")
    ap.add_argument("--cache-size", type=int, default=40000,
                    help="tool results memoised in process, LRU (default: "
                         "40000 -- about 280 MB at the 7 KB cap on a tool "
                         "result, and questions arrive grouped by article, so "
                         "a larger cache is repaid directly in scans not run)")
    ap.add_argument("--temperature", type=float, default=0.2,
                    help="sampling temperature (default: 0.2 -- facts, not "
                         "variety)")
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="per-request timeout")
    ap.add_argument("--thinking", action="store_true",
                    help="let the model reason before answering. Off by "
                         "default: it bills as output and the tool results "
                         "already carry the evidence")
    ap.add_argument("--force", action="store_true",
                    help="re-answer questions already in the output")
    ap.add_argument("--retry-failed", action="store_true",
                    help="also retry questions recorded in failures.jsonl")
    ap.add_argument("--overwrite", action="store_true",
                    help="truncate the output file instead of appending")
    ap.add_argument("--dry-run", action="store_true",
                    help="estimate tokens, cost and runtime, then exit")
    ap.add_argument("--price-in", type=float, default=pc.DEFAULT_PRICE_IN,
                    help=f"$/1M cache-miss input (default: {pc.DEFAULT_PRICE_IN})")
    ap.add_argument("--price-out", type=float, default=pc.DEFAULT_PRICE_OUT,
                    help=f"$/1M output (default: {pc.DEFAULT_PRICE_OUT})")
    ap.add_argument("--price-cached", type=float, default=pc.DEFAULT_PRICE_CACHED,
                    help=f"$/1M cache-hit input (default: {pc.DEFAULT_PRICE_CACHED})")
    args = ap.parse_args()

    records = load_questions(args)
    if not records:
        sys.exit("No questions matched the filters.")

    if args.dry_run:
        estimate(records, args)
        return

    # Both before the 171k-file corpus index, which costs a few seconds: a
    # missing key should fail immediately, not after the wait.
    key = wc.load_api_key()

    corpus = wc.Corpus(args.corpus)
    # Built here rather than on the first search_text call: this is a batch of
    # thousands of questions, so the load is amortised to nothing, and paying
    # it up front keeps it out of the first question's latency (and out of the
    # request semaphore, where every other question would wait behind it).
    t0 = time.time()
    size = corpus.build_text_index(rebuild=args.rebuild_index)
    print(f"{size / 1e6:,.0f} MB of article text indexed for search_text "
          f"in {time.time() - t0:.0f}s")
    if args.search_workers > 0:
        # The reason this run is not stuck at one core. Started before the
        # workers so the pool's own start-up is not paid inside a question.
        corpus.attach_scanner(args.search_workers)
        print(f"search_text scans across {args.search_workers} processes")

    total = len(records)
    # --overwrite throws the output away, so nothing in it can be "already done".
    if not (args.force or args.overwrite):
        done = load_done()
        records = [r for r in records if gq.dedup_key(r["q"]) not in done]
    if not args.retry_failed:
        failed = load_failures()
        records = [r for r in records if gq.dedup_key(r["q"]) not in failed]

    print(f"{len(corpus):,} articles indexed")
    print(f"{total:,} questions selected, {len(records):,} to do "
          f"({total - len(records):,} already answered or previously failed)")
    if not records:
        return

    stats = Stats()
    started = time.time()
    try:
        asyncio.run(run(records, corpus, key, args, stats))
    except KeyboardInterrupt:
        print("\ninterrupted -- rerun the same command to resume")
    finally:
        if corpus.scanner is not None:
            corpus.scanner.close()

    elapsed = time.time() - started
    if stats.fatal:
        print(f"\nRUN ABORTED -- {stats.fatal}")
        print("This is an account/config fault, not a problem with any question.")
        print("  402 Insufficient Balance -> top up at platform.deepseek.com")
        print("  401/403                  -> check DEEPSEEK_API_KEY")
        print("  404                      -> wrong --model id")
        print(f"\n{stats.answered:,} answers written before the abort.")
        sys.exit(1)

    miss = stats.tokens_in - stats.tokens_cached
    cost = (miss / 1e6 * args.price_in
            + stats.tokens_cached / 1e6 * args.price_cached
            + stats.tokens_out / 1e6 * args.price_out)
    done_n = max(stats.answered, 1)
    print(f"\n{stats.answered:,} answers | failed {stats.failed:,} | "
          f"skipped {stats.skipped:,}")
    print(f"  {stats.answer_chars/done_n:.0f} chars per answer, "
          f"{stats.tool_calls/done_n:.1f} tool calls each "
          f"({stats.extra_rounds/done_n:.0%} of questions needed a round beyond "
          f"the seed)")
    if stats.forced:
        print(f"  {stats.forced:,} ran out of rounds and were made to answer "
              f"-- raise --max-rounds if this is more than a trickle")
    if stats.repaired:
        print(f"  {stats.repaired:,} had a trailing sources block stripped")
    if stats.preambles:
        print(f"  {stats.preambles:,} had a tool-loop preamble trimmed off the front")
    print(f"requests {stats.requests:,} (retries {stats.retries:,}, re-asks "
          f"{stats.reasks:,}) | tokens {stats.tokens_in:,} in / "
          f"{stats.tokens_out:,} out | ~${cost:,.2f}")
    if stats.tokens_in:
        print(f"  input cache hits {stats.tokens_cached/stats.tokens_in:.0%} "
              f"({stats.tokens_cached:,} of {stats.tokens_in:,})")
    if stats.tokens_reasoning:
        share = stats.tokens_reasoning / max(stats.tokens_out, 1)
        print(f"  reasoning tokens {stats.tokens_reasoning:,} ({share:.0%} of "
              f"output, ~${stats.tokens_reasoning/1e6*args.price_out:,.2f} "
              f"discarded) -- drop --thinking to save it")
    if stats.tool_calls:
        print(f"  tool cache served {stats.tool_hits:,} of "
              f"{stats.tool_calls:,} calls")
    if stats.by_type:
        spread = sorted(stats.by_type.items(), key=lambda kv: -kv[1])
        print("type spread: " + ", ".join(f"{k} {v:,}" for k, v in spread))
    if stats.rejects:
        grouped: dict[str, int] = {}
        for reason, n in stats.rejects.items():
            head = reason.split(":")[0]
            grouped[head] = grouped.get(head, 0) + n
        top = sorted(grouped.items(), key=lambda kv: -kv[1])
        print(f"gates fired {sum(grouped.values()):,} times: "
              + ", ".join(f"{k} x{v:,}" for k, v in top))
        leaks = sorted(((r.split(": ", 1)[1], n) for r, n in stats.rejects.items()
                        if r.startswith("persona leak: ")),
                       key=lambda kv: -kv[1])[:5]
        if leaks:
            print("  top leaked phrases: "
                  + ", ".join(f"{p!r} x{n}" for p, n in leaks))
    if stats.unsupported:
        share = stats.unsupported / done_n
        print(f"grounding: {stats.unsupported:,} answers ({share:.1%}) name "
              f"something absent from everything they read.")
        print(f"  read the `unsupported` field on those before trusting it -- "
              f"then --reject-unsupported to drop them")
    if stats.errors:
        top = sorted(stats.errors.items(), key=lambda kv: -kv[1])[:5]
        print("top failure reasons: " + ", ".join(f"{k} x{v}" for k, v in top))
    print(f"elapsed {elapsed/60:.1f} min -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
