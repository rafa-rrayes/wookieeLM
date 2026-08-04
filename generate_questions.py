#!/usr/bin/env python3
"""Generate questions the Wookieepedia corpus can answer, with DeepSeek.

Reads ``corpus/wookieepedia/<L>/<Title>.md`` and writes one JSON object per
question to ``questions/questions.jsonl``:

    {"q": "How tall was Ahsoka Tano as an adult?",
     "type": "numeric",
     "source": "wookieepedia/A/Ahsoka_Tano.md",
     "continuity": "canon",
     "split": "train"}

Questions only -- no answers. The article they came from is recorded, so an
answer can be produced later (``wookiee_chat.py`` answers from the same corpus)
without paying for one now.

Every kind of question is wanted, so the model is given a taxonomy and told to
spread across it; ``--types`` narrows it. See QUESTION_TYPES.

Evidence
--------
The model must return, with every question, the sentence from the passage that
answers it, copied verbatim. That span never reaches the output file by default
(--keep-evidence overrides), because it exists to be *checked*:

    1. the span must occur in the passage -- a model that invented the fact
       cannot produce the sentence
    2. every proper noun and number in the question must occur near the span,
       not merely somewhere in the article

Check 2 is what catches attribute mis-binding, the failure mode that formatting
validators cannot see. A 60 KB battle article says "Jedi Master" in fifty
places, so an article-wide check licenses "Which Jedi Master joined the rescue
team...?" about someone the article calls a Padawan. Scoped to the cited
sentence, it does not.

What gets sent:

    ## Infobox    folded into the article's first chunk. paraphrase_corpus.py
                  leaves it alone because rewording key/value facts is noise,
                  but for question generation it is the densest fact source in
                  the article -- heights, homeworlds, birth and death years
    body          chunked on "## " headings, one request per chunk, so long
                  articles are covered end to end rather than truncated
    skeleton      long articles get one extra request over a digest of the
                  whole article -- lead, headings, and the opening of each
                  section. Chunking otherwise makes a cross-section question
                  impossible to ask, since no single request ever sees two
                  sections. --no-skeleton turns it off.

Thinking is disabled by default, for the same reason as paraphrase_corpus.py:
DeepSeek's V4 models reason unless told not to and bill it as output. Writing a
question from a passage in front of you needs no deliberation. Pass --thinking
to re-enable.

Requirements:
    DEEPSEEK_API_KEY in the environment

Usage:
    uv run generate_questions.py --sample 3000 --dry-run   # cost estimate only
    uv run generate_questions.py --sample 300 --keep-evidence   # smoke test
    uv run generate_questions.py                           # full corpus
    uv run generate_questions.py --types multihop,causal   # one flavour only

Resume: rerun the same command. Articles already present in the output JSONL
are skipped, and the questions already written seed the duplicate filter.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
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
OUTPUT_DIR = REPO_ROOT / "questions"
OUTPUT_FILE = OUTPUT_DIR / "questions.jsonl"
STATE_DIR = REPO_ROOT / "question_state"

API_BASE = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEFAULT_MODEL = "deepseek-v4-flash"

# Chunking. Larger than paraphrase_corpus.py's 6 KB: a question can span the
# whole passage it is drawn from, so fewer, wider chunks give better questions
# and fewer repeats of the same fact across chunk boundaries.
CHUNK_CHARS = 12000
INFOBOX_CHARS = 2000        # cap: a few articles carry enormous key/value blocks

# The cross-section pass. Articles above SKELETON_MIN_CHARS get one extra
# request over a digest of the whole article.
SKELETON_MIN_CHARS = 20000
SKELETON_SECTION_CHARS = 260    # opening of each section, so spans stay citable
SKELETON_TYPES = ("multihop", "chronology", "comparison", "causal", "open")

# How many questions to ask for, as a function of passage size.
CHARS_PER_QUESTION = 700
MIN_QUESTIONS = 2           # a two-sentence stub has about two questions in it;
MAX_QUESTIONS = 12          # a floor of 3 manufactured a third from nothing

CHARS_PER_TOKEN = 3.6       # matches paraphrase_corpus.py, for --dry-run
TOKENS_PER_QUESTION = 50    # question + evidence span; --dry-run only

# Evidence checking.
EVIDENCE_WINDOW = 300       # chars either side of the span: ~a sentence each way
DEFAULT_EVAL_FRACTION = 0.02
NEAR_DUPE_JACCARD = 0.6     # within one article only -- see is_near_dupe()

MAX_ATTEMPTS = 4
REJECT_SAMPLES = 40      # per reason, written to question_state/ at the end
# A batch where most questions fail validation means the model misread the
# instructions, not that the article is bad -- worth a cheaper, cooler retry.
MIN_YIELD = 0.4

# The taxonomy. The gloss is what the model is shown; the key is what lands in
# the "type" field, so downstream filtering has a fixed vocabulary.
QUESTION_TYPES: dict[str, str] = {
    "factoid":      "a single named fact",
    "numeric":      "a quantity, measurement, count or size",
    "date":         "a year, an era (BBY/ABY), or when something happened",
    "definition":   "what a thing is",
    "list":         "several items that have to be enumerated",
    "causal":       "why or how something happened",
    "comparison":   "a contrast between two things in the passage",
    "multihop":     "two separate facts from the passage combined into one answer",
    "chronology":   "the order of events, what came before or after what",
    "relationship": "who is related to, allied with, or serves whom",
    "description":  "appearance, traits, capabilities",
    "production":   "the real world: actor, author, release, development",
    "yesno":        "answerable yes or no, and settled by the passage",
    "open":         "needs a paragraph of explanation, not a phrase",
}

SYSTEM_PROMPT_HEAD = """\
You write quiz questions about Star Wars from encyclopedia passages.

You are given one passage from a Wookieepedia article. Write questions that \
the passage answers -- and only questions it answers. Never ask about anything \
the passage does not state.

Hard requirements:
- Every question must stand on its own. A reader who has never seen the \
passage must know what is being asked, so name the subject in full: "How tall \
was Ahsoka Tano?", never "How tall was she?" or "How tall was the subject?".
- Never refer to the passage, the article, the text, the infobox or the \
source. The question is about Star Wars, not about a document.
- Use the exact proper nouns, numbers, dates and era abbreviations (BBY, ABY) \
as the passage writes them. Never introduce a name, rank or title the passage \
does not use for that person or thing.
- Ask a real question with a definite answer. No opinion, no speculation, and \
nothing that needs knowledge from outside the passage.
- Vary them. Different facts, different shapes, different lengths. Do not ask \
the same thing twice in different words.

Question types, one per question -- spread your questions across the types \
that the passage can actually support:
"""

SYSTEM_PROMPT_TAIL = """
Output one question per line, in this exact form:

    type | question? | evidence

`evidence` is the sentence in the passage that answers the question, copied \
out word for word. Copy it exactly. Do not paraphrase it, do not trim it to an \
ellipsis, and do not stitch two sentences together -- if the answer needs two, \
copy the one that is harder to find. If no sentence in the passage answers the \
question, do not ask that question at all.

Nothing else. No numbering, no bullets, no blank lines, no commentary."""

SKELETON_NOTE = """
This passage is a digest of a long article: the opening, then each section's \
heading and first lines. Ask questions that join facts from DIFFERENT sections \
-- an order of events across the article, a comparison between two episodes in \
it, a cause in one section and its effect in another. Do not ask anything a \
single section would answer on its own."""


# ---- Reuse of paraphrase_corpus.py -------------------------------------------

def load_pc():
    """Import paraphrase_corpus.py by path.

    Same trick as count.py. The splitter, the skip rule, the chunker and the
    backoff are already written and audited there; re-implementing any of them
    here would let the two drift into disagreeing about what a body is or which
    articles are worth a request.
    """
    spec = importlib.util.spec_from_file_location(
        "pc", REPO_ROOT / "paraphrase_corpus.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["pc"] = module
    spec.loader.exec_module(module)
    return module


pc = load_pc()

_CONTINUITY_RE = re.compile(r"^continuity:\s*(\S+)\s*$", re.MULTILINE)


# ---- Response parsing --------------------------------------------------------

# The model is asked for lines rather than JSON on purpose: a truncated JSON
# array is unparseable and costs the whole batch, where a truncated last line
# costs one question and the rest still land.
_LEADIN_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_FENCE_RE = re.compile(r"^\s*```[a-z]*\s*\n(.*)\n```\s*$", re.DOTALL)
_REFUSAL_RE = re.compile(
    r"^\s*(?:i (?:can'?t|cannot|won'?t|am unable)|as an ai|i'm sorry)", re.IGNORECASE
)

# Meta-references: a question about the document rather than about Star Wars.
# These are useless in training data -- there is no document at inference time.
_META_RE = re.compile(
    r"\b(?:th(?:e|is|at)|the given|the above)\s+"
    r"(?:passage|article|text|excerpt|document|infobox|section|page|entry|source)\b"
    r"|\baccording to (?:the|this)\b"
    r"|\bas (?:described|mentioned|stated|shown) (?:above|here|in)\b",
    re.IGNORECASE,
)

# Opens on a bare pronoun -- the classic non-self-contained question.
_PRONOUN_OPEN_RE = re.compile(
    r"^(?:who|what|when|where|why|how|which)\b[^?]{0,30}?"
    r"\b(?:he|she|it|they|him|her|them|his|hers|its|their|this|that)\b",
    re.IGNORECASE,
)

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_PROPER_RE = re.compile(r"\b[A-Z][A-Za-z0-9'\-]{2,}")
_NUMBER_RE = re.compile(r"\b\d[\d,.]*\b")

# A whole capitalised name, not its words: "Relay Post Epsilon One", not
# {"Relay", "Post", "Epsilon", "One"}. Checking words in isolation cannot see a
# corrupted name -- the model's "Relay *Station* Epsilon One" passed a per-word
# check because the cited sentence happens to contain the common noun
# "station" ("the Galactic Republic station Relay Post Epsilon One"). Checking
# the phrase catches it, and stays immune to the stray capital that a
# case-sensitive per-word check would trip over.
_PHRASE_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9'\-]*"
    r"(?:\s+(?:of|the|and|for|in|a|an|to)\s+[A-Z][A-Za-z0-9'\-]*"
    r"|\s+[A-Z][A-Za-z0-9'\-]*)*"
)
_CONNECTIVE_SPLIT_RE = re.compile(r"\s+(?:of|the|and|for|in|a|an|to)\s+")
_STOPWORDS = frozenset(
    "the a an of and or in on at to for with from by as is was were are be "
    "list star wars".split()
)

# Capitalised words that name a rank or title rather than an individual. These
# are the mis-binding surface: the model writes "Jedi Master" for someone the
# article calls a Padawan, or promotes a Commander to a General. A phrase made
# only of these has to be found beside the cited evidence; a phrase carrying a
# distinctive name as well is allowed the article's own short forms, because
# the evidence sentence says "Marek" where the question must say "Galen Marek"
# to stand on its own.
_ROLE_WORDS = frozenset("""
jedi sith master padawan knight apprentice general admiral captain commander
colonel lieutenant sergeant major moff grand lord darth king queen prince
princess senator chancellor governor emperor empress doctor baron count duke
viceroy warden inquisitor overseer marshal
""".split())

MIN_Q_CHARS = 15
MAX_Q_CHARS = 300
MIN_SPAN_CHARS = 20


@dataclass
class Candidate:
    kind: str
    question: str
    evidence: str


def parse_questions(text: str) -> list[Candidate]:
    """Model output -> candidates, tolerating the usual formatting drift."""
    text = text.strip()
    if m := _FENCE_RE.match(text):
        text = m.group(1)

    out: list[Candidate] = []
    for line in text.split("\n"):
        line = _LEADIN_RE.sub("", line).strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|", 2)]
        if len(parts) == 3:
            kind, question, evidence = parts
        elif len(parts) == 2:
            # Type or evidence dropped. Which one is missing is guessable from
            # whether the first field looks like a type name.
            if parts[0].lower() in QUESTION_TYPES:
                kind, question, evidence = parts[0], parts[1], ""
            else:
                kind, question, evidence = "other", parts[0], parts[1]
        else:
            kind, question, evidence = "other", parts[0], ""
        kind = kind.lower()
        if kind not in QUESTION_TYPES:
            kind = "other"
        out.append(Candidate(kind, question.strip().strip('"'),
                             evidence.strip().strip('"')))
    return out


# ---- Evidence checking -------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def normalize_for_match(s: str) -> str:
    """Fold the differences that are not the model's fault.

    Markdown emphasis is the big one: a passage says ``*Rogue One*`` and the
    model copies the sentence without the asterisks, which is a faithful quote
    of the prose and a failed substring match. Case and whitespace go the same
    way.
    """
    s = s.replace("*", "").replace("_", " ")
    return _WS_RE.sub(" ", s).strip().casefold()


def evidence_window(span: str, norm_passage: str, radius: int) -> str | None:
    """Text around the cited span, or None if the span is not in the passage.

    A character radius rather than a sentence count: sentence splitting on
    Wookieepedia prose is unreliable (abbreviations, "Sr.", stardates), and
    getting it wrong here would reject good questions.
    """
    norm_span = normalize_for_match(span)
    if len(norm_span) < MIN_SPAN_CHARS:
        return None
    idx = norm_passage.find(norm_span)
    if idx < 0:
        return None
    start = max(0, idx - radius)
    end = min(len(norm_passage), idx + len(norm_span) + radius)
    return norm_passage[start:end]


def split_possessives(phrase: str) -> list[str]:
    """"Drohon's Tempest" -> ["Drohon", "Tempest"].

    A possessive is grammar, not part of a name: the article says "her Tempest",
    so requiring the literal string "Drohon's Tempest" near the evidence would
    reject a perfectly good question. Names that genuinely contain a possessive
    ("Plo's Bros") still match, because the check is a substring test and both
    halves occur inside it.
    """
    parts, current = [], []
    for word in phrase.split():
        if word.endswith("'s") or word.endswith("'"):
            current.append(word.rstrip("'s").rstrip("'"))
            parts.append(" ".join(current))
            current = []
        else:
            current.append(word)
    if current:
        parts.append(" ".join(current))
    return [p for p in parts if p]


def significant_terms(question: str, title: str) -> tuple[list[str], list[str], frozenset[str]]:
    """(proper-noun phrases, numbers) in a question that the evidence must support.

    Phrases made only of words from the article title are dropped: the subject
    of the article is always a legitimate thing to name, and the sentence that
    answers a question about it very often refers to it with a pronoun.
    """
    title_words = frozenset(w.casefold() for w in _WORD_RE.findall(title))
    phrases = []
    # Skip the first character: every question opens on a capital, and that
    # capital is grammar rather than a name.
    for phrase in _PHRASE_RE.findall(question[1:] if question else ""):
        for part in split_possessives(phrase):
            words = [w.casefold() for w in _WORD_RE.findall(part)]
            if not words or all(w in title_words for w in words):
                continue
            phrases.append(part)
    return phrases, _NUMBER_RE.findall(question), title_words


def occurs_in(norm: str, haystack: str) -> bool:
    """Substring test that forgives an English plural on the final word.

    A question asks "Which Jedi Masters led...?" about an article that names
    them one at a time. The plural is the question's own grammar, not a claim
    about a name, so it must not read as a missing entity.
    """
    if norm in haystack:
        return True
    return len(norm) > 1 and norm.endswith("s") and norm[:-1] in haystack


def decompose(phrase: str) -> list[str]:
    """Split a phrase on the connectives that join two names into one.

    `_PHRASE_RE` is deliberately greedy so that "Relay Post Epsilon One" is
    tested as a unit -- checking its words separately would never notice that
    the model wrote "Station" for "Post". The cost is that "Even Piell and
    Tarkin" or "Galliwig a Human" are captured as single phrases that occur
    nowhere. Only ever used as a fallback, after the whole phrase has failed to
    match, so the greedy test keeps its teeth.
    """
    parts = [p.strip() for p in _CONNECTIVE_SPLIT_RE.split(phrase)]
    return [p for p in parts if p]


def trim_title_words(phrase: str, title_words: frozenset[str]) -> str:
    """Drop the article's own name from the ends of a phrase.

    The other way greed misfires: "the Force Enclave Trel Nu was assigned to"
    glues the title onto the name beside it, and the compound occurs nowhere.
    Trimming only at the ends, and only words from the title, keeps a corrupted
    middle -- "Relay *Station* Epsilon One" -- visible.
    """
    words = phrase.split()
    while words and words[0].casefold() in title_words:
        words.pop(0)
    while words and words[-1].casefold() in title_words:
        words.pop()
    return " ".join(words)


def check_phrase(phrase: str, window: str, norm_passage: str,
                 title_words: frozenset[str] = frozenset()) -> str | None:
    """Why a proper-noun phrase in a question is unsupported, or None.

    Two different faults, kept apart because they mean different things:

    ``name not in article``   the phrase occurs nowhere in the source. A
                              corrupted or invented name -- "Relay *Station*
                              Epsilon One" for the article's "Relay Post
                              Epsilon One".
    ``term not near evidence`` the phrase exists in the article but not beside
                              the sentence the model cited, and it carries no
                              distinctive name of its own. This is attribute
                              mis-binding: "Which Jedi Master...?" about
                              someone the cited sentence calls a Padawan.
    """
    norm = normalize_for_match(phrase)
    if occurs_in(norm, window):
        return None
    if not occurs_in(norm, norm_passage):
        # Fallbacks for the two ways the greedy phrase matcher over-captures.
        # Both are tried only after the whole phrase has failed, and both
        # terminate: parts carry no connectives, trimming is idempotent.
        candidates = [decompose(phrase)]
        trimmed = trim_title_words(phrase, title_words)
        if trimmed and trimmed != phrase:
            candidates.append([trimmed])
        fallback = None
        for parts in candidates:
            if len(parts) == 1 and parts[0] == phrase:
                continue
            reasons = [check_phrase(p, window, norm_passage, title_words)
                       for p in parts]
            if not any(reasons):
                return None
            # The pieces are all real names, but one sits away from the cited
            # sentence -- that is the mis-binding fault, not a missing name.
            fallback = fallback or next(r for r in reasons if r)
        return fallback or "name not in article"
    distinctive = [w.casefold() for w in _WORD_RE.findall(phrase)
                   if w.casefold() not in _ROLE_WORDS]
    if any(w in window for w in distinctive):
        return None
    return "term not near evidence"


def validate(cand: Candidate, title: str, passage: str, args,
             norm_passage: str | None = None) -> str | None:
    """Why this question should be dropped, or None to keep it.

    ``norm_passage`` is ``normalize_for_match(passage)``; the caller passes it
    in so a 380 KB article is not re-normalised once per candidate question.
    """
    q = cand.question
    if norm_passage is None:
        norm_passage = normalize_for_match(passage)
    if not q.endswith("?"):
        return "not a question"
    if not (MIN_Q_CHARS <= len(q) <= MAX_Q_CHARS):
        return "length"
    if _META_RE.search(q):
        return "meta-reference"
    if _META_RE.search(cand.evidence):
        return "meta-reference"

    if not cand.evidence:
        return "no evidence"
    if len(normalize_for_match(cand.evidence)) < MIN_SPAN_CHARS:
        # Kept apart from the miss below: a fragment too short to locate is the
        # model misreading the format, a span that simply is not there is the
        # model paraphrasing. They call for different fixes, so the run summary
        # must not blur them into one number.
        return "evidence too short"
    window = evidence_window(cand.evidence, norm_passage, args.evidence_window)
    if window is None:
        # Either invented, or paraphrased instead of copied. Both mean the
        # question cannot be checked, and an unverifiable question is worth
        # less than the one request it would take to ask for another.
        return "evidence not in passage"

    phrases, nums, title_words = significant_terms(q, title)
    for phrase in phrases:
        if fault := check_phrase(phrase, window, norm_passage, title_words):
            return f"{fault}: {phrase}"
    if any(n not in window and n.replace(",", "") not in window for n in nums):
        return "number not near evidence"

    if _PRONOUN_OPEN_RE.match(q) and not _PROPER_RE.search(q[1:]):
        return "pronoun subject"
    return None


def dedup_key(question: str) -> str:
    return " ".join(w.lower() for w in _WORD_RE.findall(question))


def token_set(question: str) -> frozenset[str]:
    return frozenset(w.lower() for w in _WORD_RE.findall(question)) - _STOPWORDS


def is_near_dupe(candidate: frozenset[str], seen: list[frozenset[str]]) -> bool:
    """Same fact asked twice in one article.

    Within an article only. Across the corpus this test is worthless -- two
    articles asking "what species was <someone>?" are asking about different
    someones -- and at 170k articles it would be quadratic as well.
    """
    for other in seen:
        union = len(candidate | other)
        if union and len(candidate & other) / union >= NEAR_DUPE_JACCARD:
            return True
    return False


# ---- Prompt assembly ---------------------------------------------------------

def build_system_prompt(types: list[str], skeleton: bool = False) -> str:
    """One constant string per pass, so it bills at the cache-hit rate."""
    lines = [f"- {name}: {QUESTION_TYPES[name]}" for name in types]
    prompt = SYSTEM_PROMPT_HEAD + "\n".join(lines)
    if skeleton:
        prompt += "\n" + SKELETON_NOTE
    return prompt + SYSTEM_PROMPT_TAIL


def question_count(chars: int, args) -> int:
    return max(args.min_questions,
               min(args.max_questions, chars // args.chars_per_question))


def plan_chunks(doc, args) -> list[str]:
    """The passages to send for one article.

    The infobox rides along with the first chunk. It is pure fact density --
    born, died, homeworld, species, height -- and on a stub it may be most of
    what the article knows.
    """
    chunks = pc.chunk_body(doc.body, args.chunk_chars)
    if not chunks:
        return []
    if doc.infobox:
        infobox = doc.infobox[:args.infobox_chars]
        chunks = [infobox.rstrip() + "\n\n" + chunks[0]] + chunks[1:]
    return chunks


def build_skeleton(doc, args) -> str | None:
    """A digest of a long article: lead, then every heading with its opening.

    Chunking is what makes a cross-section question impossible -- no request
    ever sees two sections, so nothing can ask about both. This pass restores
    that view. Each section contributes its first lines rather than only its
    heading, so the model still has real sentences to cite as evidence.
    """
    if len(doc.body) < args.skeleton_min_chars:
        return None

    sections = re.split(r"\n(?=## )", doc.body)
    parts: list[str] = []
    if doc.infobox:
        parts.append(doc.infobox[:args.infobox_chars].rstrip())
    for section in sections:
        section = section.strip()
        if not section:
            continue
        head, _, rest = section.partition("\n")
        if not head.startswith("## "):
            # The lead, which has no heading of its own. Worth more than a
            # section opening -- it is the article's summary.
            parts.append(section[:args.skeleton_section_chars * 3].rstrip())
            continue
        opening = rest.strip()[:args.skeleton_section_chars].rstrip()
        parts.append(f"{head}\n\n{opening}" if opening else head)

    skeleton = "\n\n".join(parts)
    return skeleton[:args.chunk_chars] if len(skeleton) > MIN_SPAN_CHARS else None


# ---- API ---------------------------------------------------------------------

class GenerationError(RuntimeError):
    """Raised when a passage yielded no usable questions."""


@dataclass
class Request:
    """One call's worth of work.

    ``prompt_text`` is what the model sees; ``verify_text`` is what evidence is
    checked against. They differ only for the skeleton pass, where the model
    sees a digest but may legitimately quote any sentence the digest contains --
    checking against the full body keeps that from mattering.
    """
    prompt_text: str
    verify_text: str
    system: str
    skeleton: bool = False


@dataclass
class Stats:
    articles: int = 0
    skipped: int = 0
    failed: int = 0
    questions: int = 0
    duplicates: int = 0
    near_dupes: int = 0
    skeleton_questions: int = 0
    eval_questions: int = 0
    requests: int = 0
    retries: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cached: int = 0
    tokens_reasoning: int = 0
    fatal: str | None = None
    rejects: dict[str, int] = field(default_factory=dict)
    by_type: dict[str, int] = field(default_factory=dict)
    errors: dict[str, int] = field(default_factory=dict)
    reject_samples: dict[str, list] = field(default_factory=dict)

    def note(self, bucket: dict[str, int], key: str) -> None:
        bucket[key] = bucket.get(key, 0) + 1

    def sample_reject(self, reason: str, cand, title: str) -> None:
        """Keep a few rejected candidates per reason, to be written at the end.

        A rejection rate is a number you cannot act on. The examples are what
        tell you whether the validator caught the model or the model caught a
        bug in the validator, and the only affordable time to collect them is
        while the run is happening.
        """
        bucket = self.reject_samples.setdefault(reason.split(":")[0], [])
        if len(bucket) < REJECT_SAMPLES:
            bucket.append({"reason": reason, "title": title,
                           "type": cand.kind, "q": cand.question,
                           "evidence": cand.evidence})

    def note_usage(self, usage: dict) -> None:
        self.tokens_in += usage.get("prompt_tokens", 0)
        self.tokens_out += usage.get("completion_tokens", 0)
        self.tokens_cached += usage.get("prompt_cache_hit_tokens", 0)
        details = usage.get("completion_tokens_details") or {}
        self.tokens_reasoning += details.get("reasoning_tokens", 0)


async def questions_for(client: httpx.AsyncClient, req: Request, title: str,
                        args, stats: Stats,
                        req_sem: asyncio.Semaphore) -> list[Candidate]:
    """One passage -> validated candidates."""
    want = question_count(len(req.prompt_text), args)
    user = (f"Article: {title}\n\n"
            f"Write {want} questions about the following passage.\n\n"
            f"Passage:\n\n{req.prompt_text}")

    payload = {
        "model": args.model,
        "temperature": 1.0,     # variety across types; the passage pins the facts
        "thinking": {"type": "enabled" if args.thinking else "disabled"},
        "messages": [
            {"role": "system", "content": req.system},
            {"role": "user", "content": user},
        ],
    }

    last_error = "unknown"
    best: list[Candidate] = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
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
            raise GenerationError(f"http {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        stats.requests += 1
        if usage := data.get("usage"):
            stats.note_usage(usage)

        raw = (data["choices"][0]["message"]["content"] or "").strip()
        if not raw:
            last_error = "empty response"
        elif _REFUSAL_RE.match(raw):
            last_error = "refusal"
        else:
            candidates = parse_questions(raw)
            kept = []
            norm_passage = normalize_for_match(req.verify_text)
            for cand in candidates:
                if reason := validate(cand, title, req.verify_text, args,
                                      norm_passage):
                    # Counted on every attempt, including ones that go on to
                    # fail the yield gate. Tallying only accepted batches hid
                    # the worst batches from the summary -- exactly the ones
                    # worth seeing.
                    stats.note(stats.rejects, reason)
                    stats.sample_reject(reason, cand, title)
                    continue
                kept.append(cand)
            if len(kept) > len(best):
                best = kept
            if kept and len(kept) / max(len(candidates), 1) >= MIN_YIELD:
                return kept
            last_error = "low yield" if candidates else "unparseable"

        payload["temperature"] = max(0.4, payload["temperature"] - 0.3)
        await pc._backoff(attempt)

    stats.note(stats.errors, last_error)
    if best:
        # Every attempt fell below MIN_YIELD, but some questions did pass every
        # check. They are as good as any other passing question -- dropping the
        # passage entirely would cost more than the low yield does.
        return best
    raise GenerationError(last_error)


# ---- Orchestration -----------------------------------------------------------

def assign_split(rel_path: str, eval_fraction: float) -> str:
    """Train/eval by a stable hash of the article path.

    By article, never by question: two questions from one article share its
    facts, so splitting per question puts near-answers to the eval set into the
    training set and the eval measures memorisation. Hashing the path means the
    assignment is identical on every run and every machine, so a resumed or
    re-sampled run cannot leak an article across the boundary.
    """
    if eval_fraction <= 0:
        return "train"
    digest = hashlib.sha1(rel_path.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") / 2**32
    return "eval" if bucket < eval_fraction else "train"


def load_done() -> tuple[set[str], set[str]]:
    """-> (source paths already written, dedup keys already written).

    Output is a single append-only JSONL rather than a file per article, so
    resume reads it back. That also carries the duplicate filter across runs:
    without it, a resumed run would happily re-emit questions it already has.
    """
    if not OUTPUT_FILE.exists():
        return set(), set()
    sources: set[str] = set()
    seen: set[str] = set()
    with OUTPUT_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if src := rec.get("source"):
                sources.add(src)
            if q := rec.get("q"):
                seen.add(dedup_key(q))
    return sources, seen


def load_failures() -> set[str]:
    """Paths that genuinely failed. Account-level faults are not the article's
    fault, so they are ignored here exactly as paraphrase_corpus.py ignores them."""
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
        if path := rec.get("path"):
            failed.add(path)
    return failed


def plan_requests(doc, args, systems: tuple[str, str]) -> list[Request]:
    """Every call one article needs: its chunks, plus a skeleton if it is long."""
    body_system, skeleton_system = systems
    requests = [Request(chunk, chunk, body_system) for chunk in plan_chunks(doc, args)]
    if not requests:
        return []
    if not args.no_skeleton and (skeleton := build_skeleton(doc, args)):
        # The digest opens with the infobox, so the infobox has to be part of
        # what the span is checked against -- verifying against doc.body alone
        # made every infobox-cited span unfindable by construction.
        verify = f"{doc.infobox}\n\n{doc.body}" if doc.infobox else doc.body
        requests.append(Request(skeleton, verify, skeleton_system, skeleton=True))
    return requests


async def worker(src: Path, client: httpx.AsyncClient, sem: asyncio.Semaphore,
                 systems: tuple[str, str], args, stats: Stats, seen: set[str],
                 out, ledger, bar: tqdm, abort: asyncio.Event,
                 req_sem: asyncio.Semaphore) -> None:
    async with sem:
        if abort.is_set():
            bar.update(1)
            return
        rel = src.relative_to(CORPUS_DIR).as_posix()
        try:
            doc = pc.split_document(src, src.read_text(encoding="utf-8"))
            if pc.skip_reason(doc):
                stats.skipped += 1
                return

            requests = plan_requests(doc, args, systems)
            if not requests:
                stats.skipped += 1
                return

            m = _CONTINUITY_RE.search(doc.frontmatter)
            continuity = m.group(1) if m else "unknown"
            split = assign_split(rel, args.eval_fraction)

            tasks = [asyncio.create_task(
                         questions_for(client, r, doc.title, args, stats, req_sem))
                     for r in requests]
            try:
                batches = await asyncio.gather(*tasks)
            except BaseException:
                for t in tasks:          # don't leave siblings burning tokens
                    t.cancel()
                raise

            written = 0
            article_tokens: list[frozenset[str]] = []
            for req, batch in zip(requests, batches):
                for cand in batch:
                    key = dedup_key(cand.question)
                    if key in seen:
                        stats.duplicates += 1
                        continue
                    tokens = token_set(cand.question)
                    if is_near_dupe(tokens, article_tokens):
                        stats.near_dupes += 1
                        continue
                    seen.add(key)
                    article_tokens.append(tokens)

                    record = {
                        "q": cand.question,
                        "type": cand.kind,
                        "source": rel,
                        "continuity": continuity,
                        "split": split,
                    }
                    if args.keep_evidence:
                        record["evidence"] = cand.evidence
                    # One complete line per write, with no await in between, so
                    # concurrent workers cannot interleave inside a record.
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats.note(stats.by_type, cand.kind)
                    if req.skeleton:
                        stats.skeleton_questions += 1
                    if split == "eval":
                        stats.eval_questions += 1
                    written += 1
            out.flush()
            stats.questions += written
            stats.articles += 1
        except pc.FatalAPIError as e:
            if not abort.is_set():
                stats.fatal = str(e)
                abort.set()
        except GenerationError as e:
            stats.failed += 1
            ledger.write(json.dumps({"path": rel, "error": str(e)}) + "\n")
            ledger.flush()
        except Exception as e:  # noqa: BLE001 - one bad page must not stop the run
            stats.failed += 1
            stats.note(stats.errors, type(e).__name__)
            ledger.write(json.dumps({"path": rel,
                                     "error": f"{type(e).__name__}: {e}"}) + "\n")
            ledger.flush()
        finally:
            bar.update(1)
            bar.set_postfix(q=stats.questions, fail=stats.failed,
                            skip=stats.skipped, refresh=False)


def estimate(paths: list[Path], args, systems: tuple[str, str]) -> None:
    """--dry-run: size the job without spending anything."""
    prompt_chars = 0
    chunks = skeletons = wanted = 0
    skipped: dict[str, int] = {}
    for p in tqdm(paths, desc="measuring", unit="file"):
        doc = pc.split_document(p, p.read_text(encoding="utf-8"))
        if reason := pc.skip_reason(doc):
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        for req in plan_requests(doc, args, systems):
            chunks += 1
            skeletons += req.skeleton
            prompt_chars += len(req.prompt_text)
            wanted += question_count(len(req.prompt_text), args)

    system_chars = max(len(systems[0]), len(systems[1]))
    cached = chunks * system_chars / CHARS_PER_TOKEN
    tok_in = prompt_chars / CHARS_PER_TOKEN + cached + chunks * 25
    tok_out = wanted * TOKENS_PER_QUESTION
    if args.thinking:
        tok_out *= 5.6          # measured on the paraphrase run
    cost = ((tok_in - cached) / 1e6 * args.price_in
            + cached / 1e6 * args.price_cached
            + tok_out / 1e6 * args.price_out)

    print(f"\narticles           {len(paths):,}")
    print(f"  usable           {len(paths) - sum(skipped.values()):,}")
    for reason, n in sorted(skipped.items()):
        print(f"  - {reason:<15}{n:,}")
    print(f"requests           {chunks:,}  ({skeletons:,} cross-section passes)")
    print(f"questions asked ~  {wanted:,}  (before validation and dedup)")
    print(f"input tokens  ~    {tok_in/1e6:.1f}M  (of which {cached/1e6:.1f}M "
          f"cacheable system prompt)")
    print(f"output tokens ~    {tok_out/1e6:.1f}M  (question + evidence span)"
          f"{'  THINKING ON -- ~82% discarded' if args.thinking else ''}")
    print(f"est. cost     ~    ${cost:,.2f}   (at ${args.price_in}/${args.price_out} "
          f"per 1M in/out, ${args.price_cached} cached in)")
    eta = chunks / max(args.concurrency, 1) * 3.0 / 3600
    print(f"est. wall clock ~  {eta:.1f} h at {args.concurrency} concurrent, "
          f"3 s/request")
    print(f"eval split    ~    {args.eval_fraction:.0%} of articles, held out by "
          f"path hash")


async def run(paths: list[Path], args, systems: tuple[str, str], stats: Stats,
              seen: set[str]) -> None:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        sys.exit("DEEPSEEK_API_KEY is not set. export it and rerun.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
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
        with OUTPUT_FILE.open("a", encoding="utf-8") as out, \
             (STATE_DIR / "failures.jsonl").open("a", encoding="utf-8") as ledger:
            with tqdm(total=len(paths), desc="questioning", unit="art") as bar:
                await asyncio.gather(*(
                    worker(p, client, sem, systems, args, stats, seen, out,
                           ledger, bar, abort, req_sem)
                    for p in paths
                ))


# ---- Main --------------------------------------------------------------------

def parse_types(spec: str) -> list[str]:
    names = [t.strip().lower() for t in spec.split(",") if t.strip()]
    if unknown := [t for t in names if t not in QUESTION_TYPES]:
        sys.exit(f"unknown question type(s): {', '.join(unknown)}\n"
                 f"choose from: {', '.join(QUESTION_TYPES)}")
    return names or list(QUESTION_TYPES)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"DeepSeek model id (default: {DEFAULT_MODEL})")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="in-flight requests (default: 8, low on purpose so "
                         "this can run alongside paraphrase_corpus.py)")
    ap.add_argument("--limit", type=int,
                    help="process the first N articles (sorted; mostly stubs)")
    ap.add_argument("--sample", type=int,
                    help="process N articles drawn from across the whole "
                         "corpus -- prefer this for smoke tests")
    ap.add_argument("--seed", type=int, default=1977,
                    help="seed for --sample (default: 1977)")
    ap.add_argument("--types", default="",
                    help="comma-separated subset of question types "
                         f"(default: all -- {', '.join(QUESTION_TYPES)})")
    ap.add_argument("--chunk-chars", type=int, default=CHUNK_CHARS,
                    help=f"max passage chars per request (default: {CHUNK_CHARS})")
    ap.add_argument("--infobox-chars", type=int, default=INFOBOX_CHARS,
                    help=f"max infobox chars to include (default: {INFOBOX_CHARS})")
    ap.add_argument("--chars-per-question", type=int, default=CHARS_PER_QUESTION,
                    help=f"passage chars per question asked for "
                         f"(default: {CHARS_PER_QUESTION})")
    ap.add_argument("--min-questions", type=int, default=MIN_QUESTIONS,
                    help=f"floor per passage (default: {MIN_QUESTIONS})")
    ap.add_argument("--max-questions", type=int, default=MAX_QUESTIONS,
                    help=f"ceiling per passage (default: {MAX_QUESTIONS})")
    ap.add_argument("--no-skeleton", action="store_true",
                    help="skip the cross-section pass on long articles")
    ap.add_argument("--skeleton-min-chars", type=int, default=SKELETON_MIN_CHARS,
                    help=f"body size that earns a cross-section pass "
                         f"(default: {SKELETON_MIN_CHARS})")
    ap.add_argument("--skeleton-section-chars", type=int,
                    default=SKELETON_SECTION_CHARS,
                    help=f"chars of each section in the digest "
                         f"(default: {SKELETON_SECTION_CHARS})")
    ap.add_argument("--evidence-window", type=int, default=EVIDENCE_WINDOW,
                    help=f"chars either side of the cited span that a question's "
                         f"names and numbers must fall inside "
                         f"(default: {EVIDENCE_WINDOW})")
    ap.add_argument("--keep-evidence", action="store_true",
                    help="write the cited span into each record. Off by "
                         "default -- it is a validation artefact, not part of "
                         "the dataset -- but useful for auditing a run")
    ap.add_argument("--eval-fraction", type=float, default=DEFAULT_EVAL_FRACTION,
                    help=f"fraction of ARTICLES held out as the eval split "
                         f"(default: {DEFAULT_EVAL_FRACTION}). 0 disables it")
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="per-request timeout")
    ap.add_argument("--force", action="store_true",
                    help="re-question articles already in the output")
    ap.add_argument("--retry-failed", action="store_true",
                    help="also retry articles recorded in failures.jsonl")
    ap.add_argument("--thinking", action="store_true",
                    help="let the model reason before answering. Off by "
                         "default: it bills as output (~70%% of spend on the "
                         "paraphrase run) and writing a question from a "
                         "passage in front of you does not need it")
    ap.add_argument("--dry-run", action="store_true",
                    help="estimate questions, tokens, cost and runtime, then exit")
    ap.add_argument("--price-in", type=float, default=pc.DEFAULT_PRICE_IN,
                    help=f"$/1M cache-miss input (default: {pc.DEFAULT_PRICE_IN})")
    ap.add_argument("--price-out", type=float, default=pc.DEFAULT_PRICE_OUT,
                    help=f"$/1M output (default: {pc.DEFAULT_PRICE_OUT})")
    ap.add_argument("--price-cached", type=float, default=pc.DEFAULT_PRICE_CACHED,
                    help=f"$/1M cache-hit input (default: {pc.DEFAULT_PRICE_CACHED})")
    args = ap.parse_args()

    if not SOURCE_DIR.exists():
        sys.exit(f"No corpus at {SOURCE_DIR}. Run download_wookieepedia.py first.")

    types = parse_types(args.types)
    # The cross-section pass gets the same taxonomy filtered to the types a
    # digest can support, so --types still governs it.
    skeleton_types = [t for t in types if t in SKELETON_TYPES] or types
    systems = (build_system_prompt(types),
               build_system_prompt(skeleton_types, skeleton=True))

    paths = pc.iter_sources(args.limit, args.sample, args.seed)
    if not paths:
        sys.exit(f"No .md files under {SOURCE_DIR}.")

    if args.dry_run:
        estimate(paths, args, systems)
        return

    total = len(paths)
    done, seen = load_done()
    if not args.force:
        paths = [p for p in paths
                 if p.relative_to(CORPUS_DIR).as_posix() not in done]
    if not args.retry_failed:
        failed = load_failures()
        paths = [p for p in paths
                 if p.relative_to(CORPUS_DIR).as_posix() not in failed]

    print(f"{total:,} articles found, {len(paths):,} to do "
          f"({total - len(paths):,} already done or previously failed)")
    if seen:
        print(f"{len(seen):,} questions already written -- they seed the "
              f"duplicate filter")
    if not paths:
        return

    stats = Stats()
    started = time.time()
    try:
        asyncio.run(run(paths, args, systems, stats, seen))
    except KeyboardInterrupt:
        print("\ninterrupted -- rerun the same command to resume")

    elapsed = time.time() - started
    if stats.fatal:
        print(f"\nRUN ABORTED -- {stats.fatal}")
        print("This is an account/config fault, not a problem with any article.")
        print("  402 Insufficient Balance -> top up at platform.deepseek.com")
        print("  401/403                  -> check DEEPSEEK_API_KEY")
        print("  404                      -> wrong --model id")
        print(f"\n{stats.questions:,} questions written before the abort.")
        sys.exit(1)

    miss = stats.tokens_in - stats.tokens_cached
    cost = (miss / 1e6 * args.price_in
            + stats.tokens_cached / 1e6 * args.price_cached
            + stats.tokens_out / 1e6 * args.price_out)
    print(f"\n{stats.questions:,} questions from {stats.articles:,} articles "
          f"| skipped {stats.skipped:,} | failed {stats.failed:,}")
    if stats.articles:
        print(f"  {stats.questions / stats.articles:.1f} questions per article")
    if stats.skeleton_questions:
        print(f"  {stats.skeleton_questions:,} from the cross-section pass")
    if stats.eval_questions:
        print(f"  {stats.eval_questions:,} in the eval split "
              f"({stats.eval_questions/max(stats.questions,1):.1%})")
    print(f"requests {stats.requests:,} (retries {stats.retries:,}) | "
          f"tokens {stats.tokens_in:,} in / {stats.tokens_out:,} out "
          f"| ~${cost:,.2f}")
    if stats.tokens_in:
        print(f"  input cache hits {stats.tokens_cached/stats.tokens_in:.0%} "
              f"({stats.tokens_cached:,} of {stats.tokens_in:,})")
    if stats.tokens_reasoning:
        share = stats.tokens_reasoning / max(stats.tokens_out, 1)
        print(f"  reasoning tokens {stats.tokens_reasoning:,} ({share:.0%} of "
              f"output, ~${stats.tokens_reasoning/1e6*args.price_out:,.2f} "
              f"discarded) -- drop --thinking to save it")
    if stats.by_type:
        spread = sorted(stats.by_type.items(), key=lambda kv: -kv[1])
        print("type spread: " + ", ".join(f"{k} {v:,}" for k, v in spread))
    if stats.rejects:
        # Grouped: the per-term reasons would otherwise be one bucket per name.
        grouped: dict[str, int] = {}
        for reason, n in stats.rejects.items():
            grouped[reason.split(":")[0]] = grouped.get(reason.split(":")[0], 0) + n
        top = sorted(grouped.items(), key=lambda kv: -kv[1])
        total_rejected = sum(grouped.values())
        asked = total_rejected + stats.questions
        print(f"rejected {total_rejected:,} of {asked:,} candidates "
              f"({total_rejected/max(asked,1):.1%}): "
              + ", ".join(f"{k} x{v:,}" for k, v in top))
    if stats.reject_samples:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        path = STATE_DIR / "rejected_samples.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for bucket in stats.reject_samples.values():
                for row in bucket:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {sum(len(b) for b in stats.reject_samples.values()):,} "
              f"rejected examples -> {path}")
    if stats.duplicates or stats.near_dupes:
        print(f"dropped {stats.duplicates:,} duplicates, "
              f"{stats.near_dupes:,} near-duplicates within an article")
    if stats.errors:
        top = sorted(stats.errors.items(), key=lambda kv: -kv[1])[:5]
        print("top failure reasons: " + ", ".join(f"{k} x{v}" for k, v in top))
    print(f"elapsed {elapsed/60:.1f} min -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
