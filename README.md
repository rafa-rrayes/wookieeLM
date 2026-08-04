# wookieeLM

A pipeline that turns Wookieepedia into LLM training data, end to end: a
1.9 GB MediaWiki dump becomes 171,440 Markdown articles, those get paraphrased
for knowledge augmentation, questions are mined from them with evidence spans,
and a tool-using teacher that may only read the corpus answers those questions
into an SFT set.

Everything is packaged into Parquet by `package.py` and is reproducible from
this repository. Nothing but the code is committed — the corpus is ~3.7 GB and
is rebuilt, not stored.

## Corpus

`uv run count.py`, snapshot of 2026-08-04. The Q&A stages are still running, so
those two numbers grow; run `count.py` for the current ones.

| Source | Documents | Size | ~Tokens |
|---|---:|---:|---:|
| `wookieepedia` | 171,440 | 347.2 MB | 101.1M |
| `wookieepedia_paraphrased1` \* | 169,505 | 366.3 MB | 106.7M |
| `books` † | 50 | 29.7 MB | 8.6M |
| `wikipedia` | 1,003 | 17.0 MB | 4.9M |
| `subtitles` † | 371 | 4.2 MB | 1.2M |
| `scripts` † | 8 | 1.2 MB | 355K |
| **Total** | **172,872** | **399.3 MB** | **116.3M** |

\* A rewrite of `wookieepedia`, so it is not in the total. With it, ~208M tokens.
† Copyrighted. Never published — see [DATA_LICENSE.md](DATA_LICENSE.md).

| Q&A | Count | Notes |
|---|---:|---|
| Questions generated | 185,323 | 181,279 train / 4,044 eval, from 58,053 articles (33.9%) |
| Questions answered (SFT) | 16,437 | 16,062 train / 375 eval, 8.9% of questions |

Train and eval are split by **article**, not by question, so no eval question
comes from an article a train question also came from. Verified on every build:
0 shared source articles.

Packaged, that is **543,708 rows / ~224M tokens / 226 MB** of Parquet.

## Pipeline

```
Wookieepedia XML dump
        │
        ├─ download_wookieepedia.py ──► corpus/wookieepedia/          171,440 .md
        │                                (continuity stamped into frontmatter)
        │
        ├─ paraphrase_corpus.py ──────► corpus/wookieepedia_paraphrased1/
        │                                (prose reworded; infoboxes and tables
        │                                 kept verbatim — no sentence structure
        │                                 to vary, so rewording is just noise)
        │
        └─ generate_questions.py ─────► questions/questions.jsonl
                    │                    (each question + the evidence span
                    │                     that answers it)
                    │
                    └─ answer_questions.py ──► sft/sft.jsonl
                            │                   (answered by wookiee_chat.py's
                            │                    agentic loop over the corpus)
                            │
                            └─ package.py ──► dist/*.parquet + dataset card
                                                    │
                                                    └─ upload_hf.py ──► the Hub
```

The teacher uses tools and the student will not. That is deliberate:
`wookiee_chat.py` is told its tools *are* its memory and that the reader must
never learn they exist, so its answers read as recall and carry no citations —
the shape a student trained on this corpus should produce. The tools are how
the teacher is kept honest, not part of what is being taught.

## Quickstart

```bash
uv sync
cp .env.example .env          # add DEEPSEEK_API_KEY

uv run download_wookieepedia.py   # dump -> corpus/wookieepedia/ (~1h, no API key)
uv run paraphrase_corpus.py       # -> corpus/wookieepedia_paraphrased1/
uv run generate_questions.py      # -> questions/questions.jsonl
uv run answer_questions.py        # -> sft/sft.jsonl
uv run package.py                 # -> dist/*.parquet + README + DATASHEET
```

Every stage resumes. They keep a failure ledger under `*_state/` and skip work
whose output already exists, so a killed run costs only what was in flight.

```bash
uv run count.py --watch 10        # live inventory + paraphrase progress
uv run wookiee_chat.py            # ask the corpus something
```

## Layout

```
download_wookieepedia.py   MediaWiki XML dump -> Markdown, continuity-tagged
paraphrase_corpus.py       article -> reworded twin, structure preserved
generate_questions.py      article -> questions + evidence spans
answer_questions.py        questions -> answers, via the tool-using teacher
wookiee_chat.py            the teacher: agentic loop over the corpus, 3 tools
corpus_scan.py             the full-text scan and its process pool
package.py                 corpus + JSONL -> Parquet shards + dataset card
upload_hf.py               dist/ -> a Hugging Face dataset repo
count.py                   inventory and progress across every stage

tests/                     uv run tests/test_scan.py, tests/test_package.py
tasks/                     working notes: todo.md, lessons.md

corpus/                    all text (gitignored, rebuilt by the pipeline)
dist/                      packaged Parquet (gitignored, rebuilt by package.py)
```

## Packaging

`package.py` is what makes the corpus usable by anyone else. 341,000 loose
files are the right shape for a resumable build — a paraphrase run resumes by
asking whether a path exists — and the wrong shape for a dataloader, which pays
a syscall per document.

```bash
uv run package.py --dry-run             # the plan, writes nothing
uv run package.py                       # dist/*.parquet, ~700 MB
uv run package.py --include-restricted  # + books/scripts/subtitles, local only
```

Output is deterministic: documents are enumerated in sorted path order and
sharded on a fixed byte budget, so the same corpus yields byte-identical shards
and a stable `SHA256SUMS`.

The `text` column carries a header — `# Title` and `*Wookieepedia · Legends*` —
before the article body. Canon and Legends contradict each other constantly,
and a model can only learn which timeline a fact belongs to if the label is in
the token stream; a Parquet column it never sees teaches it nothing. The same
values are kept as columns for filtering.

## What is not finished

Stated here rather than left to be discovered, and repeated in the generated
`DATASHEET.md`:

- The paraphrase pass is **complete**: all 169,505 paraphrasable articles are
  done. The 1,935 that were not sent are index lists and one table-only
  article, which `paraphrase_corpus.py` skips on purpose — rewording a list of
  links produces noise, not variety.
- Only **8.9%** of generated questions have answers. The `questions` config
  ships the whole pool — the ~169k unanswered ones carry their evidence span
  and are usable on their own; answering them is the highest-value work left.
- Questions cover **33.9%** of articles (58,053 of 171,440).
- `unsupported` is set on ~22% of SFT rows. The check flags an answer opening
  on a bare deictic about as often as a real fabrication, so rows are kept with
  the flag rather than dropped.
- Answers are model-generated and **unaudited by a human**.
- Eval questions come from articles that are themselves in the pretraining
  corpus. Eval measures recall of trained-on material, not generalization.

## Licensing

Code is MIT ([LICENSE](LICENSE)). The text is not — Wookieepedia is CC BY-SA
3.0, Wikipedia is CC BY-SA 4.0, and share-alike reaches the paraphrases, the
questions and the answers, all of which are derivative works. Novels,
screenplays and subtitles are copyrighted and are excluded from every published
build. [DATA_LICENSE.md](DATA_LICENSE.md) has the details and the guardrails.
