# Unpin answer_questions.py from one CPU core

## Diagnosis (measured 2026-08-03, live PID 50399)

- Process sat at **89–99% of one core** on a 10-core machine (`cputime` delta
  7.14 s over 8 s wall). Nine cores idle. Throughput 12.96 q/s at
  `--concurrency 192` — more in-flight requests bought nothing.
- Cause: every corpus scan holds the GIL start to finish.
  - `bytearray.find` over the 365 MB buffer: **275 ms**, C-level memmem, does
    not release the GIL.
  - `search_titles` linear scan over 171,440 keys: **79 ms**, pure Python.
  - `asyncio.to_thread` does not help — it moves the call off the loop's stack
    while still holding the GIL.
- Demand: 12.96 q/s x 0.54 `search_*` calls/question = ~7 scans/s. At ~0.28 s
  each that is ~1.0 core-second per wall second. Exactly the measurement.

## Plan

- [x] 1. Move the full-text scan off the GIL
  - [x] Build the search buffer into a **persisted file** under `.index/`
        (`text.bin` + `starts.bin` + `meta.json`) instead of an in-RAM
        `bytearray`. Parent and workers both `mmap` it, so the OS page cache
        holds one copy. Bonus: a rerun skips the ~9 s build.
  - [x] `SearchPool`: `ProcessPoolExecutor` whose workers mmap the same file at
        init. A scan is sharded across workers on article boundaries and merged
        in the parent. Sync API, so the existing `to_thread` call now genuinely
        releases the GIL while it blocks on IPC.
  - [x] `Corpus.scanner` hook so `run_tool` -> `tool_search_text` ->
        `count_matches` parallelises with no plumbing through the tool layer.
- [x] 2. Fix the tool memo cache
  - [x] `ToolRunner.cache` never evicted — it stopped inserting at
        `cache_size` and froze. Replace with an LRU.
- [x] 3. Adjacent GIL costs found while measuring (same root cause)
  - [x] `search_titles`: scan a joined key blob with `str.find` (C speed)
        instead of 171k Python iterations.
  - [x] `tool_search_text` recomputed `normalize(path.stem)` per matching
        article; `corpus.keys[i]` already holds it.
  - [x] `seed_messages` did blocking file IO in the event loop.
- [x] 4. Verify: new scanner returns byte-identical results to the old one.

## Review

**Measured, real corpus (171,440 articles / 365 MB), 9 representative queries:**

```
in-process : 2.34s for 9 scans (261 ms each)
pooled(8)  : 0.41s for 9 scans ( 46 ms each)  ->  5.7x
identical results: YES
```

Index build 9.0 s the first time, **0.0 s** thereafter (reused from `.index/`).

**Correctness** — `tests/test_scan.py` keeps the previous `search_titles` and
`count_matches` verbatim as reference implementations and asserts identical
output:

- 91 queries, whole-range scan == reference
- 7 shard counts x 91 queries, sharded+merged == unsharded (and shards never
  double-count: their article ranges are checked disjoint)
- 7 regex patterns, 8 shards == unsharded
- 30 queries through the real `SearchPool` == in-process
- 101 queries x 4 limits, `search_titles` == reference
- index cache reused byte-identically, and invalidated by a new article

**Files**

- `corpus_scan.py` (new) — `scan_range`, the mmap opener, `SearchPool`. Its own
  module because `wookiee_chat.py` is loaded by path as `wc`, a name no worker
  process can import.
- `wookiee_chat.py` — index moved from an in-RAM `bytearray` to a persisted
  file every process mmaps; `count_matches_idx` returns article indices;
  `search_titles` scans a joined key blob; `Corpus.scanner` hook.
- `answer_questions.py` — LRU tool cache; pool started and shut down around the
  run; `get_seed` off the event loop and stampede-proof; `--search-workers`,
  `--rebuild-index`; `--cache-size` default 4000 -> 40000.

**Not done / notes**

- The `cap` is now divided across shards rather than applied globally. It only
  feeds the word "over" in front of a result count, and nothing else reads it.
- `_stamp()` does not notice an article edited **in place** (same name, same
  shard-directory mtime). `--rebuild-index` covers that.
- `.index/` holds ~365 MB. It is derived data; delete it freely.
- Worker processes re-import the main script (~97 ms each, once). Harmless
  because `main()` is `__name__`-guarded — but any script that builds a pool
  must keep that guard.

---

# Prepare the repo for publishing, and the data for distribution

Target: github.com/rafa-rrayes/wookieeLM (the v1 repo — this replaces its
`src/wookielm/` layout with the flat scripts here). Decisions taken up front:
copyrighted sources excluded from anything distributable but their build code
kept, Parquet/HF format, MIT for the code, CC BY-SA inherited by the data.

## Plan

- [x] 1. Repo files
  - [x] `README.md` — pipeline diagram, corpus tables, quickstart, layout,
        an explicit "what is not finished" section.
  - [x] `LICENSE` (MIT, code only) and `DATA_LICENSE.md` (CC BY-SA inheritance,
        including that it reaches the paraphrases and the answers).
  - [x] `.env.example`; `.gitignore` extended (`dist/`, `.index/`, `*_state/`,
        `questions/`, `sft/`, `.DS_Store`).
  - [x] `pyproject.toml` — description rewritten for the whole pipeline,
        license, authors, urls, `pyarrow`, `[upload]` extra.
- [x] 2. `package.py` — corpus + JSONL -> Parquet shards, dataset card,
      datasheet, SHA256SUMS. Reuses `paraphrase_corpus.split_document`.
- [x] 3. `upload_hf.py` — dry-run by default; refuses any tree containing
      `dist/restricted/`, with no override.
- [x] 4. `tests/test_package.py` — 37 checks against a fixture corpus.
- [x] 5. Full build + verification.

## Review

**Built** (2026-08-04): 543,708 rows, ~224M tokens, 769.9 MB raw -> 225.7 MB of
Parquet (3.4x). Row counts match `count.py` exactly for all three corpora.

**Verified**

- 37/37 packager checks; `tests/test_scan.py` still green.
- Every shard reloaded: 0 null or empty `text`/`q`, one schema per source.
- Train/eval share 0 source articles, in both `sft` and `questions`.
- Continuity histogram matches `corpus/wookieepedia/manifest.md` exactly
  (84,522 / 49,309 / 33,652 / 3,957).
- Two full builds produce byte-identical parquet shards — checked on the real
  corpus, not a fixture: all 9 shards match across independent runs.

**Three things the build caught that documentation would have shipped wrong**

- `enumerate_documents` filtered out names starting with `.`, which was meant
  for `.DS_Store` and actually dropped `_/.48-caliber_Enforcer_pistol.md`, a
  real article. The suffix test already excludes `.DS_Store`. The dotfile guard
  is gone and a fixture article now covers it.
- The datasheet called the paraphrase gap "98.9% complete". It is complete:
  all 169,505 paraphrasable articles are done, and the 1,935 others are index
  lists and one table-only article that `paraphrase_corpus.py` skips on
  purpose. Reporting a design decision as a shortfall.
- The SFT coverage denominator was `questions + answers`, on the assumption
  that `questions.jsonl` held only unanswered ones. It holds the whole pool, so
  the figure read 8.1% against a denominator of 201,760 that does not exist.
  Correct: 16,437 of 185,323 = 8.9%, which is what `count.py` says. Both the
  card and the datasheet now derive that figure from the build's own stats, so
  it cannot drift again as the answering run continues.

**Not done / notes**

- `git init` run and everything staged; **not committed**. The v1 repo has a
  `src/wookielm/` layout that a push from here replaces.
- Nothing uploaded. `uv run upload_hf.py --repo <id> --push` does that.
- `dist/` is gitignored: 226 MB, and it is the Hub's copy, not git's.
- The corpus numbers in `README.md` are a dated snapshot. The Q&A stages append
  as they run, so they go stale; `count.py` is the live view.
