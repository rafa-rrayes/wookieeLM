# Data licensing

The MIT license in `LICENSE` covers the code. It does not cover a single word
of the text this pipeline downloads, rewrites, or generates. That text carries
the license of whoever wrote it, and those licenses differ by source.

## What each source is

| Source | License | Attribution |
|---|---|---|
| `corpus/wookieepedia` | CC BY-SA 3.0 | Wookieepedia contributors, starwars.fandom.com |
| `corpus/wookieepedia_paraphrased1` | CC BY-SA 3.0 (derivative) | as above |
| `corpus/wikipedia` | CC BY-SA 4.0 | Wikipedia contributors, en.wikipedia.org |
| `questions/`, `sft/` | CC BY-SA 3.0 (derivative) | as above |
| `corpus/books` | **All rights reserved** | various publishers |
| `corpus/scripts` | **All rights reserved** | Lucasfilm Ltd. / various |
| `corpus/subtitles` | **All rights reserved** | various rights holders |

## Share-alike reaches further than people expect

Three things in this repository are derivative works of CC BY-SA text, and
therefore inherit share-alike:

- **The paraphrases.** Rewriting an article in different words does not create
  a new work free of the original's license. `wookieepedia_paraphrased1` is
  CC BY-SA 3.0, exactly as its source is.
- **The questions.** Each one is mined from a specific article and ships with
  an `evidence` span quoted from it.
- **The answers.** The teacher model was given the corpus as its only source of
  fact and was forbidden from answering from memory, which is the whole point
  of the setup — and which means the answers are built from CC BY-SA text.

If you redistribute any of these, redistribute them under CC BY-SA and keep the
attribution. Every row in the packaged Parquet carries `license` and `url`
columns so this stays possible downstream.

A model's *weights* are a separate question that this file does not try to
answer.

## The three sources that are not redistributable

`corpus/books`, `corpus/scripts` and `corpus/subtitles` are copyrighted novels,
screenplays, and subtitle rips. They are in the pipeline because they are good
training data for a personal model. They are not in any published dataset, and
they must not be:

- `package.py` excludes them by default. `--include-restricted` writes them to
  `dist/restricted/` for local use and stamps both the dataset card and the
  datasheet as non-redistributable.
- `upload_hf.py` refuses to upload any tree containing `dist/restricted/`, with
  no flag to override it.

The pipeline does not fetch these for you. You supply your own copies.

## If you publish a dataset built with this

- Keep the `license` and `url` columns, or otherwise carry the attribution.
- License the result CC BY-SA, and say which version applies to which rows.
- Ship `DATASHEET.md` — it records what is incomplete and what is unaudited.
- Do not include `restricted/`.
