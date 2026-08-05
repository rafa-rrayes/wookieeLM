#!/usr/bin/env python3
"""Upload dist/ to a Hugging Face dataset repository.

``package.py`` already writes everything the Hub needs -- the shards, a
``README.md`` whose YAML frontmatter declares the configs, a datasheet, and
checksums -- so this is only the transfer.

    uv run scripts/upload_hf.py                          # dry run: list what would go
    uv run scripts/upload_hf.py --repo user/name --push  # actually upload
    uv run scripts/upload_hf.py --repo user/name --push --private

Requires ``huggingface_hub`` and a token, from ``hf auth login`` or ``HF_TOKEN``:

    uv add huggingface_hub

Two guards, because an upload cannot be taken back once it has been indexed:

1. ``--push`` is required. Without it nothing leaves the machine.
2. A build containing ``restricted/`` is refused outright. That directory only
   exists when ``package.py --include-restricted`` put copyrighted novels,
   screenplays or subtitles in it, and no flag here overrides the refusal --
   rebuild without ``--include-restricted`` instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = REPO_ROOT / "dist"


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} TB"


def survey(dist: Path) -> tuple[list[Path], int]:
    files = sorted(p for p in dist.rglob("*") if p.is_file())
    return files, sum(p.stat().st_size for p in files)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dist", type=Path, default=DIST_DIR,
                    help="directory to upload (default: dist/)")
    ap.add_argument("--repo", type=str, default=None,
                    help="target dataset repo id, e.g. user/wookieelm-corpus")
    ap.add_argument("--push", action="store_true",
                    help="actually upload; without it this is a dry run")
    ap.add_argument("--private", action="store_true",
                    help="create the repo private")
    ap.add_argument("--message", type=str, default="Upload WookieeLM corpus",
                    help="commit message for the upload")
    args = ap.parse_args()

    dist = args.dist
    if not dist.is_dir():
        print(f"{dist} does not exist -- run `uv run scripts/package.py` first.")
        return 1

    if (dist / "restricted").is_dir():
        print(f"refusing to upload: {dist}/restricted/ exists.\n\n"
              "That directory holds copyrighted novels, screenplays or\n"
              "subtitles, added by `package.py --include-restricted`. Rebuild\n"
              "without that flag before uploading.")
        return 1

    files, total = survey(dist)
    if not files:
        print(f"{dist} is empty.")
        return 1

    if not (dist / "README.md").exists():
        print(f"warning: {dist}/README.md is missing -- the Hub will show no "
              "dataset card and will not detect the configs.")

    print(f"{dist} -> {args.repo or '<--repo not set>'}")
    print(f"  {len(files)} files, {human(total)}\n")
    for path in files:
        print(f"    {path.relative_to(dist).as_posix():<50} "
              f"{human(path.stat().st_size):>10}")

    if not args.push:
        print("\ndry run -- nothing uploaded. Add --repo <id> --push to upload.")
        return 0
    if not args.repo:
        print("\n--push needs --repo <user/name>.")
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("\nhuggingface_hub is not installed:  uv add huggingface_hub")
        return 1

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=args.private,
                    exist_ok=True)
    api.upload_folder(folder_path=str(dist), repo_id=args.repo,
                      repo_type="dataset", commit_message=args.message)
    print(f"\nuploaded -> https://huggingface.co/datasets/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
