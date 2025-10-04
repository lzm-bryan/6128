#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gh_fetch_dir.py
---------------
Download ALL files (not just .txt) under a GitHub tree URL to a local directory,
preserving the subfolder structure. No repo ZIP snapshots are used.
Uses Git Tree API (recursive) and raw.githubusercontent.com for content.

Features:
- Concurrency (ThreadPoolExecutor)
- Optional include/exclude glob patterns
- Retries with exponential backoff
- Uses GITHUB_TOKEN/GH_TOKEN for API to avoid rate limits
- Windows-friendly path handling
- Dry-run

Usage:
  python gh_fetch_dir.py \
      --tree "https://github.com/location-competition/indoor-location-competition-20/tree/master/data" \
      --out ./indoor_data \
      --workers 8 \
      --include "*" \
      --exclude "*.md" \
      --dry-run 0

By default, include="*" and exclude="" (no excludes).
"""

import os, sys, re, json, time, math, argparse, fnmatch, pathlib
from urllib.parse import urlparse
from urllib.request import urlopen, Request
from concurrent.futures import ThreadPoolExecutor, as_completed

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0 Safari/537.36"

def eprint(*a, **k):
    print(*a, **k, file=sys.stderr)

def parse_tree_url(tree_url: str):
    """
    Parse a GitHub tree URL, return (owner, repo, branch, subpath).
    Accepts forms like:
      https://github.com/<owner>/<repo>/tree/<branch>/<subpath...>
      https://github.com/<owner>/<repo>/tree/<branch>
      https://github.com/<owner>/<repo>   (branch defaults to master, subpath empty)
    """
    p = urlparse(tree_url)
    parts = [x for x in p.path.strip("/").split("/") if x]
    if len(parts) < 2:
        raise ValueError("Invalid GitHub URL: need at least owner/repo")
    owner, repo = parts[0], parts[1]
    branch, subpath = "master", ""
    if len(parts) >= 4 and parts[2] == "tree":
        branch = parts[3]
        if len(parts) > 4:
            subpath = "/".join(parts[4:])
    elif len(parts) > 2:
        # e.g., https://github.com/owner/repo/path/without/tree
        subpath = "/".join(parts[2:])
    return owner, repo, branch, subpath

def req(url: str, token: str = None, accept_json: bool = False, timeout: int = 60) -> bytes:
    headers = {"User-Agent": UA}
    if accept_json:
        headers["Accept"] = "application/vnd.github+json"
    if token and "api.github.com" in url:
        headers["Authorization"] = f"Bearer {token}"
    r = Request(url, headers=headers)
    with urlopen(r, timeout=timeout) as resp:
        return resp.read()

def github_tree_recursive(owner: str, repo: str, branch: str, token: str = None) -> list:
    """
    Call Git Tree API with recursive=1. Returns list of dicts like:
    {"path": "...", "mode": "...", "type": "blob" or "tree", "sha": "..."}
    """
    api = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    data = req(api, token=token, accept_json=True)
    obj = json.loads(data.decode("utf-8", "ignore"))
    if "tree" not in obj:
        raise RuntimeError(f"Unexpected GitHub API response: keys={list(obj.keys())}")
    return obj.get("tree", [])

def should_take(path: str, subpath: str, include_globs: list, exclude_globs: list) -> bool:
    if subpath:
        if not path.startswith(subpath.rstrip("/") + "/") and path != subpath.rstrip("/"):
            return False
    # include patterns (any match -> include)
    if include_globs:
        if not any(fnmatch.fnmatch(path, pat) for pat in include_globs):
            return False
    # exclude patterns (any match -> drop)
    if exclude_globs:
        if any(fnmatch.fnmatch(path, pat) for pat in exclude_globs):
            return False
    return True

def raw_url(owner: str, repo: str, branch: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"

def download_one(url: str, out_path: str, retries: int = 3, backoff: float = 0.8) -> tuple:
    """
    Download url to out_path with retries. Returns (ok:bool, bytes:int).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    attempt = 0
    while True:
        try:
            r = Request(url, headers={"User-Agent": UA})
            with urlopen(r, timeout=120) as resp, open(out_path, "wb") as f:
                data = resp.read()
                f.write(data)
                return True, len(data)
        except Exception as e:
            attempt += 1
            if attempt > retries:
                eprint(f"FAILED: {url} -> {out_path} : {e}")
                return False, 0
            time.sleep(backoff * (2 ** (attempt - 1)))

def human(nbytes: int) -> str:
    if nbytes < 1024: return f"{nbytes} B"
    units = ["KB","MB","GB","TB"]
    x = float(nbytes)
    for u in units:
        x /= 1024.0
        if x < 1024.0:
            return f"{x:.2f} {u}"
    return f"{x:.2f} PB"

def main():
    ap = argparse.ArgumentParser(description="Download all files under a GitHub tree URL, preserving subfolders.")
    ap.add_argument("--tree", required=True, help="GitHub tree URL, e.g. https://github.com/owner/repo/tree/branch/path")
    ap.add_argument("--out", required=True, help="Local output directory")
    ap.add_argument("--workers", type=int, default=8, help="Concurrent workers")
    ap.add_argument("--include", default="*", help="Comma-separated glob patterns to include (default '*')")
    ap.add_argument("--exclude", default="", help="Comma-separated glob patterns to exclude (default none)")
    ap.add_argument("--dry-run", type=int, default=0, help="If 1, only list what would be downloaded")
    ap.add_argument("--print", dest="just_print", action="store_true", help="Alias of --dry-run=1")
    args = ap.parse_args()

    if args.just_print:
        args.dry_run = 1

    owner, repo, branch, subpath = parse_tree_url(args.tree)
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")

    print(f"[INFO] Repo: {owner}/{repo}, branch: {branch}, subpath: '{subpath}'")
    print(f"[INFO] Output: {args.out}")
    print(f"[INFO] Using token: {'YES' if token else 'NO'}")

    try:
        tree = github_tree_recursive(owner, repo, branch, token=token)
    except Exception as e:
        eprint(f"[ERROR] Git Tree API failed: {e}")
        eprint("Tips: set GITHUB_TOKEN to increase rate limits; ensure branch exists; verify the URL.")
        sys.exit(2)

    include_globs = [p.strip() for p in args.include.split(",") if p.strip()]
    exclude_globs = [p.strip() for p in args.exclude.split(",") if p.strip()]

    # Filter blobs within subpath
    blobs = [t for t in tree if t.get("type") == "blob"]
    candidates = []
    for it in blobs:
        p = it.get("path", "")
        if should_take(p, subpath, include_globs, exclude_globs):
            candidates.append(p)

    if not candidates:
        print("[INFO] No files matched. Check subpath/include/exclude patterns.")
        return

    print(f"[INFO] {len(candidates)} files to process.")
    if args.dry_run:
        for p in candidates:
            print(p)
        return

    total_bytes = 0
    ok_count, fail_count = 0, 0

    def task(path):
        url = raw_url(owner, repo, branch, path)
        out_path = os.path.join(args.out, path.replace("/", os.sep))
        ok, n = download_one(url, out_path)
        return (ok, n, path)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(task, p) for p in candidates]
        for fu in as_completed(futs):
            ok, n, p = fu.result()
            if ok:
                ok_count += 1
                total_bytes += n
                print(f"[OK] {p} ({human(n)})")
            else:
                fail_count += 1
                print(f"[FAIL] {p}")

    print(f"[DONE] ok={ok_count}, fail={fail_count}, total={human(total_bytes)}")

if __name__ == "__main__":
    main()

# python fetch_cache.py --tree "https://github.com/location-competition/indoor-location-competition-20/tree/master/data" --out ".\indoor_data" --workers 8
