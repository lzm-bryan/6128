#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub 目录爬虫（API 优先；可选 HTML 兜底）
- 递归抓取指定目录（含子目录）下的文件
- 过滤：按扩展名 / include 正则 / exclude 正则
- 本地落盘：保持目录结构（或可扁平化）
- 仅用标准库；支持 GITHUB_TOKEN 提升速率上限

用法示例：
  1) 指定 GitHub 目录 URL（/tree/...）：
     python github_dir_crawler.py https://github.com/location-competition/indoor-location-competition-20/tree/master/data/site1/B1/path_data_files -o out -e .txt

  2) 指定 owner/repo + 路径：
     python github_dir_crawler.py location-competition/indoor-location-competition-20 --path data/site1/B1/path_data_files -o out -e .txt

  3) 指定分支（默认自动发现）：
     python github_dir_crawler.py owner/repo --path data --branch main -o out -e .json .txt

  4) 只走 API、禁用 HTML 兜底：
     python github_dir_crawler.py <src> --path ... --api-only

环境变量：
  GITHUB_TOKEN  # 可选，设置后可提升 API 限额与稳定性
"""

import os, sys, re, json, time, argparse, logging, hashlib
from urllib.parse import urlparse
from urllib.request import urlopen, Request
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("crawler")

# ---------------- HTTP helpers ----------------
def http_get(url: str, headers: Dict[str, str], timeout: int = 60) -> Tuple[bytes, Dict[str, str], int]:
    req = Request(url, headers=headers)
    with urlopen(req, timeout=timeout) as r:
        data = r.read()
        resp_headers = {k.lower(): v for k, v in r.headers.items()}
        status = r.getcode()
        return data, resp_headers, status

def gh_headers() -> Dict[str, str]:
    hdrs = {
        "User-Agent": "github-dir-crawler/1.0",
        "Accept": "application/vnd.github+json",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        hdrs["Authorization"] = f"Bearer {token}"
    return hdrs

# ---------------- GitHub parsing ----------------
def parse_source(source: str, path: Optional[str], branch: Optional[str]) -> Tuple[str, str, str, str]:
    """
    返回 (owner, repo, branch, subpath)
    - source 支持：'owner/repo' 或 'https://github.com/owner/repo[/tree/<branch>/<path>]'
    - 若未指定分支，则通过仓库 API 自动获取 default_branch
    """
    owner = repo = ""
    subpath = path or ""
    br = branch or ""

    if source.startswith("http"):
        p = urlparse(source)
        parts = [x for x in p.path.strip("/").split("/") if x]
        if len(parts) < 2:
            raise ValueError("无法解析 GitHub URL：缺少 owner/repo")
        owner, repo = parts[0], parts[1]
        if len(parts) >= 4 and parts[2] in ("tree", "blob"):
            br = br or parts[3]
            subpath = "/".join(parts[4:]) if len(parts) >= 5 else (subpath or "")
        else:
            # 仅到仓库根；subpath 仍由 --path 控制
            pass
    else:
        # 形如 'owner/repo'
        if "/" not in source:
            raise ValueError("source 需要形如 'owner/repo' 或 GitHub 仓库 URL")
        owner, repo = source.split("/", 1)

    if not br:
        br = get_default_branch(owner, repo)
        log.info(f"自动识别默认分支：{br}")

    return owner, repo, br, (subpath or "").strip("/")

def get_default_branch(owner: str, repo: str) -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}"
    data, hdrs, code = http_get(url, gh_headers())
    if code != 200:
        # 常见：私有仓库或限流；回退使用 'master' 再 'main'
        log.warning(f"获取默认分支失败（HTTP {code}），回退 'master/main'")
        return "master"
    meta = json.loads(data.decode("utf-8", "ignore"))
    return meta.get("default_branch", "master")

# ---------------- Listing via API ----------------
def list_files_via_trees_api(owner: str, repo: str, branch: str, prefix: str) -> List[str]:
    """
    使用 Git Trees 递归列出所有文件（一次 API 调用）
    参考：GET /repos/{owner}/{repo}/git/trees/{branch}?recursive=1
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    data, hdrs, code = http_get(url, gh_headers())
    if code != 200:
        raise RuntimeError(f"Trees API 失败：HTTP {code}")
    obj = json.loads(data.decode("utf-8", "ignore"))
    out = []
    for item in obj.get("tree", []):
        if item.get("type") == "blob":
            path = item.get("path", "")
            if not prefix or path.startswith(prefix + "/") or path == prefix:
                out.append(path)
    return out

def list_files_via_contents_api(owner: str, repo: str, branch: str, prefix: str) -> List[str]:
    """
    使用 Contents API 递归列出文件（多次调用）
    GET /repos/{owner}/{repo}/contents/{path}?ref={branch}
    """
    base = f"https://api.github.com/repos/{owner}/{repo}/contents"
    headers = gh_headers()
    out = []

    def walk(path: str):
        url = f"{base}/{path}?ref={branch}" if path else f"{base}?ref={branch}"
        data, hdrs, code = http_get(url, headers)
        if code == 403 and "x-ratelimit-remaining" in hdrs and hdrs["x-ratelimit-remaining"] == "0":
            reset = int(hdrs.get("x-ratelimit-reset", "0"))
            wait = max(0, reset - int(time.time()) + 1)
            log.warning(f"Rate limited. 等待 {wait}s 后重试...")
            time.sleep(wait)
            data, hdrs, code = http_get(url, headers)
        if code != 200:
            raise RuntimeError(f"Contents API 失败：HTTP {code} @ {url}")
        arr = json.loads(data.decode("utf-8", "ignore"))
        if isinstance(arr, dict) and arr.get("type") == "file":
            out.append(arr["path"])
            return
        for item in arr:
            if item.get("type") == "file":
                out.append(item["path"])
            elif item.get("type") == "dir":
                walk(item.get("path"))
    walk(prefix)
    return out

# ---------------- HTML fallback (optional) ----------------
def list_txt_links_from_html(url: str) -> List[str]:
    """
    轻量兜底：在目录页面（/tree/...）里解析 .txt 链接（只适用于简单场景）
    """
    headers = {"User-Agent": "github-dir-crawler/1.0"}
    data, hdrs, code = http_get(url, headers)
    if code != 200:
        raise RuntimeError(f"HTML 解析失败：HTTP {code}")
    html = data.decode("utf-8", "ignore")
    hrefs = re.findall(r'href="([^"]+\.txt)"', html)
    full = []
    for h in hrefs:
        if not h.startswith("http"):
            h = f"https://github.com{h}"
        if "/blob/" not in h:
            continue
        full.append(h)
    return full

# ---------------- Download ----------------
def raw_url(owner: str, repo: str, branch: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"

def save_file(url: str, dst: Path, headers: Dict[str, str], delay: float = 0.0, force: bool = False):
    if dst.exists() and dst.stat().st_size > 0 and not force:
        log.debug(f"SKIP exists: {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if delay > 0:
        time.sleep(delay)
    data, hdrs, code = http_get(url, headers)
    if code != 200:
        raise RuntimeError(f"下载失败：HTTP {code} @ {url}")
    with open(dst, "wb") as f:
        f.write(data)

# ---------------- Filters ----------------
def pass_filters(path: str, exts: List[str], include: Optional[str], exclude: Optional[str]) -> bool:
    if exts:
        ok = any(path.lower().endswith(e.lower()) for e in exts)
        if not ok:
            return False
    if include:
        if not re.search(include, path):
            return False
    if exclude:
        if re.search(exclude, path):
            return False
    return True

# ---------------- Manifest ----------------
def write_manifest(paths: List[str], owner: str, repo: str, branch: str, prefix: str, outdir: Path):
    meta = {
        "owner": owner,
        "repo": repo,
        "branch": branch,
        "prefix": prefix,
        "count": len(paths),
        "files": paths,
    }
    (outdir / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------- CLI ----------------
def build_argparser():
    ap = argparse.ArgumentParser(description="GitHub 目录爬虫（API 优先；可选 HTML 兜底）")
    ap.add_argument("source", help="GitHub 仓库 URL 或 'owner/repo'")
    ap.add_argument("--path", default="", help="仓库内子目录（例如 data/site1/B1/path_data_files）")
    ap.add_argument("--branch", default="", help="分支（默认自动发现）")
    ap.add_argument("-o", "--out", default="download_cache", help="输出目录")
    ap.add_argument("-e", "--ext", nargs="*", default=[".txt"], help="只下载这些扩展名（多个，用空格分隔）")
    ap.add_argument("--include", default="", help="包含正则（匹配完整路径）")
    ap.add_argument("--exclude", default="", help="排除正则（匹配完整路径）")
    ap.add_argument("--api-only", action="store_true", help="仅使用 GitHub API（禁用 HTML 兜底）")
    ap.add_argument("--use-contents", action="store_true", help="用 Contents API（默认 Trees API）")
    ap.add_argument("--flatten", action="store_true", help="扁平化保存（不保留子目录结构）")
    ap.add_argument("--max-files", type=int, default=0, help="最多下载前 N 个（0 表示不限）")
    ap.add_argument("--delay", type=float, default=0.0, help="每个文件下载间隔秒数（限速用）")
    ap.add_argument("--force", action="store_true", help="总是重新下载（覆盖已有文件）")
    return ap

def main():
    args = build_argparser().parse_args()
    try:
        owner, repo, branch, subpath = parse_source(args.source, args.path, args.branch)
    except Exception as e:
        log.error(f"解析 source 失败：{e}")
        sys.exit(2)

    log.info(f"目标仓库：{owner}/{repo}  分支：{branch}  子目录：{subpath or '(repo root)'}")
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) 列表
    all_paths: List[str] = []
    try:
        if args.use_contents:
            all_paths = list_files_via_contents_api(owner, repo, branch, subpath)
        else:
            all_paths = list_files_via_trees_api(owner, repo, branch, subpath)
        log.info(f"API 列出文件共 {len(all_paths)} 个")
    except Exception as api_err:
        if args.api_only:
            log.error(f"API 失败且已禁用 HTML 兜底：{api_err}")
            sys.exit(3)
        # 尝试 HTML 兜底，仅适用于 /tree/ 页面且只抓 .txt
        src_url = args.source if args.source.startswith("http") else f"https://github.com/{owner}/{repo}/tree/{branch}/{subpath}"
        try:
            txt_links = list_txt_links_from_html(src_url)
            log.warning(f"改用 HTML 兜底解析，发现 .txt 文件 {len(txt_links)} 个")
            # 直接下载这些 .txt
            headers = {"User-Agent": "github-dir-crawler/1.0"}
            files = []
            for u in txt_links:
                # 转 raw
                ru = u.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/","/")
                files.append((ru, os.path.basename(urlparse(ru).path)))
            if args.max_files and args.max_files > 0:
                files = files[:args.max_files]
            for i, (ru, name) in enumerate(files, 1):
                dst = (outdir / name) if args.flatten else (outdir / subpath / name)
                log.info(f"[{i}/{len(files)}] {ru} -> {dst}")
                save_file(ru, dst, headers, delay=args.delay, force=args.force)
            # 写 manifest
            write_manifest([f for _, f in files], owner, repo, branch, subpath, outdir)
            log.info("完成（HTML 兜底路径）。")
            return
        except Exception as e2:
            log.error(f"HTML 兜底也失败：{e2}")
            sys.exit(4)

    # 2) 过滤
    flt_paths = []
    for p in sorted(all_paths):
        if not subpath:
            rel = p
        else:
            # 只保留 prefix 子树
            if not (p == subpath or p.startswith(subpath + "/")):
                continue
            rel = p[len(subpath):].lstrip("/") if p != subpath else os.path.basename(p)
        if pass_filters(p, args.ext, args.include or None, args.exclude or None):
            flt_paths.append(p)
    if args.max_files and args.max_files > 0:
        flt_paths = flt_paths[:args.max_files]

    if not flt_paths:
        log.warning("没有匹配到任何文件。请检查 --path / --ext / include / exclude 参数。")
        sys.exit(0)

    # 3) 下载
    headers = gh_headers()
    for i, p in enumerate(flt_paths, 1):
        ru = raw_url(owner, repo, branch, p)
        dst = (outdir / os.path.basename(p)) if args.flatten else (outdir / p)
        log.info(f"[{i}/{len(flt_paths)}] {ru} -> {dst}")
        try:
            save_file(ru, dst, headers, delay=args.delay, force=args.force)
        except Exception as e:
            log.error(f"下载失败：{e}")

    # 4) manifest
    write_manifest(flt_paths, owner, repo, branch, subpath, outdir)
    log.info("全部完成。")

if __name__ == "__main__":
    main()

# python github_dir_crawler.py https://github.com/location-competition/indoor-location-competition-20/tree/master/data/site1/B1/path_data_files -o indoor_cache/site1/B1 -e .txt

# python github_dir_crawler.py location-competition/indoor-location-competition-20 --path data/site1/B1/path_data_files -o out -e .txt --include 5720e
#
# python github_dir_crawler.py owner/repo --path some/dir -o out -e .json --api-only
