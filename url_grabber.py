#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json
from urllib.parse import urlparse
from urllib.request import Request, urlopen

# ========== 可选设置 ==========
OUT_DIR = "downloaded"       # 保存目录
EXT_FILTER = [".txt"]        # 目录下载时的后缀过滤；设为 [] 或 None 表示不过滤
# ============================

def hdr():
    h = {"User-Agent": "url-grabber/2.0", "Accept": "*/*"}
    tok = os.getenv("GITHUB_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
        h["Accept"] = "application/vnd.github+json"
    return h

def http_get(url: str) -> bytes:
    with urlopen(Request(url, headers=hdr()), timeout=60) as r:
        return r.read()

def to_raw_blob(url: str) -> str:
    # 把 github.com 的 /blob/ 文件链接转为 raw
    return url.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/")

def is_github_blob(url: str) -> bool:
    return ("github.com" in url and "/blob/" in url) or ("raw.githubusercontent.com" in url)

def is_github_tree(url: str) -> bool:
    return ("github.com" in url and "/tree/" in url)

def parse_tree(url: str):
    """
    https://github.com/<owner>/<repo>/tree/<branch>/<subpath...>
    -> owner, repo, branch, subpath
    """
    p = urlparse(url)
    parts = [x for x in p.path.strip("/").split("/") if x]
    if len(parts) < 4 or parts[2] != "tree":
        raise ValueError("不是 /tree/ 目录链接")
    owner, repo, branch = parts[0], parts[1], parts[3]
    subpath = "/".join(parts[4:]) if len(parts) > 4 else ""
    return owner, repo, branch, subpath

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def save_bytes(data: bytes, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        f.write(data)

def download_file(url: str, outdir: str, filename: str = None):
    raw = url if "raw.githubusercontent.com" in url else to_raw_blob(url)
    name = filename or os.path.basename(urlparse(raw).path) or "download.bin"
    dst = os.path.join(outdir, name)
    print(f"⬇️  {raw}\n    -> {dst}")
    data = http_get(raw)
    save_bytes(data, dst)
    print(f"✅ OK ({len(data)/1024:.1f} KB)\n")

def list_dir_via_api(owner: str, repo: str, branch: str, subpath: str):
    api = f"https://api.github.com/repos/{owner}/{repo}/contents/{subpath}?ref={branch}"
    data = http_get(api)
    arr = json.loads(data.decode("utf-8", "ignore"))
    if isinstance(arr, dict) and arr.get("type") == "file":
        return [arr]
    # 只列当前目录（不递归）
    return [it for it in arr if it.get("type") == "file"]

def main():
    # 取 URL（参数或交互）
    url = sys.argv[1] if len(sys.argv) >= 2 else input("请输入 GitHub 文件或目录 URL：").strip()
    if not url:
        print("❌ 未提供 URL"); return

    ensure_dir(OUT_DIR)

    # 情况 1：单文件
    if is_github_blob(url):
        print("📄  识别为单文件，直接下载")
        download_file(url, OUT_DIR)
        print("🎉 完成。"); return

    # 情况 2：目录（用 Contents API）
    if is_github_tree(url):
        print("🗂  识别为目录页（用 GitHub API 列当前目录，不递归）")
        owner, repo, branch, subpath = parse_tree(url)
        try:
            items = list_dir_via_api(owner, repo, branch, subpath)
        except Exception as e:
            print(f"❌ API 失败：{e}\n提示：未设置 GITHUB_TOKEN 时 API 有 60 次/小时限额。可设置后重试。")
            return

        # 后缀过滤
        files = []
        for it in items:
            name = it.get("name","")
            if EXT_FILTER not in ([], None):
                if not any(name.lower().endswith(ext.lower()) for ext in EXT_FILTER):
                    continue
            files.append(it)

        if not files:
            print("⚠️  该目录未匹配到文件（或被 EXT_FILTER 过滤掉）。可把 EXT_FILTER 设为 []。")
            return

        print(f"发现文件 {len(files)} 个，将下载到 ./{OUT_DIR}/")
        for i, it in enumerate(files, 1):
            name = it["name"]
            # 优先用 API 返回的 download_url；若无，则拼 raw
            raw = it.get("download_url") or f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{it['path']}"
            print(f"[{i}/{len(files)}] {name}")
            download_file(raw, OUT_DIR, filename=name)
        print("🎉 完成。"); return

    # 其他：提示
    print("⚠️  这不是文件或 /tree/ 目录链接。请提供：")
    print("    - 单文件： https://github.com/<owner>/<repo>/blob/<branch>/<path/file.ext>")
    print("    - 目录页： https://github.com/<owner>/<repo>/tree/<branch>/<dir_path>")

if __name__ == "__main__":
    main()

# python url_grabber.py https://github.com/location-competition/indoor-location-competition-20/tree/master/data/site1/B1/path_data_files

# python url_grabber.py https://github.com/location-competition/indoor-location-competition-20/blob/master/data/site1/B1/floor_info.json
