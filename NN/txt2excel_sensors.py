#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
txt2excel_sensors.py  —  批量解析 TXT 传感器日志到 Excel/CSV（带进度显示）

新增：
  --progress {auto,bar,print,none}   进度显示方式（默认 auto）
  --log-every N                      打印模式下每 N 条输出一次（默认 50）
"""

import os
import re
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple, Optional, Iterable

import pandas as pd

# 可选 tqdm（没有也能跑）
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

# 一些常见会带 accuracy 的 TYPE（仅供“最后一个数值是 accuracy”判断的提示，不是硬规则）
LIKELY_WITH_ACCURACY = {
    "TYPE_ACCELEROMETER",
    "TYPE_ACCELEROMETER_UNCALIBRATED",
    "TYPE_GYROSCOPE",
    "TYPE_GYROSCOPE_UNCALIBRATED",
    "TYPE_MAGNETIC_FIELD",
    "TYPE_MAGNETIC_FIELD_UNCALIBRATED",
    "TYPE_ROTATION_VECTOR",
    "TYPE_SENSOR_MAGNETIC_FIELD_ACCURACY_CHANGED",
}


def iter_txt_files(root: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        yield from root.rglob("*.txt")
    else:
        yield from (p for p in root.iterdir() if p.suffix.lower() == ".txt")


def parse_meta_headers(lines: List[str]) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    for line in lines:
        if not line.startswith("#"):
            break
        payload = line[1:].strip()
        for tok in payload.split("\t"):
            if ":" in tok:
                k, v = tok.split(":", 1)
                k, v = k.strip(), v.strip()
                safe_k = re.sub(r"[^\w\-一-龥]+", "_", k)
                meta[f"meta_{safe_k}"] = v
    return meta


def ms_to_iso_utc(ms: int) -> str:
    try:
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()
    except Exception:
        return ""


def looks_like_accuracy(v: float) -> bool:
    return abs(v - round(v)) < 1e-6 and int(round(v)) in (0, 1, 2, 3)


def parse_data_line(line: str) -> Optional[Tuple[int, str, List[float]]]:
    parts = line.strip().split()
    if len(parts) < 2:
        return None
    try:
        ts = int(parts[0])
    except Exception:
        return None
    typ = parts[1]
    vals: List[float] = []
    for tok in parts[2:]:
        try:
            vals.append(float(tok))
        except Exception:
            continue
    return ts, typ, vals


def rows_from_file(path: Path) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    meta_lines: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(200):
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if line.startswith("#"):
                meta_lines.append(line.rstrip("\n"))
            else:
                f.seek(pos)
                break

        meta = parse_meta_headers(meta_lines)

        for line in f:
            if not line.strip():
                continue
            if line.startswith("#"):
                extra = parse_meta_headers([line.rstrip("\n")])
                meta.update(extra)
                continue
            parsed = parse_data_line(line)
            if parsed is None:
                continue
            ts, typ, values = parsed

            acc: Optional[int] = None
            if typ in LIKELY_WITH_ACCURACY and len(values) >= 1 and looks_like_accuracy(values[-1]):
                acc = int(round(values[-1]))
                values = values[:-1]

            row: Dict[str, Any] = {
                "file": path.name,
                "timestamp_ms": ts,
                "time_iso_utc": ms_to_iso_utc(ts),
                "type": typ,
                "n_values": len(values),
            }
            if acc is not None:
                row["accuracy"] = acc
            for i, v in enumerate(values, start=1):
                row[f"v{i}"] = v
            row.update(meta)
            rows.append(row)

    return meta, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True, help="包含若干 .txt 的目录（可用 --recursive 遍历子目录）")
    ap.add_argument("--recursive", type=int, default=0, help="是否递归子目录：1=是，0=否（默认 0）")
    ap.add_argument("--out", type=Path, default=Path("sensors.xlsx"), help="输出 Excel 文件名")
    ap.add_argument("--csv", type=Path, default=None, help="（可选）同时导出一份长表 CSV")
    ap.add_argument("--sheet-by-type", type=int, default=1,
                    help="1=每种 TYPE 一个 sheet（默认），0=只导出长表在单个 sheet")
    ap.add_argument("--max-rows-per-sheet", type=int, default=1_000_000,
                    help="防止超过 Excel 单表 1048576 行的安全阈值")
    ap.add_argument("--progress", type=str, default="auto", choices=["auto", "bar", "print", "none"],
                    help="进度显示方式")
    ap.add_argument("--log-every", type=int, default=50, help="打印模式下每 N 项输出一次")
    args = ap.parse_args()

    # 小工具：统一的进度包装器
    def progress(iterable, desc="", total=None, unit="it"):
        use_bar = (args.progress in ("bar", "auto") and tqdm is not None)
        use_print = (args.progress in ("print", "auto") and not use_bar)
        if use_bar:
            return tqdm(iterable, total=total, desc=desc, unit=unit, ncols=100)
        elif use_print:
            def gen():
                for i, x in enumerate(iterable, 1):
                    if i == 1 or (i % max(1, args.log_every) == 0) or (total and i == total):
                        if total:
                            print(f"{desc}: {i}/{total} {unit}")
                        else:
                            print(f"{desc}: {i} {unit}")
                    yield x

            return gen()
        else:
            return iterable

    root = args.input_dir
    if not root.exists():
        raise FileNotFoundError(f"目录不存在：{root}")

    files = list(iter_txt_files(root, bool(args.recursive)))
    print(f"→ 扫描到 {len(files)} 个 .txt 文件（recursive={bool(args.recursive)}）")

    all_rows: List[Dict[str, Any]] = []
    for p in progress(files, desc="Parsing files", total=len(files), unit="file"):
        _meta, rows = rows_from_file(p)
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("没有解析出任何数据行，请检查文件格式。")

    print(f"→ 已解析 {len(all_rows):,} 行记录，构建 DataFrame …")
    df_all = pd.DataFrame(all_rows)

    base_cols = ["file", "type", "timestamp_ms", "time_iso_utc", "accuracy", "n_values"]
    v_cols = sorted([c for c in df_all.columns if re.fullmatch(r"v\d+", c)], key=lambda s: int(s[1:]))
    meta_cols = [c for c in df_all.columns if c.startswith("meta_")]
    ordered = [c for c in base_cols if c in df_all.columns] + v_cols + meta_cols
    df_all = df_all[ordered]

    if args.csv:
        print(f"→ 写出 CSV：{args.csv} …")
        df_all.to_csv(args.csv, index=False, encoding="utf-8-sig")
        print("✔ CSV 完成")

    # 选引擎：优先 xlsxwriter（更快），否则 openpyxl
    try:
        writer = pd.ExcelWriter(args.out, engine="xlsxwriter")
        engine_name = "xlsxwriter"
    except Exception:
        writer = pd.ExcelWriter(args.out, engine="openpyxl")
        engine_name = "openpyxl"
    print(f"→ 写出 Excel（引擎：{engine_name}）：{args.out}")

    if args.sheet_by_type:
        groups = list(df_all.groupby("type", sort=True))
        for typ, df in progress(groups, desc="Writing sheets", total=len(groups), unit="type"):
            sheet = re.sub(r"[:\\/?*\[\]]", "_", str(typ))[:31] or "TYPE"
            start = 0
            chunk_idx = 1
            total_rows = len(df)
            while start < total_rows:
                end = min(start + args.max_rows_per_sheet, total_rows)
                chunk = df.iloc[start:end]
                name = sheet if chunk_idx == 1 else f"{sheet}_{chunk_idx}"
                chunk.to_excel(writer, index=False, sheet_name=name)
                if args.progress in ("print", "auto") and tqdm is None:
                    print(f"  · {sheet}: rows {start}-{end - 1}")
                start = end
                chunk_idx += 1

        summary = (
            df_all.groupby("type")
            .size()
            .reset_index(name="rows")
            .sort_values("rows", ascending=False)
        )
        summary.to_excel(writer, index=False, sheet_name="SUMMARY")
        meta_only = df_all[[c for c in df_all.columns if c.startswith("meta_")]].drop_duplicates()
        if len(meta_only) <= 1000:
            meta_only.to_excel(writer, index=False, sheet_name="META_SAMPLES")
    else:
        df_all.to_excel(writer, index=False, sheet_name="ALL")

    writer.close()
    print(f"✅ 已导出 Excel：{args.out.resolve()}")
    if args.csv:
        print(f"✅ 已导出 CSV：{args.csv.resolve()}")


if __name__ == "__main__":
    main()

# # 进度条（推荐，有 tqdm 就显示条，没有则自动打印）
# python txt2excel_sensors.py --input-dir .\site1\B1 --out sensors.xlsx --progress auto
#
# # 强制用 tqdm 进度条
# python txt2excel_sensors.py --input-dir .\site1\B1 --out sensors.xlsx --progress bar
#
# # 纯打印，每 100 个文件/类型提示一次
# python txt2excel_sensors.py --input-dir .\site1\B1 --out sensors.xlsx --progress print --log-every 100
