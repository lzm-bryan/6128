#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prep_fingerprint_csv.py — Build supervised dataset (CSV) for geomagnetic location fingerprinting.

For each floor directory:
  - Parse TXT logs with lines like:
      <ts_ms>  TYPE_WAYPOINT  x  y
      <ts_ms>  TYPE_MAGNETIC_FIELD(_UNCALIBRATED)  bx  by  bz  [biasx biasy biasz]  [acc]
      <ts_ms>  TYPE_ACCELEROMETER(_UNCALIBRATED)   ax  ay  az  [acc]
      <ts_ms>  TYPE_GYROSCOPE(_UNCALIBRATED)       gx  gy  gz  [acc]
  - Interpolate location at each sample/window center using WAYPOINTs
  - Apply affine (from floor_info.json or --affine) to convert meters->pixels; optional y-flip
  - Generate windowed stats features; save one-row-per-sample/window to CSV.

Usage:
  python prep_fingerprint_csv.py --floor-dir .\site1\B1 --out b1_dataset.csv
  python prep_fingerprint_csv.py --floor-dir .\site1\B1 --source uncal_debiased --window-ms 800 --hop-ms 400 --grid-px 50 --out b1_grid_dataset.csv
"""

import argparse, json, math, re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Optional tqdm
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# -------------------- I/O helpers --------------------
def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def read_floor_info(path: Path) -> Dict[str, Any]:
    obj = load_json(path)
    if isinstance(obj, dict):
        return {"map_w": float(obj["map_info"]["width"]),
                "map_h": float(obj["map_info"]["height"]),
                "raw": obj}
    if isinstance(obj, list) and obj:
        return {"map_w": float(obj[0]["map_info"]["width"]),
                "map_h": float(obj[0]["map_info"]["height"]),
                "raw": obj}
    raise TypeError("floor_info.json 应为 dict 或非空 list")

def ms_to_iso_utc(ms: int) -> str:
    try:
        return datetime.fromtimestamp(ms/1000.0, tz=timezone.utc).isoformat()
    except Exception:
        return ""

# -------------------- affine --------------------
def parse_affine_from_string(s: str) -> Optional[Tuple[float,float,float,float,float,float]]:
    try:
        a,b,c,d,e,f = [float(x.strip()) for x in s.split(",")]
        return (a,b,c,d,e,f)
    except Exception:
        return None

def compose_affine(scale=(1.0,1.0), theta_deg=0.0, translate=(0.0,0.0)):
    sx, sy = scale
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    a = c*sx; b = -s*sy
    c2 = s*sx; d =  c*sy
    e, f = translate
    return (a,b,c2,d,e,f)

def try_affine_from_floorinfo(fi_raw: Any) -> Optional[Tuple[float,float,float,float,float,float]]:
    def norm6(v):
        if isinstance(v, list) and len(v)==6:
            return tuple(float(x) for x in v)
        if isinstance(v, list) and len(v)==2 and isinstance(v[0], list) and len(v[0])==3:
            a,b,e = v[0]; c,d,f = v[1]
            return (float(a),float(b),float(c),float(d),float(e),float(f))
        return None
    raw = fi_raw
    t = (raw.get("transform") if isinstance(raw, dict) else None) or {}
    if isinstance(t, dict) and {"a","b","c","d","e","f"}.issubset(t.keys()):
        return (float(t["a"]), float(t["b"]), float(t["c"]), float(t["d"]), float(t["e"]), float(t["f"]))
    for k in ("affine","matrix"):
        if k in t:
            v = norm6(t[k])
            if v: return v
    if isinstance(t, dict) and "scale" in t and "translate" in t:
        sx, sy = t.get("scale",[1,1]) or [1,1]
        tx, ty = t.get("translate",[0,0]) or [0,0]
        theta = float(t.get("theta_deg",0.0))
        return compose_affine((float(sx),float(sy)), theta, (float(tx),float(ty)))
    mi = raw.get("map_info") if isinstance(raw, dict) else None
    if isinstance(mi, dict):
        ox, oy = 0.0, 0.0
        if "origin" in mi and isinstance(mi["origin"], (list,tuple)) and len(mi["origin"])>=2:
            ox, oy = float(mi["origin"][0]), float(mi["origin"][1])
        theta = float(mi.get("theta_deg", 0.0))
        if "pixel_per_meter" in mi:
            ppm = float(mi["pixel_per_meter"]); return compose_affine((ppm,ppm), theta, (ox,oy))
        if "meters_per_pixel" in mi:
            mpp = float(mi["meters_per_pixel"]); ppm = 1.0/mpp if mpp!=0 else 1.0
            return compose_affine((ppm,ppm), theta, (ox,oy))
    return None

def apply_affine_xy(x: float, y: float, A: Tuple[float,float,float,float,float,float]):
    a,b,c,d,e,f = A
    return a*x + b*y + e, c*x + d*y + f

# -------------------- parsing --------------------
def looks_like_accuracy(v: float) -> bool:
    return abs(v - round(v)) < 1e-6 and int(round(v)) in (0,1,2,3)

def parse_txt(path: Path):
    """
    Return dict of numpy arrays:
      way_t, way_x, way_y
      mag_t, mag_bx, mag_by, mag_bz   (+ uncal_*, bias_*)
      acc_t, acc_ax, acc_ay, acc_az
      gyr_t, gyr_gx, gyr_gy, gyr_gz
    """
    way, mag, mag_uncal, mag_bias = [], [], [], []
    acc, gyr = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                t = int(parts[0])
            except Exception:
                continue
            typ = parts[1]

            if typ == "TYPE_WAYPOINT" and len(parts) >= 4:
                try:
                    way.append((t, float(parts[2]), float(parts[3])))
                except Exception:
                    pass
                continue

            if typ == "TYPE_MAGNETIC_FIELD" and len(parts) >= 5:
                try:
                    vals = [float(parts[2]), float(parts[3]), float(parts[4])]
                    # strip trailing accuracy if present
                    if len(parts) >= 6 and looks_like_accuracy(float(parts[-1])):
                        pass
                    mag.append((t, *vals))
                except Exception:
                    pass
                continue

            if typ == "TYPE_MAGNETIC_FIELD_UNCALIBRATED":
                try:
                    bux,buy,buz = float(parts[2]), float(parts[3]), float(parts[4])
                    if len(parts) >= 8:
                        bbx,bby,bbz = float(parts[5]), float(parts[6]), float(parts[7])
                        mag_uncal.append((t, bux,buy,buz))
                        mag_bias.append((t, bbx,bby,bbz))
                    else:
                        mag_uncal.append((t, bux,buy,buz))
                except Exception:
                    pass
                continue

            if typ.startswith("TYPE_ACCELEROMETER") and len(parts) >= 5:
                try:
                    ax,ay,az = float(parts[2]), float(parts[3]), float(parts[4])
                    acc.append((t, ax,ay,az))
                except Exception: pass
                continue

            if typ.startswith("TYPE_GYROSCOPE") and len(parts) >= 5:
                try:
                    gx,gy,gz = float(parts[2]), float(parts[3]), float(parts[4])
                    gyr.append((t, gx,gy,gz))
                except Exception: pass
                continue

    def to_np(lst, ncol):
        if not lst:
            return np.empty((0,ncol), dtype=float)
        arr = np.asarray(lst, dtype=float)
        idx = np.argsort(arr[:,0])
        return arr[idx]

    return {
        "way": to_np(way, 3),                  # t, x, y
        "mag": to_np(mag, 4),                  # t, bx,by,bz
        "mag_uncal": to_np(mag_uncal, 4),      # t, bx,by,bz
        "mag_bias": to_np(mag_bias, 4),        # t, bbx,bby,bbz
        "acc": to_np(acc, 4),                  # t, ax,ay,az
        "gyr": to_np(gyr, 4),                  # t, gx,gy,gz
    }

# -------------------- interpolation & windows --------------------
def interp_pos(t: np.ndarray, way: np.ndarray, mode="linear"):
    """
    way: [N,3] (t, x, y); t: [M]
    return [M,2] positions or NaN if cannot infer.
    """
    if way.shape[0] == 0:
        return np.full((len(t),2), np.nan)
    wt = way[:,0]; wx = way[:,1]; wy = way[:,2]
    out = np.zeros((len(t),2), dtype=float)
    j = 0
    for i, ti in enumerate(t):
        while j+1 < len(wt) and wt[j+1] <= ti:
            j += 1
        if j < len(wt)-1 and wt[j] <= ti <= wt[j+1]:
            if mode == "linear":
                denom = max(1.0, (wt[j+1]-wt[j]))
                r = (ti - wt[j]) / denom
                out[i,0] = wx[j]*(1-r) + wx[j+1]*r
                out[i,1] = wy[j]*(1-r) + wy[j+1]*r
            else:
                out[i,0] = wx[j]; out[i,1] = wy[j]
        elif ti < wt[0]:
            out[i,0] = wx[0]; out[i,1] = wy[0]
        else:
            out[i,0] = wx[-1]; out[i,1] = wy[-1]
    return out

def window_indices(ts: np.ndarray, center: float, half: float):
    """Return slice mask for ts in [center-half, center+half]."""
    lo, hi = center - half, center + half
    # bool mask (simple, clear; if super large arrays we could do two-pointer)
    return (ts >= lo) & (ts <= hi)

def safe_stats(x: np.ndarray):
    if x.size == 0 or not np.isfinite(x).any():
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    x = x[np.isfinite(x)]
    m = float(np.mean(x))
    s = float(np.std(x))
    mn = float(np.min(x))
    mx = float(np.max(x))
    try:
        # compat for old numpy
        q25 = float(np.quantile(x, 0.25))
        q75 = float(np.quantile(x, 0.75))
    except TypeError:
        q25 = float(np.percentile(x, 25))
        q75 = float(np.percentile(x, 75))
    iqr = q75 - q25
    return m, s, mn, mx, q25, q75, iqr

# -------------------- dataset build --------------------
def build_rows_for_file(path: Path, A, map_w, map_h, y_flip,
                        source="cal", window_ms=1000, hop_ms=500,
                        min_mag_pts=5, grid_px: float = 0.0):
    d = parse_txt(path)
    way = d["way"]
    if way.shape[0] < 2:
        return []  # need at least two points for interpolation

    # choose magnetometer source
    if source == "cal":
        mag = d["mag"]
    elif source == "uncal_debiased" and d["mag_uncal"].shape[0] and d["mag_bias"].shape[0]:
        # align by time (simple nearest)
        mu = d["mag_uncal"]; mb = d["mag_bias"]
        idx = np.searchsorted(mb[:,0], mu[:,0], side="left")
        idx = np.clip(idx, 0, len(mb)-1)
        m = np.empty_like(mu)
        m[:,0] = mu[:,0]
        m[:,1:] = mu[:,1:] - mb[idx,1:]
        mag = m
    elif d["mag_uncal"].shape[0]:
        mag = d["mag_uncal"]
    else:
        mag = d["mag"]

    acc = d["acc"]; gyr = d["gyr"]

    if mag.shape[0] == 0:
        return []

    # define window centers from mag time range
    t0, t1 = float(mag[0,0]), float(mag[-1,0])
    centers = np.arange(t0, t1+1, hop_ms, dtype=float)
    half = window_ms/2.0

    rows = []
    # pre-extract arrays
    mag_t = mag[:,0]; bx,by,bz = mag[:,1], mag[:,2], mag[:,3]
    bmag = np.sqrt(bx*bx + by*by + bz*bz)
    acc_t = acc[:,0] if acc.shape[0] else np.empty((0,))
    ax = acc[:,1] if acc.shape[0] else np.empty((0,))
    ay = acc[:,2] if acc.shape[0] else np.empty((0,))
    az = acc[:,3] if acc.shape[0] else np.empty((0,))
    anorm = np.sqrt(ax*ax + ay*ay + az*az)

    gyr_t = gyr[:,0] if gyr.shape[0] else np.empty((0,))
    gx = gyr[:,1] if gyr.shape[0] else np.empty((0,))
    gy = gyr[:,2] if gyr.shape[0] else np.empty((0,))
    gz = gyr[:,3] if gyr.shape[0] else np.empty((0,))
    gnorm = np.sqrt(gx*gx + gy*gy + gz*gz)

    # interpolate labels at centers
    pos = interp_pos(centers, way, mode="linear")  # meters or raw units from WAYPOINT
    # apply affine to get pixel labels
    label_px = pos.copy()
    if A is not None:
        for i in range(len(label_px)):
            label_px[i,0], label_px[i,1] = apply_affine_xy(label_px[i,0], label_px[i,1], A)
    if y_flip:
        label_px[:,1] = map_h - label_px[:,1]

    # build per-window rows
    iterator = tqdm(centers, desc=f"{path.name}: windows", unit="win") if tqdm is not None else centers
    for c, (lx_m, ly_m), (lx_px, ly_px) in zip(iterator, pos, label_px):
        # window masks
        m_mask = window_indices(mag_t, c, half)
        if np.count_nonzero(m_mask) < min_mag_pts:
            continue  # skip too sparse windows

        a_mask = window_indices(acc_t, c, half) if acc_t.size else np.zeros(0, dtype=bool)
        g_mask = window_indices(gyr_t, c, half) if gyr_t.size else np.zeros(0, dtype=bool)

        # mag stats (components + magnitude)
        bx_s = safe_stats(bx[m_mask]); by_s = safe_stats(by[m_mask]); bz_s = safe_stats(bz[m_mask])
        bm_s = safe_stats(bmag[m_mask])

        # acc / gyro stats on norms（更鲁棒）
        an_s = safe_stats(anorm[a_mask]) if acc_t.size else (np.nan,)*7
        gn_s = safe_stats(gnorm[g_mask]) if gyr_t.size else (np.nan,)*7

        # label grid (optional)
        cell_i = cell_j = cell_id = ""
        if grid_px and grid_px > 0:
            i = int(lx_px // grid_px); j = int(ly_px // grid_px)
            cell_i, cell_j = i, j
            cell_id = f"{i}_{j}"

        row = {
            # id & timing
            "file": path.name,
            "center_ts_ms": int(c),
            "center_time_iso_utc": ms_to_iso_utc(int(c)),
            # labels
            "label_x_m": float(lx_m), "label_y_m": float(ly_m),
            "label_x_px": float(lx_px), "label_y_px": float(ly_px),
            "label_cell_i": cell_i, "label_cell_j": cell_j, "label_cell_id": cell_id,
            # counts
            "n_mag": int(np.count_nonzero(m_mask)),
            "n_acc": int(np.count_nonzero(a_mask)) if acc_t.size else 0,
            "n_gyr": int(np.count_nonzero(g_mask)) if gyr_t.size else 0,
            # mag features (mean,std,min,max,q25,q75,iqr)
            "bx_mean": bx_s[0], "bx_std": bx_s[1], "bx_min": bx_s[2], "bx_max": bx_s[3], "bx_q25": bx_s[4], "bx_q75": bx_s[5], "bx_iqr": bx_s[6],
            "by_mean": by_s[0], "by_std": by_s[1], "by_min": by_s[2], "by_max": by_s[3], "by_q25": by_s[4], "by_q75": by_s[5], "by_iqr": by_s[6],
            "bz_mean": bz_s[0], "bz_std": bz_s[1], "bz_min": bz_s[2], "bz_max": bz_s[3], "bz_q25": bz_s[4], "bz_q75": bz_s[5], "bz_iqr": bz_s[6],
            "bmag_mean": bm_s[0], "bmag_std": bm_s[1], "bmag_min": bm_s[2], "bmag_max": bm_s[3], "bmag_q25": bm_s[4], "bmag_q75": bm_s[5], "bmag_iqr": bm_s[6],
            # acc / gyro (norm)
            "anorm_mean": an_s[0], "anorm_std": an_s[1], "anorm_min": an_s[2], "anorm_max": an_s[3], "anorm_q25": an_s[4], "anorm_q75": an_s[5], "anorm_iqr": an_s[6],
            "gnorm_mean": gn_s[0], "gnorm_std": gn_s[1], "gnorm_min": gn_s[2], "gnorm_max": gn_s[3], "gnorm_q25": gn_s[4], "gnorm_q75": gn_s[5], "gnorm_iqr": gn_s[6],
        }
        rows.append(row)
    return rows

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor-dir", type=Path, required=True, help="包含 floor_info.json / geojson_map.json / floor_image.png / TXT 的目录")
    ap.add_argument("--txt-dir", type=Path, default=None, help="显式 TXT 目录（不传则自动在 floor 里递归找 *.txt）")
    ap.add_argument("--recursive", type=int, default=1, help="是否递归找 TXT（默认 1）")

    # label/feature settings
    ap.add_argument("--source", type=str, default="cal", choices=["cal","uncal","uncal_debiased"], help="磁力计来源")
    ap.add_argument("--window-ms", type=int, default=1000, help="聚合窗长（ms）")
    ap.add_argument("--hop-ms", type=int, default=500, help="步进（ms）")
    ap.add_argument("--min-mag-pts", type=int, default=5, help="窗口内最少磁力样本数")
    ap.add_argument("--grid-px", type=float, default=0.0, help=">0 则量化到网格，生成 label_cell_id（用于分类）")

    # coordinate transform
    ap.add_argument("--affine", type=str, default=None, help='手动 a,b,c,d,e,f（像素=仿射(米)）')
    ap.add_argument("--y-flip", type=int, default=0, help="仿射后是否额外 y->H-y（CRS.Simple 常用 0；若图像坐标原点在左上可设 1）")

    # output
    ap.add_argument("--out", type=Path, default=Path("fingerprint_dataset.csv"), help="输出 CSV 文件")
    ap.add_argument("--progress", type=str, default="auto", choices=["auto","bar","print","none"], help="进度显示方式")
    args = ap.parse_args()

    # load floor info & affine
    FLOORINFO = args.floor_dir / "floor_info.json"
    if not FLOORINFO.exists():
        raise FileNotFoundError(f"缺少 floor_info.json：{FLOORINFO}")
    fi = read_floor_info(FLOORINFO)
    map_w, map_h, fi_raw = fi["map_w"], fi["map_h"], fi["raw"]
    A = None
    if args.affine:
        A = parse_affine_from_string(args.affine)
        if A is None:
            print("⚠️ --affine 解析失败，将尝试 floor_info")
    if A is None:
        A = try_affine_from_floorinfo(fi_raw)
        if A:
            print(f"✔ 从 floor_info 提取仿射：{A}")
        else:
            print("⚠️ 未提取到仿射，将使用 WAYPOINT 原始单位作为 label_x_m/label_y_m；像素标签仅做 y_flip。")

    # collect files
    if args.txt_dir and args.txt_dir.exists():
        files = list(args.txt_dir.rglob("*.txt") if args.recursive else args.txt_dir.glob("*.txt"))
    else:
        files = list(args.floor_dir.rglob("*.txt") if args.recursive else args.floor_dir.glob("*.txt"))
    files = sorted(files)
    if not files:
        raise FileNotFoundError("未找到任何 .txt")

    # progress helper
    def progress(it, desc, total=None, unit="it"):
        use_bar = (args.progress in ("bar","auto") and tqdm is not None)
        use_print = (args.progress in ("print","auto") and not use_bar)
        if use_bar:
            return tqdm(it, total=total, desc=desc, unit=unit, ncols=100)
        elif use_print:
            count = 0
            for x in it:
                count += 1
                if count == 1 or (total and count == total) or count % 20 == 0:
                    print(f"{desc}: {count}/{total if total else '?'} {unit}")
                yield x
        else:
            return it

    # build
    all_rows: List[Dict[str, Any]] = []
    for p in progress(files, desc="Files", total=len(files), unit="file"):
        rows = build_rows_for_file(
            p, A, map_w, map_h, y_flip=bool(args.y_flip),
            source=args.source, window_ms=args.window_ms, hop_ms=args.hop_ms,
            min_mag_pts=args.min_mag_pts, grid_px=args.grid_px
        )
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("没有生成任何样本（检查 WAYPOINT / 磁力数据是否匹配）。")

    df = pd.DataFrame(all_rows)

    # basic sanity / ordering
    cols_front = [
        "file","center_ts_ms","center_time_iso_utc",
        "label_x_m","label_y_m","label_x_px","label_y_px",
        "label_cell_id","label_cell_i","label_cell_j",
        "n_mag","n_acc","n_gyr"
    ]
    other_cols = [c for c in df.columns if c not in cols_front]
    df = df[cols_front + sorted(other_cols)]

    # save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"✅ Saved dataset CSV: {args.out.resolve()}   (rows={len(df):,})")

if __name__ == "__main__":
    main()


# # 1) 最常见：每 1s 窗口、0.5s 步进，生成回归标签（像素 & 米）
# python prep_fingerprint_csv.py --floor-dir .\site1\B1 --out b1_dataset.csv
#
# # 2) 用未校准并去偏（TXT 中含 bias），更密一点的样本
# python prep_fingerprint_csv.py --floor-dir .\site1\B1 --source uncal_debiased \
#   --window-ms 800 --hop-ms 400 --out b1_uncal_debiased.csv
#
# # 3) 想做分类（指纹→网格 cell），把标签量化到 50px 网格
# python prep_fingerprint_csv.py --floor-dir .\site1\B1 \
#   --grid-px 50 --out b1_grid_50px.csv


# 想要更多？
# python prep_fingerprint_csv.py --floor-dir .\site1\B1 --source uncal_debiased --window-ms 1200 --hop-ms 200 --min-mag-pts 3 --out b1_dataset_dense.csv
