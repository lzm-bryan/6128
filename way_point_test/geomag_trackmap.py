#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
geomag_trackmap.py — Geomagnetic "track-style" map (Matplotlib, CRS.Simple)

效果：在楼层底图上，用带颜色条的粗线段表示磁场强度（或分量）沿轨迹的变化。
相比 Folium 热力图，本脚本不会“糊成一片”，而是形成贴着走廊的彩色轨迹，风格接近论文图。

目录（--floor-dir 指向该层目录）：
  floor_info.json     # map_info.width/height；可含 transform/affine/matrix/scale/translate
  geojson_map.json    # 可缺省
  floor_image.png     # 建议有（含店铺标注），用于底图
  (可选) path_data_files/*.txt  # 若无 --txt-dir 则自动搜

示例：
  python geomag_trackmap.py --floor-dir .\B1 --out b1_track.png
  python geomag_trackmap.py --floor-dir .\F1 --source cal --stat mag --q 10,90 --lw 6 --alpha 0.95

依赖：matplotlib, numpy（均为常见包；不依赖 shapely）
"""

import os, json, math, argparse, hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

# ---------- I/O ----------
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

def find_txt_dir(floor_dir: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit and explicit.exists() and explicit.is_dir():
        if any(p.suffix.lower()==".txt" for p in explicit.iterdir()):
            return explicit
        return explicit
    cand = floor_dir / "path_data_files"
    if cand.exists() and cand.is_dir() and any(p.suffix.lower()==".txt" for p in cand.iterdir()):
        return cand
    if any(p.suffix.lower()==".txt" for p in floor_dir.iterdir()):
        return floor_dir
    for sub in floor_dir.iterdir():
        if sub.is_dir() and any(p.suffix.lower()==".txt" for p in sub.iterdir()):
            return sub
    return None

# ---------- affine ----------
def parse_affine_from_string(s: str):
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

def try_affine_from_floorinfo(fi_raw: Any):
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
    if "scale" in t and "translate" in t:
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

def apply_affine_xy(x, y, A):
    a,b,c,d,e,f = A
    return a*x + b*y + e, c*x + d*y + f

# ---------- parse TXT ----------
def parse_headers(txt_path: Path) -> Dict[str,str]:
    meta = {}
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("#"): break
            for tok in line[1:].strip().split("\t"):
                if ":" in tok:
                    k,v = tok.split(":",1)
                    meta[k.strip()] = v.strip()
    return meta

def parse_waypoints_and_mags(txt_path: Path, source="cal", uncal_debias=False):
    """
    返回：
      waypoints: [(t,x,y), ...]
      mags:      [(t,bx,by,bz), ...]  # 已根据 source / uncal_debias 处理
    """
    wps = []
    mags_cal = []
    mags_uncal = []          # 仅未去偏
    mags_uncal_pairs = []    # (Bx_uncal,By_uncal,Bz_uncal,Bx_bias,By_bias,Bz_bias)

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line[0] == "#":  # 跳过头
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
                    x = float(parts[2]); y = float(parts[3])
                    wps.append((t,x,y))
                except Exception:
                    pass
                continue

            if typ == "TYPE_MAGNETIC_FIELD" and len(parts) >= 5:
                try:
                    mags_cal.append((t, float(parts[2]), float(parts[3]), float(parts[4])))
                except Exception:
                    pass
                continue

            if typ == "TYPE_MAGNETIC_FIELD_UNCALIBRATED" and len(parts) >= 8:
                try:
                    bux, buy, buz = float(parts[2]), float(parts[3]), float(parts[4])
                    bbx, bby, bbz = float(parts[5]), float(parts[6]), float(parts[7])
                    mags_uncal_pairs.append((t, bux, buy, buz, bbx, bby, bbz))
                    mags_uncal.append((t, bux, buy, buz))
                except Exception:
                    pass
                continue
            elif typ == "TYPE_MAGNETIC_FIELD_UNCALIBRATED" and len(parts) >= 5:
                try:
                    mags_uncal.append((t, float(parts[2]), float(parts[3]), float(parts[4])))
                except Exception:
                    pass
                continue

    wps.sort(key=lambda x: x[0])
    mags_cal.sort(key=lambda x: x[0])
    mags_uncal.sort(key=lambda x: x[0])
    mags_uncal_pairs.sort(key=lambda x: x[0])

    mags = []
    if source == "cal" and mags_cal:
        mags = mags_cal
    elif source == "uncal" and mags_uncal:
        mags = mags_uncal
    elif source == "uncal_debiased" and mags_uncal_pairs:
        for (t, bux,buy,buz, bbx,bby,bbz) in mags_uncal_pairs:
            mags.append((t, bux-bbx, buy-bby, buz-bbz))
    elif mags_cal:
        mags = mags_cal

    return wps, mags, parse_headers(txt_path)

def interpolate_pos_for_times(waypoints, times, mode="linear"):
    if not waypoints:
        return [None]*len(times)
    ts = np.array([w[0] for w in waypoints], dtype=np.int64)
    xs = np.array([w[1] for w in waypoints], dtype=float)
    ys = np.array([w[2] for w in waypoints], dtype=float)

    out = []
    j = 0
    for t in times:
        while j+1 < len(ts) and ts[j+1] <= t:
            j += 1
        if j < len(ts)-1:
            t0,t1 = ts[j], ts[j+1]
            if t0 <= t <= t1:
                if mode == "linear":
                    r = (t - t0)/max(1,(t1 - t0))
                    out.append((xs[j]*(1-r)+xs[j+1]*r, ys[j]*(1-r)+ys[j+1]*r))
                    continue
                elif mode == "hold":
                    out.append((xs[j], ys[j])); continue
        if t < ts[0]:
            out.append((xs[0], ys[0]) if mode=="hold" else None)
        elif t > ts[-1]:
            out.append((xs[-1], ys[-1]) if mode=="hold" else None)
        else:
            out.append((xs[j], ys[j]))
    return out

# ---------- utils ----------
def robust_minmax(values, q_low=5.0, q_high=95.0):
    """兼容老版本 NumPy 的分位数计算；对空/异常做兜底。"""
    import numpy as np
    vs = np.asarray(values, dtype=float)
    vs = vs[np.isfinite(vs)]
    if vs.size == 0:
        return 0.0, 1.0
    try:
        # NumPy >= 1.22
        lo = np.quantile(vs, q_low / 100.0, method="linear")
        hi = np.quantile(vs, q_high / 100.0, method="linear")
    except TypeError:
        # NumPy <= 1.21 (没有 method=)
        lo = np.quantile(vs, q_low / 100.0, interpolation="linear")
        hi = np.quantile(vs, q_high / 100.0, interpolation="linear")
    except AttributeError:
        # 更旧：没有 quantile，用 percentile 兜底
        lo = np.percentile(vs, q_low, interpolation="linear")
        hi = np.percentile(vs, q_high, interpolation="linear")
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = float(np.nanmin(vs)), float(np.nanmax(vs))
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def make_segments_colored(xs, ys, vals):
    """
    xs, ys, vals: 1D arrays (同长). 生成相邻点连线的线段及其对应的取值（中点值）。
    """
    xs = np.asarray(xs); ys = np.asarray(ys); vals = np.asarray(vals)
    if len(xs) < 2: return [], []
    segs = np.stack([np.stack([xs[:-1], ys[:-1]], axis=1),
                      np.stack([xs[1:],  ys[1:]],  axis=1)], axis=1)
    vline = 0.5*(vals[:-1] + vals[1:])
    return segs, vline

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor-dir", type=Path, required=True)
    ap.add_argument("--txt-dir", type=Path, default=None)
    ap.add_argument("--source", type=str, default="cal",
                    choices=["cal","uncal","uncal_debiased"], help="磁场来源（默认 cal）")
    ap.add_argument("--stat", type=str, default="mag", choices=["mag","bx","by","bz"])
    ap.add_argument("--interp", type=str, default="linear", choices=["linear","hold","skip"])
    ap.add_argument("--subsample", type=int, default=1, help="磁场样本抽稀（默认 1）")
    ap.add_argument("--affine", type=str, default=None, help='手动 a,b,c,d,e,f（像素=仿射(米)）')
    ap.add_argument("--y-flip", type=int, default=1, help="y -> H - y（Matplotlib 用 origin=lower，默认翻转=1）")

    # 视觉
    ap.add_argument("--lw", type=float, default=6.0, help="线宽（默认 6）")
    ap.add_argument("--alpha", type=float, default=0.9, help="线段透明度（默认 0.9）")
    ap.add_argument("--cmap", type=str, default="inferno", help="颜色图（默认 inferno）")
    ap.add_argument("--q", type=str, default="5,95", help="分位裁剪（默认 5,95）")
    ap.add_argument("--vminmax", type=str, default=None, help="指定绝对 vmin,vmax，覆盖 --q。例：'20,70'")
    ap.add_argument("--show-waypoints", type=int, default=0, help="是否叠加稀疏waypoints散点")
    ap.add_argument("--scatter-every", type=int, default=50)

    # 背景/输出
    ap.add_argument("--no-image", action="store_true", help="不铺底图 floor_image.png")
    ap.add_argument("--image-alpha", type=float, default=0.7)
    ap.add_argument("--no-geojson", action="store_true")
    ap.add_argument("--figsize", type=str, default="8,6", help="宽,高（英寸）")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--out", type=Path, default=Path("geomag_track.png"))

    args = ap.parse_args()

    floor_dir = args.floor_dir
    FLOORINFO = floor_dir / "floor_info.json"
    GEOJSON   = floor_dir / "geojson_map.json"
    FLOORIMG  = floor_dir / "floor_image.png"
    if not FLOORINFO.exists():
        raise FileNotFoundError(f"缺少 floor_info.json：{FLOORINFO}")

    # floor meta
    fi = read_floor_info(FLOORINFO)
    W, H, fi_raw = fi["map_w"], fi["map_h"], fi["raw"]

    # affine
    A = None
    if args.affine:
        A = parse_affine_from_string(args.affine)
        if A: print("✔ 手动仿射：", A)
        else: print("⚠️ --affine 解析失败，尝试 floor_info")
    if A is None:
        A = try_affine_from_floorinfo(fi_raw)
        if A: print("✔ floor_info 仿射：", A)
        else: print("⚠️ 未提取到仿射，使用原始坐标 + 可选 y_flip")

    # TXT
    txt_dir = find_txt_dir(floor_dir, args.txt_dir)
    if not txt_dir:
        raise FileNotFoundError("未找到含 .txt 的目录")
    txts = sorted([p for p in txt_dir.iterdir() if p.suffix.lower()==".txt"])
    if not txts:
        raise FileNotFoundError("TXT 目录下没有 .txt 文件")

    # 收集所有轨迹线段 & 数值
    all_segments = []
    all_v = []
    meta_any = {}
    total_mag = []
    total_wp  = 0

    for txt in txts:
        wps, mags, meta = parse_waypoints_and_mags(txt, source=args.source,
                                                   uncal_debias=(args.source=="uncal_debiased"))
        if len(wps) < 2 or len(mags)==0:
            continue
        meta_any.update(meta)

        times = [t for (t, *_r) in mags]
        pos = interpolate_pos_for_times(wps, times, mode=args.interp)
        xs, ys, vs = [], [], []
        keep = 0
        for p, (t,bx,by,bz) in zip(pos, mags):
            if p is None:
                continue
            x, y = p
            if A is not None: x,y = apply_affine_xy(x,y,A)
            if args.y_flip:  y = H - y
            if not (0 <= x <= W and 0 <= y <= H):
                continue
            if args.stat == "mag": v = math.sqrt(bx*bx + by*by + bz*bz)
            elif args.stat == "bx": v = bx
            elif args.stat == "by": v = by
            else: v = bz
            xs.append(x); ys.append(y); vs.append(v); keep += 1

        if keep == 0:
            continue

        if args.subsample > 1:
            xs = xs[::args.subsample]; ys = ys[::args.subsample]; vs = vs[::args.subsample]

        segs, vline = make_segments_colored(xs, ys, vs)
        if len(vline) > 0:
            all_segments.append(segs)
            all_v.append(vline)
            total_mag.extend(vs)
            total_wp += len(wps)

    if not all_v:
        raise RuntimeError("没有可绘制的线段（检查仿射/坐标/时间对齐）。")

    segs = np.concatenate(all_segments, axis=0)
    vline = np.concatenate(all_v, axis=0)

    # 颜色范围
    if args.vminmax:
        vmin, vmax = [float(x.strip()) for x in args.vminmax.split(",")]
    else:
        ql, qh = [float(x.strip()) for x in args.q.split(",")]
        vmin, vmax = robust_minmax(vline, q_low=ql, q_high=qh)
    print(f"颜色范围 vmin={vmin:.3f}, vmax={vmax:.3f}")

    # 画图
    fw, fh = [float(x) for x in args.figsize.split(",")]
    fig, ax = plt.subplots(figsize=(fw, fh), dpi=args.dpi)

    # 底图（可选）
    if (not args.no_image) and FLOORIMG.exists():
        img = plt.imread(str(FLOORIMG))
        ax.imshow(img, extent=[0, W, 0, H], origin="lower", alpha=args.image_alpha)

    # 简单填充 geojson 外形（可选）
    if (not args.no_geojson) and GEOJSON.exists():
        try:
            gj = load_json(GEOJSON)
            def draw_geom(geom):
                t = geom.get("type")
                if t == "Polygon":
                    for ring in geom["coordinates"][:1]:  # 只画外环
                        pts = np.array(ring, dtype=float)
                        ax.add_patch(Polygon(pts, closed=True, facecolor=(0.55,0.8,0.9,0.25),
                                             edgecolor=(0.2,0.2,0.2,0.8), linewidth=1.0))
                elif t == "MultiPolygon":
                    for poly in geom["coordinates"]:
                        if len(poly) == 0: continue
                        pts = np.array(poly[0], dtype=float)
                        ax.add_patch(Polygon(pts, closed=True, facecolor=(0.55,0.8,0.9,0.25),
                                             edgecolor=(0.2,0.2,0.2,0.8), linewidth=1.0))
            if isinstance(gj, dict):
                if gj.get("type") == "FeatureCollection":
                    for feat in gj.get("features", []):
                        geom = feat.get("geometry", {})
                        if geom: draw_geom(geom)
                elif gj.get("type") in ("Polygon","MultiPolygon"):
                    draw_geom(gj)
        except Exception as e:
            print("⚠️ 解析/绘制 GeoJSON 失败：", e)

    # 线段着色
    lc = LineCollection(segs, cmap=args.cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax),
                        linewidths=args.lw, alpha=args.alpha, capstyle='round', joinstyle='round')
    lc.set_array(vline)
    ax.add_collection(lc)

    # 可选散点
    if args.show_waypoints:
        # 稀疏画（防止过密）
        pts = segs.reshape(-1,2)
        pts = pts[::max(1,int(args.scatter_every))]
        ax.scatter(pts[:,0], pts[:,1], s=6, c='k', alpha=0.4, zorder=3)

    # 轴/范围
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    # 标题（提取一次 meta 即可）
    site = meta_any.get("SiteName", meta_any.get("SiteID", "Site"))
    floor = meta_any.get("FloorName", "Floor")
    title = f"{site} - {floor} -- {len(total_mag)} Mag: Avg {np.mean(total_mag):.0f} Waypoints: ~{total_wp}"
    ax.set_title(title)

    # colorbar
    cbar = fig.colorbar(lc, ax=ax, shrink=0.9)
    label_unit = "|B| (μT)" if args.stat=="mag" else f"{args.stat} (μT)"
    cbar.set_label(label_unit)

    plt.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print("✅ 已输出：", args.out.resolve())

if __name__ == "__main__":
    main()


# # 用“校准磁力计 + |B|”，线宽更粗、对比更强，分位裁剪 10–90
# python geomag_trackmap.py --floor-dir .\site1\B1 \
#   --source cal --stat mag --q 10,90 --lw 6 --alpha 0.95 --out b1_track.png
#
# # 未校准但想去掉硬铁偏置（txt里包含 bias 三元组）：
# python geomag_trackmap.py --floor-dir .\site1\F1 \
#   --source uncal_debiased --stat mag --q 15,85 --lw 6 --alpha 0.95 --out f1_track.png
#
# # 强制颜色范围与论文一致（例如 20~70 μT）：
# python geomag_trackmap.py --floor-dir .\site1\B1 \
#   --vminmax 20,70 --lw 6 --alpha 0.95 --out b1_track_20_70.png

#
# python geomag_trackmap.py --floor-dir .\site1\B1  --source cal --stat mag --q 10,90 --lw 6 --alpha 0.95 --out b1_track.png