#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
geomag_trackmap_folium.py — Interactive track-colored geomagnetic map (Leaflet/Folium, CRS.Simple)

- 解析 TXT 中的 TYPE_WAYPOINT 与磁力计（cal/uncal/uncal_debiased）
- 按时间把磁场样本插值到轨迹位置
- 应用 floor_info.json 的仿射（或手动 --affine），可选 y 翻转
- 以“彩色轨迹（分段 PolyLine）+ 颜色条”的方式在网页中展示（非热力云团）
- 可叠加楼层底图/GeoJSON/轨迹

用法示例：
  python geomag_trackmap_folium.py --floor-dir .\site1\B1 --out b1_track.html
  python geomag_trackmap_folium.py --floor-dir .\site1\B1 --source cal --stat mag --q 10,90 --lw 4 --alpha 0.95
  python geomag_trackmap_folium.py --floor-dir .\site1\B1 --vminmax 20,70 --cmap inferno

依赖：folium, branca（自带颜色条）；不依赖 matplotlib
"""

import os, json, math, argparse, hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import folium
from folium import FeatureGroup
from branca.colormap import LinearColormap

# --------- I/O ---------
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
    raise TypeError("floor_info.json 类型异常，应为 dict 或非空 list")

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

# --------- affine ---------
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

# --------- TXT parse ---------
def parse_waypoints_and_mags(txt_path: Path, source="cal"):
    """
    返回：
      waypoints: [(t,x,y), ...]
      mags:      [(t,bx,by,bz), ...]  # 已按 source 选择
    支持 source=cal | uncal | uncal_debiased（若行中含 bias 三元组）
    """
    wps = []; mags_cal = []; mags_uncal = []; mags_uncal_pairs = []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line[0] == "#":  # 跳过头部
                continue
            parts = line.strip().split()
            if len(parts) < 2: continue
            try:
                t = int(parts[0])
            except Exception:
                continue
            typ = parts[1]

            if typ == "TYPE_WAYPOINT" and len(parts) >= 4:
                try:
                    x = float(parts[2]); y = float(parts[3])
                    wps.append((t,x,y))
                except Exception: pass
                continue

            if typ == "TYPE_MAGNETIC_FIELD" and len(parts) >= 5:
                try:
                    mags_cal.append((t, float(parts[2]), float(parts[3]), float(parts[4])))
                except Exception: pass
                continue

            if typ == "TYPE_MAGNETIC_FIELD_UNCALIBRATED":
                if len(parts) >= 8:
                    try:
                        bux,buy,buz = float(parts[2]), float(parts[3]), float(parts[4])
                        bbx,bby,bbz = float(parts[5]), float(parts[6]), float(parts[7])
                        mags_uncal_pairs.append((t, bux,buy,buz, bbx,bby,bbz))
                    except Exception: pass
                    continue
                elif len(parts) >= 5:
                    try:
                        mags_uncal.append((t, float(parts[2]), float(parts[3]), float(parts[4])))
                    except Exception: pass
                    continue

    wps.sort(key=lambda x: x[0]); mags_cal.sort(key=lambda x: x[0])
    mags_uncal.sort(key=lambda x: x[0]); mags_uncal_pairs.sort(key=lambda x: x[0])

    # 选择来源
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
    return wps, mags

def interpolate_pos_for_times(waypoints, times, mode="linear"):
    if not waypoints:
        return [None]*len(times)
    ts = [w[0] for w in waypoints]
    xs = [w[1] for w in waypoints]
    ys = [w[2] for w in waypoints]
    out = []; j = 0; n = len(waypoints)
    for t in times:
        while j+1 < n and ts[j+1] <= t:
            j += 1
        if j < n-1:
            t0, t1 = ts[j], ts[j+1]
            if t0 <= t <= t1:
                if mode == "linear":
                    r = (t - t0)/max(1, (t1 - t0))
                    out.append((xs[j]*(1-r)+xs[j+1]*r, ys[j]*(1-r)+ys[j+1]*r)); continue
                elif mode == "hold":
                    out.append((xs[j], ys[j])); continue
        if t < ts[0]:
            out.append((xs[0], ys[0]) if mode=="hold" else None)
        elif t > ts[-1]:
            out.append((xs[-1], ys[-1]) if mode=="hold" else None)
        else:
            out.append((xs[j], ys[j]))
    return out

# --------- utils ---------
def robust_minmax(values, q_low=5.0, q_high=95.0):
    """兼容老 NumPy：优先 quantile(method=)，否则用 interpolation=，再退回 percentile。"""
    import numpy as np
    vs = np.asarray(values, dtype=float)
    vs = vs[np.isfinite(vs)]
    if vs.size == 0:
        return 0.0, 1.0
    try:
        lo = np.quantile(vs, q_low/100.0, method="linear")
        hi = np.quantile(vs, q_high/100.0, method="linear")
    except TypeError:
        lo = np.quantile(vs, q_low/100.0, interpolation="linear")
        hi = np.quantile(vs, q_high/100.0, interpolation="linear")
    except AttributeError:
        lo = np.percentile(vs, q_low, interpolation="linear")
        hi = np.percentile(vs, q_high, interpolation="linear")
    if not (math.isfinite(lo) and math.isfinite(hi)):
        lo, hi = float(vs.min()), float(vs.max())
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)

def make_linear_colormap(name: str, vmin: float, vmax: float) -> LinearColormap:
    # 轻量内置几套常用配色
    cmaps = {
        "inferno": ['#000004','#1b0c41','#4a0c6b','#781c6d','#a52c60','#cf4446','#ed6925','#fb9b06','#f7d13d','#fcffa4'],
        "magma":   ['#000004','#1c1044','#51127c','#822681','#b63679','#e65164','#fb8761','#fec287','#fff1a8'],
        "plasma":  ['#0d0887','#5b02a3','#9a179b','#cb4679','#ed7953','#fb9f3a','#fdca26','#f0f921'],
        "viridis": ['#440154','#3b528b','#21918c','#5ec962','#fde725'],
        "turbo":   ['#30123b','#4145ab','#2fb4f3','#32f1c5','#7ef66f','#f6d54c','#f98e42','#d33b29','#8a0943']
    }
    colors = cmaps.get(name.lower(), cmaps["inferno"])
    return LinearColormap(colors=colors, vmin=vmin, vmax=vmax)

def color_for_name(name: str) -> str:
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    return f"#{h[4:10]}"

# --------- main ---------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor-dir", type=Path, required=True)
    ap.add_argument("--txt-dir", type=Path, default=None)

    ap.add_argument("--source", type=str, default="cal",
                    choices=["cal","uncal","uncal_debiased"])
    ap.add_argument("--stat", type=str, default="mag", choices=["mag","bx","by","bz"])
    ap.add_argument("--interp", type=str, default="linear", choices=["linear","hold","skip"])
    ap.add_argument("--subsample", type=int, default=1, help="磁场样本抽稀（默认 1）")
    ap.add_argument("--segment-decim", type=int, default=1, help="线段抽稀（每 N 段取 1 段）")

    ap.add_argument("--affine", type=str, default=None, help='手动 a,b,c,d,e,f（像素=仿射(米)）')
    ap.add_argument("--y-flip", type=int, default=0, help="仿射后是否 y->H-y（Folium 默认 0）")

    # 视觉参数
    ap.add_argument("--lw", type=float, default=4.0, help="线宽（像素）")
    ap.add_argument("--alpha", type=float, default=0.95, help="透明度 0~1")
    ap.add_argument("--cmap", type=str, default="inferno", help="配色：inferno/viridis/plasma/magma/turbo")
    ap.add_argument("--q", type=str, default="5,95", help="分位裁剪（默认 5,95）")
    ap.add_argument("--vminmax", type=str, default=None, help="直接给定范围 '20,70' 覆盖 --q")

    # 叠加 & 输出
    ap.add_argument("--no-image", action="store_true")
    ap.add_argument("--no-geojson", action="store_true")
    ap.add_argument("--show-traj", type=int, default=0, help="额外用单色细线显示轨迹（参考）")
    ap.add_argument("--out", type=Path, default=Path("geomag_track.html"))

    args = ap.parse_args()

    # floor
    floor_dir = args.floor_dir
    GEOJSON = floor_dir / "geojson_map.json"
    FLOORINFO = floor_dir / "floor_info.json"
    FLOORIMG = floor_dir / "floor_image.png"
    if not FLOORINFO.exists():
        raise FileNotFoundError(f"缺少 floor_info.json：{FLOORINFO}")

    fi = read_floor_info(FLOORINFO)
    W, H, fi_raw = fi["map_w"], fi["map_h"], fi["raw"]

    # affine
    A = None
    if args.affine:
        A = parse_affine_from_string(args.affine)
        if A: print("✔ 使用手动仿射：", A)
        else: print("⚠️ --affine 解析失败，将尝试 floor_info")
    if A is None:
        A = try_affine_from_floorinfo(fi_raw)
        if A: print("✔ 从 floor_info 提取仿射：", A)
        else: print("⚠️ 未提取到仿射，使用原始坐标 + 可选 y_flip")

    # map
    m = folium.Map(location=[H/2, W/2], zoom_start=0, tiles=None, crs="Simple")
    if not args.no_image and FLOORIMG.exists():
        folium.raster_layers.ImageOverlay(
            name="floor_image", image=str(FLOORIMG),
            bounds=[[0,0],[H,W]], opacity=1.0, interactive=False, cross_origin=False
        ).add_to(m)
    if not args.no_geojson and GEOJSON.exists():
        gj = load_json(GEOJSON)
        def style_fn(feat):
            props = feat.get("properties", {}) or {}
            fid = props.get("floor_id", None)
            try: fid = int(fid)
            except Exception: fid = None
            color_map = {1:"#3186cc", 2:"#2ecc71", 3:"#e74c3c"}
            return {"fillColor": color_map.get(fid, "#95a5a6"),
                    "color": "#2c3e50", "weight": 1, "fillOpacity": 0.25}
        folium.GeoJson(gj, name="floor_geojson", style_function=style_fn).add_to(m)

    # TXT
    txt_dir = find_txt_dir(floor_dir, args.txt_dir)
    if not txt_dir:
        raise FileNotFoundError("未找到含 .txt 的目录")
    txts = sorted([p for p in txt_dir.iterdir() if p.suffix.lower()==".txt"])
    if not txts:
        raise FileNotFoundError("TXT 目录下没有 .txt 文件")

    # 收集所有线段的取值，用于统一颜色范围
    all_v_mid = []
    per_file_segments = []

    for txt in txts:
        wps, mags = parse_waypoints_and_mags(txt, source=args.source)
        if len(wps) < 2 or len(mags) == 0:
            continue
        times = [t for (t, *_r) in mags]
        pos = interpolate_pos_for_times(wps, times, mode=args.interp)

        xs, ys, vs = [], [], []
        for p, (t,bx,by,bz) in zip(pos, mags):
            if p is None: continue
            x,y = p
            if A is not None:
                x,y = apply_affine_xy(x,y,A)
            if args.y_flip:
                y = H - y
            if not (0 <= x <= W and 0 <= y <= H):
                continue
            if args.subsample > 1 and (len(vs) % args.subsample != 0):
                # 简单抽稀：保留每 N 个样本
                pass
            xs.append(x); ys.append(y)
            if args.stat == "mag":
                v = math.sqrt(bx*bx + by*by + bz*bz)
            elif args.stat == "bx": v = bx
            elif args.stat == "by": v = by
            else: v = bz
            vs.append(v)

        if len(xs) < 2:
            continue

        segs = []
        v_mid = []
        for i in range(1, len(xs), max(1, args.segment_decim)):
            x0,y0 = xs[i-1], ys[i-1]
            x1,y1 = xs[i],   ys[i]
            segs.append(((y0,x0),(y1,x1)))  # Leaflet lat,lon = y,x
            v_mid.append(0.5*(vs[i-1]+vs[i]))
        per_file_segments.append((txt.name, segs, v_mid))
        all_v_mid.extend(v_mid)

    if not all_v_mid:
        raise RuntimeError("没有可绘制的线段（检查仿射/坐标/时间对齐）。")

    # 颜色范围
    if args.vminmax:
        vmin, vmax = [float(x.strip()) for x in args.vminmax.split(",")]
    else:
        ql, qh = [float(x.strip()) for x in args.q.split(",")]
        vmin, vmax = robust_minmax(all_v_mid, q_low=ql, q_high=qh)
    print(f"颜色范围 vmin={vmin:.3f}, vmax={vmax:.3f}")

    cmap = make_linear_colormap(args.cmap, vmin, vmax)
    cmap.caption = f"{'|B|' if args.stat=='mag' else args.stat} (μT)"
    cmap.add_to(m)

    # 画“彩色轨迹”（很多小段 PolyLine）
    tracks_group = FeatureGroup(name=f"Tracks ({args.source}/{args.stat})", show=True)
    for fname, segs, v_mid in per_file_segments:
        fg = FeatureGroup(name=fname, show=False)
        for ((lat0,lon0),(lat1,lon1)), v in zip(segs, v_mid):
            color = cmap(v)  # hex
            folium.PolyLine(
                locations=[(lat0,lon0),(lat1,lon1)],
                color=color, weight=args.lw, opacity=args.alpha
            ).add_to(fg)
        fg.add_to(tracks_group)
    tracks_group.add_to(m)

    # 可选：叠加单色细轨迹（辅助对照）
    if args.show_traj:
        traj_group = FeatureGroup(name="Traj (thin)", show=False)
        for fname, segs, _ in per_file_segments:
            for ((lat0,lon0),(lat1,lon1)) in segs:
                folium.PolyLine(
                    locations=[(lat0,lon0),(lat1,lon1)],
                    color="#333333", weight=1.5, opacity=0.5
                ).add_to(traj_group)
        traj_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    folium.LatLngPopup().add_to(m)
    m.save(args.out)
    print(f"✅ 已生成：{args.out.resolve()}")

if __name__ == "__main__":
    main()


# 1) 最常用：校准磁力计 + |B|，颜色分位 10–90，粗线
python geomag_trackmap_folium.py --floor-dir .\site1\B1 \
  --source cal --stat mag --q 10,90 --lw 5 --alpha 0.95 --out b1_track.html

# 2) 固定颜色范围，保证不同楼层可比（例如 20~70 μT）
python geomag_trackmap_folium.py --floor-dir .\site1\B1 \
  --vminmax 20,70 --lw 5 --alpha 0.95 --out b1_track_20_70.html

# 3) 未校准并去偏（TXT 含 bias 三元组时）
python geomag_trackmap_folium.py --floor-dir .\site1\F1 \
  --source uncal_debiased --q 15,85 --lw 5 --alpha 0.95 --out f1_track.html

还没测试
