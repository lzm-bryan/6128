#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
geomag_heatmap.py  —  Visualize geomagnetic heat map (indoor, CRS.Simple)
从 TXT 中解析 WAYPOINT + 磁力计，将磁场按时间插值到轨迹位置，生成热力图。
独立运行，不依赖其它脚本。

目录结构（--floor-dir 指向该层目录）：
  floor_image.png
  floor_info.json    # map_info.width/height；若含 transform/affine/matrix/scale/translate 等会自动解析仿射
  geojson_map.json
  (可选) path_data_files/  # 里边若干 .txt

示例：
  python geomag_heatmap.py --floor-dir .\B1 --out b1_heat.html
  python geomag_heatmap.py --floor-dir .\B1 --heat-source uncal --heat-stat mag --heat-q 5,95 --heat-subsample 3
  python geomag_heatmap.py --floor-dir .\B1 --affine "1,0,0,-1,0,10800" --y-flip 0 --show-traj 1

注意：默认使用 Folium 的 CRS.Simple（本地平面/像素坐标），非经纬度。
"""

import os, json, math, argparse, hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import folium
from folium.plugins import HeatMap


# ========== 基础 I/O ==========
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
    if explicit:
        if explicit.exists() and explicit.is_dir() and any(p.suffix.lower()==".txt" for p in explicit.iterdir()):
            return explicit
        return explicit if explicit.exists() else None
    cand = floor_dir / "path_data_files"
    if cand.exists() and cand.is_dir() and any(p.suffix.lower()==".txt" for p in cand.iterdir()):
        return cand
    if any(p.suffix.lower()==".txt" for p in floor_dir.iterdir()):
        return floor_dir
    for sub in floor_dir.iterdir():
        if sub.is_dir() and any(p.suffix.lower()==".txt" for p in sub.iterdir()):
            return sub
    return None


# ========== 仿射 ==========
def parse_affine_from_string(s: str) -> Optional[Tuple[float,float,float,float,float,float]]:
    try:
        parts = [float(x.strip()) for x in s.split(",")]
        if len(parts) == 6:
            return tuple(parts)
    except Exception:
        pass
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
            a,b,e = v[0]; c,d,f = v[1]; return (float(a),float(b),float(c),float(d),float(e),float(f))
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
            ppm = float(mi["pixel_per_meter"])
            return compose_affine((ppm,ppm), theta, (ox,oy))
        if "meters_per_pixel" in mi:
            mpp = float(mi["meters_per_pixel"]); ppm = 1.0/mpp if mpp!=0 else 1.0
            return compose_affine((ppm,ppm), theta, (ox,oy))
    return None

def apply_affine_xy(x: float, y: float, A: Tuple[float,float,float,float,float,float]):
    a,b,c,d,e,f = A
    return a*x + b*y + e, c*x + d*y + f


# ========== 解析 TXT ==========
def parse_waypoints_and_mags(txt_path: Path, prefer: str = "cal"):
    """
    返回：
      waypoints: List[(t:int, x:float, y:float)]
      mags:      List[(t:int, bx:float, by:float, bz:float, src:str)]  # src in {"cal","uncal"}
    """
    wps, mags_cal, mags_uncal = [], [], []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line[0] == "#":  # 跳过头部元数据
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
                    bx,by,bz = float(parts[2]), float(parts[3]), float(parts[4])
                    mags_cal.append((t,bx,by,bz))
                except Exception:
                    pass
                continue

            if typ == "TYPE_MAGNETIC_FIELD_UNCALIBRATED" and len(parts) >= 5:
                try:
                    bx,by,bz = float(parts[2]), float(parts[3]), float(parts[4])
                    mags_uncal.append((t,bx,by,bz))
                except Exception:
                    pass
                continue

    wps.sort(key=lambda x: x[0])
    mags_cal.sort(key=lambda x: x[0])
    mags_uncal.sort(key=lambda x: x[0])

    use_src = "cal"
    mags = mags_cal
    if prefer == "uncal" and mags_uncal:
        mags, use_src = mags_uncal, "uncal"
    elif prefer == "cal" and not mags_cal and mags_uncal:
        mags, use_src = mags_uncal, "uncal"

    return wps, [(t,bx,by,bz,use_src) for (t,bx,by,bz) in mags]


def interpolate_pos_for_times(waypoints, times, mode="linear"):
    """
    waypoints: 按时间升序 [(t,x,y), ...]
    times:     [t1, t2, ...]
    mode: "linear" | "hold" | "skip"
    """
    if not waypoints:
        return [None]*len(times)

    ts = [w[0] for w in waypoints]
    xs = [w[1] for w in waypoints]
    ys = [w[2] for w in waypoints]

    out = []
    j = 0; n = len(waypoints)
    for t in times:
        while j+1 < n and ts[j+1] <= t:
            j += 1
        if j < n-1:
            t0, t1 = ts[j], ts[j+1]
            if t0 <= t <= t1:
                if mode == "linear":
                    r = (t - t0)/max(1, (t1 - t0))
                    out.append((xs[j]*(1-r) + xs[j+1]*r, ys[j]*(1-r) + ys[j+1]*r))
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


# ========== 归一化/热力点 ==========
def robust_minmax(values: List[float], q_low=5.0, q_high=95.0):
    if not values: return 0.0, 1.0
    vs = sorted(values)
    def q(p):
        if not vs: return 0.0
        k = (len(vs)-1)*(p/100.0)
        f = int(math.floor(k)); c = int(math.ceil(k))
        if f==c: return vs[f]
        return vs[f] + (vs[c]-vs[f])*(k-f)
    lo, hi = q(q_low), q(q_high)
    if hi <= lo: hi = lo + 1e-6
    return lo, hi

def make_geomag_heat_points(txt_files: List[Path], A, map_h, y_flip,
                            prefer_src="cal", stat="mag", subsample=1,
                            interp_mode="linear", q_low=5.0, q_high=95.0):
    """
    返回 Folium HeatMap 点：[lat, lon, weight]   (CRS.Simple: lat=Y, lon=X, weight∈[0,1])
    """
    raw_vals = []
    temp_pts = []  # (x_px, y_px, v_raw)

    for txt in txt_files:
        wps, mags = parse_waypoints_and_mags(txt, prefer=prefer_src)
        if len(wps) < 2 or len(mags) == 0:
            continue
        times = [t for (t, *_r) in mags]
        pos = interpolate_pos_for_times(wps, times, mode=interp_mode)

        keep = []
        for p, m in zip(pos, mags):
            if p is None: continue
            t, bx, by, bz, _src = m
            if stat == "mag":
                v = math.sqrt(bx*bx + by*by + bz*bz)
            elif stat == "bx":
                v = bx
            elif stat == "by":
                v = by
            elif stat == "bz":
                v = bz
            else:
                v = math.sqrt(bx*bx + by*by + bz*bz)
            x, y = p
            if A is not None:
                x, y = apply_affine_xy(x, y, A)
            if y_flip:
                y = map_h - y
            keep.append((x,y,v))

        if subsample > 1:
            keep = keep[::subsample]
        for (x,y,v) in keep:
            temp_pts.append((x,y,v))
            raw_vals.append(v)

    if not temp_pts:
        return []

    lo, hi = robust_minmax(raw_vals, q_low=q_low, q_high=q_high)
    def norm(v): v = max(lo, min(hi, v)); return (v - lo)/(hi - lo + 1e-9)

    return [[y, x, norm(v)] for (x,y,v) in temp_pts]


# ========== 其它小工具 ==========
def color_for_name(name: str) -> str:
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    return f"#{h[4:10]}"


# ========== 主流程 ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor-dir", type=Path, required=True, help="包含 floor_info / geojson / floor_image 的目录")
    ap.add_argument("--txt-dir", type=Path, default=None, help="TXT 所在目录（不传则自动探测）")
    ap.add_argument("--use-crs-simple", type=int, default=1, help="1=CRS.Simple，0=经纬度（默认 1）")
    ap.add_argument("--y-flip", type=int, default=0, help="仿射后额外做 y->H-y（默认 0）")
    ap.add_argument("--affine", type=str, default=None, help='手动仿射 a,b,c,d,e,f 覆盖，例如 "1,0,0,-1,0,10800"')
    ap.add_argument("--out", type=Path, default=Path("geomag_heatmap.html"), help="输出 HTML")

    # 可选叠加层
    ap.add_argument("--no-image", action="store_true", help="不叠加楼层底图")
    ap.add_argument("--no-geojson", action="store_true", help="不叠加 GeoJSON")
    ap.add_argument("--show-traj", type=int, default=0, help="是否叠加轨迹折线（默认 0 不显示）")
    ap.add_argument("--traj-subsample", type=int, default=1, help="轨迹抽稀（仅显示用）")

    # 热力参数
    ap.add_argument("--heat-source", type=str, default="cal", choices=["cal","uncal"], help="磁场来源（默认 cal）")
    ap.add_argument("--heat-stat", type=str, default="mag", choices=["mag","bx","by","bz"], help="权重取值（默认 |B|）")
    ap.add_argument("--heat-subsample", type=int, default=1, help="磁场样本抽稀（默认 1）")
    ap.add_argument("--heat-interp", type=str, default="linear", choices=["linear","hold","skip"], help="WAYPOINT 插值策略")
    ap.add_argument("--heat-q", type=str, default="5,95", help="归一化分位数范围（默认 '5,95'）")
    ap.add_argument("--heat-radius", type=int, default=16, help="HeatMap 半径（默认 16）")
    ap.add_argument("--heat-blur", type=int, default=18, help="HeatMap 模糊（默认 18）")
    ap.add_argument("--heat-max-zoom", type=int, default=18, help="HeatMap 最大缩放（默认 18）")
    ap.add_argument("--heat-min-pts", type=int, default=50, help="不足该点数则不绘制热力图（默认 50）")
    ap.add_argument("--heat-opacity", type=float, default=0.85, help="热力图层最小不透明度（默认 0.85）")

    args = ap.parse_args()

    floor_dir = args.floor_dir
    GEOJSON = floor_dir / "geojson_map.json"
    FLOORINFO = floor_dir / "floor_info.json"
    FLOORIMG = floor_dir / "floor_image.png"

    print("→ 楼层目录：", floor_dir.resolve())
    if not FLOORINFO.exists():
        raise FileNotFoundError(f"缺少 floor_info.json：{FLOORINFO}")
    if not GEOJSON.exists() and not args.no-geojson:
        print("⚠️  未找到 geojson_map.json，将不叠加 GeoJSON。")
        args.no_geojson = True  # 兜底

    txt_dir = find_txt_dir(floor_dir, args.txt_dir)
    if not txt_dir:
        print("⚠️  未找到含 .txt 的目录，仅输出底图/GeoJSON（无热力图）。")
    else:
        print("→ TXT 目录：", txt_dir.resolve())

    fi = read_floor_info(FLOORINFO)
    map_w, map_h, fi_raw = fi["map_w"], fi["map_h"], fi["raw"]

    # 仿射
    A = None
    if args.affine:
        A = parse_affine_from_string(args.affine)
        if A: print(f"✔ 使用手动仿射：{A}")
        else: print("⚠️  --affine 解析失败，改用 floor_info 自动解析")
    if A is None:
        A = try_affine_from_floorinfo(fi_raw)
        if A: print(f"✔ 从 floor_info 提取仿射：{A}")
        else: print("⚠️  未提取到仿射，将使用原始坐标 + 可选 y 翻转")

    # 地图
    if args.use_crs_simple:
        m = folium.Map(location=[map_h/2, map_w/2], zoom_start=0, tiles=None, crs="Simple")
        if not args.no_image and FLOORIMG.exists():
            folium.raster_layers.ImageOverlay(
                name="floor_image", image=str(FLOORIMG),
                bounds=[[0,0],[map_h,map_w]], opacity=1.0, interactive=False, cross_origin=False
            ).add_to(m)
        elif not args.no_image:
            print("⚠️  未找到 floor_image.png，跳过底图。")
    else:
        print("⚠️  use_crs_simple=0 需要真实经纬度 bounds，这里仅占位示例。")
        m = folium.Map(location=[map_h/2, map_w/2], zoom_start=18, tiles="CartoDB positron")

    # GeoJSON
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

    # 热力图
    if txt_dir and txt_dir.exists():
        txts = sorted([p for p in txt_dir.iterdir() if p.suffix.lower()==".txt"])
        try:
            ql, qh = [float(x.strip()) for x in args.heat_q.split(",")]
        except Exception:
            ql, qh = 5.0, 95.0

        heat_pts = make_geomag_heat_points(
            txts, A, map_h, args.y_flip,
            prefer_src=args.heat_source, stat=args.heat_stat,
            subsample=args.heat_subsample, interp_mode=args.heat_interp,
            q_low=ql, q_high=qh
        )
        print(f"→ 收集到热力点：{len(heat_pts)}")
        if len(heat_pts) >= args.heat_min_pts:
            HeatMap(
                heat_pts,
                name=f"Geomag Heat ({args.heat_source}/{args.heat_stat})",
                radius=args.heat_radius,
                blur=args.heat_blur,
                max_zoom=args.heat_max_zoom,
                min_opacity=args.heat_opacity
            ).add_to(m)
        else:
            print(f"⚠️  点数不足（<{args.heat_min_pts}），跳过热力图。")

        # 可选叠加轨迹（仅展示）
        if args.show_traj:
            for txt in txts:
                # 只取 WAYPOINT 画线
                wps, _ = parse_waypoints_and_mags(txt, prefer=args.heat_source)
                if len(wps) < 2: continue
                pts = []
                for (_,x,y) in wps:
                    if A is not None:
                        x,y = apply_affine_xy(x,y,A)
                    if args.y_flip:
                        y = map_h - y
                    pts.append([y,x])  # lat,lon
                if len(pts) >= 2:
                    folium.PolyLine(pts, color=color_for_name(txt.name), weight=2, opacity=0.9,
                                    tooltip=txt.name).add_to(m)
    else:
        print("ℹ️  无 TXT，无法生成热力图。")

    folium.LayerControl(collapsed=False).add_to(m)
    folium.LatLngPopup().add_to(m)
    m.save(args.out)
    print(f"✅ 已生成：{args.out.resolve()}")


if __name__ == "__main__":
    main()

# # 基本：自动读取同层 TXT，|B| 热力图
# python geomag_heatmap.py --floor-dir .\site1\B1 --out b1_heat.html
#
# # 用未校准磁力计 + 分位数拉伸 + 抽稀
# python geomag_heatmap.py --floor-dir .\site1\B1 --heat-source uncal --heat-q 5,95 --heat-subsample 3
#
# # 只看 Bz 分量，减小半径和模糊
# python geomag_heatmap.py --floor-dir .\site1\B1 --heat-stat bz --heat-radius 10 --heat-blur 12
#
# # 指定手动仿射（像素=仿射(米)），并显示轨迹
# python geomag_heatmap.py --floor-dir .\site1\B1 --affine "1,0,0,-1,0,10800" --show-traj 1
