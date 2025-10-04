#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
indoor.py  (affine alignment)
在 Folium (CRS.Simple) 中叠加楼层底图/GeoJSON 与 TXT 轨迹，并自动应用 floor_info.json 中的仿射变换，
使 (x,y) 轨迹与底图像素坐标对齐。若 floor_info 无法解析变换，可用 --affine 手动传入 a,b,c,d,e,f。

目录要求（--floor-dir 指向的文件夹）：
  floor_image.png
  floor_info.json     # 建议包含 map_info.width/height，且最好包含 transform/affine 之类字段
  geojson_map.json
  (可选) path_data_files/  # 里面是若干 .txt（含 'TYPE_WAYPOINT x y'）

示例：
  # 自动读取并对齐（优先使用 floor_info.json 的 transform）
  python indoor.py --floor-dir .\B1

  # TXT 不在 path_data_files 里
  python indoor.py --floor-dir .\B1 --txt-dir .\B1

  # 手动提供仿射矩阵（a,b,c,d,e,f）
  # 例如：仅做 y 轴翻转 + 平移到像素坐标：a=1,b=0,c=0,d=-1,e=0,f=H
  python indoor.py --floor-dir .\B1 --affine "1,0,0,-1,0,10800"  # H=10800 举例
"""

import os
import json
import math
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import folium


# ----------------------- 基础工具 -----------------------
def color_for_name(name: str) -> str:
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    return f"#{h[4:10]}"

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

def load_xy_from_txt(txt_path: Path) -> List[List[float]]:
    pts = []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4 and parts[1] == "TYPE_WAYPOINT":
                try:
                    x = float(parts[2]); y = float(parts[3])
                    pts.append([x, y])
                except Exception:
                    continue
    return pts

def find_txt_dir(floor_dir: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit:
        if explicit.exists() and explicit.is_dir():
            if any(p.suffix.lower()==".txt" for p in explicit.iterdir()):
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


# ----------------------- 仿射变换构建 -----------------------
def parse_affine_from_string(s: str) -> Optional[Tuple[float,float,float,float,float,float]]:
    try:
        parts = [float(x.strip()) for x in s.split(",")]
        if len(parts) == 6:
            return tuple(parts)  # a,b,c,d,e,f
    except Exception:
        pass
    return None

def compose_affine(scale: Tuple[float,float]=(1,1),
                   theta_deg: float=0.0,
                   translate: Tuple[float,float]=(0,0)) -> Tuple[float,float,float,float,float,float]:
    sx, sy = scale
    th = math.radians(theta_deg)
    cos_t, sin_t = math.cos(th), math.sin(th)
    # A = T * R * S  作用在列向量 [x,y,1]^T
    a = cos_t*sx; b = -sin_t*sy
    c = sin_t*sx; d =  cos_t*sy
    e, f = translate
    return (a,b,c,d,e,f)

def try_affine_from_floorinfo(fi_raw: Any) -> Optional[Tuple[float,float,float,float,float,float]]:
    """
    兼容若干常见写法：
      1) raw["transform"] = {"a":..,"b":..,"c":..,"d":..,"e":..,"f":..}
      2) raw["transform"] = {"affine":[[a,b,e],[c,d,f]]} 或 {"affine":[a,b,c,d,e,f]}
      3) raw["transform"] = {"matrix":[[a,b,e],[c,d,f]]} 或 {"matrix":[a,b,c,d,e,f]}
      4) raw["transform"] = {"scale":[sx,sy], "translate":[tx,ty], "theta_deg":..}
      5) raw["map_info"] = {"meters_per_pixel": mpp, "origin":[ox,oy], "theta_deg":..}
         或 {"pixel_per_meter": ppm, "origin":[ox,oy], ...}
    返回 (a,b,c,d,e,f)；若无法解析，返回 None。
    """
    def normalize_6(v):
        if isinstance(v, list) and len(v) == 6:
            return tuple(float(x) for x in v)
        if isinstance(v, list) and len(v) == 2 and isinstance(v[0], list) and len(v[0])==3:
            # [[a,b,e],[c,d,f]]
            a,b,e = v[0]; c,d,f = v[1]
            return (float(a),float(b),float(c),float(d),float(e),float(f))
        return None

    raw = fi_raw
    # 1) 直接 a..f
    t = (raw.get("transform") if isinstance(raw, dict) else None) or {}
    keys = set(k for k in t.keys()) if isinstance(t, dict) else set()

    if {"a","b","c","d","e","f"}.issubset(keys):
        return (float(t["a"]), float(t["b"]), float(t["c"]),
                float(t["d"]), float(t["e"]), float(t["f"]))

    # 2) affine / 3) matrix
    for k in ("affine", "matrix"):
        if k in t:
            af = t[k]
            v = normalize_6(af)
            if v: return v

    # 4) scale + translate (+ theta_deg)
    if "scale" in t and "translate" in t:
        sx, sy = t.get("scale", [1,1]) or [1,1]
        tx, ty = t.get("translate", [0,0]) or [0,0]
        theta = float(t.get("theta_deg", 0.0))
        return compose_affine((float(sx), float(sy)), theta, (float(tx), float(ty)))

    # 5) meters_per_pixel / pixel_per_meter + origin + theta
    mi = raw.get("map_info") if isinstance(raw, dict) else None
    if isinstance(mi, dict):
        ox, oy = 0.0, 0.0
        if "origin" in mi and isinstance(mi["origin"], (list,tuple)) and len(mi["origin"])>=2:
            ox, oy = float(mi["origin"][0]), float(mi["origin"][1])
        theta = float(mi.get("theta_deg", 0.0))
        if "pixel_per_meter" in mi:
            ppm = float(mi["pixel_per_meter"])
            return compose_affine((ppm, ppm), theta, (ox, oy))
        if "meters_per_pixel" in mi:
            mpp = float(mi["meters_per_pixel"])
            ppm = 1.0 / mpp if mpp != 0 else 1.0
            return compose_affine((ppm, ppm), theta, (ox, oy))

    return None

def apply_affine(pts: List[List[float]], A: Tuple[float,float,float,float,float,float]) -> List[List[float]]:
    a,b,c,d,e,f = A
    out = []
    for x,y in pts:
        xp = a*x + b*y + e
        yp = c*x + d*y + f
        out.append([xp, yp])
    return out


# ----------------------- 主流程 -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor-dir", type=Path, required=True,
                    help="包含 geojson_map.json / floor_info.json / floor_image.png 的目录，如 .\\B1")
    ap.add_argument("--txt-dir", type=Path, default=None,
                    help="轨迹 TXT 所在目录（不传则自动探测）")
    ap.add_argument("--use-crs-simple", type=int, default=1,
                    help="1=平面坐标（CRS.Simple），0=经纬度（默认 1）")
    ap.add_argument("--y-flip", type=int, default=0,
                    help="仿射之后额外做 y 翻转（y -> H - y）。若仅靠 affine 已对齐，建议设 0（默认 0）")
    ap.add_argument("--sample-every", type=int, default=1,
                    help="轨迹抽稀步长，默认 1")
    ap.add_argument("--out", type=Path, default=Path("floor_overlay_trajectories.html"),
                    help="输出 HTML 文件名")
    ap.add_argument("--preview", action="store_true",
                    help="同时用 matplotlib 画简图预览")
    ap.add_argument("--affine", type=str, default=None,
                    help='手动传 a,b,c,d,e,f 覆盖（像素=仿射(米)）。示例："1,0,0,-1,0,10800"')

    args = ap.parse_args()

    floor_dir = args.floor_dir
    GEOJSON_PATH = floor_dir / "geojson_map.json"
    FLOORINFO_PATH = floor_dir / "floor_info.json"
    FLOOR_IMAGE = floor_dir / "floor_image.png"

    print("→ 使用楼层目录：", floor_dir.resolve())
    if not GEOJSON_PATH.exists():
        raise FileNotFoundError(f"缺少 GeoJSON：{GEOJSON_PATH}")
    if not FLOORINFO_PATH.exists():
        raise FileNotFoundError(f"缺少 floor_info.json：{FLOORINFO_PATH}")

    txt_dir = find_txt_dir(floor_dir, args.txt_dir)
    if txt_dir:
        print("→ 使用轨迹目录：", txt_dir.resolve())
    else:
        print("⚠️  未找到含 .txt 的轨迹目录，将仅显示底图/GeoJSON。")
    fi = read_floor_info(FLOORINFO_PATH)
    map_w, map_h, fi_raw = fi["map_w"], fi["map_h"], fi["raw"]

    geojson_data = load_json(GEOJSON_PATH)

    # 构建仿射
    A = None
    if args.affine:
        A = parse_affine_from_string(args.affine)
        if A:
            print(f"✔ 使用手动仿射矩阵 a,b,c,d,e,f = {A}")
        else:
            print("⚠️  --affine 解析失败，将继续尝试从 floor_info.json 自动提取")

    if A is None:
        A = try_affine_from_floorinfo(fi_raw)
        if A:
            print(f"✔ 从 floor_info.json 提取到仿射矩阵 a,b,c,d,e,f = {A}")
        else:
            print("⚠️  floor_info.json 中未检测到可用仿射，将仅使用简单翻转/原样坐标")

    # 地图
    if args.use_crs_simple:
        m = folium.Map(location=[map_h/2, map_w/2], zoom_start=0, tiles=None, crs="Simple")
        if FLOOR_IMAGE.exists():
            folium.raster_layers.ImageOverlay(
                name="floor_image", image=str(FLOOR_IMAGE),
                bounds=[[0, 0], [map_h, map_w]], opacity=1.0, interactive=False, cross_origin=False
            ).add_to(m)
        else:
            print("⚠️  未找到 floor_image.png，将只显示 GeoJSON 与轨迹。")
    else:
        # 若真是经纬度，这里可改成计算重心
        m = folium.Map(location=[map_h/2, map_w/2], zoom_start=18, tiles="CartoDB positron")

    # GeoJSON（淡色）
    def style_function(feature):
        props = feature.get("properties", {}) or {}
        fid = props.get("floor_id", None)
        try: fid = int(fid)
        except Exception: fid = None
        color_map = {1: "#3186cc", 2: "#2ecc71", 3: "#e74c3c"}
        return {"fillColor": color_map.get(fid, "#95a5a6"),
                "color": "#2c3e50", "weight": 1, "fillOpacity": 0.25}
    folium.GeoJson(geojson_data, name="floor_geojson", style_function=style_function).add_to(m)

    # 轨迹
    if txt_dir and txt_dir.exists():
        txt_files = sorted([p for p in txt_dir.iterdir() if p.suffix.lower()==".txt"])
        for txt in txt_files:
            pts = load_xy_from_txt(txt)
            if len(pts) < 2:
                continue
            if args.sample_every > 1:
                pts = pts[::args.sample_every]
            # 仿射 -> 像素
            if A is not None:
                pts = apply_affine(pts, A)
            # 额外 y 翻转（可选）
            if args.y_flip:
                pts = [[x, (map_h - y)] for (x,y) in pts]
            latlngs = [[p[1], p[0]] for p in pts]  # [lat(y), lon(x)]
            folium.PolyLine(
                locations=latlngs, color=color_for_name(txt.name),
                weight=2, opacity=0.9, tooltip=txt.name
            ).add_to(m)
    else:
        print("ℹ️  未绘制任何轨迹。")

    folium.LayerControl(collapsed=False).add_to(m)
    folium.LatLngPopup().add_to(m)
    m.save(args.out)
    print(f"✅ 已生成：{args.out.resolve()}")

    # 可选 matplotlib 预览
    if args.preview and txt_dir and txt_dir.exists():
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(7,6))
            for txt in sorted([p for p in txt_dir.iterdir() if p.suffix.lower()==".txt"]):
                pts = load_xy_from_txt(txt)
                if len(pts) < 2: continue
                if args.sample_every > 1:
                    pts = pts[::args.sample_every]
                if A is not None:
                    pts = apply_affine(pts, A)
                xs = [p[0] for p in pts]
                ys = [ (map_h - p[1]) if args.y_flip else p[1] for p in pts ]
                plt.plot(xs, ys, ".", ms=2, label=txt.name)
            plt.title("Trajectories (pixel space)")
            plt.xlabel("x_px"); plt.ylabel("y_px" + (" [flipped]" if args.y_flip else ""))
            plt.grid(True); plt.legend(bbox_to_anchor=(1.05,1), loc="upper left", fontsize=6)
            plt.tight_layout(); plt.show()
        except Exception as e:
            print(f"⚠️  Matplotlib 预览失败：{e}")


if __name__ == "__main__":
    main()

# python .\indoor_plus.py --floor-dir .\site1\B1 --out b1_map.html --preview
