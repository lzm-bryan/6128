#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
indoor.py  (with --txt-dir and auto-detect)
-------------------------------------------
在 Folium 里把楼层底图/GeoJSON 与 TXT 轨迹叠加到同一张交互地图上。

--floor-dir  指向包含 floor_image.png / floor_info.json / geojson_map.json 的目录（如 .\B1）
--txt-dir    可显式指定轨迹目录（里面放 *.txt）。若不指定，将自动探测：
             1) <floor-dir>/path_data_files
             2) <floor-dir>（根目录）
             3) <floor-dir> 的子目录（第一个含 .txt 的子目录）
其它参数见 argparse。
"""

import os
import json
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import folium


# ----------------------- 实用函数 -----------------------
def color_for_name(name: str) -> str:
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    return f"#{h[4:10]}"

def read_floor_info(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        return {"map_w": float(obj["map_info"]["width"]),
                "map_h": float(obj["map_info"]["height"]),
                "raw": obj}
    if isinstance(obj, list) and obj:
        return {"map_w": float(obj[0]["map_info"]["width"]),
                "map_h": float(obj[0]["map_info"]["height"]),
                "raw": obj}
    raise TypeError("floor_info.json 类型异常，应为 dict 或非空 list")

def get_geojson_center_ll(geojson: Dict[str, Any]) -> List[float]:
    coords = []
    for feat in geojson.get("features", []):
        geom = feat.get("geometry") or {}
        t = geom.get("type")
        if t == "Polygon":
            rings = geom.get("coordinates", [])
            if rings: coords += rings[0]
        elif t == "MultiPolygon":
            for poly in geom.get("coordinates", []):
                if poly: coords += poly[0]
    if not coords:
        return [1.3, 103.8]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return [sum(lats)/len(lats), sum(lons)/len(lons)]

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
    """返回包含 .txt 的目录路径；若找不到返回 None。"""
    if explicit:
        if explicit.exists() and explicit.is_dir():
            if any(p.suffix.lower()==".txt" for p in explicit.iterdir()):
                return explicit
        return explicit if explicit.exists() else None

    # 1) floor_dir/path_data_files
    cand = floor_dir / "path_data_files"
    if cand.exists() and cand.is_dir() and any(p.suffix.lower()==".txt" for p in cand.iterdir()):
        return cand

    # 2) floor_dir 根目录
    if any(p.suffix.lower()==".txt" for p in floor_dir.iterdir()):
        return floor_dir

    # 3) 子目录中第一个含 .txt 的目录
    for sub in floor_dir.iterdir():
        if sub.is_dir() and any(p.suffix.lower()==".txt" for p in sub.iterdir()):
            return sub

    return None


# ----------------------- 主流程 -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor-dir", type=Path, required=True,
                    help="包含 geojson_map.json / floor_info.json / floor_image.png 的目录，如 .\\B1")
    ap.add_argument("--txt-dir", type=Path, default=None,
                    help="轨迹 TXT 所在目录（可不传，脚本自动探测）")
    ap.add_argument("--use-crs-simple", type=int, default=1,
                    help="1=平面坐标（CRS.Simple），0=经纬度（默认 1）")
    ap.add_argument("--y-flip", type=int, default=1,
                    help="是否做 y 翻转（y -> H - y），默认 1")
    ap.add_argument("--sample-every", type=int, default=1,
                    help="轨迹抽稀步长，默认 1")
    ap.add_argument("--out", type=Path, default=Path("floor_overlay_trajectories.html"),
                    help="输出 HTML 文件名")
    ap.add_argument("--preview", action="store_true",
                    help="同时用 matplotlib 画简图预览")
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

    # 轨迹目录探测 / 使用
    txt_dir = find_txt_dir(floor_dir, args.txt_dir)
    if txt_dir is None:
        print(f"⚠️  未找到含 .txt 的轨迹目录（尝试了 path_data_files/、B1 根目录及其子目录）。")
        print("   你可以用 --txt-dir 指定，例如：--txt-dir .\\B1 或 --txt-dir .\\B1\\path_data_files")
    else:
        print("→ 使用轨迹目录：", txt_dir.resolve())

    # 读取 floor_info & geojson
    fi = read_floor_info(FLOORINFO_PATH)
    map_w, map_h = fi["map_w"], fi["map_h"]

    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    # 创建地图
    if args.use_crs_simple:
        m = folium.Map(location=[map_h/2, map_w/2], zoom_start=0, tiles=None, crs="Simple")
        if FLOOR_IMAGE.exists():
            folium.raster_layers.ImageOverlay(
                name="floor_image",
                image=str(FLOOR_IMAGE),
                bounds=[[0, 0], [map_h, map_w]],
                opacity=1.0,
                interactive=False,
                cross_origin=False,
            ).add_to(m)
        else:
            print("⚠️  未找到 floor_image.png，将只显示 GeoJSON 与轨迹。")
    else:
        center = get_geojson_center_ll(geojson_data)
        m = folium.Map(location=center, zoom_start=18, tiles="CartoDB positron")

    # GeoJSON 样式
    def style_function(feature):
        props = feature.get("properties", {}) or {}
        fid = props.get("floor_id", None)
        try: fid = int(fid)
        except Exception: fid = None
        color_map = {1: "#3186cc", 2: "#2ecc71", 3: "#e74c3c"}
        return {"fillColor": color_map.get(fid, "#95a5a6"),
                "color": "#2c3e50", "weight": 1, "fillOpacity": 0.25}

    folium.GeoJson(geojson_data, name="floor_geojson", style_function=style_function).add_to(m)

    # 画轨迹
    if txt_dir and txt_dir.exists():
        txt_files = sorted([p for p in txt_dir.iterdir() if p.suffix.lower() == ".txt"])
        if not txt_files:
            print(f"ℹ️  轨迹目录 {txt_dir} 下没有 .txt 文件。")
        for txt in txt_files:
            pts = load_xy_from_txt(txt)
            if len(pts) < 2:  # 至少两个点才画线
                continue
            if args.sample_every > 1:
                pts = pts[::args.sample_every]
            if args.y_flip:
                pts = [[x, (map_h - y)] for (x, y) in pts]
            latlngs = [[p[1], p[0]] for p in pts]  # [lat(y), lon(x)]
            folium.PolyLine(
                locations=latlngs,
                color=color_for_name(txt.name),
                weight=2, opacity=0.9,
                tooltip=txt.name
            ).add_to(m)
    else:
        print("ℹ️  未绘制任何轨迹。")

    folium.LayerControl(collapsed=False).add_to(m)
    folium.LatLngPopup().add_to(m)
    m.save(args.out)
    print(f"✅ 已生成：{args.out.resolve()}")

    # 可选：Matplotlib 预览
    if args.preview and txt_dir and txt_dir.exists():
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(7, 6))
            for txt in sorted([p for p in txt_dir.iterdir() if p.suffix.lower() == ".txt"]):
                pts = load_xy_from_txt(txt)
                if len(pts) < 2:
                    continue
                if args.sample_every > 1:
                    pts = pts[::args.sample_every]
                xs = [p[0] for p in pts]
                ys = [(map_h - p[1]) if args.y_flip else p[1] for p in pts]
                plt.plot(xs, ys, ".", linewidth=1, markersize=2, label=txt.name)
            plt.title("Trajectories (XY plane)")
            plt.xlabel("X"); plt.ylabel("Y" + (" [flipped]" if args.y_flip else ""))
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=6)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"⚠️  Matplotlib 预览失败：{e}")


if __name__ == "__main__":
    main()

# python .\indoor.py --floor-dir .\B1 --out b1_map.html --preview
