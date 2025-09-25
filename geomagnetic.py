#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io, os, re, json, time, math, random
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from urllib.request import urlopen, Request
import folium
from folium.features import GeoJsonTooltip
from folium.plugins import HeatMap, MarkerCluster

# ========== 配置（按需修改） ==========
GEOJSON_URL    = "https://github.com/location-competition/indoor-location-competition-20/blob/master/data/site1/F1/geojson_map.json"
FLOOR_INFO_URL = "https://github.com/location-competition/indoor-location-competition-20/blob/master/data/site1/F1/floor_info.json"
FLOOR_IMG_URL  = "https://github.com/location-competition/indoor-location-competition-20/blob/master/data/site1/F1/floor_image.png"
GT_FOLDER_URL  = "https://github.com/location-competition/indoor-location-competition-20/tree/master/data/site1/F1/path_data_files"

CACHE_DIR      = os.path.join("indoor_cache", "site1", "F1")
OUT_HTML       = "F1_magnetic_heatmap.html"

STYLE_FIELD    = ""       # GeoJSON properties 里想用于着色的字段（留空则统一样式）
NAME_FILTER    = ""       # 只处理文件名包含该子串的 .txt（留空=全部）
MAX_FILES      = 0        # 0=全部；>0 只取前 N 个
SNAP_WP_M      = 0.05     # 对 waypoint 做 5cm 栅格去重（=0 关闭）
ACC_FILTER     = 0        # 仅保留磁场样本 accuracy>=此阈值（0/1/2/3；设 0 不筛）
HEAT_MAX_PTS   = 20000    # 热力图最多点数（过多会卡；超出则随机下采样）
HEAT_RADIUS    = 7        # HeatMap 点半径（像素）
ADD_WAYPOINTS  = True     # 同时把 waypoint 折线/散点叠加到图上以对比
ROBUST_CLIP_P  = (5, 95)  # 权重归一化的分位裁剪（抗异常值）

# ========== 工具函数 ==========
def _to_raw(url: str) -> str:
    return url.replace("https://github.com/","https://raw.githubusercontent.com/").replace("/blob/","/") \
           if "github.com" in url and "/blob/" in url else url

def _req(url: str) -> bytes:
    req = Request(url, headers={"User-Agent":"Mozilla/5.0","Accept":"*/*"})
    with urlopen(req, timeout=60) as r:
        return r.read()

def _download_to_cache(url: str, cache_dir: str, filename: str = None, force: bool = False) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    raw_url = _to_raw(url)
    if filename is None:
        filename = os.path.basename(urlparse(raw_url).path) or f"file_{int(time.time())}"
    path = os.path.join(cache_dir, filename)
    if (not force) and os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"↩️  缓存命中：{path}")
        return path
    print(f"⬇️  下载：{raw_url}")
    data = _req(raw_url)
    with open(path, "wb") as f: f.write(data)
    print(f"✅ 完成：{path}（{len(data)/1024:.1f} KB）")
    return path

def _parse_github_dir_api(url: str):
    """优先用 GitHub API 列目录；失败再用 HTML 兜底。只返回 .txt。"""
    p = urlparse(url)
    parts = [x for x in p.path.strip("/").split("/") if x]
    assert len(parts) >= 2, "不是合法的 GitHub 仓库链接"
    owner, repo = parts[0], parts[1]
    branch, subpath = "master", ""
    if len(parts) >= 4 and parts[2] in ("tree","blob"):
        branch = parts[3] if len(parts) >= 4 else "master"
        subpath = "/".join(parts[4:])
    else:
        subpath = "/".join(parts[2:])
    api = f"https://api.github.com/repos/{owner}/{repo}/contents/{subpath}?ref={branch}"
    try:
        data = json.loads(_req(api).decode("utf-8", "ignore"))
        files = []
        for it in data:
            if it.get("type") == "file" and it.get("name","").endswith(".txt"):
                files.append({"name": it["name"], "download_url": it.get("download_url") or _to_raw(f"https://github.com/{owner}/{repo}/blob/{branch}/{it['path']}")})
        return (owner, repo, branch, subpath), files
    except Exception as e:
        print(f"⚠️ API 列目录失败，改用 HTML 解析：{e}")

    html = _req(url).decode("utf-8","ignore")
    hrefs = re.findall(r'href="([^"]+\.txt)"', html)
    files = []
    for h in hrefs:
        if not h.startswith("http"):
            h = f"https://github.com{h}"
        if "/blob/" not in h:  # 保证能转 raw
            continue
        name = os.path.basename(urlparse(h).path)
        files.append({"name": name, "download_url": _to_raw(h)})
    if not files:
        raise RuntimeError("未能从目录页解析出 .txt 文件链接")
    return (owner, repo, branch, subpath), files

def _read_json(path: str):
    with open(path,"r",encoding="utf-8") as f:
        return json.load(f)

# —— 解析 WAYPOINT 与 MAGNETIC_FIELD ——
def _read_waypoints(path: str) -> pd.DataFrame:
    rows = []
    with open(path,"r",encoding="utf-8",errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): continue
            s = s.replace(",", " ")
            p = s.split()
            if len(p) >= 4 and p[1] == "TYPE_WAYPOINT":
                ts = int(p[0])
                x  = float(p[2]); y = float(p[3])
                rows.append((ts,x,y))
    if not rows:
        return pd.DataFrame(columns=["ts","x","y"])
    df = pd.DataFrame(rows, columns=["ts","x","y"]).sort_values("ts", kind="mergesort").reset_index(drop=True)
    if SNAP_WP_M and SNAP_WP_M > 0:
        df["_ix"] = (df["x"]/SNAP_WP_M).round().astype(int)
        df["_iy"] = (df["y"]/SNAP_WP_M).round().astype(int)
        df = df.drop_duplicates(["_ix","_iy"]).drop(columns=["_ix","_iy"])
    return df

def _read_magnetic(path: str) -> pd.DataFrame:
    """优先已校准 TYPE_MAGNETIC_FIELD；若没有则退化用 TYPE_MAGNETIC_FIELD_UNCALIBRATED 的前三列。"""
    mag, mag_unc = [], []
    with open(path,"r",encoding="utf-8",errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): continue
            s = s.replace(",", " ")
            p = s.split()
            if len(p) >= 5 and p[1] == "TYPE_MAGNETIC_FIELD":
                ts = int(p[0]); mx, my, mz = float(p[2]), float(p[3]), float(p[4])
                acc = int(p[5]) if len(p) >= 6 and p[5].isdigit() else 0
                mag.append((ts,mx,my,mz,acc))
            elif len(p) >= 5 and p[1] == "TYPE_MAGNETIC_FIELD_UNCALIBRATED":
                ts = int(p[0]); mx, my, mz = float(p[2]), float(p[3]), float(p[4])
                acc = int(p[-1]) if p[-1].isdigit() else 0
                mag_unc.append((ts,mx,my,mz,acc))
    rows = mag if len(mag)>0 else mag_unc
    if not rows:
        return pd.DataFrame(columns=["ts","mx","my","mz","acc","B"])
    df = pd.DataFrame(rows, columns=["ts","mx","my","mz","acc"]).sort_values("ts").reset_index(drop=True)
    if ACC_FILTER>0:
        df = df[df["acc"]>=ACC_FILTER]
    df["B"] = np.sqrt(df["mx"]**2 + df["my"]**2 + df["mz"]**2)
    return df

# —— 坐标与 GeoJSON 变换 ——
def _xy_to_leaflet(x,y,h):     # meters -> CRS.Simple (lon=x, lat=h-y)
    return [x, h - y]

def _transform_geojson(gj: dict, map_h: float):
    def _tx(coords):
        if isinstance(coords[0], (int,float)):
            x,y = coords[:2]
            lon, lat = _xy_to_leaflet(x,y,map_h)
            return [lon,lat]
        return [_tx(c) for c in coords]
    if gj.get("type") == "FeatureCollection":
        out = {"type":"FeatureCollection","features":[]}
        for ft in gj.get("features", []):
            g = ft.get("geometry") or {}
            if not g: continue
            f2 = dict(ft)
            f2["geometry"] = {"type": g.get("type"), "coordinates": _tx(g.get("coordinates"))}
            out["features"].append(f2)
        return out
    return {"type": gj.get("type","GeometryCollection"), "coordinates": _tx(gj.get("coordinates",[]))}

# —— 磁场样本时间插值到 (x,y) ——
def interpolate_magnetic_to_xy(mf: pd.DataFrame, wp: pd.DataFrame) -> pd.DataFrame:
    """给磁场样本补 (x,y)。只保留处于任意相邻 waypoint 区间 [ts_prev, ts_next] 内的样本。"""
    if mf.empty or len(wp) < 2:
        return pd.DataFrame(columns=["ts","x","y","B"])
    wp = wp.sort_values("ts")
    mf = mf.sort_values("ts")
    prev = pd.merge_asof(
        mf[["ts","B"]],
        wp[["ts","x","y"]].rename(columns={"ts":"ts_prev","x":"x_prev","y":"y_prev"}),
        left_on="ts", right_on="ts_prev", direction="backward"
    )
    nxt = pd.merge_asof(
        mf[["ts"]],
        wp[["ts","x","y"]].rename(columns={"ts":"ts_next","x":"x_next","y":"y_next"}),
        left_on="ts", right_on="ts_next", direction="forward"
    )
    z = prev.join(nxt[["ts_next","x_next","y_next"]])
    z = z[(~z["ts_prev"].isna()) & (~z["ts_next"].isna()) & (z["ts_next"]>z["ts_prev"])]
    t = (z["ts"] - z["ts_prev"]) / (z["ts_next"] - z["ts_prev"])
    z["x"] = z["x_prev"] + t * (z["x_next"] - z["x_prev"])
    z["y"] = z["y_prev"] + t * (z["y_next"] - z["y_prev"])
    return z[["ts","x","y","B"]].reset_index(drop=True)

# ========== 主流程 ==========
def main():
    print(f"📁 缓存目录：{CACHE_DIR}")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 1) 下载底图三件套
    local_geojson = _download_to_cache(GEOJSON_URL, CACHE_DIR)
    local_floor   = _download_to_cache(FLOOR_INFO_URL, CACHE_DIR)
    local_img     = _download_to_cache(FLOOR_IMG_URL, CACHE_DIR)

    # 2) 列出并下载 .txt
    (_, _, _, _), file_list = _parse_github_dir_api(GT_FOLDER_URL)
    file_list = [f for f in file_list if (NAME_FILTER in f["name"])]
    file_list.sort(key=lambda x: x["name"])
    if MAX_FILES and MAX_FILES>0:
        file_list = file_list[:MAX_FILES]
    print(f"🗂  发现 .txt：{len(file_list)} 个")
    local_paths = [ _download_to_cache(f["download_url"], CACHE_DIR, filename=f["name"]) for f in file_list ]

    # 3) 读取楼层与 GeoJSON
    floor = _read_json(local_floor); map_w = float(floor["map_info"]["width"]); map_h = float(floor["map_info"]["height"])
    gj    = _read_json(local_geojson); gj_s = _transform_geojson(gj, map_h)

    # 4) 累积所有文件的 (x,y,B)
    xyB_all = []
    n_wp_used = n_mf_used = 0
    for p in local_paths:
        wp = _read_waypoints(p)
        mf = _read_magnetic(p)
        if wp.empty or mf.empty:
            continue
        xyB = interpolate_magnetic_to_xy(mf, wp)
        if xyB.empty:
            continue
        xyB_all.append(xyB)
        n_wp_used += len(wp); n_mf_used += len(mf)
    if not xyB_all:
        print("⚠️ 没拿到任何可插值的磁场样本（检查文件内是否同时含 WAYPOINT 与 MAGNETIC_FIELD）")
        return
    xyB = pd.concat(xyB_all, ignore_index=True)

    # 可选：限点数以提升前端性能
    if len(xyB) > HEAT_MAX_PTS:
        xyB = xyB.sample(HEAT_MAX_PTS, random_state=42).sort_values("ts")

    # 归一化权重（Robust：按分位裁剪后 0~1）
    p_lo, p_hi = np.percentile(xyB["B"], ROBUST_CLIP_P)
    denom = max(1e-6, p_hi - p_lo)
    xyB["w"] = (xyB["B"].clip(p_lo, p_hi) - p_lo) / denom

    # 5) Folium 画图
    m = folium.Map(location=[map_h/2, map_w/2], zoom_start=0, tiles=None, crs="Simple", control_scale=True)
    bounds = [[0,0],[map_h,map_w]]
    folium.raster_layers.ImageOverlay(name="Floor image", image=local_img, bounds=bounds, opacity=1.0, interactive=False).add_to(m)
    m.fit_bounds(bounds)

    # GeoJSON 轮廓
    palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
    tooltip_fields = []
    for ft in gj.get("features", []):
        for k in (ft.get("properties") or {}).keys():
            if k not in tooltip_fields: tooltip_fields.append(k)
            if len(tooltip_fields) >= 6: break
        if len(tooltip_fields) >= 6: break
    def style_fn(feat):
        props = feat.get("properties") or {}
        color = "#000000"
        if STYLE_FIELD and STYLE_FIELD in props:
            color = palette[hash(str(props[STYLE_FIELD])) % len(palette)]
        gtype = (feat.get("geometry") or {}).get("type","")
        if gtype in ("Polygon","MultiPolygon"):
            return {"fillOpacity":0.0, "color":color, "weight":1.2}
        return {"color":color, "weight":2.0}
    folium.GeoJson(gj_s, name="GeoJSON features", style_function=style_fn,
                   tooltip=GeoJsonTooltip(fields=tooltip_fields) if tooltip_fields else None).add_to(m)

    # 热力图图层
    heat_points = []
    for _, r in xyB.iterrows():
        lon, lat = _xy_to_leaflet(r["x"], r["y"], map_h)   # (lon,lat)
        heat_points.append([lat, lon, float(r["w"])])      # HeatMap 需要 [lat,lon,weight]
    HeatMap(
        heat_points, name="Geomagnetic heat", radius=HEAT_RADIUS,
        blur=HEAT_RADIUS*2, max_zoom=18, control=True
    ).add_to(m)

    # 可选：叠加 waypoint 折线作参考
    if ADD_WAYPOINTS:
        for idx, p in enumerate(local_paths, 1):
            wp = _read_waypoints(p)
            if wp.empty: continue
            coords = [[_xy_to_leaflet(x,y,map_h)[1], _xy_to_leaflet(x,y,map_h)[0]] for x,y in zip(wp["x"], wp["y"])]
            name = os.path.basename(p)
            layer = folium.FeatureGroup(name=f"GT: {name}", show=False)
            if len(coords) >= 2:
                folium.PolyLine(coords, weight=2, opacity=0.7, color=palette[idx % len(palette)]).add_to(layer)
            layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(OUT_HTML)
    print(f"\n🎉 生成完成：{OUT_HTML}")
    print(f"   参与插值的文件：{len(local_paths)} 个，样本点：{len(xyB)}，wp数≈{n_wp_used}，mag数≈{n_mf_used}")

if __name__ == "__main__":
    main()
