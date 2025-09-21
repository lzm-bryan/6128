#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io, os, re, json, time
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from urllib.request import urlopen, Request
import folium
from folium.features import GeoJsonTooltip
from folium.plugins import MarkerCluster

# ========== 配置（按需修改） ==========
GEOJSON_URL   = "https://github.com/location-competition/indoor-location-competition-20/blob/master/data/site1/B1/geojson_map.json"
FLOOR_INFO_URL= "https://github.com/location-competition/indoor-location-competition-20/blob/master/data/site1/B1/floor_info.json"
FLOOR_IMG_URL = "https://github.com/location-competition/indoor-location-competition-20/blob/master/data/site1/B1/floor_image.png"

# 指向 .txt 轨迹文件所在“目录”的 GitHub 链接（/tree/）
GT_FOLDER_URL = "https://github.com/location-competition/indoor-location-competition-20/tree/master/data/site1/B1/path_data_files"

CACHE_DIR     = os.path.join("indoor_cache", "site1", "B1")
OUT_HTML      = "b1_waypoints_all.html"
STYLE_FIELD   = ""      # 若 GeoJSON properties 有分类字段（如 "type"/"feature_type"），填字段名可自动配色
MAX_FILES     = 0       # 0 表示全下；>0 则只取前 N 个
NAME_FILTER   = ""      # 只下载文件名包含该子串的 .txt（留空=不过滤）
SNAP_METERS   = 0.05    # 轨迹点 5cm 栅格去重（=0 关闭去重）

# ========== 工具函数 ==========
def _to_raw(url: str) -> str:
    return url.replace("https://github.com/","https://raw.githubusercontent.com/").replace("/blob/","/") if "github.com" in url and "/blob/" in url else url

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
    """
    优先用 GitHub API 列目录；失败再用 HTML 兜底解析
    返回 (owner, repo, branch, path) 与文件清单 [{'name':..., 'download_url':...}, ...]
    """
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

    # HTML 兜底：在页面里找 .txt 链接
    html = _req(url).decode("utf-8","ignore")
    hrefs = re.findall(r'href="([^"]+\.txt)"', html)
    files = []
    for h in hrefs:
        if not h.startswith("http"):
            h = f"https://github.com{h}"
        if "/blob/" not in h:
            continue
        name = os.path.basename(urlparse(h).path)
        files.append({"name": name, "download_url": _to_raw(h)})
    if not files:
        raise RuntimeError("未能从目录页解析出 .txt 文件链接")
    return (owner, repo, branch, subpath), files

def _read_json(path: str):
    with open(path,"r",encoding="utf-8") as f:
        return json.load(f)

def _read_waypoints(path: str) -> pd.DataFrame:
    """
    仅解析：<ts> TYPE_WAYPOINT <x> <y> （米）
    其它类型行（加速度/磁场/WiFi/Beacon）忽略
    """
    with open(path,"r",encoding="utf-8",errors="ignore") as f:
        raw = f.read()
    rows = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        s = s.replace(",", " ")
        p = s.split()
        if len(p) >= 4 and p[1] == "TYPE_WAYPOINT":
            try:
                ts = int(p[0])
            except Exception:
                ts = None
            try:
                x = float(p[2]); y = float(p[3])
                rows.append((ts,x,y))
            except Exception:
                continue
    if not rows:
        raise ValueError(f"[{path}] 未找到 TYPE_WAYPOINT 行")
    df = pd.DataFrame(rows, columns=["ts","x","y"]).sort_values("ts", kind="mergesort").reset_index(drop=True)
    if SNAP_METERS and SNAP_METERS > 0:
        df["_ix"] = (df["x"]/SNAP_METERS).round().astype(int)
        df["_iy"] = (df["y"]/SNAP_METERS).round().astype(int)
        df = df.drop_duplicates(["_ix","_iy"]).drop(columns=["_ix","_iy"])
    return df

def _xy_to_leaflet(x,y,h):  # meters -> CRS.Simple (lon=x, lat=h-y)
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
            if not g:
                continue
            f2 = dict(ft)
            f2["geometry"] = {"type": g.get("type"), "coordinates": _tx(g.get("coordinates"))}
            out["features"].append(f2)
        return out
    return {"type": gj.get("type","GeometryCollection"), "coordinates": _tx(gj.get("coordinates",[]))}

# ========== 主流程 ==========
def main():
    print(f"📁 缓存目录：{CACHE_DIR}")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 1) 下载基底三件套
    local_geojson = _download_to_cache(GEOJSON_URL, CACHE_DIR)
    local_floor  = _download_to_cache(FLOOR_INFO_URL, CACHE_DIR)
    local_img    = _download_to_cache(FLOOR_IMG_URL, CACHE_DIR)

    # 2) 列出并下载目录里的 .txt
    (owner, repo, branch, subpath), file_list = _parse_github_dir_api(GT_FOLDER_URL)
    file_list = [f for f in file_list if (NAME_FILTER in f["name"])]
    file_list.sort(key=lambda x: x["name"])
    if MAX_FILES and MAX_FILES > 0:
        file_list = file_list[:MAX_FILES]
    print(f"🗂  发现 .txt：{len(file_list)} 个")
    local_gt = []
    for f in file_list:
        local_gt.append(_download_to_cache(f["download_url"], CACHE_DIR, filename=f["name"]))

    # 3) 读取 meta、变换坐标
    floor = _read_json(local_floor); map_w = float(floor["map_info"]["width"]); map_h = float(floor["map_info"]["height"])
    gj = _read_json(local_geojson); gj_s = _transform_geojson(gj, map_h)

    # 4) 画 Folium 图
    m = folium.Map(location=[map_h/2, map_w/2], zoom_start=0, tiles=None, crs="Simple", control_scale=True)
    bounds = [[0,0],[map_h,map_w]]
    folium.raster_layers.ImageOverlay(name="Floor image", image=local_img, bounds=bounds, opacity=1.0, interactive=False).add_to(m)
    m.fit_bounds(bounds)

    # GeoJSON 要素
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

    # 轨迹分图层绘制
    total_pts = 0
    for idx, path in enumerate(local_gt, 1):
        print(idx)
        print(path)
        try:
            df = _read_waypoints(path)
        except Exception as e:
            print(f"⚠️ 跳过 {os.path.basename(path)}：{e}")
            continue

        color = palette[idx % len(palette)]
        coords = [[_xy_to_leaflet(x,y,map_h)[1], _xy_to_leaflet(x,y,map_h)[0]] for x,y in zip(df["x"], df["y"])]  # [lat,lon]
        name = os.path.basename(path)
        layer = folium.FeatureGroup(name=f"GT: {name}", show=(idx<=5))  # 前5个默认展开
        if len(coords) >= 2:
            folium.PolyLine(coords, weight=3, opacity=0.9, color=color).add_to(layer)
        cluster = MarkerCluster(name=f"GT cluster: {name}", show=False)
        for i,(lat,lon) in enumerate(coords):
            pop = f"{name} • idx={i}"
            folium.CircleMarker([lat,lon], radius=2.0, weight=1, fill=True, fill_opacity=0.9, color=color, popup=pop).add_to(layer)
            # 如点太多，可只画折线不画 Marker
        layer.add_to(m)
        total_pts += len(coords)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(OUT_HTML)
    print(f"\n🎉 生成完成：{OUT_HTML}")
    print(f"   轨迹文件：{len(local_gt)} 个，合计点数（下采样后）：{total_pts}")
    print(f"   缓存目录：{CACHE_DIR}")

if __name__ == "__main__":
    main()
