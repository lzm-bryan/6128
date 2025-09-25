#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re, io, json, time, shutil, zipfile
import numpy as np
import pandas as pd
from urllib.parse import urlparse, unquote
from urllib.request import urlopen, Request
import folium
from folium.features import GeoJsonTooltip
from folium.plugins import HeatMap

# ================= 配置：多站点多楼层 =================
REPO_BASE = "https://github.com/location-competition/indoor-location-competition-20/blob/master/data"

# 楼层清单：site1: B1 + F1~F4；site2: B1 + F1~F8
FLOOR_SETS = {
    "site1": ["B1", "F1", "F2", "F3", "F4"],
    "site2": ["B1", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"],
}

CACHE_ROOT = "indoor_cache"
OUT_HTML   = "multi_floor_gt_heat.html"

# 过滤与性能
NAME_FILTER             = ""      # 只处理包含该子串的 txt（留空=不过滤）
MAX_FILES_PER_FLOOR     = 0       # 0=该层全部；>0 仅取前 N 个
SNAP_WAYPOINT_METERS    = 0.05    # 轨迹 5cm 栅格去重（=0 关闭）
DRAW_POINT_SAMPLE_EVERY = 0       # >0 抽样画点；0=不画点（仅折线）

STYLE_FIELD             = ""      # GeoJSON properties 分类字段（填了就按字段自动配色）
VERBOSE                 = True    # 打印每个文件的统计

# —— 地磁热力图参数 ——（与单层脚本对齐）
ENABLE_HEATMAP          = True
PREFER_UNCALIBRATED     = False   # True=优先 UNCAL；默认优先 CALIBRATED
ACC_FILTER              = 0       # 仅保留磁场样本 accuracy>=此阈值（0/1/2/3；0=不过滤）
MAX_MAG_POINTS          = 20000   # 每层热力点总量上限（超过将随机抽样）
ROBUST_CLIP_P           = (5, 95) # 权重归一化的分位裁剪（抗异常值）
NEAREST_TOL_MS          = 0       # 严格插值失败时，最近路标兜底阈值（毫秒）；0=关闭
HEAT_RADIUS             = 6
HEAT_BLUR               = 15
HEAT_MIN_OPACITY        = 0.4

# ================= 基础工具 =================
def _is_github_blob(url: str) -> bool:
    return "github.com" in url and "/blob/" in url

def _to_raw(url: str) -> str:
    return url.replace("https://github.com/","https://raw.githubusercontent.com/").replace("/blob/","/") if _is_github_blob(url) else url

def _req(url: str) -> bytes:
    headers = {
        "User-Agent":"Mozilla/5.0",
        "Accept":"application/vnd.github+json" if "api.github.com" in url else "*/*"
    }
    # 支持 GitHub Token，显著降低 API 限速（GITHUB_TOKEN 或 GH_TOKEN）
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if "api.github.com" in url and token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, headers=headers)
    with urlopen(req, timeout=60) as r:
        return r.read()

def _file_url_to_path(url: str) -> str:
    """file:///C:/a/b.txt -> C:\\a\\b.txt（兼容 Windows）"""
    p = urlparse(url)
    path = unquote(p.path)
    if os.name == "nt":
        # windows: /C:/path → C:\path
        if path.startswith("/") and len(path) >= 4 and path[2] == ":":
            path = path[1:]
        path = path.replace("/", "\\")
    return path

def _download_to_cache(url: str, cache_dir: str, filename: str = None, force: bool = False) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    raw_url = _to_raw(url)
    if filename is None:
        filename = os.path.basename(urlparse(raw_url).path) or f"file_{int(time.time())}"
    path = os.path.join(cache_dir, filename)
    if (not force) and os.path.exists(path) and os.path.getsize(path) > 0:
        if VERBOSE: print(f"↩️  缓存命中：{path}")
        return path

    # 支持 file:// 直接复制（仓库 ZIP 快照兜底）
    if raw_url.startswith("file://"):
        src = _file_url_to_path(raw_url)
        shutil.copyfile(src, path)
        if VERBOSE: print(f"📄  复制本地：{src} → {path}")
        return path

    print(f"⬇️  下载：{raw_url}")
    data = _req(raw_url)
    with open(path, "wb") as f: f.write(data)
    print(f"✅ 完成：{path}（{len(data)/1024:.1f} KB）")
    return path

def _parse_repo_and_path(tree_url: str):
    p = urlparse(tree_url)
    parts = [x for x in p.path.strip("/").split("/") if x]
    assert len(parts) >= 2, "非法 GitHub 链接"
    owner, repo = parts[0], parts[1]
    branch = "master"
    if len(parts) >= 4 and parts[2] == "tree":
        branch = parts[3]
        subpath = "/".join(parts[4:])
    else:
        subpath = "/".join(parts[2:])
    return owner, repo, branch, subpath

# ---- 仓库 ZIP 快照兜底（不受 API 限速影响） ----
def _get_repo_snapshot_root(owner: str, repo: str, branch: str) -> str:
    """下载并解压仓库 ZIP 到缓存目录，返回解压后的根目录（通常是 repo-branch）"""
    base = os.path.join(CACHE_ROOT, "_repo_snapshots", f"{owner}_{repo}_{branch}")
    os.makedirs(base, exist_ok=True)
    marker = os.path.join(base, ".extracted")
    if not os.path.exists(marker):
        url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{branch}"
        print(f"📦  下载仓库快照：{url}")
        data = _req(url)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(base)
        # 写标记文件
        with open(marker, "w", encoding="utf-8") as f:
            f.write(str(time.time()))
        print(f"✅ 快照解压完成：{base}")
    # 选择解压后的根目录（通常只有一个子目录）
    candidates = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)) and not d.startswith(".")]
    if not candidates:
        raise RuntimeError("仓库快照解压失败：未找到根目录")
    # 优先 repo-branch 命名
    pref = f"{repo}-{branch}"
    for c in candidates:
        if os.path.basename(c) == pref:
            return c
    return candidates[0]

def _list_txt_in_snapshot(owner: str, repo: str, branch: str, subpath: str):
    """从本地仓库快照中列举 subpath 下的 .txt 文件"""
    root = _get_repo_snapshot_root(owner, repo, branch)
    target_dir = os.path.join(root, subpath.replace("/", os.sep))
    if not os.path.isdir(target_dir):
        return []
    out = []
    for name in sorted(os.listdir(target_dir)):
        if name.lower().endswith(".txt"):
            local_path = os.path.join(target_dir, name)
            out.append({"name": name, "download_url": f"file:///{local_path}" if os.name == "nt" else f"file://{local_path}"})
    return out

def _list_txt_in_github_dir(tree_url: str):
    """
    目录列举顺序：
      1) Contents API: /repos/{owner}/{repo}/contents/{subpath}?ref={branch}
      2) Git Tree API (recursive): /repos/{owner}/{repo}/git/trees/{branch}?recursive=1
      3) HTML 兜底：解析 /blob/... .txt 的链接
      4) ZIP 快照兜底：下载仓库快照并从本地列举
    任一步拿到结果即返回。
    """
    owner, repo, branch, subpath = _parse_repo_and_path(tree_url)

    # 1) Contents API
    api = f"https://api.github.com/repos/{owner}/{repo}/contents/{subpath}?ref={branch}"
    try:
        data = json.loads(_req(api).decode("utf-8", "ignore"))
        files = []
        for it in data:
            if it.get("type") == "file" and it.get("name","").lower().endswith(".txt"):
                dl = it.get("download_url") or f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{it['path']}"
                files.append({"name": it["name"], "download_url": dl})
        if files:
            return files
        else:
            print("ℹ️ Contents API 返回空目录。")
    except Exception as e:
        print(f"⚠️ API 失败：{e}；尝试 Git Tree API。")

    # 2) Git Tree API（全树递归，再按前缀过滤）
    tree_api = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    try:
        data = json.loads(_req(tree_api).decode("utf-8", "ignore"))
        tree = data.get("tree", []) or []
        prefix = subpath.rstrip("/") + "/"
        files = []
        for it in tree:
            if it.get("type") == "blob":
                path = it.get("path","")
                if path.startswith(prefix) and path.lower().endswith(".txt"):
                    files.append({
                        "name": os.path.basename(path),
                        "download_url": f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
                    })
        if files:
            return files
        else:
            print("ℹ️ Git Tree API 未找到 .txt。")
    except Exception as e:
        print(f"⚠️ Git Tree API 失败：{e}；改用 HTML 解析。")

    # 3) HTML 兜底：抓取 /blob/... .txt（兼容 ?plain=1）
    try:
        html = _req(tree_url).decode("utf-8","ignore")
        # 捕获 /owner/repo/blob/branch/.../*.txt 或完整 https 链接（含 ?plain=1 等）
        hrefs = re.findall(r'href="((?:https://github\.com)?/[^"]+/blob/[^"]+?\.txt(?:\?[^"]*)?)"', html)
        files = []
        seen = set()
        for h in hrefs:
            if not h.startswith("http"):
                h = f"https://github.com{h}"
            name = os.path.basename(urlparse(h).path)
            if name.lower().endswith(".txt") and (name, h) not in seen:
                files.append({"name": name, "download_url": _to_raw(h)})
                seen.add((name, h))
        if files:
            return files
        else:
            print("ℹ️ HTML 兜底未匹配到 .txt，转用 ZIP 快照。")
    except Exception as e:
        print(f"⚠️ HTML 解析失败：{e}；转用 ZIP 快照。")

    # 4) ZIP 快照兜底
    try:
        files = _list_txt_in_snapshot(owner, repo, branch, subpath)
        if files:
            print(f"📦  来自本地仓库快照：找到 {len(files)} 个 .txt")
        return files
    except Exception as e:
        print(f"❌ ZIP 快照兜底失败：{e}")
        return []

def _read_json(path: str):
    with open(path,"r",encoding="utf-8") as f:
        return json.load(f)

# ================= 解析：WAYPOINT & MAG =================
def _read_waypoints(path: str) -> pd.DataFrame:
    """仅解析 WAYPOINT 行：ts TYPE_WAYPOINT x y"""
    rows = []
    with open(path,"r",encoding="utf-8",errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): continue
            p = s.replace(",", " ").split()
            if len(p) >= 4 and p[1] == "TYPE_WAYPOINT":
                try:
                    ts = int(p[0]); x=float(p[2]); y=float(p[3])
                    rows.append((ts,x,y))
                except Exception:
                    pass
    if not rows:
        return pd.DataFrame(columns=["ts","x","y"])
    df = pd.DataFrame(rows, columns=["ts","x","y"]).sort_values("ts").reset_index(drop=True)
    if SNAP_WAYPOINT_METERS and SNAP_WAYPOINT_METERS > 0:
        df["_ix"] = (df["x"]/SNAP_WAYPOINT_METERS).round().astype(int)
        df["_iy"] = (df["y"]/SNAP_WAYPOINT_METERS).round().astype(int)
        df = df.drop_duplicates(["_ix","_iy"]).drop(columns=["_ix","_iy"])
    return df

def _read_magnetometer(path: str) -> pd.DataFrame:
    """
    优先 CALIBRATED；无则回退 UNCALIBRATED。
    支持 ACC_FILTER（0/1/2/3；0=不过滤）。
    返回列：ts,mx,my,mz,acc
    """
    rec_cal, rec_uncal = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            p = s.replace(",", " ").split()
            if len(p) < 5:
                continue
            typ = p[1]
            try:
                ts = int(p[0])
            except Exception:
                continue
            if typ == "TYPE_MAGNETIC_FIELD":
                try:
                    mx, my, mz = float(p[2]), float(p[3]), float(p[4])
                    acc = int(p[5]) if len(p) >= 6 and p[5].isdigit() else 0
                    rec_cal.append((ts, mx, my, mz, acc))
                except Exception:
                    pass
            elif typ == "TYPE_MAGNETIC_FIELD_UNCALIBRATED":
                try:
                    mx, my, mz = float(p[2]), float(p[3]), float(p[4])
                    acc = int(p[-1]) if p[-1].isdigit() else 0
                    rec_uncal.append((ts, mx, my, mz, acc))
                except Exception:
                    pass

    df_cal   = pd.DataFrame(rec_cal,   columns=["ts","mx","my","mz","acc"]).sort_values("ts").reset_index(drop=True)
    df_uncal = pd.DataFrame(rec_uncal, columns=["ts","mx","my","mz","acc"]).sort_values("ts").reset_index(drop=True)

    if PREFER_UNCALIBRATED and not df_uncal.empty:
        df = df_uncal
    elif (not PREFER_UNCALIBRATED) and not df_cal.empty:
        df = df_cal
    else:
        df = df_uncal if not df_uncal.empty else df_cal

    if df is None or df.empty:
        return pd.DataFrame(columns=["ts","mx","my","mz","acc"])

    if ACC_FILTER and ACC_FILTER > 0:
        df = df[df["acc"] >= ACC_FILTER].copy()

    return df

# ================= 坐标变换 & GeoJSON =================
def _xy_to_leaflet(x, y, h):  # meters -> CRS.Simple (lon=x, lat=h-y)
    return [x, h - y]

def _transform_geojson(gj: dict, map_h: float):
    def _tx(coords):
        if isinstance(coords[0], (int,float)):
            x,y = coords[:2]; lon, lat = _xy_to_leaflet(x,y,map_h)
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

# ================= Tooltip 字段推断（过滤 style 等非法字段） =================
def _infer_tooltip_fields(gj: dict, max_fields: int = 6) -> list:
    feats = gj.get("features") or []
    if not feats:
        return []
    blacklist = {"style","styles","stroke","fill","stroke-width","stroke-opacity","fill-opacity"}
    def scalar_keys(props: dict):
        out = set()
        for k, v in (props or {}).items():
            if k in blacklist:
                continue
            if isinstance(v, (str, int, float, bool)) or v is None:
                out.add(k)
        return out
    common = None
    for ft in feats:
        keys = scalar_keys(ft.get("properties") or {})
        common = keys if common is None else (common & keys)
    if not common:
        return []
    return sorted(common)[:max_fields]

# —— 磁场样本时间插值到 (x,y) ——（单层脚本原实现）
def interpolate_magnetic_to_xy(mf: pd.DataFrame, wp: pd.DataFrame) -> pd.DataFrame:
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
    z = z[(~z["ts_prev"].isna()) & (~z["ts_next"].isna()) & (z["ts_next"] > z["ts_prev"])]

    t = (z["ts"] - z["ts_prev"]) / (z["ts_next"] - z["ts_prev"])
    z["x"] = z["x_prev"] + t * (z["x_next"] - z["x_prev"])
    z["y"] = z["y_prev"] + t * (z["y_next"] - z["y_prev"])
    return z[["ts","x","y","B"]].reset_index(drop=True)

# ================= 主流程 =================
def main():
    # 统一的地图（CRS.Simple）
    m = folium.Map(location=[0,0], zoom_start=0, tiles=None, crs="Simple", control_scale=True)

    palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
               "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]

    # 记录每层的 JS 变量名与边界，供控件使用
    floor_entries = []
    default_floor_key = None
    default_bounds = None

    for site, floors in FLOOR_SETS.items():
        for floor_name in floors:
            key = f"{site}-{floor_name}"
            if default_floor_key is None:
                default_floor_key = key

            cache_dir = os.path.join(CACHE_ROOT, site, floor_name)
            base_url  = f"{REPO_BASE}/{site}/{floor_name}"

            # 下载三件套
            geojson_path   = _download_to_cache(f"{base_url}/geojson_map.json", cache_dir)
            floorinfo_path = _download_to_cache(f"{base_url}/floor_info.json", cache_dir)
            img_path       = _download_to_cache(f"{base_url}/floor_image.png", cache_dir)

            # 尺寸/边界
            floorinfo = _read_json(floorinfo_path)
            map_w = float(floorinfo["map_info"]["width"])
            map_h = float(floorinfo["map_info"]["height"])
            bounds = [[0,0],[map_h,map_w]]
            if default_bounds is None:
                default_bounds = bounds

            # GeoJSON -> Simple CRS
            gj   = _read_json(geojson_path)
            gj_s = _transform_geojson(gj, map_h)

            # —— 楼层“BASE”组：底图 + GeoJSON ——（作为一个整体被显示/隐藏）
            base_group = folium.FeatureGroup(name=f"{key} | BASE", show=False)

            folium.raster_layers.ImageOverlay(
                image=img_path, bounds=bounds, opacity=1.0, interactive=False, name=f"{key} floor"
            ).add_to(base_group)

            tooltip_fields = _infer_tooltip_fields(gj, max_fields=6)

            def style_fn(feat):
                props = feat.get("properties") or {}
                color = "#000000"
                if STYLE_FIELD and STYLE_FIELD in props:
                    color = palette[hash(str(props[STYLE_FIELD])) % len(palette)]
                gtype = (feat.get("geometry") or {}).get("type","")
                if gtype in ("Polygon","MultiPolygon"):
                    return {"fillOpacity":0.0, "color":color, "weight":1.2}
                return {"color":color, "weight":2.0}

            folium.GeoJson(
                gj_s, name=f"{key} geojson",
                style_function=style_fn,
                tooltip=GeoJsonTooltip(fields=tooltip_fields) if tooltip_fields else None
            ).add_to(base_group)
            base_group.add_to(m)

            # —— 楼层“GT”组：该层所有文件汇总的一组轨迹折线 ——（便于总控/本层控）
            gt_group = folium.FeatureGroup(name=f"{key} | GT", show=False)

            # 列出并下载该层所有 txt（多级兜底，最终可走 ZIP 快照）
            txt_dir_url = f"https://github.com/location-competition/indoor-location-competition-20/tree/master/data/{site}/{floor_name}/path_data_files"
            files = _list_txt_in_github_dir(txt_dir_url)
            files = [f for f in files if NAME_FILTER in f["name"]]
            files.sort(key=lambda x: x["name"])
            if MAX_FILES_PER_FLOOR and MAX_FILES_PER_FLOOR > 0:
                files = files[:MAX_FILES_PER_FLOOR]
            print(f"🗂  {key} 发现 txt：{len(files)} 个")

            # 汇总绘制 GT；并收集热力点的原始 (x,y,B)（合并后统一归一化）
            color_i = 0
            total_pts = 0
            xyB_all = []
            n_wp_used = n_mf_used = 0

            for it in files:
                try:
                    local_txt = _download_to_cache(it["download_url"], cache_dir, filename=it["name"])
                except Exception as e:
                    if VERBOSE: print(f"      · 下载失败 {it['name']}: {e}")
                    continue

                wp = _read_waypoints(local_txt)
                mg = _read_magnetometer(local_txt)

                # 画 GT 折线
                if not wp.empty:
                    coords = [[ _xy_to_leaflet(x,y,map_h)[1], _xy_to_leaflet(x,y,map_h)[0] ] for x,y in zip(wp["x"], wp["y"])]
                    if len(coords) >= 2:
                        folium.PolyLine(coords, weight=3, opacity=0.9, color=palette[color_i % len(palette)],
                                        tooltip=it["name"]).add_to(gt_group)
                    if DRAW_POINT_SAMPLE_EVERY and DRAW_POINT_SAMPLE_EVERY > 0:
                        for lat, lon in coords[::DRAW_POINT_SAMPLE_EVERY]:
                            folium.CircleMarker([lat,lon], radius=2, weight=1, fill=True, fill_opacity=0.9,
                                                color="#333333").add_to(gt_group)
                    color_i += 1
                    total_pts += len(coords)

                # 收集 (x,y,B)
                if wp.empty or mg.empty:
                    if VERBOSE: print(f"      · {it['name']} 无法插值（waypoints={len(wp)}, mag={len(mg)})")
                    continue

                mf = mg.copy()
                mf["B"] = np.sqrt(mf["mx"]**2 + mf["my"]**2 + mf["mz"]**2)
                xyB = interpolate_magnetic_to_xy(mf[["ts","B"]], wp[["ts","x","y"]])

                # 兜底：最近路标（仅当严格插值没有一个点时）
                if xyB.empty and NEAREST_TOL_MS and NEAREST_TOL_MS > 0:
                    near = pd.merge_asof(
                        mf[["ts","B"]].sort_values("ts"),
                        wp.sort_values("ts"),
                        on="ts", direction="nearest", tolerance=NEAREST_TOL_MS
                    ).dropna(subset=["x","y","B"])
                    if not near.empty:
                        xyB = near[["ts","x","y","B"]].reset_index(drop=True)
                        if VERBOSE:
                            print(f"        ↳ 回退(最近路标±{NEAREST_TOL_MS}ms)：{len(xyB)} 点")

                if not xyB.empty:
                    xyB_all.append(xyB)
                    n_wp_used += len(wp); n_mf_used += len(mg)
                elif VERBOSE:
                    print(f"      · {it['name']} 没有得到可用的 (x,y,B) 点")

            gt_group.add_to(m)

            # —— 楼层“HEAT”组 ——（合并后统一做鲁棒归一化 → 权重 0~1）
            heat_group = folium.FeatureGroup(name=f"{key} | Heat", show=False)
            if ENABLE_HEATMAP and len(xyB_all) > 0:
                xyB = pd.concat(xyB_all, ignore_index=True)

                # 权重：鲁棒分位
                Bvals = xyB["B"].to_numpy(dtype=float)
                plo, phi = np.percentile(Bvals, ROBUST_CLIP_P)
                if not np.isfinite(plo) or not np.isfinite(phi) or (phi - plo) <= 1e-9:
                    weights = np.ones_like(Bvals, dtype=float)  # 防退化：全部置 1
                else:
                    weights = np.clip((Bvals - plo) / (phi - plo), 0.0, 1.0)
                xyB = xyB.assign(w=weights)

                # 限量（随机抽样更稳，避免时间步进偏置）
                if len(xyB) > MAX_MAG_POINTS:
                    xyB = xyB.sample(MAX_MAG_POINTS, random_state=42)

                # 转成 HeatMap 需要的 [lat, lon, weight]
                heat_points = []
                for _, r in xyB.iterrows():
                    lon, lat = _xy_to_leaflet(float(r["x"]), float(r["y"]), map_h)
                    heat_points.append([lat, lon, float(r["w"])])

                HeatMap(
                    heat_points, radius=HEAT_RADIUS, blur=HEAT_BLUR,
                    min_opacity=HEAT_MIN_OPACITY, max_zoom=18
                ).add_to(heat_group)

                print(f"   ↳ {key} 参与插值的文件：{len(files)}，样本点：{len(xyB)}，wp≈{n_wp_used}，mag≈{n_mf_used}")
            else:
                print(f"⚠️  {key} 无可用热力点")

            heat_group.add_to(m)

            # 收集 JS 引用
            floor_entries.append({
                "key": key,
                "base_var": base_group.get_name(),
                "gt_var": gt_group.get_name(),
                "heat_var": heat_group.get_name(),
                "bounds_js": f"[[0,0],[{map_h},{map_w}]]",
            })

    # 初始视野更稳：Python 端先适配第一层边界
    if default_bounds is not None:
        try:
            m.fit_bounds(default_bounds)
        except Exception:
            pass

    folium.LayerControl(collapsed=False).add_to(m)

    # ================= 自定义控件（楼层切换 + GT/Heat 本层开关 + 全楼总控） =================
    js_lines = ["var floorGroups = {};"]
    for ent in floor_entries:
        js_lines.append(
            'floorGroups["{k}"] = {{base:{b}, gt:{g}, heat:{h}, bounds:{bd}}};'.format(
                k=ent["key"], b=ent["base_var"], g=ent["gt_var"], h=ent["heat_var"], bd=ent["bounds_js"]
            )
        )
    js_floor_groups = "\n".join(js_lines)
    first_key = (floor_entries[0]["key"] if floor_entries else "")

    floor_options_html = "".join([f'<option value="{ent["key"]}">{ent["key"]}</option>' for ent in floor_entries])

    ctrl_tpl = r"""
    %%GROUPS%%
    (function() {
        var map = %%MAP%%;
        // 控件 UI
        var ctrl = L.control({position:'topright'});
        ctrl.onAdd = function() {
            var div = L.DomUtil.create('div', 'leaflet-bar');
            div.style.background = 'white';
            div.style.padding = '8px';
            div.style.lineHeight = '1.4';
            div.style.userSelect = 'none';
            div.innerHTML = `
                <div style="font-weight:600;margin-bottom:6px;">Floor / Layers</div>
                <div style="margin-bottom:6px;">
                  <label>楼层：</label>
                  <select id="floorSel" style="max-width:200px;">
                    %%OPTIONS%%
                  </select>
                </div>
                <div style="margin-bottom:6px;">
                  <label><input type="checkbox" id="chkFloorGT" checked /> 本层 GT</label>
                  &nbsp;&nbsp;
                  <label><input type="checkbox" id="chkFloorHeat" checked /> 本层 Heat</label>
                </div>
                <div style="display:flex;gap:6px;align-items:center;">
                  <button id="btnToggleAllGT" class="leaflet-control-zoom-in" title="切换所有楼层 GT">GT 总控</button>
                  <span id="lblAllGT" style="margin-left:4px;">（全楼：隐藏）</span>
                </div>
                <div style="display:flex;gap:6px;align-items:center;margin-top:6px;">
                  <button id="btnToggleAllHeat" class="leaflet-control-zoom-in" title="切换所有楼层 Heat">Heat 总控</button>
                  <span id="lblAllHeat" style="margin-left:4px;">（全楼：隐藏）</span>
                </div>
            `;
            L.DomEvent.disableClickPropagation(div);
            return div;
        };
        ctrl.addTo(map);

        var allGTOn = false;
        var allHeatOn = false;
        var currentFloor = "%%FIRST%%";

        function hideAllBases() {
            for (var k in floorGroups) {
                if (map.hasLayer(floorGroups[k].base)) map.removeLayer(floorGroups[k].base);
            }
        }

        function updatePerFloorCheckboxes() {
            var g = floorGroups[currentFloor];
            var chkGT = document.getElementById('chkFloorGT');
            var chkHeat = document.getElementById('chkFloorHeat');
            if (chkGT) chkGT.checked = map.hasLayer(g.gt);
            if (chkHeat) chkHeat.checked = map.hasLayer(g.heat);
        }

        function showFloor(key) {
            currentFloor = key;
            hideAllBases();
            var g = floorGroups[key];
            map.addLayer(g.base);

            var chkGT = document.getElementById('chkFloorGT');
            var chkHeat = document.getElementById('chkFloorHeat');
            if (chkGT && chkGT.checked) map.addLayer(g.gt); else if (map.hasLayer(g.gt)) map.removeLayer(g.gt);
            if (chkHeat && chkHeat.checked) map.addLayer(g.heat); else if (map.hasLayer(g.heat)) map.removeLayer(g.heat);

            try { map.fitBounds(g.bounds); } catch(e) {}
        }

        function setAllGT(on) {
            allGTOn = on;
            for (var k in floorGroups) {
                if (on) map.addLayer(floorGroups[k].gt);
                else if (map.hasLayer(floorGroups[k].gt)) map.removeLayer(floorGroups[k].gt);
            }
            var lbl = document.getElementById('lblAllGT');
            if (lbl) lbl.textContent = '（全楼：' + (on ? '显示' : '隐藏') + '）';
            updatePerFloorCheckboxes();
        }

        function setAllHeat(on) {
            allHeatOn = on;
            for (var k in floorGroups) {
                if (on) map.addLayer(floorGroups[k].heat);
                else if (map.hasLayer(floorGroups[k].heat)) map.removeLayer(floorGroups[k].heat);
            }
            var lbl = document.getElementById('lblAllHeat');
            if (lbl) lbl.textContent = '（全楼：' + (on ? '显示' : '隐藏') + '）';
            updatePerFloorCheckboxes();
        }

        document.getElementById('floorSel').addEventListener('change', function() {
            showFloor(this.value);
        });
        document.getElementById('chkFloorGT').addEventListener('change', function() {
            var g = floorGroups[currentFloor];
            if (this.checked) map.addLayer(g.gt); else if (map.hasLayer(g.gt)) map.removeLayer(g.gt);
        });
        document.getElementById('chkFloorHeat').addEventListener('change', function() {
            var g = floorGroups[currentFloor];
            if (this.checked) map.addLayer(g.heat); else if (map.hasLayer(g.heat)) map.removeLayer(g.heat);
        });
        document.getElementById('btnToggleAllGT').addEventListener('click', function() {
            setAllGT(!allGTOn);
        });
        document.getElementById('btnToggleAllHeat').addEventListener('click', function() {
            setAllHeat(!allHeatOn);
        });

        setTimeout(function() {
            var sel = document.getElementById('floorSel');
            if (sel) sel.value = "%%FIRST%%";
            if ("%%FIRST%%") showFloor("%%FIRST%%");
        }, 50);
    })();
    """

    ctrl_html = (ctrl_tpl
                 .replace("%%GROUPS%%", js_floor_groups)
                 .replace("%%MAP%%", m.get_name())
                 .replace("%%FIRST%%", first_key)
                 .replace("%%OPTIONS%%", floor_options_html))

    folium.Element(f"<script>{ctrl_html}</script>").add_to(m)

    m.save(OUT_HTML)
    print(f"\n🎉 生成完成：{OUT_HTML}")

if __name__ == "__main__":
    main()
