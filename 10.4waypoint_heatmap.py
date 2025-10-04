#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re, io, json, time, shutil, zipfile, math
import numpy as np
import pandas as pd
from urllib.parse import urlparse, unquote
from urllib.request import urlopen, Request
import folium
from folium.features import GeoJsonTooltip
from folium.plugins import HeatMap

# ================= 配置：多站点多楼层 =================
REPO_BASE = "https://github.com/location-competition/indoor-location-competition-20/blob/master/data"

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
VERBOSE                 = True    # 打印统计

# —— 地磁热力图参数 ——
ENABLE_HEATMAP          = True
PREFER_UNCALIBRATED     = False   # True=优先 UNCAL；默认优先 CALIBRATED
ACC_FILTER              = 0       # 仅保留磁场样本 accuracy>=此阈值（0/1/2/3；0=不过滤）
MAX_MAG_POINTS          = 20000   # 每层热力点总量上限（超过将随机抽样）
ROBUST_CLIP_P           = (5, 95) # 权重归一化分位裁剪（抗异常）
NEAREST_TOL_MS          = 0       # 最近路标兜底阈值（毫秒）；0=关闭
HEAT_RADIUS             = 6
HEAT_BLUR               = 15
HEAT_MIN_OPACITY        = 0.4

# —— 仿射修正（像素 = A·(米) + t） ——
def _env_flag(name: str, default: bool=False) -> bool:
    v = os.getenv(name)
    if v is None: return default
    return str(v).strip().lower() in ("1","true","t","yes","y","on")

AFFINE_OVERRIDE_STR     = os.environ.get("INDOOR_AFFINE", "").strip()
X_FLIP_AFTER_AFFINE     = _env_flag("INDOOR_XFLIP", False)  # 新增：左右镜像
Y_FLIP_AFTER_AFFINE     = _env_flag("INDOOR_YFLIP", False)  # 仍可控制：上下镜像
FORCE_ISOTROPIC         = _env_flag("INDOOR_FORCE_ISO", False)  # 新增：强制等比去剪切

# —— 网络兜底策略 ——
DISABLE_SNAPSHOT_FALLBACK = True  # True=禁用 ZIP 仓库快照兜底；False=允许

# ================= 基础工具 =================
def _is_github_blob(url: str) -> bool:
    return isinstance(url, str) and ("github.com" in url) and ("/blob/" in url)

def _to_raw(url: str) -> str:
    if _is_github_blob(url):
        return url.replace("https://github.com/","https://raw.githubusercontent.com/").replace("/blob/","/")
    return url

def _req(url: str) -> bytes:
    headers = {
        "User-Agent":"Mozilla/5.0",
        "Accept":"application/vnd.github+json" if "api.github.com" in url else "*/*"
    }
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if "api.github.com" in url and token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, headers=headers)
    with urlopen(req, timeout=60) as r:
        return r.read()

def _file_url_to_path(url: str) -> str:
    p = urlparse(url)
    path = unquote(p.path)
    if os.name == "nt":
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

    if raw_url.startswith("file://"):
        src = _file_url_to_path(raw_url)
        shutil.copyfile(src, path)
        if VERBOSE: print(f"📄  复制本地：{src} → {path}")
        return path

    print(f"⬇️  下载：{raw_url}")
    data = _req(raw_url)
    with open(path, "wb") as f:
        f.write(data)
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

# ---- ZIP 快照兜底（是否启用由 DISABLE_SNAPSHOT_FALLBACK 控制） ----
def _get_repo_snapshot_root(owner: str, repo: str, branch: str) -> str:
    base = os.path.join(CACHE_ROOT, "_repo_snapshots", f"{owner}_{repo}_{branch}")
    os.makedirs(base, exist_ok=True)
    marker = os.path.join(base, ".extracted")
    if not os.path.exists(marker):
        url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{branch}"
        print(f"📦  下载仓库快照：{url}")
        data = _req(url)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(base)
        with open(marker, "w", encoding="utf-8") as f:
            f.write(str(time.time()))
        print(f"✅ 快照解压完成：{base}")
    candidates = [
        os.path.join(base, d) for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and not d.startswith(".")
    ]
    if not candidates:
        raise RuntimeError("仓库快照解压失败：未找到根目录")
    pref = f"{repo}-{branch}"
    for c in candidates:
        if os.path.basename(c) == pref:
            return c
    return candidates[0]

def _list_txt_in_snapshot(owner: str, repo: str, branch: str, subpath: str):
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
        print(f"⚠️ Contents API 失败：{e}；尝试 Git Tree API。")

    # 2) Git Tree API（全树递归）
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
        print(f"⚠️ Git Tree API 失败：{e}；尝试 HTML 解析。")

    # 3) HTML 兜底（只解析，不下快照）
    try:
        html = _req(tree_url).decode("utf-8","ignore")
        hrefs = re.findall(r'href=\"((?:https://github\.com)?/[^"]+/blob/[^"]+?\.txt(?:\?[^"]*)?)\"', html)
        files, seen = [], set()
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
            print("ℹ️ HTML 兜底未匹配到 .txt。")
    except Exception as e:
        print(f"⚠️ HTML 解析失败：{e}。")

    # 4) ZIP 快照兜底（按开关决定是否启用）
    if DISABLE_SNAPSHOT_FALLBACK:
        print("⏩ 已禁用 ZIP 仓库快照兜底；返回空列表。")
        return []
    else:
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
    rec_cal, rec_uncal = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): continue
            p = s.replace(",", " ").split()
            if len(p) < 5: continue
            typ = p[1]
            try:
                ts = int(p[0])
            except Exception:
                continue
            if typ == "TYPE_MAGNETIC_FIELD":
                try:
                    mx, my, mz = float(p[2]), float(p[3]), float(p[4])
                    try:
                        acc = int(float(p[5])) if len(p) >= 6 else 0
                    except Exception:
                        acc = 0
                    rec_cal.append((ts, mx, my, mz, acc))
                except Exception:
                    pass
            elif typ == "TYPE_MAGNETIC_FIELD_UNCALIBRATED":
                try:
                    mx, my, mz = float(p[2]), float(p[3]), float(p[4])
                    try:
                        acc = int(float(p[-1]))
                    except Exception:
                        acc = 0
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

# ================= 仿射 & GeoJSON =================
def _compose_affine(scale=(1,1), theta_deg=0.0, translate=(0,0)):
    sx, sy = scale
    th = math.radians(theta_deg)
    ct, st = math.cos(th), math.sin(th)
    a = ct*sx; b = -st*sy
    c = st*sx; d =  ct*sy
    e, f = translate
    return (a,b,c,d,e,f)

def _try_affine_from_floorinfo(fi_raw: dict):
    def norm6(v):
        if isinstance(v, list) and len(v)==6:
            return tuple(float(x) for x in v)
        if isinstance(v, list) and len(v)==2 and isinstance(v[0], list) and len(v[0])==3:
            a,b,e = v[0]; c,d,f = v[1]
            return (float(a),float(b),float(c),float(d),float(e),float(f))
        return None
    raw = fi_raw if isinstance(fi_raw, dict) else (fi_raw[0] if isinstance(fi_raw, list) and fi_raw else {})
    t = raw.get("transform") or {}
    if isinstance(t, dict) and {"a","b","c","d","e","f"}.issubset(t.keys()):
        return (float(t["a"]), float(t["b"]), float(t["c"]), float(t["d"]), float(t["e"]), float(t["f"]))
    for k in ("affine","matrix"):
        if k in t:
            v = norm6(t[k])
            if v: return v
    if "scale" in t and "translate" in t:
        sx, sy = t.get("scale",[1,1]) or [1,1]
        tx, ty = t.get("translate",[0,0]) or [0,0]
        theta  = float(t.get("theta_deg",0.0))
        return _compose_affine((float(sx),float(sy)), theta, (float(tx),float(ty)))
    mi = raw.get("map_info") or {}
    if isinstance(mi, dict):
        ox, oy = 0.0, 0.0
        if "origin" in mi and isinstance(mi["origin"], (list,tuple)) and len(mi["origin"])>=2:
            ox, oy = float(mi["origin"][0]), float(mi["origin"][1])
        theta = float(mi.get("theta_deg", 0.0))
        if "pixel_per_meter" in mi:
            ppm = float(mi["pixel_per_meter"])
            return _compose_affine((ppm, ppm), theta, (ox, oy))
        if "meters_per_pixel" in mi:
            mpp = float(mi["meters_per_pixel"]); ppm = 1.0/mpp if mpp else 1.0
            return _compose_affine((ppm, ppm), theta, (ox, oy))
    return None

def _parse_affine_from_string(s: str):
    try:
        parts = [float(x.strip()) for x in s.split(",")]
        if len(parts) == 6:
            return tuple(parts)  # a,b,c,d,e,f
    except Exception:
        pass
    return None

def _apply_affine_xy(x, y, A):
    a,b,c,d,e,f = A
    xp = a*x + b*y + e
    yp = c*x + d*y + f
    return xp, yp

def _xy_to_pixel(x, y, map_w: float, map_h: float, A):
    # A 必不为 None；默认 A=(1,0,0,-1,0,map_h)
    xp, yp = _apply_affine_xy(x, y, A)
    if X_FLIP_AFTER_AFFINE:
        xp = map_w - xp
    if Y_FLIP_AFTER_AFFINE:
        yp = map_h - yp
    return xp, yp

def _tx_geojson_xy_to_simple(gj: dict, map_w: float, map_h: float, A):
    def _tx(coords):
        if isinstance(coords[0], (int,float)):
            x,y = coords[:2]
            xp, yp = _xy_to_pixel(x, y, map_w, map_h, A)
            return [xp, yp]   # GeoJSON = [lon, lat]
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

# ================= Tooltip 字段推断 =================
def _infer_tooltip_fields(gj: dict, max_fields: int = 6) -> list:
    feats = gj.get("features") or []
    if not feats:
        return []
    blacklist = {"style","styles","stroke","fill","stroke-width","stroke-opacity","fill-opacity"}
    def scalar_keys(props: dict):
        out = set()
        for k, v in (props or {}).items():
            if k in blacklist: continue
            if isinstance(v, (str, int, float, bool)) or v is None: out.add(k)
        return out
    common = None
    for ft in feats:
        keys = scalar_keys(ft.get("properties") or {})
        common = keys if common is None else (common & keys)
    if not common:
        return []
    return sorted(common)[:max_fields]

# ================= 主流程 =================
def main():
    m = folium.Map(location=[0,0], zoom_start=0, tiles=None, crs="Simple", control_scale=True)

    palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
               "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]

    floor_entries = []
    default_floor_key = None
    default_bounds = None

    for site, floors in FLOOR_SETS.items():
        for floor_name in floors:
            key = f"{site}-{floor_name}"
            is_first = default_floor_key is None
            if is_first:
                default_floor_key = key

            cache_dir = os.path.join(CACHE_ROOT, site, floor_name)
            base_url  = f"{REPO_BASE}/{site}/{floor_name}"

            # 下载三件套
            geojson_path   = _download_to_cache(f"{base_url}/geojson_map.json", cache_dir)
            floorinfo_path = _download_to_cache(f"{base_url}/floor_info.json", cache_dir)
            img_path       = _download_to_cache(f"{base_url}/floor_image.png", cache_dir)

            # 尺寸/边界
            floorinfo = _read_json(floorinfo_path)
            fi_dict = floorinfo if isinstance(floorinfo, dict) else (floorinfo[0] if isinstance(floorinfo, list) and floorinfo else {})
            map_w = float(fi_dict["map_info"]["width"])
            map_h = float(fi_dict["map_info"]["height"])
            bounds = [[0,0],[map_h,map_w]]
            if default_bounds is None:
                default_bounds = bounds

            # —— 取 A：环境变量 → floor_info → 默认
            A = None
            if AFFINE_OVERRIDE_STR:
                A = _parse_affine_from_string(AFFINE_OVERRIDE_STR)
                if A and VERBOSE: print(f"✔ {key} 使用手动仿射：{A}")
            if A is None:
                A = _try_affine_from_floorinfo(floorinfo)
                if A and VERBOSE: print(f"✔ {key} 从 floor_info.json 提取仿射：{A}")
            if A is None:
                A = (1.0, 0.0, 0.0, -1.0, 0.0, map_h)  # 默认=翻转+上移（米→像素）
                if VERBOSE: print(f"ℹ️ {key} 使用默认仿射：{A}")

            # —— 诊断：仿射几何性质
            a,b,c,d,e,f = A
            sx = math.hypot(a, c)     # 第一列范数
            sy = math.hypot(b, d)     # 第二列范数
            orth = a*b + c*d          # 列向量点积（正交性）
            det = a*d - b*c           # 行列式（<0 含镜像）
            if VERBOSE:
                print(f"[A:{key}] scale_x={sx:.5f} scale_y={sy:.5f} orth={orth:.3e} det={det:.5f}")

            # —— 可选：强制等比去剪切（保持旋转/镜像）
            if FORCE_ISOTROPIC:
                M = np.array([[a,b],[c,d]], dtype=float)
                U,S,Vt = np.linalg.svd(M)
                Q = U @ Vt                   # 正交矩阵（可能含镜像 det=±1）
                s_iso = float(S.mean())      # 等比尺度（两奇异值均值）
                M_iso = Q * s_iso
                a,b = float(M_iso[0,0]), float(M_iso[0,1])
                c,d = float(M_iso[1,0]), float(M_iso[1,1])
                A = (a,b,c,d,e,f)
                if VERBOSE:
                    sx2 = math.hypot(a,c); sy2 = math.hypot(b,d); orth2 = a*b + c*d; det2 = a*d - b*c
                    print(f"→ FORCE_ISO 后: scale_x={sx2:.5f} scale_y={sy2:.5f} orth={orth2:.3e} det={det2:.5f}")

            # —— 映射基向量（便于肉眼核对）
            if VERBOSE:
                t0 = (0,0)
                t1 = (1,0)
                t2 = (0,1)
                m0 = _xy_to_pixel(*t0, map_w, map_h, A)
                m1 = _xy_to_pixel(*t1, map_w, map_h, A)
                m2 = _xy_to_pixel(*t2, map_w, map_h, A)
                print(f"    映射基向量: (0,0)->{m0}  (1,0)->{m1}  (0,1)->{m2}")
                if X_FLIP_AFTER_AFFINE or Y_FLIP_AFTER_AFFINE:
                    print(f"    额外镜像: X_FLIP={X_FLIP_AFTER_AFFINE}  Y_FLIP={Y_FLIP_AFTER_AFFINE}")

            # GeoJSON 也用同一 A
            gj   = _read_json(geojson_path)
            gj_s = _tx_geojson_xy_to_simple(gj, map_w, map_h, A)

            # —— BASE（首层 show=True）
            base_group = folium.FeatureGroup(name=f"{key} | BASE", show=is_first)
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

            # —— GT（首层 show=True）
            gt_group = folium.FeatureGroup(name=f"{key} | GT", show=is_first)

            txt_dir_url = f"https://github.com/location-competition/indoor-location-competition-20/tree/master/data/{site}/{floor_name}/path_data_files"
            files = _list_txt_in_github_dir(txt_dir_url)
            files = [f for f in files if NAME_FILTER in f["name"]]
            files.sort(key=lambda x: x["name"])
            if MAX_FILES_PER_FLOOR and MAX_FILES_PER_FLOOR > 0:
                files = files[:MAX_FILES_PER_FLOOR]
            print(f"🗂  {key} 发现 txt：{len(files)} 个")

            color_i = 0
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

                # 轨迹折线（米→像素→Simple）
                if not wp.empty:
                    coords = []
                    for x, y in zip(wp["x"], wp["y"]):
                        xp, yp = _xy_to_pixel(x, y, map_w, map_h, A)
                        coords.append([yp, xp])  # [lat, lon]
                    if len(coords) >= 2:
                        folium.PolyLine(coords, weight=3, opacity=0.9,
                                        color=palette[color_i % len(palette)],
                                        tooltip=it["name"]).add_to(gt_group)
                    if DRAW_POINT_SAMPLE_EVERY and DRAW_POINT_SAMPLE_EVERY > 0:
                        for lat, lon in coords[::DRAW_POINT_SAMPLE_EVERY]:
                            folium.CircleMarker([lat,lon], radius=2, weight=1, fill=True, fill_opacity=0.9,
                                                color="#333333").add_to(gt_group)
                    color_i += 1

                # 热力准备
                if wp.empty or mg.empty:
                    if VERBOSE: print(f"      · {it['name']} 无法插值（waypoints={len(wp)}, mag={len(mg)})")
                    continue

                mf = mg.copy()
                mf["B"] = np.sqrt(mf["mx"]**2 + mf["my"]**2 + mf["mz"]**2)
                xyB = interpolate_magnetic_to_xy(mf[["ts","B"]], wp[["ts","x","y"]])

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

            gt_group.add_to(m)

            # —— Heat（首层 show=True）
            heat_group = folium.FeatureGroup(name=f"{key} | Heat", show=is_first)
            if ENABLE_HEATMAP and len(xyB_all) > 0:
                xyB = pd.concat(xyB_all, ignore_index=True)

                Bvals = xyB["B"].to_numpy(dtype=float)
                plo, phi = np.percentile(Bvals, ROBUST_CLIP_P)
                if not np.isfinite(plo) or not np.isfinite(phi) or (phi - plo) <= 1e-9:
                    weights = np.ones_like(Bvals, dtype=float)
                else:
                    weights = np.clip((Bvals - plo) / (phi - plo), 0.0, 1.0)
                xyB = xyB.assign(w=weights)

                if len(xyB) > MAX_MAG_POINTS:
                    xyB = xyB.sample(MAX_MAG_POINTS, random_state=42)

                heat_points = []
                for _, r in xyB.iterrows():
                    xp, yp = _xy_to_pixel(float(r["x"]), float(r["y"]), map_w, map_h, A)
                    heat_points.append([yp, xp, float(r["w"])])

                HeatMap(
                    heat_points, radius=HEAT_RADIUS, blur=HEAT_BLUR,
                    min_opacity=HEAT_MIN_OPACITY, max_zoom=18
                ).add_to(heat_group)

                print(f"   ↳ {key} 参与插值的文件：{len(files)}，样本点：{len(xyB)}，wp≈{n_wp_used}，mag≈{n_mf_used}")
            else:
                print(f"⚠️  {key} 无可用热力点")

            heat_group.add_to(m)

            floor_entries.append({
                "key": key,
                "base_var": base_group.get_name(),
                "gt_var": gt_group.get_name(),
                "heat_var": heat_group.get_name(),
                "bounds_js": f"[[0,0],[{map_h},{map_w}]]",
            })

    # 初始视野
    if default_bounds is not None:
        try: m.fit_bounds(default_bounds)
        except Exception: pass

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
        document.getElementById('floorSel').addEventListener('change', function(){ showFloor(this.value); });
        document.getElementById('chkFloorGT').addEventListener('change', function(){
            var g = floorGroups[currentFloor];
            if (this.checked) map.addLayer(g.gt); else if (map.hasLayer(g.gt)) map.removeLayer(g.gt);
        });
        document.getElementById('chkFloorHeat').addEventListener('change', function(){
            var g = floorGroups[currentFloor];
            if (this.checked) map.addLayer(g.heat); else if (map.hasLayer(g.heat)) map.removeLayer(g.heat);
        });
        document.getElementById('btnToggleAllGT').addEventListener('click', function(){ setAllGT(!allGTOn); });
        document.getElementById('btnToggleAllHeat').addEventListener('click', function(){ setAllHeat(!allHeatOn); });

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
