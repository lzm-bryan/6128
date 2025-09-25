#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re, json, time
import numpy as np
import pandas as pd
from urllib.parse import urlparse
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

# —— 地磁热力图参数 ——（对齐单层脚本的思路）
ENABLE_HEATMAP          = True
PREFER_UNCALIBRATED     = False   # 和单层版一致：默认优先 CALIBRATED；需要时可改 True
ACC_FILTER              = 0       # 仅保留磁场样本 accuracy>=此阈值（0/1/2/3；0=不过滤）
MAX_MAG_POINTS          = 20000   # 每层热力点总量上限（超过会等步长抽样）
ROBUST_CLIP_P           = (5, 95) # 权重归一化的分位裁剪（抗异常值）
NEAREST_TOL_MS          = 0       # 宽松回退：若不在两路标之间，允许用“最近路标”且|Δt|<=该阈值（毫秒）；0=关闭
HEAT_RADIUS             = 6
HEAT_BLUR               = 15
HEAT_MIN_OPACITY        = 0.4

# ================= 基础工具 =================
def _is_github_blob(url: str) -> bool:
    return "github.com" in url and "/blob/" in url

def _to_raw(url: str) -> str:
    return url.replace("https://github.com/","https://raw.githubusercontent.com/").replace("/blob/","/") if _is_github_blob(url) else url

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
        if VERBOSE: print(f"↩️  缓存命中：{path}")
        return path
    print(f"⬇️  下载：{raw_url}")
    data = _req(raw_url)
    with open(path, "wb") as f: f.write(data)
    print(f"✅ 完成：{path}（{len(data)/1024:.1f} KB）")
    return path

def _list_txt_in_github_dir(tree_url: str):
    """优先 GitHub API；失败用 HTML 兜底解析 `.txt`。"""
    p = urlparse(tree_url)
    parts = [x for x in p.path.strip("/").split("/") if x]
    assert len(parts) >= 2, "非法 GitHub 链接"
    owner, repo = parts[0], parts[1]
    if len(parts) >= 4 and parts[2] == "tree":
        branch = parts[3]
        subpath = "/".join(parts[4:])
    else:
        branch = "master"
        subpath = "/".join(parts[2:])
    api = f"https://api.github.com/repos/{owner}/{repo}/contents/{subpath}?ref={branch}"
    try:
        data = json.loads(_req(api).decode("utf-8", "ignore"))
        files = []
        for it in data:
            if it.get("type") == "file" and it.get("name","").endswith(".txt"):
                files.append({"name": it["name"], "download_url": it.get("download_url")})
        return files
    except Exception as e:
        print(f"⚠️ API 失败：{e}；改用 HTML 解析。")
    # HTML 兜底
    html = _req(tree_url).decode("utf-8","ignore")
    hrefs = re.findall(r'href="([^"]+\.txt)"', html)
    files = []
    for h in hrefs:
        if not h.startswith("http"):
            h = f"https://github.com{h}"
        if "/blob/" not in h:
            continue
        files.append({"name": os.path.basename(urlparse(h).path), "download_url": _to_raw(h)})
    return files

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

    df = None
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
def _xy_to_leaflet(x,y,h):  # meters -> CRS.Simple (lon=x, lat=h-y)
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

# ================= 从单个 txt 生成“未归一化”的热力样本（lat,lon,B） =================
def _heat_points_for_file(txt_path: str, map_h: float, verbose: bool = False) -> list:
    wp = _read_waypoints(txt_path)
    mg = _read_magnetometer(txt_path)

    if verbose:
        print(f"      · {os.path.basename(txt_path)}  waypoints={len(wp)}  magnetometer={len(mg)}")

    if wp.empty or mg.empty:
        return []

    wp2  = wp[["ts","x","y"]].drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    mg   = mg.sort_values("ts").reset_index(drop=True)
    wp2n = wp2.rename(columns={"ts":"ts2","x":"x2","y":"y2"})

    # asof 前后路标
    prev = pd.merge_asof(mg, wp2, on="ts", direction="backward", allow_exact_matches=True)
    nxt  = pd.merge_asof(mg.assign(ts2=mg["ts"]), wp2n, on="ts2", direction="forward", allow_exact_matches=True)

    df = mg.copy()
    df["x0"] = prev["x"];  df["y0"] = prev["y"];  df["t0"] = prev["ts"]
    df["x1"] = nxt["x2"];  df["y1"] = nxt["y2"];  df["t1"] = nxt["ts2"]

    before = len(df)
    df = df.dropna(subset=["x0","y0","x1","y1","t0","t1"])
    df = df[df["t1"] > df["t0"]]

    if verbose:
        print(f"        ↳ 可插值样本 {len(df)}/{before}")

    latlonB = []

    # ① 严格插值：落在两路标之间
    if len(df) > 0:
        alpha = (df["ts"] - df["t0"]) / (df["t1"] - df["t0"])
        x = df["x0"] + alpha * (df["x1"] - df["x0"])
        y = df["y0"] + alpha * (df["y1"] - df["y0"])
        B = np.sqrt(df["mx"]**2 + df["my"]**2 + df["mz"]**2).astype(float)
        for xi, yi, bi in zip(x, y, B):
            lon, lat = _xy_to_leaflet(float(xi), float(yi), map_h)
            latlonB.append([lat, lon, float(bi)])

    # ② 可选宽松回退：最近路标（|Δt|<=NEAREST_TOL_MS）
    if len(latlonB) == 0 and NEAREST_TOL_MS and NEAREST_TOL_MS > 0:
        near = pd.merge_asof(
            mg[["ts","mx","my","mz"]].sort_values("ts"),
            wp2.sort_values("ts"),
            on="ts", direction="nearest", tolerance=NEAREST_TOL_MS
        ).dropna(subset=["x","y"])
        if not near.empty:
            B = np.sqrt(near["mx"]**2 + near["my"]**2 + near["mz"]**2).astype(float)
            for xi, yi, bi in zip(near["x"], near["y"], B):
                lon, lat = _xy_to_leaflet(float(xi), float(yi), map_h)
                latlonB.append([lat, lon, float(bi)])
            if verbose:
                print(f"        ↳ 回退(最近路标±{NEAREST_TOL_MS}ms)：{len(latlonB)} 点")

    return latlonB

# ================= 主流程 =================
def main():
    # 统一的地图（CRS.Simple）
    m = folium.Map(location=[0,0], zoom_start=0, tiles=None, crs="Simple", control_scale=True)

    palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
               "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]

    # 记录每层的 JS 变量名与边界，供控件使用
    floor_entries = []  # dict: {key, base_var, gt_var, heat_var, bounds_js}
    default_floor_key = None

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

            # GeoJSON -> Simple CRS
            gj   = _read_json(geojson_path)
            gj_s = _transform_geojson(gj, map_h)

            # —— 楼层“BASE”组：底图 + GeoJSON ——（作为一个整体被显示/隐藏）
            base_group = folium.FeatureGroup(name=f"{key} | BASE}", show=False)
            folium.raster_layers.ImageOverlay(
                image=img_path, bounds=bounds, opacity=1.0, interactive=False, name=f"{key} floor"
            ).add_to(base_group)

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

            folium.GeoJson(
                gj_s, name=f"{key} geojson",
                style_function=style_fn,
                tooltip=GeoJsonTooltip(fields=tooltip_fields) if tooltip_fields else None
            ).add_to(base_group)
            base_group.add_to(m)

            # —— 楼层“GT”组：该层所有文件汇总的一组轨迹折线 ——（便于总控/本层控）
            gt_group = folium.FeatureGroup(name=f"{key} | GT", show=False)

            # 列出并下载该层所有 txt
            txt_dir_url = f"https://github.com/location-competition/indoor-location-competition-20/tree/master/data/{site}/{floor_name}/path_data_files"
            files = _list_txt_in_github_dir(txt_dir_url)
            files = [f for f in files if NAME_FILTER in f["name"]]
            files.sort(key=lambda x: x["name"])
            if MAX_FILES_PER_FLOOR and MAX_FILES_PER_FLOOR > 0:
                files = files[:MAX_FILES_PER_FLOOR]
            print(f"🗂  {key} 发现 txt：{len(files)} 个")

            # 汇总绘制 GT & 收集热力点（先收集 |B|，稍后统一做鲁棒归一化）
            color_i = 0
            total_pts = 0
            heat_points_latlonB = []

            for it in files:
                local_txt = _download_to_cache(it["download_url"], cache_dir, filename=it["name"])
                df = _read_waypoints(local_txt)
                if df.empty:
                    if VERBOSE: print(f"      · {it['name']} 无 WAYPOINT")
                    continue
                # 折线
                coords = [[_xy_to_leaflet(x,y,map_h)[1], _xy_to_leaflet(x,y,map_h)[0]] for x,y in zip(df["x"], df["y"])]
                if len(coords) >= 2:
                    folium.PolyLine(coords, weight=3, opacity=0.9, color=palette[color_i % len(palette)],
                                    tooltip=it["name"]).add_to(gt_group)
                # 可选：抽样画点
                if DRAW_POINT_SAMPLE_EVERY and DRAW_POINT_SAMPLE_EVERY > 0:
                    for lat, lon in coords[::DRAW_POINT_SAMPLE_EVERY]:
                        folium.CircleMarker([lat,lon], radius=2, weight=1, fill=True, fill_opacity=0.9,
                                            color="#333333").add_to(gt_group)
                color_i += 1
                total_pts += len(coords)

                # 热力点（来自地磁 + waypoint 时间插值）
                if ENABLE_HEATMAP:
                    hp = _heat_points_for_file(local_txt, map_h, verbose=VERBOSE)
                    heat_points_latlonB.extend(hp)

            gt_group.add_to(m)

            # —— 楼层“HEAT”组 ——（统一对 |B| 做鲁棒归一化 → 权重 0~1）
            heat_group = folium.FeatureGroup(name=f"{key} | Heat", show=False)
            if ENABLE_HEATMAP and len(heat_points_latlonB) > 0:
                # 限量（再次兜底）
                hp = heat_points_latlonB
                if len(hp) > MAX_MAG_POINTS:
                    step = int(np.ceil(len(hp) / MAX_MAG_POINTS))
                    hp = hp[::step]

                # 鲁棒归一化（分位裁剪）
                Bvals = np.array([w for _,_,w in hp], dtype=float)
                plo, phi = np.percentile(Bvals, ROBUST_CLIP_P)
                denom = max(1e-6, (phi - plo))
                weights = np.clip((Bvals - plo) / denom, 0.0, 1.0)
                heat_points = [[lat, lon, float(w)] for (lat,lon,_), w in zip(hp, weights)]

                HeatMap(
                    heat_points, radius=HEAT_RADIUS, blur=HEAT_BLUR,
                    min_opacity=HEAT_MIN_OPACITY, max_zoom=18
                ).add_to(heat_group)
            heat_group.add_to(m)

            print(f"   ↳ GT 折线点数：{total_pts}，热力点：{len(heat_points_latlonB)}")

            # 收集 JS 引用
            floor_entries.append({
                "key": key,
                "base_var": base_group.get_name(),
                "gt_var": gt_group.get_name(),
                "heat_var": heat_group.get_name(),
                "bounds_js": f"[[0,0],[{map_h},{map_w}]]",
            })

    folium.LayerControl(collapsed=False).add_to(m)

    # ================= 自定义控件（楼层切换 + GT/Heat 本层开关 + 全楼总控） =================
    js_lines = ["var floorGroups = {};"]
    for ent in floor_entries:
        js_lines.append(
            f'floorGroups["{ent["key"]}"] = {{base:{ent["base_var"]}, gt:{ent["gt_var"]}, heat:{ent["heat_var"]}, bounds:{ent["bounds_js"]}}};'
        )
    js_floor_groups = "\n".join(js_lines)
    first_key = (floor_entries[0]["key"] if floor_entries else "")

    floor_options_html = "".join([f'<option value="{ent["key"]}">{ent["key"]}</option>' for ent in floor_entries])

    ctrl_html = f"""
    {js_floor_groups}
    (function() {{
        var map = {m.get_name()};
        // 控件 UI
        var ctrl = L.control({{position:'topright'}});
        ctrl.onAdd = function() {{
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
                    {floor_options_html}
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
        }};
        ctrl.addTo(map);

        var allGTOn = false;
        var allHeatOn = false;
        var currentFloor = "{first_key}";

        function hideAllBases() {{
            for (var k in floorGroups) {{
                if (map.hasLayer(floorGroups[k].base)) map.removeLayer(floorGroups[k].base);
            }}
        }}

        function updatePerFloorCheckboxes() {{
            var g = floorGroups[currentFloor];
            var chkGT = document.getElementById('chkFloorGT');
            var chkHeat = document.getElementById('chkFloorHeat');
            if (chkGT) chkGT.checked = map.hasLayer(g.gt);
            if (chkHeat) chkHeat.checked = map.hasLayer(g.heat);
        }}

        function showFloor(key) {{
            currentFloor = key;
            hideAllBases();
            var g = floorGroups[key];
            map.addLayer(g.base);

            // 按本层开关决定是否显示
            var chkGT = document.getElementById('chkFloorGT');
            var chkHeat = document.getElementById('chkFloorHeat');
            if (chkGT && chkGT.checked) map.addLayer(g.gt); else if (map.hasLayer(g.gt)) map.removeLayer(g.gt);
            if (chkHeat && chkHeat.checked) map.addLayer(g.heat); else if (map.hasLayer(g.heat)) map.removeLayer(g.heat);

            try {{ map.fitBounds(g.bounds); }} catch(e) {{}}
        }}

        function setAllGT(on) {{
            allGTOn = on;
            for (var k in floorGroups) {{
                if (on) map.addLayer(floorGroups[k].gt);
                else if (map.hasLayer(floorGroups[k].gt)) map.removeLayer(floorGroups[k].gt);
            }}
            var lbl = document.getElementById('lblAllGT');
            if (lbl) lbl.textContent = '（全楼：' + (on ? '显示' : '隐藏') + '）';
            updatePerFloorCheckboxes();
        }}

        function setAllHeat(on) {{
            allHeatOn = on;
            for (var k in floorGroups) {{
                if (on) map.addLayer(floorGroups[k].heat);
                else if (map.hasLayer(floorGroups[k].heat)) map.removeLayer(floorGroups[k].heat);
            }}
            var lbl = document.getElementById('lblAllHeat');
            if (lbl) lbl.textContent = '（全楼：' + (on ? '显示' : '隐藏') + '）';
            updatePerFloorCheckboxes();
        }}

        // 事件
        document.getElementById('floorSel').addEventListener('change', function() {{
            showFloor(this.value);
        }});
        document.getElementById('chkFloorGT').addEventListener('change', function() {{
            var g = floorGroups[currentFloor];
            if (this.checked) map.addLayer(g.gt); else if (map.hasLayer(g.gt)) map.removeLayer(g.gt);
        }});
        document.getElementById('chkFloorHeat').addEventListener('change', function() {{
            var g = floorGroups[currentFloor];
            if (this.checked) map.addLayer(g.heat); else if (map.hasLayer(g.heat)) map.removeLayer(g.heat);
        }});
        document.getElementById('btnToggleAllGT').addEventListener('click', function() {{
            setAllGT(!allGTOn);
        }});
        document.getElementById('btnToggleAllHeat').addEventListener('click', function() {{
            setAllHeat(!allHeatOn);
        }});

        // 初始化：显示第一层，默认本层 GT/Heat 为“勾选→显示”，全楼总控保持“隐藏”
        setTimeout(function() {{
            var sel = document.getElementById('floorSel');
            if (sel) sel.value = "{first_key}";
            showFloor("{first_key}");
        }}, 50);
    }})();
    """

    folium.Element(f"<script>{ctrl_html}</script>").add_to(m)

    m.save(OUT_HTML)
    print(f"\n🎉 生成完成：{OUT_HTML}")

if __name__ == "__main__":
    main()
