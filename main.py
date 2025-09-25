#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re, json, time
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from urllib.request import urlopen, Request
import folium
from folium.features import GeoJsonTooltip

# ================= 配置：多楼层 =================
REPO_BASE = "https://github.com/location-competition/indoor-location-competition-20/blob/master/data"
# 楼层清单：site1: B1 + F1~F4；site2: B1 + F1~F8
FLOOR_SETS = {
    "site1": ["B1", "F1", "F2", "F3", "F4"],
    "site2": ["B1", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"],
}

CACHE_ROOT = "indoor_cache"
OUT_HTML   = "multi_floor_gt.html"

# 过滤与性能
NAME_FILTER           = ""     # 只处理包含该子串的 txt（留空=不过滤）
MAX_FILES_PER_FLOOR   = 0      # 0=该层全部；>0 仅取前 N 个
DRAW_MARKERS          = False  # 为每个点画圆点，点多建议关闭，仅画折线
SNAP_WAYPOINT_METERS  = 0.05   # 轨迹 5cm 栅格去重（=0 关闭）
STYLE_FIELD           = ""     # GeoJSON properties 的分类字段（填了就按字段自动配色）

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
        print(f"↩️  缓存命中：{path}")
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
    # 解析分支与子路径
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

def _read_waypoints(path: str) -> pd.DataFrame:
    rows = []
    with open(path,"r",encoding="utf-8",errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): continue
            s = s.replace(",", " ")
            p = s.split()
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

# ---------- 关键补丁：安全挑选 GeoJSON tooltip 字段 ----------
def _pick_tooltip_fields(gj: dict, max_fields: int = 6):
    """
    只返回“真实存在且稳定”的字段：
    - 先取前 N 个要素 properties 的交集；若交集为空，退化为首要素字段
    - 再按偏好顺序挑选一批
    """
    feats = [ft for ft in gj.get("features", []) if isinstance(ft, dict)]
    if not feats:
        return []

    N = min(20, len(feats))
    common = set((feats[0].get("properties") or {}).keys())
    for ft in feats[1:N]:
        common &= set((ft.get("properties") or {}).keys())
    if not common:
        common = set((feats[0].get("properties") or {}).keys())

    preferred = ["name", "name_chinese", "store_id", "poi_no", "floor", "two_class", "center"]
    fields = [k for k in preferred if k in common][:max_fields]
    if not fields:
        fields = sorted(common)[:max_fields]
    return fields

# ================= 主流程 =================
def main():
    # 统一的地图（CRS.Simple）
    m = folium.Map(location=[0,0], zoom_start=0, tiles=None, crs="Simple", control_scale=True)

    palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
               "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]

    # 记录每层的 JS 变量名与边界，供自定义控件使用
    floor_entries = []  # list of dicts: {key, base_var, gt_var, bounds_js}
    default_floor_key = None

    for site, floors in FLOOR_SETS.items():
        for floor_name in floors:
            key = f"{site}-{floor_name}"
            if default_floor_key is None:
                default_floor_key = key

            cache_dir = os.path.join(CACHE_ROOT, site, floor_name)
            base_url  = f"{REPO_BASE}/{site}/{floor_name}"

            # 下载三件套
            geojson_path    = _download_to_cache(f"{base_url}/geojson_map.json", cache_dir)
            floorinfo_path  = _download_to_cache(f"{base_url}/floor_info.json", cache_dir)
            img_path        = _download_to_cache(f"{base_url}/floor_image.png", cache_dir)

            # 读取尺寸并准备边界
            floorinfo = _read_json(floorinfo_path)
            map_w = float(floorinfo["map_info"]["width"])
            map_h = float(floorinfo["map_info"]["height"])
            bounds = [[0,0],[map_h,map_w]]

            # GeoJSON 转换到 Simple CRS
            gj   = _read_json(geojson_path)
            gj_s = _transform_geojson(gj, map_h)

            # —— 楼层“BASE”组：底图 + GeoJSON ——（作为一个整体被显示/隐藏）
            base_group = folium.FeatureGroup(name=f"{key} | BASE", show=False)
            folium.raster_layers.ImageOverlay(
                image=img_path, bounds=bounds, opacity=1.0, interactive=False, name=f"{key} floor"
            ).add_to(base_group)

            tooltip_fields = _pick_tooltip_fields(gj, max_fields=6)

            def style_fn(feat):
                props = feat.get("properties") or {}
                color = "#000000"
                if STYLE_FIELD and STYLE_FIELD in props:
                    color = palette[hash(str(props[STYLE_FIELD])) % len(palette)]
                gtype = (feat.get("geometry") or {}).get("type","")
                if gtype in ("Polygon","MultiPolygon"):
                    return {"fillOpacity":0.0, "color":color, "weight":1.2}
                return {"color":color, "weight":2.0}

            # Tooltip 字段有时和首要素不一致会触发断言，这里兜底
            try:
                folium.GeoJson(
                    gj_s, name=f"{key} geojson",
                    style_function=style_fn,
                    tooltip=GeoJsonTooltip(fields=tooltip_fields) if tooltip_fields else None
                ).add_to(base_group)
            except AssertionError:
                folium.GeoJson(
                    gj_s, name=f"{key} geojson",
                    style_function=style_fn
                ).add_to(base_group)

            base_group.add_to(m)

            # —— 楼层“GT”组：该层所有文件汇总的一组轨迹折线（便于总控） ——
            gt_group = folium.FeatureGroup(name=f"{key} | GT", show=False)

            # 列出并下载该层所有 txt
            txt_dir_url = f"https://github.com/location-competition/indoor-location-competition-20/tree/master/data/{site}/{floor_name}/path_data_files"
            files = _list_txt_in_github_dir(txt_dir_url)
            files = [f for f in files if NAME_FILTER in f["name"]]
            files.sort(key=lambda x: x["name"])
            if MAX_FILES_PER_FLOOR and MAX_FILES_PER_FLOOR > 0:
                files = files[:MAX_FILES_PER_FLOOR]
            print(f"🗂  {key} 发现 txt：{len(files)} 个")

            color_i = 0
            total_pts = 0
            for it in files:
                local_txt = _download_to_cache(it["download_url"], cache_dir, filename=it["name"])
                df = _read_waypoints(local_txt)
                if df.empty:
                    continue
                # 折线
                coords = [[_xy_to_leaflet(x,y,map_h)[1], _xy_to_leaflet(x,y,map_h)[0]] for x,y in zip(df["x"], df["y"])]
                if len(coords) >= 2:
                    folium.PolyLine(coords, weight=3, opacity=0.9, color=palette[color_i % len(palette)],
                                    tooltip=it["name"]).add_to(gt_group)
                color_i += 1
                total_pts += len(coords)
                # 可选：画点
                if DRAW_MARKERS:
                    step = max(1, len(coords)//500)  # 采样画点防卡
                    for lat, lon in coords[::step]:
                        folium.CircleMarker([lat,lon], radius=2, weight=1, fill=True, fill_opacity=0.9,
                                            color="#333333").add_to(gt_group)

            gt_group.add_to(m)
            print(f"   ↳ 汇总绘制：{total_pts} 点")

            # 收集 JS 引用
            floor_entries.append({
                "key": key,
                "base_var": base_group.get_name(),
                "gt_var": gt_group.get_name(),
                "bounds_js": f"[[0,0],[{map_h},{map_w}]]",
            })

    # 图层控件
    folium.LayerControl(collapsed=False).add_to(m)

    # ================= 自定义控件（楼层切换 + GT 总控） =================
    js_lines = ["var floorGroups = {};"]
    for ent in floor_entries:
        js_lines.append(
            f'floorGroups["{ent["key"]}"] = {{base:{ent["base_var"]}, gt:{ent["gt_var"]}, bounds:{ent["bounds_js"]}}};'
        )
    js_floor_groups = "\n".join(js_lines)
    first_key = (floor_entries[0]["key"] if floor_entries else "") or ""

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
                <div style="font-weight:600;margin-bottom:6px;">Floor & GT 控制</div>
                <div style="margin-bottom:6px;">
                  <label>楼层：</label>
                  <select id="floorSel" style="max-width:180px;">
                    {floor_options_html}
                  </select>
                </div>
                <div style="margin-bottom:6px;">
                  <label><input type="checkbox" id="chkFloorGT" checked /> 显示本层 GT</label>
                </div>
                <div>
                  <button id="btnToggleAllGT" class="leaflet-control-zoom-in" title="切换当前层 GT">GT 总控</button>
                  <span id="lblAllGT" style="margin-left:6px;">（当前：隐藏）</span>
                </div>
            `;
            L.DomEvent.disableClickPropagation(div);
            return div;
        }};
        ctrl.addTo(map);

        var allGTOn = false;
        var currentFloor = "{first_key}";

        function hideAll() {{
            for (var k in floorGroups) {{
                if (map.hasLayer(floorGroups[k].base)) map.removeLayer(floorGroups[k].base);
                if (map.hasLayer(floorGroups[k].gt))   map.removeLayer(floorGroups[k].gt);
            }}
        }}

        function showFloor(key) {{
            currentFloor = key;
            hideAll();
            var g = floorGroups[key];
            map.addLayer(g.base);
            if (document.getElementById('chkFloorGT').checked) {{
                map.addLayer(g.gt);
            }}
            try {{ map.fitBounds(g.bounds); }} catch(e) {{}}
            // 同步总控文字
            document.getElementById('lblAllGT').textContent = '（当前：' + (document.getElementById('chkFloorGT').checked ? '显示' : '隐藏') + '）';
        }}

        function setAllGT(on) {{
            allGTOn = on;
            var g = floorGroups[currentFloor];
            if (on) map.addLayer(g.gt);
            else if (map.hasLayer(g.gt)) map.removeLayer(g.gt);
            var lbl = document.getElementById('lblAllGT');
            if (lbl) lbl.textContent = '（当前：' + (on ? '显示' : '隐藏') + '）';
            var chk = document.getElementById('chkFloorGT');
            if (chk) chk.checked = on;
        }}

        document.getElementById('floorSel').addEventListener('change', function() {{
            showFloor(this.value);
        }});
        document.getElementById('chkFloorGT').addEventListener('change', function() {{
            setAllGT(this.checked);
        }});
        document.getElementById('btnToggleAllGT').addEventListener('click', function() {{
            setAllGT(!allGTOn);
        }});

        setTimeout(function() {{
            var sel = document.getElementById('floorSel');
            if (sel) sel.value = "{first_key}";
            showFloor("{first_key}");
            setAllGT(true);
        }}, 50);
    }})();
    """

    folium.Element(f"<script>{ctrl_html}</script>").add_to(m)

    m.save(OUT_HTML)
    print(f"\n🎉 生成完成：{OUT_HTML}")

if __name__ == "__main__":
    main()
