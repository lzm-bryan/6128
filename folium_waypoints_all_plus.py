#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io, os, re, json, time
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from urllib.request import urlopen, Request
import folium
from folium.features import GeoJsonTooltip
from folium.plugins import MarkerCluster, Fullscreen, MousePosition, MeasureControl, Draw, Search
from branca.element import Element

# ========== 配置（按需修改） ==========
GEOJSON_URL   = "https://github.com/location-competition/indoor-location-competition-20/blob/master/data/site1/B1/geojson_map.json"
FLOOR_INFO_URL= "https://github.com/location-competition/indoor-location-competition-20/blob/master/data/site1/B1/floor_info.json"
FLOOR_IMG_URL = "https://github.com/location-competition/indoor-location-competition-20/blob/master/data/site1/B1/floor_image.png"

# 指向包含 .txt 轨迹文件的“目录”页面（/tree/）
GT_FOLDER_URL = "https://github.com/location-competition/indoor-location-competition-20/tree/master/data/site1/B1/path_data_files"

CACHE_DIR     = os.path.join("indoor_cache", "site1", "B1")
OUT_HTML      = "b1_waypoints_all.html"
STYLE_FIELD   = ""      # 若 GeoJSON properties 有分类字段（如 "type"/"feature_type"），填字段名可自动配色
MAX_FILES     = 0       # 0=全部；>0 仅取前 N 个
NAME_FILTER   = ""      # 仅下载文件名包含该子串的 .txt（留空=不过滤）
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

def _parse_github_dir(url: str):
    """优先用 GitHub API 列目录；失败再用 HTML 兜底解析"""
    p = urlparse(url); parts = [x for x in p.path.strip("/").split("/") if x]
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
        return files
    except Exception as e:
        print(f"⚠️ API 列目录失败，改用 HTML 解析：{e}")
    html = _req(url).decode("utf-8","ignore")
    hrefs = re.findall(r'href="([^"]+\.txt)"', html)
    files = []
    for h in hrefs:
        if not h.startswith("http"): h = f"https://github.com{h}"
        if "/blob/" not in h: continue
        files.append({"name": os.path.basename(urlparse(h).path), "download_url": _to_raw(h)})
    if not files: raise RuntimeError("未能从目录页解析出 .txt 文件链接")
    return files

def _read_json(path: str):
    with open(path,"r",encoding="utf-8") as f:
        return json.load(f)

def _read_waypoints(path: str) -> pd.DataFrame:
    """仅解析：<ts> TYPE_WAYPOINT <x> <y> （米）；其它类型行忽略"""
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
            try: ts = int(p[0])
            except: ts = None
            try:
                x = float(p[2]); y = float(p[3])
                rows.append((ts,x,y))
            except:
                continue
    if not rows: raise ValueError(f"[{path}] 未找到 TYPE_WAYPOINT 行")
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
            if not g: continue
            f2 = dict(ft); f2["geometry"] = {"type": g.get("type"), "coordinates": _tx(g.get("coordinates"))}
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
    file_list = _parse_github_dir(GT_FOLDER_URL)
    file_list = [f for f in file_list if (NAME_FILTER in f["name"])]
    file_list.sort(key=lambda x: x["name"])
    if MAX_FILES and MAX_FILES > 0:
        file_list = file_list[:MAX_FILES]
    print(f"🗂  发现 .txt：{len(file_list)} 个")
    local_gt = [ _download_to_cache(f["download_url"], CACHE_DIR, filename=f["name"]) for f in file_list ]

    # 3) 读取 meta、变换坐标
    floor = _read_json(local_floor); map_w = float(floor["map_info"]["width"]); map_h = float(floor["map_info"]["height"])
    gj = _read_json(local_geojson); gj_s = _transform_geojson(gj, map_h)

    # 4) 画 Folium 图（带 UI 插件与 API）
    m = folium.Map(location=[map_h/2, map_w/2], zoom_start=0, tiles=None, crs="Simple", control_scale=True)
    bounds = [[0,0],[map_h,map_w]]

    img_layer = folium.raster_layers.ImageOverlay(
        name="Floor image", image=local_img, bounds=bounds, opacity=1.0, interactive=False
    )
    img_layer.add_to(m)
    m.fit_bounds(bounds)

    # GeoJSON
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

    gj_layer = folium.GeoJson(gj_s, name="GeoJSON features",
                              style_function=style_fn,
                              tooltip=GeoJsonTooltip(fields=tooltip_fields) if tooltip_fields else None)
    gj_layer.add_to(m)

    # 轨迹（每条一个图层；默认展开前 5 条）
    total_pts = 0
    for idx, path in enumerate(local_gt, 1):
        try:
            df = _read_waypoints(path)
        except Exception as e:
            print(f"⚠️ 跳过 {os.path.basename(path)}：{e}")
            continue

        color = palette[idx % len(palette)]
        coords = []
        for x, y in zip(df["x"], df["y"]):
            lon, lat = _xy_to_leaflet(float(x), float(y), map_h)
            coords.append([lat, lon])  # folium: [lat, lon]

        name = os.path.basename(path)
        layer = folium.FeatureGroup(name=f"GT: {name}", show=(idx<=5))
        if len(coords) >= 2:
            folium.PolyLine(coords, weight=3, opacity=0.9, color=color).add_to(layer)
        layer.add_to(m)
        total_pts += len(coords)

    # ==== 交互增强：按钮 / 工具 / API ====
    Fullscreen(position="topright").add_to(m)
    MousePosition(
        position="bottomleft",
        separator=" , ",
        prefix="x (m), y (m):",
        lng_first=True,
        num_digits=2,
        formatter="{"
                  "lng: function(num) {return L.Util.formatNum(num, 2);},"
                  f"lat: function(num) {{return L.Util.formatNum({map_h} - num, 2);}}"
                  "}"
    ).add_to(m)
    MeasureControl(
        position="topleft",
        primary_length_unit="meters",
        secondary_length_unit=None,
        primary_area_unit="sqmeters"
    ).add_to(m)
    # 重要：Draw 不要传 edit_options（否则会 JSON 序列化失败）
    Draw(
        export=True,
        filename="annotations.geojson",
        position="topleft",
        draw_options={"polyline": True, "polygon": True, "rectangle": True,
                      "marker": True, "circle": False, "circlemarker": False}
    ).add_to(m)
    # 搜索 GeoJSON（按第一个属性字段）
    if tooltip_fields:
        Search(layer=gj_layer, search_label=tooltip_fields[0], placeholder="Search features", collapsed=True).add_to(m)
    # 比例尺
    m.get_root().html.add_child(Element(
        f"""<script>
        L.control.scale({{metric:true, imperial:false, position:'bottomright'}}).addTo({m.get_name()});
        </script>"""
    ))

    # 自定义控制面板 + 简易 JS API
    img_var = img_layer.get_name()
    gj_var  = gj_layer.get_name()
    control_html = f"""
    <style>
      .custom-panel {{
        position: absolute; top: 10px; left: 10px; z-index: 9999;
        background: rgba(255,255,255,0.92); padding: 8px 10px; border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2); font-family: system-ui, Arial, sans-serif; font-size: 12px;
      }}
      .custom-panel button, .custom-panel input[type=range] {{ margin: 3px 0; width: 160px; }}
      .custom-panel .row {{ display:flex; align-items:center; gap:6px; }}
      .custom-panel .title {{ font-weight:600; margin-bottom:4px; }}
    </style>
    <div class="custom-panel" id="indoorPanel">
      <div class="title">Indoor Controls</div>
      <div class="row"><button id="btnReset">Reset view</button></div>
      <div class="row">
        <label>Floor opacity</label>
        <input id="opacitySlider" type="range" min="0" max="1" step="0.05" value="1">
      </div>
      <div class="row"><button id="btnPNG">Export PNG</button></div>
      <div class="row">
        <input id="fileCSV" type="file" accept=".csv,.txt" style="width:160px">
      </div>
    </div>
    <script>
    (function() {{
      var map = {m.get_name()};
      var floor = {img_var};
      var gj = {gj_var};
      var mapH = {map_h}, mapW = {map_w};
      var predGroup = L.featureGroup().addTo(map);

      function fitAll() {{
        var b = L.latLngBounds([[0,0],[mapH,mapW]]);
        map.fitBounds(b);
      }}
      document.getElementById('btnReset').onclick = fitAll;

      var slider = document.getElementById('opacitySlider');
      slider.oninput = function() {{ floor.setOpacity(parseFloat(this.value)); }};

      function exportPNG() {{
        var id = map.getContainer().id;
        if (!window.html2canvas) {{
          var s = document.createElement('script');
          s.src = "https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js";
          s.onload = doShot; document.body.appendChild(s);
        }} else doShot();
        function doShot() {{
          html2canvas(document.getElementById(id)).then(function(canvas){{
            var a = document.createElement('a');
            a.href = canvas.toDataURL('image/png');
            a.download = 'indoor_map.png';
            a.click();
          }});
        }}
      }}
      document.getElementById('btnPNG').onclick = exportPNG;

      function parseCSV(text) {{
        var lines = text.split(/\\r?\\n/), pts=[];
        for (var i=0;i<lines.length;i++) {{
          var s = lines[i].trim(); if(!s||s[0]=='#') continue;
          s = s.replace(/,/g,' ');
          var p = s.split(/\\s+/);
          if (p.length>=2 && !isNaN(parseFloat(p[0])) && !isNaN(parseFloat(p[1]))) {{
            var x = parseFloat(p[0]), y = parseFloat(p[1]); pts.push([x,y]); continue;
          }}
          if (p.length>=4 && p[1]=='TYPE_WAYPOINT') {{
            var x2 = parseFloat(p[2]), y2 = parseFloat(p[3]); if(!isNaN(x2)&&!isNaN(y2)) pts.push([x2,y2]);
          }}
        }}
        return pts;
      }}
      function drawPred(points) {{
        if (!points || points.length==0) return;
        var latlngs = points.map(function(pt){{ var x=pt[0], y=pt[1]; return [mapH - y, x]; }});
        L.polyline(latlngs, {{color:'#3b82f6', weight:3, opacity:0.9}}).addTo(predGroup);
      }}
      document.getElementById('fileCSV').onchange = function(e){{
        var f = e.target.files[0]; if(!f) return;
        var reader = new FileReader();
        reader.onload = function(evt){{
          var txt = evt.target.result;
          var pts = parseCSV(txt);
          drawPred(pts);
        }};
        reader.readAsText(f);
      }};

      // 简易 API
      window.IndoorMapAPI = {{
        fitAll: fitAll,
        setFloorOpacity: function(a){{ floor.setOpacity(a); }},
        clearPred: function(){{ predGroup.clearLayers(); }},
        addPredCSV: function(text){{ var pts=parseCSV(text); drawPred(pts); }},
        addPredPoints: function(arr){{ drawPred(arr); }}
      }};
    }})();
    </script>
    """
    m.get_root().html.add_child(Element(control_html))

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(OUT_HTML)
    print(f"\n🎉 生成完成：{OUT_HTML}")
    print(f"   轨迹文件：{len(local_gt)} 个，合计点数（下采样后）：{total_pts}")
    print(f"   缓存目录：{CACHE_DIR}")

if __name__ == "__main__":
    main()
