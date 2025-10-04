import folium
import json
from pathlib import Path

# ---------------------- 1. 路径 ----------------------
geojson_path = Path("geojson_map.json")
floor_info_path = Path("floor_info.json")

if not geojson_path.exists():
    raise FileNotFoundError(f"GeoJSON文件不存在：{geojson_path}")
if not floor_info_path.exists():
    raise FileNotFoundError(f"楼层信息文件不存在：{floor_info_path}")

# ---------------------- 2. 读取数据 ----------------------
with open(geojson_path, "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

with open(floor_info_path, "r", encoding="utf-8") as f:
    floor_info_raw = json.load(f)

# 兼容：list 或 dict
# 期望 floor_info_dict: {floor_id: {floor_name, usage, area, key_points, ...}}
floor_info_dict = {}
if isinstance(floor_info_raw, list):
    for it in floor_info_raw:
        fid = it.get("floor_id")
        if fid is not None:
            floor_info_dict[fid] = it
elif isinstance(floor_info_raw, dict):
    fid = floor_info_raw.get("floor_id", 1)  # 若没有 floor_id，默认 1
    floor_info_dict[fid] = floor_info_raw
else:
    raise TypeError("floor_info.json 类型不支持（需 list 或 dict）")

# ---------------------- 3. 几何中心（Polygon/MultiPolygon） ----------------------
def get_geojson_center(geojson):
    coords = []
    for feature in geojson.get("features", []):
        geom = feature.get("geometry") or {}
        gtype = geom.get("type")
        if gtype == "Polygon":
            rings = geom.get("coordinates", [])
            if rings:
                coords += rings[0]  # 外环
        elif gtype == "MultiPolygon":
            for poly in geom.get("coordinates", []):
                if poly:
                    coords += poly[0]
    if not coords:
        raise ValueError("GeoJSON 中没有可用坐标（或不是 Polygon/MultiPolygon）")
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return [sum(lats)/len(lats), sum(lons)/len(lons)]

map_center = get_geojson_center(geojson_data)
m = folium.Map(location=map_center, zoom_start=18, tiles="CartoDB positron")

# ---------------------- 4. 合并楼层属性到 GeoJSON properties ----------------------
# 规则：按 feature.properties.floor_id 匹配；若缺失且 floor_info_dict 只有 1 层，则统一赋值
candidate_merge_keys = ["floor_name", "usage", "area"]
all_features = geojson_data.get("features", [])
only_one_floor = (len(floor_info_dict) == 1)
default_floor_id = next(iter(floor_info_dict.keys())) if only_one_floor else None

for feat in all_features:
    props = feat.setdefault("properties", {})
    fid = props.get("floor_id", default_floor_id)
    if fid is None:
        continue
    # 可能 fid 是字符串，统一转 int 或保持一致
    try:
        fid_int = int(fid)
    except Exception:
        fid_int = fid
    props["floor_id"] = fid_int
    info = floor_info_dict.get(fid_int) or floor_info_dict.get(str(fid_int))
    if isinstance(info, dict):
        for k in candidate_merge_keys:
            if (k in info) and (k not in props):
                props[k] = info[k]

# ---------------------- 5. 样式 & Tooltip ----------------------
def style_function(feature):
    fid = feature.get("properties", {}).get("floor_id")
    try:
        fid = int(fid)
    except Exception:
        pass
    color_map = {1: "#3186cc", 2: "#2ecc71", 3: "#e74c3c"}
    return {
        "fillColor": color_map.get(fid, "#95a5a6"),
        "color": "#2c3e50",
        "weight": 2,
        "fillOpacity": 0.4,
    }

# Tooltip 字段尽量从 properties 中自动选择可用的
available_keys = set()
for feat in all_features:
    available_keys.update(list((feat.get("properties") or {}).keys()))
candidate_fields = ["floor_id", "floor_name", "usage", "area", "name"]
fields = [k for k in candidate_fields if k in available_keys]
aliases_map = {
    "floor_id": "楼层ID：",
    "floor_name": "楼层：",
    "usage": "用途：",
    "area": "面积：",
    "name": "名称：",
}
aliases = [aliases_map[k] for k in fields]

tooltip = folium.GeoJsonTooltip(fields=fields, aliases=aliases, labels=True, sticky=False)

folium.GeoJson(
    geojson_data,
    name="楼层区域",
    style_function=style_function,
    tooltip=tooltip,
).add_to(m)

# ---------------------- 6. 关键点（来自 floor_info） ----------------------
for fid, info in floor_info_dict.items():
    key_points = info.get("key_points", []) or []
    for point in key_points:
        lat = point.get("lat")
        lon = point.get("lon")
        name = point.get("name", "")
        if lat is None or lon is None:
            continue
        color_map = {1: "blue", 2: "green", 3: "red"}
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(f"<b>{info.get('floor_name', fid)}</b><br>{name}", max_width=200),
            icon=folium.Icon(color=color_map.get(fid, "gray"), icon="map-marker", prefix="fa"),
        ).add_to(m)

# ---------------------- 7. 控件 & 保存 ----------------------
folium.LayerControl(collapsed=False).add_to(m)
folium.LatLngPopup().add_to(m)

output_path = Path("local_geojson_floor_map.html")
m.save(output_path)
print(f"地图已保存至：{output_path.resolve()}")
