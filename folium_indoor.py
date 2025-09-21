#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io, os, json, argparse
from urllib.parse import urlparse
from urllib.request import urlopen
import pandas as pd
import folium
from folium.features import GeoJsonTooltip
from folium.plugins import MarkerCluster, HeatMap


# ---------- utils ----------
def _is_url(s: str) -> bool:
    try:
        p = urlparse(s)
        return p.scheme in ("http", "https")
    except Exception:
        return False


def _to_raw(url: str) -> str:
    # turn GitHub /blob/ into raw
    if "github.com" in url and "/blob/" in url:
        return url.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/")
    return url


def _read_json(path_or_url: str):
    if _is_url(path_or_url):
        url = _to_raw(path_or_url)
        with urlopen(url) as r:
            return json.load(io.TextIOWrapper(r, encoding="utf-8"))
    else:
        with open(path_or_url, "r", encoding="utf-8") as f:
            return json.load(f)


# meters (x,y)  ->  Leaflet CRS.Simple coords (lon=x, lat=y_flipped)
def xy_m_to_leaflet_xy(x, y, map_h_m):
    return [x, map_h_m - y]  # [lon, lat]


def transform_geojson_m_to_simple(gj: dict, map_h_m: float):
    def _tx_coords(coords):
        if isinstance(coords[0], (int, float)):
            x, y = coords[:2]
            lon, lat = xy_m_to_leaflet_xy(x, y, map_h_m)  # lon=x, lat=y'
            return [lon, lat]
        return [_tx_coords(c) for c in coords]

    out = {"type": gj.get("type", "FeatureCollection")}
    if out["type"] == "FeatureCollection":
        feats = []
        for ft in gj.get("features", []):
            geom = ft.get("geometry", {})
            if not geom:
                continue
            g2 = dict(ft)
            g2["geometry"] = {"type": geom.get("type"), "coordinates": _tx_coords(geom.get("coordinates"))}
            feats.append(g2)
        out["features"] = feats
    else:
        # plain geometry
        out["coordinates"] = _tx_coords(gj.get("coordinates", []))
    return out


def safe_fields_from_properties(gj: dict, limit=6):
    # pick up to N property keys for tooltip
    keys = []
    for ft in gj.get("features", []):
        props = ft.get("properties") or {}
        for k in props.keys():
            if k not in keys:
                keys.append(k)
            if len(keys) >= limit:
                return keys
    return keys


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Indoor map (meters) overlay with Folium CRS.Simple")
    ap.add_argument("--geojson", required=True, help="Path or URL to geojson_map.json")
    ap.add_argument("--floor_info", required=True, help="Path or URL to floor_info.json")
    ap.add_argument("--floor_img", required=True, help="Path or URL to floor_image.png")
    ap.add_argument("--points_csv", action="append", default=[],
                    help="CSV with columns x,y[,name] (meters). Can repeat.")
    ap.add_argument("--lines_csv", action="append", default=[],
                    help="CSV with columns track_id,x,y (meters). Can repeat.")
    ap.add_argument("--heatmap_csv", action="append", default=[],
                    help="CSV with columns x,y[,weight] (meters). Can repeat.")
    ap.add_argument("--style_field", default="", help="GeoJSON properties key for coloring (optional)")
    ap.add_argument("--out_html", default="indoor_map.html")
    args = ap.parse_args()

    # --- load meta & transform ---
    floor = _read_json(args.floor_info)
    map_w_m = float(floor["map_info"]["width"])
    map_h_m = float(floor["map_info"]["height"])

    gj = _read_json(args.geojson)
    gj_simple = transform_geojson_m_to_simple(gj, map_h_m)
    tooltip_fields = safe_fields_from_properties(gj, limit=6)

    # --- prepare map ---
    # CRS.Simple: coordinates are arbitrary plane units; we'll use meters with (0,0) at top-left after flip
    m = folium.Map(location=[map_h_m / 2, map_w_m / 2], zoom_start=0, tiles=None, crs="Simple", control_scale=True)

    # floor image overlay
    img_url = _to_raw(args.floor_img) if _is_url(args.floor_img) else args.floor_img
    # bounds = [[lat_min, lon_min], [lat_max, lon_max]] i.e., [[0,0],[map_h,map_w]]
    bounds = [[0, 0], [map_h_m, map_w_m]]
    folium.raster_layers.ImageOverlay(
        name="Floor image",
        image=img_url,
        bounds=bounds,
        opacity=1.0,
        interactive=False,
        cross_origin=True
    ).add_to(m)

    # fit view
    m.fit_bounds(bounds)

    # --- style function for geojson ---
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    def style_function(feat):
        props = feat.get("properties") or {}
        color = "#000000"
        if args.style_field and args.style_field in props:
            try:
                idx = hash(str(props[args.style_field])) % len(palette)
                color = palette[idx]
            except Exception:
                pass
        gtype = (feat.get("geometry") or {}).get("type", "")
        if gtype in ("Polygon", "MultiPolygon"):
            return {"fillOpacity": 0.0, "color": color, "weight": 1.2}
        elif gtype in ("LineString", "MultiLineString"):
            return {"color": color, "weight": 2.0}
        else:
            return {"color": color, "weight": 2.0}

    # --- add transformed GeoJSON layer ---
    gj_layer = folium.GeoJson(
        gj_simple,
        name="GeoJSON features",
        style_function=style_function,
        tooltip=GeoJsonTooltip(fields=tooltip_fields) if tooltip_fields else None
    )
    gj_layer.add_to(m)

    # --- add many points (one or multiple CSVs) ---
    for pcsv in args.points_csv:
        df = pd.read_csv(pcsv)
        grp = folium.FeatureGroup(name=f"Points: {os.path.basename(pcsv)}", show=True)
        cluster = MarkerCluster(name=f"Cluster: {os.path.basename(pcsv)}", show=True)
        for _, r in df.iterrows():
            x, y = float(r["x"]), float(r["y"])
            lon, lat = xy_m_to_leaflet_xy(x, y, map_h_m)
            name = str(r.get("name", ""))
            folium.CircleMarker(
                location=[lat, lon], radius=3, weight=1, fill=True, fill_opacity=0.8,
                popup=name or None
            ).add_to(grp)
            folium.Marker(location=[lat, lon], tooltip=name or None).add_to(cluster)
        grp.add_to(m);
        cluster.add_to(m)

    # --- add polylines by track (one or multiple CSVs) ---
    for lcsv in args.lines_csv:
        df = pd.read_csv(lcsv)
        if not {"track_id", "x", "y"}.issubset(set(df.columns)):
            raise ValueError(f"{lcsv} needs columns: track_id,x,y")
        layer = folium.FeatureGroup(name=f"Lines: {os.path.basename(lcsv)}", show=True)
        for tid, g in df.groupby("track_id"):
            coords = [xy_m_to_leaflet_xy(float(x), float(y), map_h_m)[::-1] for x, y in zip(g["x"], g["y"])]
            # coords need [lat, lon]
            folium.PolyLine(locations=coords, weight=2, opacity=0.9, tooltip=f"track {tid}").add_to(layer)
        layer.add_to(m)

    # --- heatmaps (one or multiple CSVs) ---
    for hcsv in args.heatmap_csv:
        df = pd.read_csv(hcsv)
        weights = df["weight"].values.tolist() if "weight" in df.columns else None
        pts = []
        for _, r in df.iterrows():
            lon, lat = xy_m_to_leaflet_xy(float(r["x"]), float(r["y"]), map_h_m)
            pts.append([lat, lon] if weights is None else [lat, lon, float(r["weight"])])
        HeatMap(pts, name=f"Heat: {os.path.basename(hcsv)}", radius=12, blur=18, max_zoom=22).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(args.out_html)
    print(f"✅ Saved {args.out_html}")
    print("打开后可切换图层、缩放、查看 tooltip。")


if __name__ == "__main__":
    main()


# python folium_indoor.py --geojson https://github.com/location-competition/indoor-location-competition-20/blob/master/data/site1/B1/geojson_map.json --floor_info https://github.com/location-competition/indoor-location-competition-20/blob/master/data/site1/B1/floor_info.json --floor_img https://github.com/location-competition/indoor-location-competition-20/blob/master/data/site1/B1/floor_image.png --out_html b1_folium.html
