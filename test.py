#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot indoor floor GeoJSON over floor image (Indoor Location Competition 2.0)
- Reads floor_info.json (map width/height in meters)
- Reads floor_image.png (pixel size)
- Reads geojson_map.json (vector features)
- Converts (x[m], y[m]) --> pixel coords, overlays on image
- Outputs b1_overlay.png

Usage:
  python plot_indoor_geojson.py \
    --geojson https://github.com/location-competition/indoor-location-competition-20/blob/master/data/site1/B1/geojson_map.json \
    --floor_info https://github.com/location-competition/indoor-location-competition-20/blob/master/data/site1/B1/floor_info.json \
    --floor_img https://github.com/location-competition/indoor-location-competition-20/blob/master/data/site1/B1/floor_image.png
"""

import io, os, json, argparse, math
from urllib.request import urlopen
from urllib.parse import urlparse
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import LineCollection

def _is_url(s: str) -> bool:
    try:
        p = urlparse(s)
        return p.scheme in ("http", "https")
    except Exception:
        return False

def _read_json(path_or_url: str):
    if _is_url(path_or_url):
        # Use GitHub's "raw" if user pasted a web UI URL
        if "github.com" in path_or_url and "/blob/" in path_or_url:
            path_or_url = path_or_url.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/","/")
        with urlopen(path_or_url) as r:
            return json.load(io.TextIOWrapper(r, encoding="utf-8"))
    else:
        with open(path_or_url, "r", encoding="utf-8") as f:
            return json.load(f)

def _read_image(path_or_url: str) -> Image.Image:
    if _is_url(path_or_url):
        if "github.com" in path_or_url and "/blob/" in path_or_url:
            path_or_url = path_or_url.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/","/")
        with urlopen(path_or_url) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGBA")
    else:
        return Image.open(path_or_url).convert("RGBA")

def meters_to_pixels(xm: float, ym: float, img_w: int, img_h: int, map_w_m: float, map_h_m: float):
    """
    Dataset uses local map coordinates in meters.
    Image pixel origin is at top-left; map origin usually at bottom-left → need y flip.
    """
    sx = img_w / map_w_m
    sy = img_h / map_h_m
    px = xm * sx
    py = img_h - (ym * sy)  # flip Y
    return px, py

def transform_coords(coords, img_w, img_h, map_w_m, map_h_m):
    """Recursively transform nested coordinate arrays."""
    if isinstance(coords[0], (float, int)):  # [x, y]
        x, y = coords[:2]
        return meters_to_pixels(x, y, img_w, img_h, map_w_m, map_h_m)
    else:
        return [transform_coords(c, img_w, img_h, map_w_m, map_h_m) for c in coords]

def draw_geojson(ax, gj, img_w, img_h, map_w_m, map_h_m):
    polys_drawn = lines_drawn = points_drawn = 0

    for feat in gj.get("features", []):
        geom = feat.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if not gtype or coords is None:
            continue

        try:
            if gtype == "Polygon":
                rings = transform_coords(coords, img_w, img_h, map_w_m, map_h_m)
                # exterior ring first
                exterior = rings[0]
                poly = MplPolygon(exterior, closed=True,
                                  facecolor=(0, 0, 0, 0),  # transparent fill
                                  edgecolor="black", linewidth=0.8)
                ax.add_patch(poly)
                # holes (if any)
                for hole in rings[1:]:
                    poly_hole = MplPolygon(hole, closed=True,
                                           facecolor=(0, 0, 0, 0),
                                           edgecolor="black", linewidth=0.5, linestyle="--")
                    ax.add_patch(poly_hole)
                polys_drawn += 1

            elif gtype == "MultiPolygon":
                for poly_coords in coords:
                    rings = transform_coords(poly_coords, img_w, img_h, map_w_m, map_h_m)
                    exterior = rings[0]
                    poly = MplPolygon(exterior, closed=True,
                                      facecolor=(0, 0, 0, 0),
                                      edgecolor="black", linewidth=0.8)
                    ax.add_patch(poly)
                    for hole in rings[1:]:
                        poly_hole = MplPolygon(hole, closed=True,
                                               facecolor=(0, 0, 0, 0),
                                               edgecolor="black", linewidth=0.5, linestyle="--")
                        ax.add_patch(poly_hole)
                    polys_drawn += 1

            elif gtype == "LineString":
                line = transform_coords(coords, img_w, img_h, map_w_m, map_h_m)
                lc = LineCollection([line], linewidths=1.2)
                ax.add_collection(lc)
                lines_drawn += 1

            elif gtype == "MultiLineString":
                segs = [transform_coords(seg, img_w, img_h, map_w_m, map_h_m) for seg in coords]
                lc = LineCollection(segs, linewidths=1.0)
                ax.add_collection(lc)
                lines_drawn += len(segs)

            elif gtype == "Point":
                x, y = transform_coords(coords, img_w, img_h, map_w_m, map_h_m)
                ax.plot(x, y, "o", markersize=2)
                points_drawn += 1

            elif gtype == "MultiPoint":
                pts = [transform_coords(pt, img_w, img_h, map_w_m, map_h_m) for pt in coords]
                xs, ys = zip(*pts)
                ax.plot(xs, ys, "o", markersize=2)
                points_drawn += len(pts)

            else:
                # Unsupported type
                pass

        except Exception as e:
            print(f"[warn] failed to draw feature {gtype}: {e}")

    return polys_drawn, lines_drawn, points_drawn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geojson", required=True, help="Path or URL to geojson_map.json")
    ap.add_argument("--floor_info", required=True, help="Path or URL to floor_info.json")
    ap.add_argument("--floor_img", required=True, help="Path or URL to floor_image.png")
    ap.add_argument("--out", default="b1_overlay.png")
    args = ap.parse_args()

    # Load inputs
    floor_info = _read_json(args.floor_info)
    gj = _read_json(args.geojson)
    img = _read_image(args.floor_img)

    img_w, img_h = img.size
    map_w_m = float(floor_info["map_info"]["width"])
    map_h_m = float(floor_info["map_info"]["height"])
    print(f"Image px (W×H): {img_w} × {img_h}")
    print(f"Map meters (W×H): {map_w_m:.3f} × {map_h_m:.3f}")

    # Plot
    dpi = 150
    fig_w_in = img_w / dpi
    fig_h_in = img_h / dpi
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    ax = plt.axes([0, 0, 1, 1])  # full-bleed
    ax.imshow(img)
    ax.set_xlim(0, img_w); ax.set_ylim(img_h, 0)  # y axis down
    ax.set_xticks([]); ax.set_yticks([])

    polys, lines, points = draw_geojson(ax, gj, img_w, img_h, map_w_m, map_h_m)

    ax.set_title(f"Indoor Map Overlay (polys={polys}, lines={lines}, points={points})", fontsize=10)
    plt.savefig(args.out, dpi=dpi)
    print(f"✅ Saved: {args.out}")

if __name__ == "__main__":
    main()
# python test.py --geojson https://github.com/location-competition/indoor-location-competition-20/blob/master/data/site1/B1/geojson_map.json --floor_info https://github.com/location-competition/indoor-location-competition-20/blob/master/data/site1/B1/floor_info.json --floor_img https://github.com/location-competition/indoor-location-competition-20/blob/master/data/site1/B1/floor_image.png
