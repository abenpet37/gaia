import glob, json, os, io, base64, numpy as np, rasterio, geopandas as gpd
from rasterio.warp import transform
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPoly
from shapely.geometry import box, shape

ROOT = r"c:/Users/abene/OneDrive/Documents/GitHub/nasa-proj/"
OUT = r"C:/Users/abene/AppData/Local/Temp/claude/bb2dc992b56aac75/cc61ab74-38c9-4373-b17d-784ee0479d4c/scratchpad/pres/footprints.png"
polys = {"A": [], "D": []}
for f in sorted(glob.glob(ROOT + "manipulate/output/*_los_displacement_color.tif")):
    with rasterio.open(f) as s:
        l, b, r, t = s.bounds
        xs, ys = transform(s.crs, "EPSG:4326", [l, r, r, l], [t, t, b, b])
    polys["A" if "_A_" in os.path.basename(f) else "D"].append(list(zip(xs, ys)))
print({k: len(v) for k, v in polys.items()})
allp = [p for v in polys.values() for p in v]
lons = [x for p in allp for x, _ in p]; lats = [y for p in allp for _, y in p]
print("extent", min(lons), max(lons), min(lats), max(lats))

W, H = 440, 700                      # display size on the slide (px); render at 2x
x0, x1, y0, y1 = min(lons) - 1.0, max(lons) + 1.0, min(lats) - 1.0, max(lats) + 1.0
# match the box aspect (equal degrees on both axes)
need_w = (y1 - y0) * W / H
if need_w > (x1 - x0):
    c = (x0 + x1) / 2; x0, x1 = c - need_w / 2, c + need_w / 2
fig = plt.figure(figsize=(W * 2 / 200, H * 2 / 200), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("auto")
ax.set_facecolor("#0d1a31"); fig.patch.set_facecolor("#0d1a31"); ax.axis("off")

import pyogrio, sysconfig
cands = glob.glob(os.path.dirname(pyogrio.__file__) + "/tests/fixtures/naturalearth_lowres/naturalearth_lowres.shp")
land = gpd.read_file(cands[0]).clip(box(x0 - 1, y0 - 1, x1 + 1, y1 + 1))
land.plot(ax=ax, facecolor="#1b2c4d", edgecolor="#4a6ba3", linewidth=0.8)
# graticule
for g in range(int(np.ceil(x0 / 5) * 5), int(x1) + 1, 5): ax.plot([g, g], [y0, y1], color="#ffffff", alpha=0.07, lw=0.6)
for g in range(int(np.ceil(y0 / 5) * 5), int(y1) + 1, 5): ax.plot([x0, x1], [g, g], color="#ffffff", alpha=0.07, lw=0.6)
# plate boundaries (amber)
pl = json.load(open(ROOT + "map/tectonic_plates.json", encoding="utf-8"))
view = box(x0, y0, x1, y1)
for ft in pl["features"]:
    g = shape(ft["geometry"])
    if g.intersects(view):
        gg = g.intersection(view)
        for part in (gg.geoms if hasattr(gg, "geoms") else [gg]):
            xs, ys = part.xy; ax.plot(xs, ys, color="#f2a33a", lw=1.6, alpha=0.9, solid_capstyle="round")
# footprints
for key, col in (("D", "#56b4e9"), ("A", "#3cc8be")):
    for p in polys[key]:
        ax.add_patch(MPoly(p, closed=True, facecolor=col+"22", edgecolor=col+"cc", lw=0.7))
fig.savefig(OUT, dpi=200, facecolor=fig.get_facecolor())
print("saved", OUT)
