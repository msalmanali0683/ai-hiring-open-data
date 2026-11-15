"""Draw Figure 6 of the paper from the NB5 outputs (country_change.csv, section174_series.csv,
synth_paths.csv). Standard library only. Writes fig6.html and fig6.svg; open fig6.html in a
browser and print or screenshot it to obtain a PDF or PNG."""
import csv, math, datetime as dt, pathlib, sys, html

if len(sys.argv) != 3:
    sys.exit("usage: python redraw_figure6.py <folder with NB5 outputs> <output folder>")
D = pathlib.Path(sys.argv[1])   # folder with country_change.csv, section174_series.csv, synth_paths.csv
OUT = pathlib.Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

cc = list(csv.DictReader(open(D / "country_change.csv", encoding="utf-8")))
exp = {}
for r in cc:
    exp.setdefault(r["display_name"], float(r["exp_main"]) if r["exp_main"] not in ("", "nan") else None)
cols = sorted(exp, key=lambda k: (exp[k] is None, -(exp[k] or 0)))
rows = [c for c in ["US", "GB", "CA", "DE", "FR", "AU"] if any(r["jobcountry"] == c for r in cc)]
val = {}
for r in cc:
    if r["pct_change"] not in ("", "nan"):
        val.setdefault((r["jobcountry"], r["display_name"]), []).append(float(r["pct_change"]))
val = {k: sum(v) / len(v) for k, v in val.items()}
a = sorted(abs(v) for k, v in val.items() if k[0] in rows and k[1] in cols)
pos = 0.95 * (len(a) - 1); lo = int(math.floor(pos))
lim = a[lo] + (a[min(lo + 1, len(a) - 1)] - a[lo]) * (pos - lo)

RDBU = ["#67001f", "#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#f7f7f7",
        "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#053061"]
def hexrgb(h): return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))
def cmap(x):
    x = min(max(x, 0.0), 1.0) * (len(RDBU) - 1); i = min(int(x), len(RDBU) - 2); f = x - i
    c0, c1 = hexrgb(RDBU[i]), hexrgb(RDBU[i + 1])
    return "#%02x%02x%02x" % tuple(round(c0[j] + (c1[j] - c0[j]) * f) for j in range(3))

def dates(rs): return [dt.date.fromisoformat(r["date"][:10]) for r in rs]
s174 = list(csv.DictReader(open(D / "section174_series.csv", encoding="utf-8")))
sp = [r for r in csv.DictReader(open(D / "synth_paths.csv", encoding="utf-8"))
      if r["treatment"] == "Dec 2022" and r["pool"].startswith("A")]

W, H = 1500, 1480
FONT = "Arial, Helvetica, sans-serif"
svg = []
def T(x, y, s, size=15, anchor="start", rot=None, color="#000"):
    tr = f' transform="rotate({rot} {x:.1f} {y:.1f})"' if rot is not None else ""
    svg.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
               f'fill="{color}" font-family="{FONT}"{tr}>{html.escape(s)}</text>')
def LN(x1, y1, x2, y2, color="#000", w=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    svg.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{w}"{d}/>')
def poly(pts, color, w, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    svg.append(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{w}" stroke-linejoin="round"{d}/>')

# panel (a)
ax0, ay0, cw, rh = 70, 60, 24.0, 40.0
T(20, 32, "(a) Change in postings (%), 2025–26 versus Feb 2020 to Nov 2022, occupations from most to least exposed", 19)
for i, c in enumerate(rows):
    T(ax0 - 10, ay0 + rh * i + rh / 2 + 5, c, 15, "end")
    for j, o in enumerate(cols):
        v = val.get((c, o))
        if v is None:
            continue
        svg.append(f'<rect x="{ax0 + cw * j:.2f}" y="{ay0 + rh * i:.2f}" width="{cw + 0.4:.2f}" '
                   f'height="{rh + 0.4:.2f}" fill="{cmap((v + lim) / (2 * lim))}"/>')
hb = ay0 + rh * len(rows)
svg.append(f'<rect x="{ax0}" y="{ay0}" width="{cw * len(cols):.1f}" height="{rh * len(rows):.1f}" fill="none" stroke="#000" stroke-width="0.8"/>')
for j, o in enumerate(cols):
    x = ax0 + cw * j + cw / 2
    LN(x, hb, x, hb + 5, w=0.8)
    T(x + 4.5, hb + 9, o, 12.5, "end", rot=-90)
cbx, cby, cbw, cbh = ax0 + cw * len(cols) + 22, ay0, 18, rh * len(rows)
for k in range(120):
    svg.append(f'<rect x="{cbx}" y="{cby + cbh * k / 120:.2f}" width="{cbw}" height="{cbh / 120 + 0.6:.2f}" fill="{cmap(1 - (k + 0.5) / 120)}"/>')
svg.append(f'<rect x="{cbx}" y="{cby}" width="{cbw}" height="{cbh}" fill="none" stroke="#000" stroke-width="0.8"/>')
for t in range(-int(lim // 20) * 20, int(lim) + 1, 20):
    y = cby + cbh * (1 - (t + lim) / (2 * lim))
    LN(cbx + cbw, y, cbx + cbw + 5, y, w=0.8)
    T(cbx + cbw + 8, y + 5, ("−" if t < 0 else "") + str(abs(t)), 13)

D0, D1 = dt.date(2020, 1, 1), dt.date(2026, 12, 31)
def panel(px, py, pw, ph, title, series, ymin, ymax, ystep, vlines, legend, legcols):
    def X(d): return px + pw * (d - D0).days / (D1 - D0).days
    def Y(v): return py + ph * (1 - (v - ymin) / (ymax - ymin))
    T(px - 60, py - 22, title, 18)
    for yv in range(int(math.ceil(ymin / ystep) * ystep), int(ymax) + 1, ystep):
        LN(px, Y(yv), px + pw, Y(yv), "#e3e3e3", 0.8)
        LN(px - 5, Y(yv), px, Y(yv), w=0.8); T(px - 9, Y(yv) + 5, str(yv), 13, "end")
    for yr in range(2020, 2027):
        d = dt.date(yr, 1, 1); LN(X(d), py, X(d), py + ph, "#e3e3e3", 0.8)
        LN(X(d), py + ph, X(d), py + ph + 5, w=0.8); T(X(d), py + ph + 22, str(yr), 13, "middle")
    for d, color, w, dash in vlines:
        LN(X(d), py, X(d), py + ph, color, w, dash)
    for name, ds, vs, color, w, dash in series:
        poly([(X(d), Y(v)) for d, v in zip(ds, vs)], color, w, dash)
    LN(px, py + ph, px + pw, py + ph, w=0.9); LN(px, py, px, py + ph, w=0.9)
    T(px - 50, py + ph / 2, "Index, Feb 2020 = 100", 13, "middle", rot=-90)
    colw = 140 if legcols == 1 else 70
    lx, ly = px + pw - 10 - colw * legcols, py + 8
    for k, (name, color, w, dash) in enumerate(legend):
        cx, cy = lx + colw * (k % legcols), ly + 22 * (k // legcols) + 10
        LN(cx, cy, cx + 26, cy, color, w, dash); T(cx + 32, cy + 5, name, 13)

OK = {"AU": "#E69F00", "CA": "#56B4E9", "DE": "#009E73", "FR": "#F0E442", "GB": "#0072B2"}
CHATGPT, S174 = dt.date(2022, 11, 30), dt.date(2022, 1, 1)
ds = dates(s174)
ser, leg = [], []
for c in ["AU", "CA", "DE", "FR", "GB", "US"]:
    color, w = ("#000", 3.2) if c == "US" else (OK[c], 1.6)
    ser.append((c, ds, [float(r[c]) for r in s174], color, w, None)); leg.append((c, color, w, None))
allv = [v for s in ser for v in s[2]]
py0 = hb + 300
panel(100, py0, 590, 430, "(b) Software development postings", ser,
      25 * math.floor(min(allv) / 25), 25 * math.ceil(max(allv) / 25) + 25, 50,
      [(S174, "#777", 1.3, "3,4"), (CHATGPT, "#000", 1.3, None)], leg, 3)
dsp = dates(sp)
us = [math.exp(float(r["us_software"])) for r in sp]; sy = [math.exp(float(r["synthetic"])) for r in sp]
panel(860, py0, 590, 430, "(c) Synthetic control (donor pool A)",
      [("US software", dsp, us, "#000", 2.4, None), ("synthetic control", dsp, sy, "#0072B2", 2.4, "9,5")],
      25 * math.floor(min(us + sy) / 25), 25 * math.ceil(max(us + sy) / 25) + 25, 25,
      [(CHATGPT, "#000", 1.3, None)],
      [("US software", "#000", 2.4, None), ("synthetic control", "#0072B2", 2.4, "9,5")], 1)
H = int(py0 + 430 + 85)
T(W / 2, py0 + 430 + 64, "Solid vertical line marks the release of ChatGPT (30 November 2022). Dotted vertical line in (b) marks the start of the Section 174 change (January 2022).", 13.5, "middle", color="#333")

doc = (f'<!doctype html><html><head><meta charset="utf-8"><style>@page{{size:{W}px {H}px;margin:0}}'
       f'html,body{{margin:0;padding:0;background:#fff;overflow:hidden}}svg{{display:block}}</style></head><body>'
       f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
       f'<rect width="{W}" height="{H}" fill="#fff"/>' + "".join(svg) + "</svg></body></html>")
(OUT / "fig6.html").write_text(doc, encoding="utf-8")
svg_only = doc[doc.index("<svg"):doc.index("</svg>") + 6]
(OUT / "fig6.svg").write_text(svg_only, encoding="utf-8")
print("lim", round(lim, 1), "cols", len(cols), "rows", rows, "synth points", len(sp), "label top", hb + 9, "panels top", py0 - 40)
