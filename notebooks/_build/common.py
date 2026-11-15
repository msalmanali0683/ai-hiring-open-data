"""Shared pieces for building the Kaggle notebooks NB2-NB6 as nbformat-4 JSON.

Usage: python build_all.py   (writes ../02-*.ipynb ... ../06-*.ipynb and validates every code cell)
Nothing here needs third-party packages.
"""
import ast
import json


class NB:
    def __init__(self, gpu=False, internet=True):
        self.cells = []
        self.gpu = gpu
        self.internet = internet

    def md(self, s):
        self.cells.append(("markdown", s.strip("\n")))

    def code(self, s):
        self.cells.append(("code", s.strip("\n")))

    def validate(self):
        """ast.parse every code cell; raise with the cell number on failure."""
        n = 0
        for i, (kind, src) in enumerate(self.cells):
            if kind != "code":
                continue
            n += 1
            try:
                ast.parse(src)
            except SyntaxError as e:
                lines = src.splitlines()
                ctx = lines[e.lineno - 1] if e.lineno and e.lineno <= len(lines) else ""
                raise SyntaxError(f"cell {i} (code cell {n}) line {e.lineno}: {e.msg}\n    {ctx}") from e
        return n

    def write(self, path):
        n_code = self.validate()
        nb = {
            "cells": [
                ({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}
                 if kind == "markdown" else
                 {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": src.splitlines(keepends=True)})
                for kind, src in self.cells
            ],
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"},
                "kaggle": {"accelerator": "nvidiaTeslaT4" if self.gpu else "none",
                           "isInternetEnabled": self.internet, "isGpuEnabled": self.gpu},
            },
            "nbformat": 4, "nbformat_minor": 5,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        return len(self.cells), n_code


# ---------------------------------------------------------------- shared cell code
# Pip helper: plain Python (no "!pip" magics) so every cell passes ast.parse.
PIP = r'''
import subprocess, sys
def pip(*pkgs):
    """Quiet pip install inside Kaggle (needs Settings -> Internet: On)."""
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs],
                       capture_output=True, text=True)
    print("pip install", " ".join(pkgs), "->", "ok" if r.returncode == 0 else "FAILED\n" + r.stderr[-1500:])
'''

FIND = r'''
import glob, os
def find_file(name, prefer=None, required=True):
    """Find a file anywhere under /kaggle/input (Kaggle mounts datasets at different depths)."""
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True), key=len)
    if prefer:
        hits = [h for h in hits if f"/{prefer}/" in h] or hits
    if not hits:
        if required:
            raise FileNotFoundError(f"'{name}' not found under /kaggle/input. "
                                    f"Add the dataset{' ' + repr(prefer) if prefer else ''} via 'Add Input'.")
        return None
    return hits[0]

def find_dir(dataset, marker, required=True):
    """Folder of input dataset `dataset`, identified by a file it must contain."""
    p = find_file(marker, prefer=dataset, required=required)
    return os.path.dirname(p) if p else None
'''

SEEDS = r'''
import random
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except ImportError:
    pass
'''

OUTPUT_LIST = r'''
# Final cell — list every output file
rows = []
for root, _, fs in os.walk(OUT):
    if "/." in root:
        continue
    for f in fs:
        p = os.path.join(root, f)
        rows.append({"file": os.path.relpath(p, OUT), "MB": round(os.path.getsize(p) / 1e6, 2)})
outputs = pd.DataFrame(rows).sort_values("file") if rows else pd.DataFrame(columns=["file", "MB"])
outputs.to_csv(f"{OUT}/outputs_manifest.csv", index=False)
print(f"{len(outputs)} files, {outputs['MB'].sum():.1f} MB in {OUT}")
print(outputs.to_string(index=False))
'''

VERSIONS = r'''
def save_versions(pkgs, path):
    import importlib
    lines = [f"python {sys.version.split()[0]}"]
    for p in pkgs:
        try:
            m = importlib.import_module(p)
            lines.append(f"{p} {getattr(m, '__version__', '?')}")
        except Exception:
            lines.append(f"{p} (not installed)")
    open(path, "w").write("\n".join(lines) + "\n")
    print(" | ".join(lines))
'''

# Wild cluster restricted (WCR) bootstrap, numpy only. Replaces the `wildboottest` package, which fails on
# Kaggle (installs as 0.0.0; UFuncTypeError on every call). Needs the pyfixest wrapper in NB3/NB5, which stores
# (formula, data, cluster column) on each fit as `fit._jb`.
WCR = r'''
import re as _re

def _num(s):
    try:
        return float(s)
    except Exception:
        return s

def _design(fml, data):
    """Build y, X (dict name -> array) and FE keys from a pyfixest-style formula: y ~ a + b:c + i(v, w, ref=r) | fe1 + f^g."""
    lhs, rest = fml.split("~", 1)
    rhs, fe = (rest.split("|", 1) + [""])[:2]
    y = data[lhs.strip()].astype(float).to_numpy()
    X = {}
    for term in [t.strip() for t in _re.split(r"\+(?![^(]*\))", rhs) if t.strip()]:
        m = _re.match(r"i\(\s*(\w+)\s*,\s*(\w+)\s*,\s*ref\s*=\s*([^)]+)\)$", term)
        if m:
            v, w, ref = m.group(1), m.group(2), _num(m.group(3).strip())
            wv = data[w].astype(float).to_numpy()
            for L in sorted(data[v].dropna().unique()):
                if _num(L) == ref:
                    continue
                X[f"{v}::{L}:{w}"] = (data[v] == L).to_numpy().astype(float) * wv
            continue
        if not _re.fullmatch(r"[\w:]+", term):
            raise ValueError(f"term not supported by the bootstrap: {term}")
        cols = term.split(":")
        X[term] = np.prod([data[c].astype(float).to_numpy() for c in cols], axis=0)
    fes = []
    for f in [t.strip() for t in fe.split("+") if t.strip()]:
        parts = f.split("^")
        fes.append(data[parts].astype(str).agg("|".join, axis=1).to_numpy() if len(parts) > 1 else data[f].astype(str).to_numpy())
    return y, X, fes

def _demean(M, fes, tol=1e-9, it=500):
    if not fes:
        return M - M.mean(0)
    M = M.copy()
    codes = [pd.factorize(f)[0] for f in fes]
    for _ in range(it):
        delta = 0.0
        for c in codes:
            cnt = np.bincount(c)
            means = np.vstack([np.bincount(c, weights=M[:, j]) / cnt for j in range(M.shape[1])]).T
            M -= means[c]
            delta = max(delta, np.abs(means).max())
        if delta < tol:
            break
    return M

def _match(names, param):
    if param in names:
        return names.index(param)
    def key(n):
        return tuple(_num(p) for p in _re.split(r"::|:", n))
    for i, n in enumerate(names):
        if key(n) == key(param):
            return i
    raise KeyError(f"{param} not in design {names[:8]}")

def wcr_boot_p(fml, data, cluster, param, reps=9999, weights="rademacher", seed=42, batch=500):
    """WCR bootstrap p-value (null imposed) for one coefficient, CRV1 t-statistic, Rademacher or Webb weights."""
    y, X, fes = _design(fml, data)
    names = list(X)
    g_raw = data[cluster].to_numpy()
    M = np.column_stack([y] + [X[n] for n in names])
    ok = np.isfinite(M).all(1) & pd.notna(g_raw)
    for f in fes:
        ok &= pd.notna(f) & (f != "nan")
    M, g_raw, fes = M[ok], g_raw[ok], [f[ok] for f in fes]
    M = _demean(M, fes)
    y, Xm = M[:, 0], M[:, 1:]
    keep = Xm.std(0) > 1e-10                           # drop columns absorbed by the FEs
    names = [n for n, k in zip(names, keep) if k]; Xm = Xm[:, keep]
    j = _match(names, param)
    g = pd.factorize(g_raw)[0]; G = g.max() + 1; n, k = Xm.shape
    if G < 3:
        raise ValueError("fewer than 3 clusters")
    A = np.linalg.pinv(Xm)                             # k x n
    c = G / (G - 1) * (n - 1) / max(n - k, 1)
    H = np.zeros((G, n)); H[g, np.arange(n)] = 1.0     # cluster indicator matrix (sums within clusters)
    def tstats(Y):                                     # Y: n x B
        B_ = A @ Y; E = Y - Xm @ B_
        S = H @ (A[j][:, None] * E)
        return B_[j] / np.sqrt(c * (S ** 2).sum(0))
    t0 = tstats(y[:, None])[0]
    Xr = np.delete(Xm, j, axis=1)
    fr = Xr @ np.linalg.lstsq(Xr, y, rcond=None)[0] if Xr.shape[1] else np.zeros(n)
    ur = y - fr
    rng = np.random.default_rng(seed); hits = 0; done = 0
    webb = np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])
    while done < reps:
        b = min(batch, reps - done)
        W = rng.choice(webb, (G, b)) if weights == "webb" else rng.choice([-1.0, 1.0], (G, b))
        ts = tstats(fr[:, None] + ur[:, None] * W[g])
        hits += (np.abs(ts) >= abs(t0)).sum(); done += b
    return float(hits / reps)

def boot_p(fit, param, weights="rademacher", reps=9999):
    """Wild cluster bootstrap p-value for `param` of a pyfixest fit made through the wrapper (fit._jb)."""
    try:
        fml, data, cluster = fit._jb
        if cluster is None:
            raise ValueError("no single cluster variable")
        return wcr_boot_p(fml, data, cluster, param, reps=reps, weights=weights)
    except Exception as e:
        print(f"  wild bootstrap failed for {param}: {type(e).__name__} {str(e)[:120]}")
        return np.nan

# self-test: no effect -> p roughly uniform; real effect -> p small
_r = np.random.default_rng(0); _n = 40 * 30
_df = pd.DataFrame({"g": np.repeat(np.arange(40), 30), "t": np.tile(np.arange(30), 40)})
_df["treat"] = (_df["g"] < 12).astype(float); _df["post"] = (_df["t"] >= 15).astype(float)
_df["u"] = _r.normal(size=_n) + np.repeat(_r.normal(size=40), 30)
_df["y0"] = _df["u"]; _df["y1"] = _df["u"] + 0.8 * _df["post"] * _df["treat"]
print("WCR self-test p (null, effect):",
      round(wcr_boot_p("y0 ~ post:treat | g + t", _df, "g", "post:treat", reps=999), 3),
      round(wcr_boot_p("y1 ~ post:treat | g + t", _df, "g", "post:treat", reps=999), 3))
'''
