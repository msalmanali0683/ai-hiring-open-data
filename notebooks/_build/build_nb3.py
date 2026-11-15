"""NB3 — 03-event-study."""
from common import NB, PIP, FIND, SEEDS, OUTPUT_LIST, VERSIONS, WCR

HELPERS = r'''
import re

def parse_term(name, var):
    """Event-time value from a pyfixest coefficient name, whatever the naming style:
    'rel_q::-4:high_exposure', 'C(rel_q, contr.treatment(base=-1))[T.-4]:high_exposure', 'rel_q[T.-4]:x'."""
    s = re.sub(r"base\s*=\s*-?\d+(?:\.\d+)?", "", name)
    m = re.search(re.escape(var) + r"\D*?(-?\d+(?:\.\d+)?)", s)
    if not m:
        return None
    v = float(m.group(1))
    return int(v) if v == int(v) else v

def find_coef(names, wanted):
    """Find 'a:b' among coefficient names regardless of the order of the interacted parts."""
    want = set(wanted.split(":"))
    for n in names:
        if set(n.split(":")) == want:
            return n
    for n in names:                      # looser: every wanted part appears in the name
        if all(w in n for w in want):
            return n
    return None

def event_label(k, months_per_bin=3, start_year=2022, start_month=12):
    """Calendar months covered by event bin k (bin 0 starts Dec 2022). k=-3 -> 'Mar–May 2022'."""
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    def shift(n):
        t = start_year * 12 + (start_month - 1) + n
        return t // 12, t % 12
    y0, m0 = shift(k * months_per_bin)
    y1, m1 = shift(k * months_per_bin + months_per_bin - 1)
    if months_per_bin == 1:
        return f"{names[m0]} {y0}"
    return f"{names[m0]} {y0}–{names[m1]} {y1}" if y0 != y1 else f"{names[m0]}–{names[m1]} {y1}"
'''


def build():
    nb = NB(gpu=False, internet=True)
    nb.md(r'''
# NB3 — Event study and difference-in-differences (`03-event-study`) → RQ1
**Role in the paper:** an open-data *check* on the World Bank (−12 %) vs NY Fed (null) dispute — not the headline (`RESEARCH_PLAN.md` §2, §5.1, §6).

| | |
|---|---|
| **Input** | **`jobai-processed`** (`panel_occ_month.csv`, `exposure_table.csv`, `sector_exposure.csv`, `hn_clean.csv`) and **`jobai-raw`** (`fred_series.csv`, `indeed_sector.csv`) |
| **Optional input** | a file `rate_sens.csv` (`soc6, rate_sens`) with a published occupation rate-sensitivity measure (e.g. built from OEWS/CPS 2010–19 or Zens et al. 2020). Without it, a pre-period HN proxy is used and labelled as such. |
| **Output** (`/kaggle/working/`) | `fig1_raw_trends.png`, `fig2_event_study.png`, `figA_event_study_monthly.png`, `figA_indeed_sectors.png`, `event_study_coefs.csv` (+ `_rate_control`, `_poisson`), `event_study_monthly_coefs.csv`, `pretrend_tests.csv`, `post_year_effects.csv`, `table2_did_robustness.csv`, `placebo_tests.csv`, `wild_bootstrap.csv`, `rate_sensitivity.csv`, `indeed_sector_did.csv`, `indeed_sector_event.csv`, `hn_representativeness.csv`, `hn_representativeness_corr.csv` |
| **Kaggle settings** | Accelerator **None (CPU)** · **Internet On** (only for `pip install pyfixest wildboottest`) |
| **Run time** | ~5–15 min (the 9,999-draw wild bootstraps take most of it) |

**Design.** `log(1 + ads)_ot = α_o + γ_t + Σ_k β_k·(High_o × 1[event quarter k]) + ε_ot`, occupation and month fixed effects, SEs clustered by occupation, reference = event quarter −1 (Sep–Nov 2022). Event quarters are 3-month bins **aligned to the treatment month** (bin 0 = Dec 2022–Feb 2023), so the "2022 quarters" the plan asks for are bins −4, −3, −2 = Dec 2021–Feb 2022, Mar–May 2022, Jun–Aug 2022. The table labels every bin with its months.

Nothing here has been run yet; no result is written in this notebook.
''')
    nb.code(PIP + r'''
# Cell 1 — installs, imports, settings
pip("pyfixest", "wildboottest")
''' + FIND + r'''
import re, json, math, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import pyfixest as pf
# pyfixest >= 0.70 rejects object-dtype columns in fixed effects: cast object columns to 'string' on every call
def _pf_strings(df):
    if df is None:
        return df
    obj = [c for c in df.columns if df[c].dtype == object]
    return df.astype({c: "string" for c in obj}) if obj else df
for _name in ("feols", "fepois", "feglm"):
    if hasattr(pf, _name):
        _orig = getattr(pf, _name)
        def _wrapped(*a, _f=_orig, **k):
            if "data" not in k and len(a) > 1:
                k["data"] = a[1]; a = (a[0],) + tuple(a[2:])
            d = k.get("data")
            v = k.get("vcov")
            if d is not None and isinstance(v, dict):   # cluster on integer codes: wildboottest can't read string ids
                v = dict(v)
                for key, cv in v.items():
                    if isinstance(cv, str) and cv in d.columns and not np.issubdtype(d[cv].dtype, np.number):
                        d = d.assign(**{f"{cv}__id": pd.factorize(d[cv])[0]})
                        v[key] = f"{cv}__id"
                k["vcov"] = v
            k["data"] = _pf_strings(d)
            res = _f(*a, **k)
            try:                                        # what boot_p needs to rebuild this model
                fml = a[0] if a else k.get("fml")
                cl = [x for x in (v.values() if isinstance(v, dict) else []) if isinstance(x, str) and "+" not in x]
                res._jb = (fml, k["data"], cl[0] if len(cl) == 1 else None)
            except Exception:
                pass
            return res
        setattr(pf, _name, _wrapped)
''' + SEEDS + VERSIONS + r'''
warnings.filterwarnings("ignore")
OUT = "/kaggle/working"
PROC = find_dir("jobai-processed", "panel_occ_month.csv")
RAW = find_dir("jobai-raw", "fred_series.csv")
print("processed:", PROC, "| raw:", RAW)
BOOT_REPS = 9999
TREAT = "high_exp_hn"             # within-HN split (main: HN has only ~1% population-low ads); population split = robustness
VCOV = {"CRV1": "soc6"}
save_versions(["pandas", "numpy", "pyfixest", "wildboottest", "scipy"], f"{OUT}/versions_nb3.txt")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})
''')
    nb.code("# Cell 2 — helpers (plain Python, unit-tested locally)" + HELPERS + r'''
def tidy(fit):
    t = fit.tidy()
    t.index = [str(i) for i in t.index]
    return t

def get_coef(fit, wanted):
    t = tidy(fit)
    name = find_coef(list(t.index), wanted)
    if name is None:
        raise KeyError(f"{wanted} not among {list(t.index)[:10]}…")
    r = t.loc[name]
    return {"term": name, "estimate": r["Estimate"], "se": r["Std. Error"], "p": r["Pr(>|t|)"],
            "ci_low": r["2.5%"], "ci_high": r["97.5%"], "nobs": getattr(fit, "_N", np.nan)}

def wald_pre(fit, names, n_clusters):
    """Joint test that the listed coefficients are 0 (F with cluster-robust vcov, df2 = G - 1)."""
    allnames = [str(c) for c in getattr(fit, "_coefnames", tidy(fit).index)]
    idx = [allnames.index(n) for n in names]
    b = np.asarray(fit._beta_hat)[idx] if hasattr(fit, "_beta_hat") else tidy(fit).loc[names, "Estimate"].values
    if hasattr(fit, "_vcov"):
        V = np.asarray(fit._vcov)[np.ix_(idx, idx)]
        note = "full cluster-robust vcov"
    else:
        V = np.diag(tidy(fit).loc[names, "Std. Error"].values ** 2)
        note = "diagonal only (vcov not exposed) — treat as approximate"
    W = float(b @ np.linalg.pinv(V) @ b)
    k = len(idx)
    F = W / k
    p = 1 - stats.f.cdf(F, k, max(n_clusters - 1, 1))
    return {"k": k, "F": F, "p": p, "df2": n_clusters - 1, "note": note}

''' + WCR + r'''

print(parse_term("C(rel_q, contr.treatment(base=-1))[T.-4]:high_exposure", "rel_q"),
      parse_term("rel_q::3:high_exposure", "rel_q"), event_label(-3))
''')
    nb.md("## Load the panel and check group sizes")
    nb.code(r'''
# Cell 3 — data
panel = pd.read_csv(f"{PROC}/panel_occ_month.csv", dtype={"soc6": str, "month": str})
panel["date"] = pd.to_datetime(panel["date"])
d_all = panel.dropna(subset=["exp_main"]).copy()
d = d_all.dropna(subset=[TREAT]).copy()
d["rel_q"] = np.floor(d["rel_month"] / 3).astype(int)
d_all["rel_q"] = np.floor(d_all["rel_month"] / 3).astype(int)
G = d["soc6"].nunique()
print(f"main sample: {G} occupations ({d.drop_duplicates('soc6')[TREAT].value_counts().to_dict()}), "
      f"{d['month'].nunique()} months, {len(d):,} rows, {int(d['ads'].sum()):,} ads")
print("event quarters:", d["rel_q"].min(), "→", d["rel_q"].max())
if G < 30:
    print("⚠ fewer than 30 occupation clusters: rely on the wild-bootstrap p-values, not the CRV1 ones.")
''')
    nb.md("## Fig 1 — raw trends, high vs low exposure")
    nb.code(r'''
# Cell 4 — Fig 1
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
for ax, col, title in [(axes[0], "high_exp_hn", "Terciles among HN occupations (main)"),
                       (axes[1], "high_exposure", "Population terciles")]:
    s = panel.dropna(subset=[col]).groupby(["date", col])["ads"].sum().unstack()
    base = s[s.index < "2022-12-01"].mean()
    idx = (100 * s / base).rolling(3, min_periods=1).mean()
    for g, lab, ls in [(1.0, "High exposure", "-"), (0.0, "Low exposure", "--")]:
        if g in idx:
            ax.plot(idx.index, idx[g], ls, label=lab)
    ax.axvline(pd.Timestamp("2022-11-30"), color="k", lw=0.8)
    ax.axvline(pd.Timestamp("2022-03-16"), color="grey", lw=0.6, ls=":")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("HN ads, pre-ChatGPT mean = 100 (3-mo avg)")
axes[0].legend(frameon=False)
axes[0].annotate("first Fed hike", (pd.Timestamp("2022-03-16"), 0.95), xycoords=("data", "axes fraction"),
                 fontsize=7, color="grey")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_raw_trends.png", bbox_inches="tight"); plt.show()
''')
    nb.md(r'''
## Fig 2 — event study (quarterly bins) + joint pre-trend test
The 2022 bins are reported explicitly (NY Fed / EIG critique: the decline started in early 2022). Two pre-trend tests: all pre-period bins, and the 2022 bins only.
''')
    nb.code(r'''
# Cell 5 — event study
fit_es = pf.feols(f"log_ads ~ i(rel_q, {TREAT}, ref=-1) | soc6 + month", data=d, vcov=VCOV)
t = tidy(fit_es)
es = t[t.index.str.contains("rel_q")].copy()
es["rel_q"] = [parse_term(n, "rel_q") for n in es.index]
es = es.dropna(subset=["rel_q"])
es = pd.concat([es, pd.DataFrame({"rel_q": [-1], "Estimate": [0.0], "Std. Error": [0.0], "2.5%": [0.0],
                                  "97.5%": [0.0], "Pr(>|t|)": [np.nan]}, index=["reference"])]).sort_values("rel_q")
es["months"] = es["rel_q"].astype(int).apply(event_label)
es["model"] = "OLS log(1+ads), occupation + month FE"
es.rename(columns={"Estimate": "estimate", "Std. Error": "se", "2.5%": "ci_low", "97.5%": "ci_high",
                   "Pr(>|t|)": "p"})[["rel_q", "months", "estimate", "se", "ci_low", "ci_high", "p", "model"]] \
  .to_csv(f"{OUT}/event_study_coefs.csv", index_label="term")

y2022 = es[es["rel_q"].isin([-4, -3, -2])]
print("2022 bins:\n", y2022[["months", "Estimate", "Std. Error", "Pr(>|t|)"]].round(4).to_string())
pre_names = [n for n, k in zip(es.index, es["rel_q"]) if n != "reference" and k <= -2]
names_2022 = [n for n, k in zip(es.index, es["rel_q"]) if n != "reference" and k in (-4, -3, -2)]
pt = [dict(test="all pre-period bins = 0", **wald_pre(fit_es, pre_names, G)),
      dict(test="2022 bins (Dec21–Aug22) = 0", **wald_pre(fit_es, names_2022, G))]
pd.DataFrame(pt).to_csv(f"{OUT}/pretrend_tests.csv", index=False)
print(pd.DataFrame(pt).round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 3.8))
x = es["rel_q"].values
ax.axhline(0, color="k", lw=0.6)
ax.axvline(-0.5, color="k", lw=0.8)
ax.axvspan(-4.5, -1.5, color="orange", alpha=0.12, label="2022 bins (pre-ChatGPT)")
ax.errorbar(x, es["Estimate"], yerr=[es["Estimate"] - es["2.5%"], es["97.5%"] - es["Estimate"]],
            fmt="o", ms=3.5, capsize=2, lw=1)
ticks = [k for k in x if k % 4 == 0]
ax.set_xticks(ticks)
ax.set_xticklabels([event_label(int(k)) for k in ticks], rotation=45, ha="right", fontsize=7)
ax.set_ylabel("Δ log(1+ads), high vs low exposure")
ax.set_title("Event study: high- vs low-exposure occupations (ref. Sep–Nov 2022)", fontsize=10)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_event_study.png", bbox_inches="tight"); plt.show()
''')
    nb.md("## Simple DiD and separate post-year effects (2023–2026 rebound test)")
    nb.code(r'''
# Cell 6 — DiD + post-year effects
fit_did = pf.feols(f"log_ads ~ post:{TREAT} | soc6 + month", data=d, vcov=VCOV)
main = get_coef(fit_did, f"post:{TREAT}")
print("DiD (log points):", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in main.items()})

d["post_year"] = np.where(d["rel_month"] >= 0, d["date"].dt.year, 0)
fit_y = pf.feols(f"log_ads ~ i(post_year, {TREAT}, ref=0) | soc6 + month", data=d, vcov=VCOV)
ty = tidy(fit_y)
py = ty[ty.index.str.contains("post_year")].copy()
py["year"] = [parse_term(n, "post_year") for n in py.index]
py = py.rename(columns={"Estimate": "estimate", "Std. Error": "se", "2.5%": "ci_low", "97.5%": "ci_high",
                        "Pr(>|t|)": "p"})
py["note"] = np.where(py["year"] == 2022, "Dec 2022 only", "")
# rebound test: is the 2025-26 effect different from the 2023-24 effect?
try:
    n = {int(y): nm for nm, y in zip(py.index, py["year"])}
    allnames = [str(c) for c in fit_y._coefnames]
    V = np.asarray(fit_y._vcov); b = np.asarray(fit_y._beta_hat)
    c = np.zeros(len(b))
    for y in (2025, 2026):
        if y in n: c[allnames.index(n[y])] += 0.5
    for y in (2023, 2024):
        if y in n: c[allnames.index(n[y])] -= 0.5
    diff, se = float(c @ b), float(np.sqrt(c @ V @ c))
    p_reb = 2 * (1 - stats.t.cdf(abs(diff / se), G - 1))
    print(f"rebound contrast (mean 2025–26 minus mean 2023–24): {diff:.4f} (se {se:.4f}, p {p_reb:.3f})")
    py.loc["rebound: mean(2025,2026) - mean(2023,2024)", ["estimate", "se", "p", "note"]] = \
        [diff, se, p_reb, "contrast, t(G-1)"]
except Exception as e:
    print("rebound contrast skipped:", e)
py.to_csv(f"{OUT}/post_year_effects.csv", index_label="term")
py[["year", "estimate", "se", "p"]].round(4)
''')
    nb.md(r'''
## Interest-rate sensitivity control
Preferred: a published occupation rate-sensitivity measure (`rate_sens.csv`). **Fallback proxy** (used only if no file is supplied): each occupation's slope of month-demeaned `log(1+ads)` on the fed funds rate over Jan 2019–Nov 2022 (pre-ChatGPT; the rate fell in 2019–20 and rose from Mar 2022). The proxy is estimated on the outcome itself, so report it as a sensitivity check, not as an exogenous control.
''')
    nb.code(r'''
# Cell 7 — rate sensitivity
fred = pd.read_csv(f"{RAW}/fred_series.csv", parse_dates=["date"])
ff = fred.set_index("date")["FEDFUNDS"].dropna().resample("MS").mean()
for frame in (d, d_all):
    frame["fedfunds"] = frame["date"].map(ff)
ext = find_file("rate_sens.csv", required=False)
if ext:
    rs = pd.read_csv(ext, dtype={"soc6": str})[["soc6", "rate_sens"]]
    RS_SOURCE = f"external file {ext}"
else:
    pre = d_all[d_all["rel_month"] < 0].copy()
    pre["y_dm"] = pre["log_ads"] - pre.groupby("month")["log_ads"].transform("mean")
    rs = (pre.groupby("soc6").apply(lambda g: np.polyfit(g["fedfunds"], g["y_dm"], 1)[0]
                                    if g["fedfunds"].std() > 0 else np.nan)
             .rename("rate_sens").reset_index())
    RS_SOURCE = "HN pre-period proxy (slope of demeaned log ads on FEDFUNDS, 2019-01..2022-11)"
rs["rate_sens_z"] = (rs["rate_sens"] - rs["rate_sens"].mean()) / rs["rate_sens"].std()
rs.to_csv(f"{OUT}/rate_sensitivity.csv", index=False)
d = d.merge(rs[["soc6", "rate_sens_z"]], on="soc6", how="left")
d_all = d_all.merge(rs[["soc6", "rate_sens_z"]], on="soc6", how="left")
print("rate sensitivity source:", RS_SOURCE)
print("mean rate_sens_z by group:", d.drop_duplicates("soc6").groupby(TREAT)["rate_sens_z"].mean().round(3).to_dict())

first_month = sorted(d["month"].unique())[0]
fit_es_r = pf.feols(f"log_ads ~ i(rel_q, {TREAT}, ref=-1) + i(month, rate_sens_z, ref='{first_month}') | soc6 + month",
                    data=d.dropna(subset=["rate_sens_z"]), vcov=VCOV)
tr = tidy(fit_es_r)
tr = tr[tr.index.str.contains("rel_q")].copy()
tr["rel_q"] = [parse_term(n, "rel_q") for n in tr.index]
tr = tr.dropna(subset=["rel_q"])
tr["months"] = tr["rel_q"].astype(int).apply(event_label)
tr["model"] = "with rate_sens × month"
tr.to_csv(f"{OUT}/event_study_coefs_rate_control.csv", index_label="term")
print("2022 bins with the rate control:\n",
      tr[tr["rel_q"].isin([-4, -3, -2])][["months", "Estimate", "Std. Error", "Pr(>|t|)"]].round(4).to_string())
''')
    nb.md(r'''
## Placebo dates (estimated on pre-ChatGPT data only)
Nov 2021 = main placebo; **Mar 2022 = first Fed hike** (a rate story predicts an "effect" here); Nov 2020 falls in COVID — read with care. Each placebo uses only months before Dec 2022.
''')
    nb.code(r'''
# Cell 8 — placebos
pre_only = d[d["date"] < "2022-12-01"].copy()
plac = []
for label, date in [("Nov 2021 (main placebo)", "2021-11-01"), ("Mar 2022 (first Fed hike)", "2022-03-01"),
                    ("Nov 2020 (COVID, read with care)", "2020-11-01")]:
    pre_only["fake_post"] = (pre_only["date"] >= date).astype(int)
    f = pf.feols(f"log_ads ~ fake_post:{TREAT} | soc6 + month", data=pre_only, vcov=VCOV)
    r = get_coef(f, f"fake_post:{TREAT}")
    r["placebo"] = label
    r["p_wild"] = boot_p(f, r["term"])
    plac.append(r)
plac = pd.DataFrame(plac)[["placebo", "estimate", "se", "p", "p_wild", "ci_low", "ci_high", "nobs"]]
plac.to_csv(f"{OUT}/placebo_tests.csv", index=False)
plac.round(4)
''')
    nb.md(r'''
## Table 2 — DiD with robustness checks (`RESEARCH_PLAN.md` §6)
Each row re-estimates the post × exposure coefficient under one change. Continuous measures are z-scores (coefficient = effect of +1 SD of exposure). Anthropic usage is measured after ChatGPT → reported as heterogeneity only. Remote status is an ad-level control and is handled in NB5's ad-level regressions.
''')
    nb.code(r'''
# Cell 9 — robustness table
def z(s):
    return (s - s.mean()) / s.std()

ex = pd.read_csv(f"{PROC}/exposure_table.csv", dtype={"soc6": str})
extra = [c for c in ["el_human_rating_beta", "aioe", "log_aei_usage", "exp_ensemble_all"] if c in ex and c not in d_all]
cont = d_all.merge(ex[["soc6"] + extra], on="soc6", how="left")
for c in ["exp_main", "exp_ensemble", "el_human_rating_beta", "aioe", "log_aei_usage", "exp_ensemble_all"]:
    if c in cont:
        cont[c + "_z"] = z(cont[c])
d["post_gpt4"] = (d["rel_month"] >= 3).astype(int)
d["log_ads_hc"] = np.log1p(d["ads_hc"])
d["log_ads_w"] = np.log1p(d["ads_w"])
d["share_pp"] = 100 * d["share"]
covid = (d["date"] >= "2020-03-01") & (d["date"] <= "2021-12-01")

def sub(df, col):
    return df.dropna(subset=[col]) if col in df else None

specs = [
    ("Main: log(1+ads), within-HN Eloundou β terciles", "feols", f"log_ads ~ post:{TREAT} | soc6 + month", d, f"post:{TREAT}"),
    ("Poisson (PPML) on raw counts", "fepois", f"ads ~ post:{TREAT} | soc6 + month", d, f"post:{TREAT}"),
    ("Population terciles (high_exposure)", "feols", "log_ads ~ post:high_exposure | soc6 + month",
     sub(d_all, "high_exposure"), "post:high_exposure"),
    ("Continuous Eloundou β (z)", "feols", "log_ads ~ post:exp_main_z | soc6 + month", cont, "post:exp_main_z"),
    ("Ensemble PC1 of pre-ChatGPT measures (z)", "feols", "log_ads ~ post:exp_ensemble_z | soc6 + month",
     sub(cont, "exp_ensemble_z"), "post:exp_ensemble_z"),
    ("Human-rated β (z)", "feols", "log_ads ~ post:el_human_rating_beta_z | soc6 + month",
     sub(cont, "el_human_rating_beta_z"), "post:el_human_rating_beta_z"),
    ("Felten AIOE (z)", "feols", "log_ads ~ post:aioe_z | soc6 + month", sub(cont, "aioe_z"), "post:aioe_z"),
    ("Anthropic usage, log (z) — post-treatment measure", "feols", "log_ads ~ post:log_aei_usage_z | soc6 + month",
     sub(cont, "log_aei_usage_z"), "post:log_aei_usage_z"),
    ("Treatment Mar 2023 (GPT-4)", "feols", f"log_ads ~ post_gpt4:{TREAT} | soc6 + month", d, f"post_gpt4:{TREAT}"),
    ("Drop computer occupations (SOC 15-12xx)", "feols", f"log_ads ~ post:{TREAT} | soc6 + month",
     d[d["tech"] == 0], f"post:{TREAT}"),
    ("Tech × month FE", "feols", f"log_ads ~ post:{TREAT} | soc6 + month^tech", d, f"post:{TREAT}"),
    ("Rate sensitivity × month", "feols",
     f"log_ads ~ post:{TREAT} + i(month, rate_sens_z, ref='{first_month}') | soc6 + month",
     d.dropna(subset=["rate_sens_z"]), f"post:{TREAT}"),
    ("Rate sensitivity × fed funds rate", "feols", f"log_ads ~ post:{TREAT} + rate_sens_z:fedfunds | soc6 + month",
     d.dropna(subset=["rate_sens_z", "fedfunds"]), f"post:{TREAT}"),
    ("Drop COVID months (Mar 2020–Dec 2021)", "feols", f"log_ads ~ post:{TREAT} | soc6 + month", d[~covid],
     f"post:{TREAT}"),
    ("High-confidence SOC matches only", "feols", f"log_ads_hc ~ post:{TREAT} | soc6 + month", d, f"post:{TREAT}"),
    ("Fractional counts (multi-role ads split)", "feols", f"log_ads_w ~ post:{TREAT} | soc6 + month", d,
     f"post:{TREAT}"),
    ("Outcome = share of HN ads (pp)", "feols", f"share_pp ~ post:{TREAT} | soc6 + month", d, f"post:{TREAT}"),
]
rows = []
for name, kind, fml, data, coef in specs:
    if data is None or len(data) == 0:
        rows.append({"spec": name, "note": "measure not available"}); continue
    try:
        fit = (pf.feols if kind == "feols" else pf.fepois)(fml, data=data, vcov=VCOV)
        r = get_coef(fit, coef)
        r.update(spec=name, n_occ=data["soc6"].nunique(),
                 p_wild=boot_p(fit, r["term"]) if kind == "feols" else np.nan)
        rows.append(r)
    except Exception as e:
        rows.append({"spec": name, "note": f"failed: {type(e).__name__} {str(e)[:100]}"})
table2 = pd.DataFrame(rows)
cols = [c for c in ["spec", "estimate", "se", "p", "p_wild", "ci_low", "ci_high", "nobs", "n_occ", "note"] if c in table2]
table2 = table2[cols]
table2.to_csv(f"{OUT}/table2_did_robustness.csv", index=False)
table2.round(4)
''')
    nb.code(r'''
# Cell 10 — wild-cluster bootstrap for the main DiD, post-year terms and 2022 bins
wb = [{"model": "DiD main", "term": main["term"], "estimate": main["estimate"], "p_crv1": main["p"],
       "p_wild": boot_p(fit_did, main["term"])}]
for nm in py.index:
    if "post_year" in nm:
        wb.append({"model": "post-year", "term": nm, "estimate": py.loc[nm, "estimate"], "p_crv1": py.loc[nm, "p"],
                   "p_wild": boot_p(fit_y, nm)})
for nm in names_2022:
    wb.append({"model": "event study 2022 bin", "term": nm, "estimate": es.loc[nm, "Estimate"],
               "p_crv1": es.loc[nm, "Pr(>|t|)"], "p_wild": boot_p(fit_es, nm)})
wb = pd.DataFrame(wb)
wb.to_csv(f"{OUT}/wild_bootstrap.csv", index=False)
wb.round(4)
''')
    nb.code(r'''
# Cell 11 — monthly event study + Poisson event study (appendix)
d["rel_m"] = d["rel_month"].clip(-36, None)          # bin everything before Jan 2020 into -36
fit_m = pf.feols(f"log_ads ~ i(rel_m, {TREAT}, ref=-1) | soc6 + month", data=d, vcov=VCOV)
tm = tidy(fit_m)
tm = tm[tm.index.str.contains("rel_m")].copy()
tm["rel_month"] = [parse_term(n, "rel_m") for n in tm.index]
tm = tm.dropna(subset=["rel_month"]).sort_values("rel_month")
tm["month_label"] = tm["rel_month"].astype(int).apply(lambda k: event_label(k, months_per_bin=1))
tm.to_csv(f"{OUT}/event_study_monthly_coefs.csv", index_label="term")
try:
    fit_pq = pf.fepois(f"ads ~ i(rel_q, {TREAT}, ref=-1) | soc6 + month", data=d, vcov=VCOV)
    tp = tidy(fit_pq); tp = tp[tp.index.str.contains("rel_q")].copy()
    tp["rel_q"] = [parse_term(n, "rel_q") for n in tp.index]
    tp.to_csv(f"{OUT}/event_study_coefs_poisson.csv", index_label="term")
except Exception as e:
    print("Poisson event study failed:", e)
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.axhline(0, color="k", lw=0.6); ax.axvline(-0.5, color="k", lw=0.8)
ax.fill_between(tm["rel_month"], tm["2.5%"], tm["97.5%"], alpha=0.25)
ax.plot(tm["rel_month"], tm["Estimate"], lw=1)
ax.set_xlabel("months since Dec 2022 (≤ −36 binned)"); ax.set_ylabel("Δ log(1+ads)")
fig.tight_layout(); fig.savefig(f"{OUT}/figA_event_study_monthly.png", bbox_inches="tight"); plt.show()
''')
    nb.md(r'''
## Whole-market check: Indeed US sectors (Feb 2020 = 100; no 2019 data)
Sectors are split into high vs low exposure with NB2's `sector_exposure.csv`. This shows whether the HN result is "just a Hacker News effect". About 15 sectors per group → wild-bootstrap p-values.
''')
    nb.code(r'''
# Cell 12 — Indeed sector DiD + event study
sec = pd.read_csv(f"{RAW}/indeed_sector.csv", parse_dates=["date"])
sx = pd.read_csv(f"{PROC}/sector_exposure.csv")
var = "total postings" if (sec["variable"] == "total postings").any() else sec["variable"].iloc[0]
us = sec[(sec["jobcountry"] == "US") & (sec["variable"] == var)]
us = (us.groupby(["display_name", pd.Grouper(key="date", freq="MS")])["indeed_job_postings_index"].mean()
        .rename("idx").reset_index())
us = us.merge(sx[["display_name", "exp_main", "high_sector"]], on="display_name", how="left")
print("unmatched sectors:", sorted(us.loc[us["exp_main"].isna(), "display_name"].unique()))
us = us.dropna(subset=["high_sector"]).copy()
us["log_idx"] = np.log(us["idx"])
us["month"] = us["date"].dt.strftime("%Y-%m")
us["post"] = (us["date"] >= "2022-12-01").astype(int)
us["rel_q"] = np.floor(((us["date"].dt.year - 2022) * 12 + us["date"].dt.month - 12) / 3).astype(int)
sv = {"CRV1": "display_name"}
f_sec = pf.feols("log_idx ~ post:high_sector | display_name + month", data=us, vcov=sv)
r_sec = get_coef(f_sec, "post:high_sector"); r_sec["p_wild"] = boot_p(f_sec, r_sec["term"])
us["post_year"] = np.where(us["post"] == 1, us["date"].dt.year, 0)
f_secy = pf.feols("log_idx ~ i(post_year, high_sector, ref=0) | display_name + month", data=us, vcov=sv)
pd.concat([pd.DataFrame([r_sec]).assign(model="DiD"),
           tidy(f_secy).reset_index().rename(columns={"index": "term", "Estimate": "estimate", "Std. Error": "se",
                                                       "Pr(>|t|)": "p", "2.5%": "ci_low", "97.5%": "ci_high"})
           .assign(model="post-year")],
          ignore_index=True).to_csv(f"{OUT}/indeed_sector_did.csv", index=False)
f_sece = pf.feols("log_idx ~ i(rel_q, high_sector, ref=-1) | display_name + month", data=us, vcov=sv)
te = tidy(f_sece); te = te[te.index.str.contains("rel_q")].copy()
te["rel_q"] = [parse_term(n, "rel_q") for n in te.index]
te = te.dropna(subset=["rel_q"])
te["months"] = te["rel_q"].astype(int).apply(event_label)
te.to_csv(f"{OUT}/indeed_sector_event.csv", index_label="term")
print(f"Indeed US sectors ({us['display_name'].nunique()}): DiD {r_sec['estimate']:.4f} "
      f"(CRV1 p {r_sec['p']:.3f}, wild p {r_sec['p_wild']:.3f})")

fig, ax = plt.subplots(figsize=(8, 3.5))
for g, lab, ls in [(1.0, "High-exposure sectors", "-"), (0.0, "Low-exposure sectors", "--")]:
    s = us[us["high_sector"] == g].groupby("date")["idx"].mean()
    ax.plot(s.index, s.values, ls, label=lab)
ax.axvline(pd.Timestamp("2022-11-30"), color="k", lw=0.8)
ax.set_ylabel("Indeed postings index (Feb 2020 = 100)"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/figA_indeed_sectors.png", bbox_inches="tight"); plt.show()
''')
    nb.code(r'''
# Cell 13 — HN representativeness: HN monthly ads vs Indeed US index and Indeed US software sector
hn = pd.read_csv(f"{PROC}/hn_clean.csv", usecols=["month", "remote", "software"], dtype={"month": str})
hn_m = hn.groupby("month").agg(hn_ads=("remote", "size"), remote_share=("remote", "mean"),
                               software_share=("software", "mean"))
hn_m.index = pd.to_datetime(hn_m.index + "-01")
fred_m = fred.set_index("date")[["IHLIDXUS", "JTSJOL"]].resample("MS").mean()
sw = sec[(sec["jobcountry"] == "US") & (sec["variable"] == var) &
         sec["display_name"].str.contains("Software", case=False, na=False)]
sw = sw.groupby(pd.Grouper(key="date", freq="MS"))["indeed_job_postings_index"].mean().rename("indeed_us_software")
rep = hn_m.join(fred_m, how="left").join(sw, how="left")
rep.to_csv(f"{OUT}/hn_representativeness.csv", index_label="month")
rows = []
for c in ["IHLIDXUS", "JTSJOL", "indeed_us_software"]:
    x = rep[["hn_ads", c]].dropna()
    if len(x) > 12:
        lv = np.corrcoef(np.log(x["hn_ads"]), np.log(x[c]))[0, 1]
        g = np.log(x).diff(12).dropna()
        gr = np.corrcoef(g["hn_ads"], g[c])[0, 1] if len(g) > 12 else np.nan
        rows.append({"series": c, "months": len(x), "corr_log_levels": lv, "corr_12m_log_change": gr})
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("HN remote share by year:", rep.groupby(rep.index.year)["remote_share"].mean().round(3).to_dict())
pd.DataFrame(rows).to_csv(f"{OUT}/hn_representativeness_corr.csv", index=False)
''')
    nb.code(OUTPUT_LIST)
    return nb
