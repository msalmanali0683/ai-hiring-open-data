"""NB6 — 06-forecast-and-figures."""
from common import NB, PIP, FIND, SEEDS, OUTPUT_LIST, VERSIONS

HELPERS = r'''
import math

def dm_test(e1, e2, h=1, power=2):
    """Diebold-Mariano statistic with the Harvey-Leybourne-Newbold small-sample correction.
    e1, e2: forecast errors of model 1 and 2 on the same targets. Negative = model 1 more accurate.
    Returns (statistic, degrees of freedom); compare with Student t(df)."""
    d = [abs(a) ** power - abs(b) ** power for a, b in zip(e1, e2)]
    T = len(d)
    if T < 3:
        return float("nan"), T - 1
    mean = sum(d) / T
    def acov(k):
        return sum((d[t] - mean) * (d[t - k] - mean) for t in range(k, T)) / T
    var = acov(0) + 2 * sum(acov(k) for k in range(1, h))
    if var <= 0:
        return float("nan"), T - 1
    stat = mean / math.sqrt(var / T)
    hln = math.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return stat * hln, T - 1

def forecast_metrics(y_true_log, y_pred_log):
    """MAE / RMSE on log(1+ads); MAPE and sMAPE (%) on ads + 1 (avoids dividing by zero)."""
    n = len(y_true_log)
    err = [p - t for t, p in zip(y_true_log, y_pred_log)]
    mae = sum(abs(e) for e in err) / n
    rmse = math.sqrt(sum(e * e for e in err) / n)
    a = [math.exp(t) for t in y_true_log]             # = ads + 1
    f = [math.exp(p) for p in y_pred_log]
    mape = 100 * sum(abs(x - y) / x for x, y in zip(a, f)) / n
    smape = 100 * sum(2 * abs(x - y) / (abs(x) + abs(y)) for x, y in zip(a, f)) / n
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape, "n": n}
'''

STYLE = r'''
# One style for every paper figure (colour-blind-safe Okabe-Ito palette)
OKABE = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
HIGH_C, LOW_C, AUTO_C, AUG_C, NA_C = "#D55E00", "#0072B2", "#D55E00", "#0072B2", "#999999"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.prop_cycle": plt.cycler(color=OKABE[1:]), "savefig.dpi": 300, "figure.dpi": 110,
    "savefig.bbox": "tight", "lines.linewidth": 1.3})
FIG = f"{OUT}/figures"
os.makedirs(FIG, exist_ok=True)
CHATGPT, FED_HIKE = pd.Timestamp("2022-11-30"), pd.Timestamp("2022-03-16")

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG}/{name}.{ext}")
    plt.show(); plt.close(fig)
    print("saved", name)

def mark_events(ax, fed=True):
    ax.axvline(CHATGPT, color="k", lw=0.8)
    if fed:
        ax.axvline(FED_HIKE, color="grey", lw=0.7, ls=":")

def need(name):
    """Path of an input produced by an earlier notebook, or None (the figure is then skipped)."""
    p = find_file(name, required=False)
    if p is None:
        print(f"  skipped: {name} not found — add that notebook's output via Add Input → Notebook Output")
    return p
'''


def build():
    nb = NB(gpu=False, internet=True)
    nb.md(r'''
# NB6 — Forecasting appendix (RQ5) and all final paper figures (`06-forecast-and-figures`)
RQ5 is **appendix only** (`RESEARCH_PLAN.md` §5.5): "does AI exposure carry forecasting signal for next-quarter postings?" If time is short, set `RUN_FORECAST = False` and use this notebook just for the figures.

| | |
|---|---|
| **Input** | **`jobai-processed`**, **`jobai-raw`** (`fred_series.csv`); optional **`jobai-skills`**; the **outputs of NB3 and NB5** (right panel → *Add Input* → *Notebook Output Files* → `03-event-study`, `05-seniority-countries`) |
| **Output** | `forecast_metrics.csv` (Table A1), `forecast_dm_tests.csv`, `forecast_predictions.csv`, `shap_importance.csv`, `figures/figA1_shap.png`; `figures/fig1…fig5*.png/.pdf` in one style; `tables/*.csv` + `tables/*.tex` |
| **Kaggle settings** | Accelerator **None (CPU)** · **Internet On** (pip only) |
| **Run time** | ~10–25 min (ARIMA per occupation and the small LSTM take most of it) |

**Forecast design.** Quarterly occupation panel (calendar quarters; the incomplete last quarter is dropped). Target = `log(1 + ads)` next quarter. **Time split only:** train on target quarters ≤ 2024Q4, test 2025Q1 → last full quarter; no shuffling. Models: naive (no change), seasonal-naive (same quarter last year), ARIMA per occupation (order by AIC, 1-step-ahead with parameters fixed after training), LightGBM global model (with and **without** exposure/skill features — the RQ5 ablation), and a small LSTM. Metrics: MAE, RMSE (log scale), MAPE, sMAPE (on ads + 1); Diebold–Mariano tests vs seasonal-naive; SHAP for LightGBM.

Nothing has been run; no result is written here.
''')
    nb.code(PIP + r'''
# Cell 1 — installs, imports, settings
RUN_FORECAST = True
RUN_LSTM = True
pip("lightgbm", "shap", "statsmodels")
''' + FIND + r'''
import re, json, math, time, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
''' + SEEDS + VERSIONS + r'''
warnings.filterwarnings("ignore")
OUT = "/kaggle/working"
PROC = find_dir("jobai-processed", "panel_occ_month.csv")
RAW = find_dir("jobai-raw", "fred_series.csv")
SKL = find_dir("jobai-skills", "ad_skill_summary.csv", required=False)
os.makedirs(f"{OUT}/tables", exist_ok=True)
TRAIN_END = pd.Period("2024Q4", freq="Q")
MIN_ADS_OCC = 60        # occupations with fewer ads in total are left out of the forecast panel
save_versions(["pandas", "numpy", "lightgbm", "shap", "statsmodels", "torch"], f"{OUT}/versions_nb6.txt")
''' + STYLE)
    nb.code("# Cell 2 — forecast helpers (plain Python, unit-tested locally)" + HELPERS + r'''
print(dm_test([0.1, -0.2, 0.3, 0.1], [0.3, -0.4, 0.5, 0.2]), forecast_metrics([1.0, 2.0], [1.1, 1.8]))
''')
    nb.md("## Quarterly panel and features")
    nb.code(r'''
# Cell 3 — build the quarterly occupation panel
panel = pd.read_csv(f"{PROC}/panel_occ_month.csv", dtype={"soc6": str, "month": str}, parse_dates=["date"])
panel["q"] = panel["date"].dt.to_period("Q")
nmonths = panel.groupby("q")["month"].nunique()
full_q = nmonths[nmonths == 3].index
qp = (panel[panel["q"].isin(full_q)].groupby(["soc6", "q"])
          .agg(ads=("ads", "sum"), hn_total=("hn_total", "sum")).reset_index())
keep = qp.groupby("soc6")["ads"].sum()
qp = qp[qp["soc6"].isin(keep[keep >= MIN_ADS_OCC].index)].sort_values(["soc6", "q"]).reset_index(drop=True)
static = panel.drop_duplicates("soc6").set_index("soc6")[["exp_main", "exp_ensemble", "tech"]]
qp = qp.join(static, on="soc6")
qp["y"] = np.log1p(qp["ads"])
g = qp.groupby("soc6")["y"]
for k in range(0, 4):
    qp[f"y_lag{k}"] = g.shift(k)                 # y_lag0 = this quarter (known when forecasting next quarter)
qp["dy"] = qp["y_lag0"] - qp["y_lag1"]
qp["y_roll4"] = g.transform(lambda s: s.rolling(4, min_periods=2).mean())
qp["target"] = g.shift(-1)                        # next quarter
qp["target_q"] = qp["q"] + 1
qp["target_qtr"] = qp["target_q"].dt.quarter
qp["seasonal_naive"] = g.shift(3)                 # same quarter one year before the target
qp["log_hn_total"] = np.log(qp["hn_total"])

fred = pd.read_csv(f"{RAW}/fred_series.csv", parse_dates=["date"])
fq = fred.set_index("date").resample("QS").mean()
fq.index = fq.index.to_period("Q")
fq = pd.DataFrame({"log_jolts": np.log(fq["JTSJOL"]), "fedfunds": fq["FEDFUNDS"],
                   "log_indeed_us": np.log(fq["IHLIDXUS"])})
fq["d_log_jolts"] = fq["log_jolts"].diff()
qp = qp.join(fq, on="q")

SKILL_FEATS = []
if SKL:
    ss = pd.read_csv(f"{SKL}/ad_skill_summary.csv", dtype={"soc6": str, "month": str})
    ss["q"] = pd.to_datetime(ss["month"] + "-01").dt.to_period("Q")
    sf = ss.groupby(["soc6", "q"]).agg(ai_skill_share=("has_ai_skill", "mean"), skill_beta=("mean_skill_beta", "mean"),
                                       auto_share=("auto_share", "mean")).reset_index()
    qp = qp.merge(sf, on=["soc6", "q"], how="left")
    SKILL_FEATS = ["ai_skill_share", "skill_beta", "auto_share"]
BASE_FEATS = ["y_lag0", "y_lag1", "y_lag2", "y_lag3", "dy", "y_roll4", "target_qtr", "log_hn_total",
              "log_jolts", "d_log_jolts", "fedfunds", "log_indeed_us"]
EXPO_FEATS = ["exp_main", "exp_ensemble", "tech"] + SKILL_FEATS
data = qp.dropna(subset=["target", "y_lag3", "seasonal_naive"]).copy()
train = data[data["target_q"] <= TRAIN_END]
test = data[data["target_q"] > TRAIN_END]
print(f"{data['soc6'].nunique()} occupations | train target quarters {train['target_q'].min()}–{train['target_q'].max()} "
      f"({len(train)}) | test {test['target_q'].min()}–{test['target_q'].max()} ({len(test)})")
''')
    nb.code(r'''
# Cell 4 — baselines, ARIMA, LightGBM (+ ablation), LSTM
preds = test[["soc6", "target_q", "target"]].copy()
if RUN_FORECAST:
    preds["naive"] = test["y_lag0"].values
    preds["seasonal_naive"] = test["seasonal_naive"].values

    # ARIMA per occupation: order by AIC on the training quarters, then 1-step-ahead with fixed parameters
    from statsmodels.tsa.arima.model import ARIMA
    ar = {}
    t0 = time.time()
    for soc, s in qp.groupby("soc6"):
        s = s.set_index("q")["y"].sort_index()
        tr_s = s[s.index <= TRAIN_END]
        if len(tr_s) < 12:
            continue
        best = None
        for order in [(1, 0, 0), (1, 1, 0), (0, 1, 1), (2, 0, 0)]:
            try:
                r = ARIMA(tr_s.values, order=order, trend="c" if order[1] == 0 else "n").fit()
                if best is None or r.aic < best.aic:
                    best = r
            except Exception:
                pass
        if best is None:
            continue
        res, hist = best, tr_s.index[-1]
        for tq in sorted(preds.loc[preds["soc6"] == soc, "target_q"]):
            while hist < tq - 1:                     # append the observations known before the target quarter
                hist = hist + 1
                if hist in s.index:
                    res = res.append([s[hist]], refit=False)
            ar[(soc, tq)] = float(res.forecast(1)[0])
    preds["arima"] = [ar.get((a, b), np.nan) for a, b in zip(preds["soc6"], preds["target_q"])]
    print(f"ARIMA done in {time.time() - t0:.0f}s; coverage {preds['arima'].notna().mean():.1%}")

    import lightgbm as lgb
    def fit_lgb(feats):
        va_q = sorted(train["target_q"].unique())[-4:]          # last 4 training quarters = early-stopping set
        tr, va = train[~train["target_q"].isin(va_q)], train[train["target_q"].isin(va_q)]
        params = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=20,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                      seed=SEED, deterministic=True, verbose=-1)
        m = lgb.train(params, lgb.Dataset(tr[feats], tr["target"]), 2000,
                      valid_sets=[lgb.Dataset(va[feats], va["target"])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        m = lgb.train(params, lgb.Dataset(train[feats], train["target"]), m.best_iteration or 200)  # refit on all train
        return m
    lgb_full = fit_lgb(BASE_FEATS + EXPO_FEATS)
    lgb_base = fit_lgb(BASE_FEATS)
    preds["lgbm"] = lgb_full.predict(test[BASE_FEATS + EXPO_FEATS])
    preds["lgbm_no_exposure"] = lgb_base.predict(test[BASE_FEATS])

    if RUN_LSTM:
        import torch, torch.nn as nn
        SEQ = 8
        macro = ["log_jolts", "fedfunds", "log_indeed_us", "log_hn_total"]
        mu, sd = train[macro].mean(), train[macro].std().replace(0, 1)
        qz = qp.copy()
        qz[macro] = (qz[macro] - mu) / sd
        def windows(df):
            X, S, Y, keys = [], [], [], []
            for soc, s in df.groupby("soc6"):
                s = s.sort_values("q")
                arr = s[["y"] + macro].fillna(0).values.astype("float32")
                st = s[["exp_main", "tech"]].fillna(0).values[0].astype("float32")
                for i in range(SEQ - 1, len(s) - 1):
                    X.append(arr[i - SEQ + 1:i + 1]); S.append(st); Y.append(arr[i + 1, 0])
                    keys.append((soc, s["q"].iloc[i + 1]))
            return np.stack(X), np.stack(S), np.array(Y, dtype="float32"), keys
        X, S, Yw, keys = windows(qz)
        tq = np.array([k[1] for k in keys])
        tr_m, te_m = tq <= TRAIN_END, tq > TRAIN_END
        class Net(nn.Module):
            def __init__(self, f, s):
                super().__init__()
                self.lstm = nn.LSTM(f, 32, batch_first=True)
                self.head = nn.Sequential(nn.Linear(32 + s, 32), nn.ReLU(), nn.Linear(32, 1))
            def forward(self, x, s):
                h = self.lstm(x)[0][:, -1]
                return self.head(torch.cat([h, s], 1)).squeeze(1) + x[:, -1, 0]   # predict change from last value
        torch.manual_seed(SEED)
        net = Net(X.shape[2], S.shape[1]); opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
        Xt, St, Yt = map(torch.tensor, (X[tr_m], S[tr_m], Yw[tr_m]))
        for ep in range(60):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 256):
                b = perm[i:i + 256]
                loss = nn.functional.mse_loss(net(Xt[b], St[b]), Yt[b])
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            p_l = net(torch.tensor(X[te_m]), torch.tensor(S[te_m])).numpy()
        lk = {k: v for k, v in zip([keys[i] for i in np.where(te_m)[0]], p_l)}
        preds["lstm"] = [lk.get((a, b), np.nan) for a, b in zip(preds["soc6"], preds["target_q"])]
    preds.to_csv(f"{OUT}/forecast_predictions.csv", index=False)
''')
    nb.code(r'''
# Cell 5 — Table A1 metrics + Diebold–Mariano tests vs seasonal-naive
if RUN_FORECAST:
    import statsmodels.api as sm
    models = [c for c in ["naive", "seasonal_naive", "arima", "lgbm", "lgbm_no_exposure", "lstm"] if c in preds]
    common = preds.dropna(subset=models)
    print(f"evaluated on {len(common)} occupation-quarters where every model has a forecast")
    met = pd.DataFrame([dict(model=m, **forecast_metrics(common["target"].tolist(), common[m].tolist()))
                        for m in models]).sort_values("MAE")
    met.to_csv(f"{OUT}/forecast_metrics.csv", index=False)
    print(met.round(4).to_string(index=False))
    dm_rows = []
    for m, ref in [(m, "seasonal_naive") for m in models if m != "seasonal_naive"] + [("lgbm", "lgbm_no_exposure")]:
        e1, e2 = common[m] - common["target"], common[ref] - common["target"]
        # (a) loss differential averaged over occupations per quarter -> time series DM (few test quarters!)
        per_q = pd.DataFrame({"q": common["target_q"], "e1": e1 ** 2, "e2": e2 ** 2}).groupby("q").mean()
        stat_q, df_q = dm_test(np.sqrt(per_q["e1"]).tolist(), np.sqrt(per_q["e2"]).tolist())
        # (b) pooled over occupation-quarters, SE clustered by occupation
        dd = (e1 ** 2 - e2 ** 2).values
        ols = sm.OLS(dd, np.ones(len(dd))).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(common["soc6"])[0]})
        dm_rows.append({"model": m, "vs": ref, "dm_quarterly": stat_q, "p_quarterly": 2 * stats.t.sf(abs(stat_q), df_q)
                        if stat_q == stat_q else np.nan, "test_quarters": df_q + 1, "mean_loss_diff": float(dd.mean()),
                        "t_pooled_clustered": float(ols.tvalues[0]), "p_pooled_clustered": float(ols.pvalues[0])})
    dm = pd.DataFrame(dm_rows); dm.to_csv(f"{OUT}/forecast_dm_tests.csv", index=False)
    print(dm.round(4).to_string(index=False))
    print("Negative statistics = first model more accurate. 'lgbm vs lgbm_no_exposure' is the RQ5 question.")
''')
    nb.code(r'''
# Cell 6 — SHAP for the LightGBM model (Fig A1)
if RUN_FORECAST:
    import shap
    feats = BASE_FEATS + EXPO_FEATS
    sv = shap.TreeExplainer(lgb_full).shap_values(test[feats])
    imp = pd.DataFrame({"feature": feats, "mean_abs_shap": np.abs(sv).mean(0)}).sort_values("mean_abs_shap", ascending=False)
    imp["group"] = np.where(imp["feature"].isin(EXPO_FEATS), "exposure / skills", "lags / macro")
    imp.to_csv(f"{OUT}/shap_importance.csv", index=False)
    shap.summary_plot(sv, test[feats], show=False, max_display=15)
    fig = plt.gcf(); fig.set_size_inches(6.5, 4.5)
    save(fig, "figA1_shap")
    print(imp.round(4).to_string(index=False))
''')
    nb.md("## Final paper figures (one style). Each figure is skipped with a message if its input is missing.")
    nb.code(r'''
# Cell 7 — Fig 1 raw trends + Table 1 data summary
hn = pd.read_csv(f"{PROC}/hn_clean.csv", usecols=["comment_id", "month", "year", "company", "soc6", "high_conf",
                                                  "high_exposure", "high_exp_hn", "remote", "n_roles"],
                 dtype={"soc6": str, "month": str})
t1 = hn.groupby("year").agg(ads=("comment_id", "size"), companies=("company", lambda s: s[s.fillna("") != ""].nunique()),
                            matched_to_SOC=("soc6", lambda s: s.notna().mean()), high_conf=("high_conf", "mean"),
                            high_exposure_share=("high_exposure", lambda s: (s == 1).mean()),
                            low_exposure_share=("high_exposure", lambda s: (s == 0).mean()),
                            remote_share=("remote", "mean"), roles_per_ad=("n_roles", "mean"))
t1.loc["all"] = [len(hn), hn.loc[hn["company"].fillna("") != "", "company"].nunique(), hn["soc6"].notna().mean(),
                 hn["high_conf"].mean(), (hn["high_exposure"] == 1).mean(), (hn["high_exposure"] == 0).mean(),
                 hn["remote"].mean(), hn["n_roles"].mean()]
t1.to_csv(f"{OUT}/tables/table1_data_summary.csv")
try:
    open(f"{OUT}/tables/table1_data_summary.tex", "w").write(t1.round(3).to_latex())
except Exception as e:
    print("LaTeX export skipped:", e)

# One panel: within-HN terciles. The population-tercile split leaves too few low-exposure ads to plot (Section 4.1).
fig, ax = plt.subplots(figsize=(5.2, 2.8))
col = "high_exp_hn"
s = panel.dropna(subset=[col]).groupby(["date", col])["ads"].sum().unstack()
idx = (100 * s / s[s.index < "2022-12-01"].mean()).rolling(3, min_periods=1).mean()
if 1.0 in idx: ax.plot(idx.index, idx[1.0], color=HIGH_C, label="high exposure (top tercile)")
if 0.0 in idx: ax.plot(idx.index, idx[0.0], color=LOW_C, ls="--", label="low exposure (bottom tercile)")
mark_events(ax)
ax.set_ylabel("HN ads (pre-ChatGPT mean = 100)"); ax.legend(frameon=False, fontsize=7)
save(fig, "fig1_raw_trends")
''')
    nb.code(r'''
# Cell 8 — Fig 2 event study + post-year effects
p_es, p_py, p_rc = need("event_study_coefs.csv"), need("post_year_effects.csv"), find_file("event_study_coefs_rate_control.csv", required=False)
if p_es:
    es = pd.read_csv(p_es).sort_values("rel_q")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [3, 1.2]})
    ax = axes[0]
    ax.axhline(0, color="k", lw=0.6); ax.axvline(-0.5, color="k", lw=0.8)
    ax.axvspan(-4.5, -1.5, color=OKABE[1], alpha=0.12, lw=0)
    ax.errorbar(es["rel_q"], es["estimate"], yerr=[es["estimate"] - es["ci_low"], es["ci_high"] - es["estimate"]],
                fmt="o", ms=2.8, capsize=1.5, color=OKABE[5], label="main")
    if p_rc:
        rc = pd.read_csv(p_rc).sort_values("rel_q")
        ax.plot(rc["rel_q"] + 0.25, rc["Estimate"], "s", ms=2.3, color=OKABE[6], alpha=0.8, label="+ rate sensitivity × month")
    ticks = [k for k in es["rel_q"] if k % 4 == 0]
    ax.set_xticks(ticks); ax.set_xticklabels(es.set_index("rel_q").loc[ticks, "months"], rotation=40, ha="right", fontsize=6)
    ax.set_ylabel("Δ log(1+ads), high vs low exposure"); ax.set_title("(a) Event study, ref. Sep–Nov 2022", loc="left")
    ax.legend(frameon=False, fontsize=6.5)
    if p_py:
        py = pd.read_csv(p_py)
        py = py[py["year"].between(2023, 2026)]
        axes[1].bar(py["year"].astype(int).astype(str), py["estimate"], color=OKABE[5],
                    yerr=1.96 * py["se"], capsize=2)
        axes[1].axhline(0, color="k", lw=0.6); axes[1].set_title("(b) Post-year effects", loc="left")
    save(fig, "fig2_event_study")
''')
    nb.code(r'''
# Cell 9 — Fig 3 rising / falling skills (main result)
p_sc = need("skill_change.csv")
if p_sc:
    ch = pd.read_csv(p_sc)
    ch["skill"] = ch["skill"].fillna(ch["skill_id"].astype(str).str.replace(r"^oov:", "", regex=True)).astype(str)
    ch["skill"] = ch["skill"].str.replace(" (not in taxonomy)", "*", regex=False)   # * = outside the inventory
    # known mapping error (Section 4.3): LLM output "Tailwind CSS" was mapped to an unrelated O*NET product
    ch.loc[ch["skill"].str.startswith("Tailwind Management"), "skill"] = "Tailwind CSS†"
    ch["skill"] = ch["skill"].where(ch["skill"].str.len() <= 28, ch["skill"].str[:27] + "…")
    tag = next((c for c in ["tag_occ", "tag_task"] if c in ch), None)
    colors = {"automation": AUTO_C, "augmentation": AUG_C, "no data": NA_C}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.4))
    for ax, dfp, title in [(axes[0], ch.nsmallest(20, "diff_pp")[::-1], "(a) Largest falls"),
                           (axes[1], ch.nlargest(20, "diff_pp")[::-1], "(b) Largest rises")]:
        c = dfp[tag].map(colors).fillna(NA_C) if tag else NA_C
        ax.barh(dfp["skill"], dfp["diff_pp"], color=c)
        ax.axvline(0, color="k", lw=0.6); ax.set_title(title, loc="left")
        ax.set_xlabel("change in share of ads (pp)")
        ax.tick_params(axis="y", labelsize=6.5)
    if tag:
        axes[1].legend([plt.Rectangle((0, 0), 1, 1, color=v) for v in colors.values()], colors.keys(),
                       frameon=False, loc="lower right")
    fig.subplots_adjust(wspace=0.75)
    fig.text(0.01, 0.005, "Share of ads, 2025–26 minus 2019–Nov 2022. * = skill outside the O*NET-based inventory "
             "(LLM wording). † mapped by the pipeline to an unrelated O*NET product (Section 4.3).", fontsize=6)
    save(fig, "fig3_skills")
''')
    nb.code(r'''
# Cell 10 — Fig 4 junior vs senior (main result)
p_rl, p_ev = need("role_levels.csv"), need("rq3_event_coefs.csv")
if p_rl:
    rl = pd.read_csv(p_rl, usecols=["month", "level_final", "high_exp_hn"], dtype={"month": str})
    rl = rl[rl["level_final"].isin(["entry", "senior"])].dropna(subset=["high_exp_hn"])
    s = rl.assign(e=(rl["level_final"] == "entry").astype(int)).groupby(["month", "high_exp_hn"])["e"].mean().unstack()
    s.index = pd.to_datetime(s.index + "-01"); s = 100 * s.rolling(3, min_periods=1).mean()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    if 1.0 in s: axes[0].plot(s.index, s[1.0], color=HIGH_C, label="high exposure")
    if 0.0 in s: axes[0].plot(s.index, s[0.0], color=LOW_C, ls="--", label="low exposure")
    mark_events(axes[0], fed=False)
    axes[0].set_ylabel("entry roles, % of entry + senior"); axes[0].set_title("(a) Entry-level share", loc="left")
    axes[0].legend(frameon=False)
    if p_ev:
        ev = pd.read_csv(p_ev)
        for (name, g), dx, col in zip(ev.groupby("model"), [-0.1, 0.1], [HIGH_C, LOW_C]):
            axes[1].errorbar(g["year"] + dx, g["Estimate"], yerr=1.96 * g["Std. Error"], fmt="o-", ms=2.5,
                             capsize=1.5, color=col, label=name)
        axes[1].axhline(0, color="k", lw=0.6); axes[1].axvline(2022.5, color="k", lw=0.8, ls=":")
        axes[1].set_title("(b) Junior vs senior within company × month", loc="left")
        axes[1].set_ylabel("log points, ref. 2022"); axes[1].legend(frameon=False)
    save(fig, "fig4_junior_senior")
''')
    nb.code(r'''
# Cell 11 — Fig 5 countries + Section 174 + synthetic control
p_cc, p_s174, p_sp = need("country_change.csv"), need("section174_series.csv"), need("synth_paths.csv")
if p_cc and p_s174:
    cc = pd.read_csv(p_cc)
    hm = cc.pivot_table(index="jobcountry", columns="display_name", values="pct_change")
    order = cc.drop_duplicates("display_name").sort_values("exp_main", ascending=False)["display_name"]
    hm = hm.reindex(index=[c for c in ["US", "GB", "CA", "DE", "FR", "AU"] if c in hm.index],
                    columns=[c for c in order if c in hm.columns])
    fig = plt.figure(figsize=(7.2, 5.6))
    ax = fig.add_axes([0.08, 0.55, 0.84, 0.38])
    lim = np.nanpercentile(np.abs(hm.values), 95)
    im = ax.imshow(hm.values, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_yticks(range(len(hm.index))); ax.set_yticklabels(hm.index)
    ax.set_xticks(range(len(hm.columns))); ax.set_xticklabels(hm.columns, rotation=75, ha="right", fontsize=5)
    ax.grid(False); ax.set_title("(a) % change in postings, 2025–26 vs Feb 2020–Nov 2022 (most → least exposed)", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    s = pd.read_csv(p_s174, parse_dates=["date"]).set_index("date")
    ax2 = fig.add_axes([0.08, 0.06, 0.4, 0.3])
    for c in s.columns:
        ax2.plot(s.index, s[c], lw=1.8 if c == "US" else 0.8, color="k" if c == "US" else None, label=c)
    ax2.axvline(pd.Timestamp("2022-01-01"), color="grey", ls=":", lw=0.8); mark_events(ax2, fed=False)
    ax2.set_title("(b) Software development postings", loc="left"); ax2.legend(ncol=3, frameon=False, fontsize=6)
    if p_sp:
        sp = pd.read_csv(p_sp, parse_dates=["date"])
        sp = sp[(sp["treatment"] == "Dec 2022") & sp["pool"].str.startswith("A")]
        ax3 = fig.add_axes([0.56, 0.06, 0.4, 0.3])
        ax3.plot(sp["date"], np.exp(sp["us_software"]), "k", label="US software")
        ax3.plot(sp["date"], np.exp(sp["synthetic"]), "--", color=OKABE[5], label="synthetic control")
        mark_events(ax3, fed=False); ax3.set_title("(c) Synthetic control (donor pool A)", loc="left")
        ax3.legend(frameon=False)
    save(fig, "fig5_countries_section174")
''')
    nb.code(r'''
# Cell 12 — copy result tables (CSV + LaTeX) into tables/
for name in ["table2_did_robustness.csv", "placebo_tests.csv", "pretrend_tests.csv", "post_year_effects.csv",
             "table3_skill_eval.csv", "eval_by_year.csv", "kappa.csv", "skill_change_tests.csv",
             "rq3_ppml_results.csv", "rq3_outcomes.csv", "did_by_level.csv", "seniority_eval.csv",
             "country_gap.csv", "section174_tests.csv", "synth_placebos.csv", "forecast_metrics.csv",
             "forecast_dm_tests.csv"]:
    p = find_file(name, required=False) or (f"{OUT}/{name}" if os.path.exists(f"{OUT}/{name}") else None)
    if not p:
        print("  not found:", name); continue
    t = pd.read_csv(p)
    t.to_csv(f"{OUT}/tables/{name}", index=False)
    try:
        open(f"{OUT}/tables/{name[:-4]}.tex", "w").write(t.round(4).to_latex(index=False))
    except Exception as e:
        print("  LaTeX skipped for", name, e)
print("tables:", sorted(os.listdir(f"{OUT}/tables")))
''')
    nb.code(OUTPUT_LIST)
    return nb
