"""NB5 — 05-seniority-countries."""
from common import NB, PIP, FIND, SEEDS, OUTPUT_LIST, VERSIONS, WCR

HELPERS = r'''
import re

# Level words are read from the job TITLE / header only ("work with senior engineers" must not count).
SENIOR_RE = re.compile(r"\b(?:senior|sr\.?|staff|principal|lead(?!\s+gen)|head\s+of|director|vp|vice\s+president|"
                       r"chief|cto|cfo|coo|ceo)\b", re.I)
ENTRY_RE = re.compile(r"\b(?:junior|jr\.?|intern(?:ship)?s?|new[- ]?grad(?:uate)?s?|graduate|entry[- ]level|"
                      r"apprentice(?:ship)?|trainee|early[- ]career)\b", re.I)
MID_RE = re.compile(r"\b(?:mid[- ]?level|mid[- ]senior|intermediate|ii)\b", re.I)
YEARS_RES = [
    re.compile(r"(\d{1,2})\s*\+?\s*(?:-|–|to)?\s*(?:\d{1,2})?\s*\+?\s*(?:years?|yrs?)\b[^.\n]{0,40}?\bexperience", re.I),
    re.compile(r"\bexperience\b[^.\n]{0,30}?\b(\d{1,2})\s*\+?\s*(?:years?|yrs?)\b", re.I),
]

def level_from_title(title):
    t = title or ""
    e, s = bool(ENTRY_RE.search(t)), bool(SENIOR_RE.search(t))
    if e and s:
        return "mixed"
    if e:
        return "entry"
    if s:
        return "senior"
    if MID_RE.search(t):
        return "mid"
    return None

def years_required(text, max_years=15):
    """Smallest 'N years ... experience' requirement in the ad (None if none). Numbers > 15 are ignored
    ('founded 20 years ago')."""
    found = []
    for rx in YEARS_RES:
        for m in rx.finditer(text or ""):
            v = int(m.group(1))
            if 0 <= v <= max_years:
                found.append(v)
    return min(found) if found else None

def level_rule(title, years):
    """Title words first, then required years (<=2 entry, >=5 senior), else 'mid' by default."""
    t = level_from_title(title)
    if t:
        return t, "title"
    if years is not None:
        return ("entry" if years <= 2 else "senior" if years >= 5 else "mid"), "years"
    return "mid", "default"

def mask_level_words(text):
    """Remove the words the rule uses, so a classifier trained on rule labels must learn from other cues."""
    t = SENIOR_RE.sub(" ", text or "")
    t = ENTRY_RE.sub(" ", t)
    t = MID_RE.sub(" ", t)
    for rx in YEARS_RES:
        t = rx.sub(" ", t)
    t = re.sub(r"\b\d{1,2}\s*\+?\s*(?:years?|yrs?)\b", " ", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()
'''


def build():
    nb = NB(gpu=True, internet=True)
    nb.md(r'''
# NB5 — Seniority (RQ3, headline) and countries / Section 174 (RQ4) (`05-seniority-countries`)

| | |
|---|---|
| **Input** | **`jobai-processed`** (`ad_roles.csv`, `hn_clean.csv`, `panel_occ_month.csv`, `sector_exposure.csv`), **`jobai-raw`** (`indeed_sector.csv`, `indeed_aggregate.csv`) |
| **Optional input** | **`jobai-skills`** (`ad_skill_summary.csv` → skill-content outcomes); **`jobai-seniority-labels`** with your filled `seniority_handcheck_labeled.csv` (2nd run) |
| **Output** (`/kaggle/working/`) | `role_levels.csv`, `seniority_source_counts.csv`, `seniority_clf_val.csv`, `seniority_handcheck_sample.csv` (+ `_key`), `seniority_eval.csv`, `fig4a_entry_share.png`, `fig4b_junior_event.png`, `rq3_ppml_results.csv`, `rq3_event_coefs.csv`, `rq3_outcomes.csv`, `did_by_level.csv`, `country_change.csv`, `country_gap.csv`, `country_panel_regs.csv`, `fig5a_country_heatmap.png`, `figA_country_totals.png`, `section174_series.csv`, `section174_tests.csv`, `fig5b_section174.png`, `synth_weights.csv`, `synth_paths.csv`, `synth_placebos.csv`, `fig5c_synth.png` |
| **Kaggle settings** | Accelerator **GPU T4** (for the classifier; on CPU it trains on a small subset and says so) · **Internet On** (pip + model download) |
| **Run time** | ~25–40 min on T4 (classifier ~10–20 min) |

**RQ3 design** (`RESEARCH_PLAN.md` §5.3): `ads(level, exposure, company, month) = exp(Junior×Post·β₁ + Post×High·β₂ + Junior×Post×High·β₃ + company×month FE + level×exposure FE)`, PPML, SEs clustered by company. Company × month fixed effects absorb anything that hits a whole firm in a month (rates, funding, Section 174). Unit = one **role** (an ad listing a junior and a senior role contributes one of each). Exposure split for RQ3 = `high_exp_hn` (terciles among HN occupations), because HN has few population-low-exposure jobs; the population split and the continuous score are robustness rows.

**Seniority labels**: title/header words → required years (≤ 2 entry, ≥ 5 senior) → default "mid", plus a **DistilBERT classifier fine-tuned on weak labels with the level words masked out**, which labels ads that have no level words. Compared on 200 hand-labelled roles.

Nothing has been run; no result is written here.
''')
    nb.code(PIP + r'''
# Cell 1 — installs, imports, settings
pip("pyfixest", "wildboottest", "pysyncon")
''' + FIND + r'''
import re, json, math, time, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize
import torch
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
PROC = find_dir("jobai-processed", "ad_roles.csv")
RAW = find_dir("jobai-raw", "indeed_sector.csv")
SKL = find_dir("jobai-skills", "ad_skill_summary.csv", required=False)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("processed:", PROC, "| raw:", RAW, "| skills:", SKL or "not attached", "| device:", DEVICE)

CLF_MODEL = "distilbert/distilbert-base-uncased"   # or "microsoft/deberta-v3-small" (slower, fp32)
USE_CLASSIFIER = False         # off: on the 200 hand-labelled roles the classifier scored 0.40 accuracy (0.12 on roles
                               # without level words) vs 0.83 for the title → years → mid rule, so final levels use the rule.
                               # The NB5 v5 run (classifier on, CONF_MIN 0.60) is kept as a robustness check.
CONF_MIN = 0.60 if USE_CLASSIFIER else 9.0   # classifier labels used only when its top probability is at least this
RQ3_EXP = "high_exp_hn"        # within-HN exposure split for RQ3 (population split = robustness)
HANDCHECK_N = 200
SECTOR_COUNTRIES = ["US", "GB", "CA", "DE", "FR", "AU"]
PRE_IDX = ("2020-02-01", "2022-11-30")   # Indeed starts Feb 2020 = 100
LATE_FROM = "2025-01-01"
EVENTS = {"Section 174 amortisation (tax years from Jan 2022)": "2022-01-01", "First Fed hike": "2022-03-16",
          "ChatGPT": "2022-11-30", "Domestic R&E expensing restored (OBBBA, Jul 2025; verify)": "2025-07-04"}
save_versions(["pandas", "numpy", "torch", "transformers", "pyfixest", "pysyncon", "sklearn"], f"{OUT}/versions_nb5.txt")
plt.rcParams.update({"savefig.dpi": 300, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
''')
    nb.code("# Cell 2 — seniority helpers (plain Python, unit-tested locally)" + HELPERS + r'''
def tidy(fit):
    t = fit.tidy(); t.index = [str(i) for i in t.index]; return t

def coef_rows(fit, label, keep=None):
    t = tidy(fit).reset_index().rename(columns={"index": "term"})
    if keep:
        t = t[t["term"].str.contains(keep)]
    t["model"] = label
    t["nobs"] = getattr(fit, "_N", np.nan)
    return t

''' + WCR + r'''

for t in ["Senior Backend Engineer", "Software Engineering Intern", "Lead Generation Specialist", "Engineer II"]:
    print(t, "->", level_from_title(t))
print(years_required("You have 3+ years of professional experience. Founded 20 years ago."))
''')
    nb.md("## Step 1 — rule-based levels (title/header words → years → default)")
    nb.code(r'''
# Cell 3 — rule labels for every role
roles = pd.read_csv(f"{PROC}/ad_roles.csv", dtype={"soc6": str, "month": str, "company": str})
hn = pd.read_csv(f"{PROC}/hn_clean.csv", dtype={"soc6": str, "month": str, "company": str})
hn["min_years"] = hn["text"].apply(years_required)
roles = roles.merge(hn[["comment_id", "header", "text", "min_years", "remote", "year", "post"]], on="comment_id", how="left")
roles["company"] = roles["company"].fillna("")
lab = [level_rule(t, (None if pd.isna(y) else y)) for t, y in zip(roles["role_raw"].fillna(""), roles["min_years"])]
roles["level_regex"] = [l for l, _ in lab]
roles["level_source"] = [s for _, s in lab]
src = (roles.groupby(["level_source", "level_regex"]).size().rename("roles").reset_index())
src["share"] = src["roles"] / src["roles"].sum()
src.to_csv(f"{OUT}/seniority_source_counts.csv", index=False)
print(src.round(3).to_string(index=False))
print("years-of-experience found in", f"{hn['min_years'].notna().mean():.1%}", "of ads")
''')
    nb.md(r'''
## Step 2 — seniority classifier (weak supervision)
Training labels = rule labels from **title words** and **years** (not the "default" bucket). Input = role title + ad text with every level word and "N years" phrase **removed**, so the model must use other cues (responsibilities, scope, "mentor", "own the roadmap", "learn from"). Validation split is **by company** (no company in both). The model then labels the roles that have no level words.
''')
    nb.code(r'''
# Cell 4 — fine-tune DistilBERT on masked weak labels
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

LABELS = ["entry", "mid", "senior"]
roles["clf_text"] = (roles["role_raw"].fillna("").apply(mask_level_words) + " [SEP] " +
                     roles["text"].fillna("").str[:1500].apply(mask_level_words))
train_pool = roles[roles["level_source"].isin(["title", "years"]) & roles["level_regex"].isin(LABELS)].copy()
cap = 15000 if DEVICE == "cuda" else 1500
train_pool = (train_pool.groupby("level_regex", group_keys=False)
                        .apply(lambda g: g.sample(min(len(g), cap), random_state=SEED)))
if DEVICE != "cuda":
    print("⚠ no GPU: training on a small subset (results are a smoke test, not for the paper)")
gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
tr_idx, va_idx = next(gss.split(train_pool, groups=train_pool["company"].replace("", "unknown")))
tr, va = train_pool.iloc[tr_idx], train_pool.iloc[va_idx]
print("train", tr["level_regex"].value_counts().to_dict(), "| val", va["level_regex"].value_counts().to_dict())

tok = AutoTokenizer.from_pretrained(CLF_MODEL)
clf = AutoModelForSequenceClassification.from_pretrained(CLF_MODEL, num_labels=3).to(DEVICE)
USE_AMP = DEVICE == "cuda" and "deberta" not in CLF_MODEL.lower()
y_tr = tr["level_regex"].map({l: i for i, l in enumerate(LABELS)}).values
w = torch.tensor(len(y_tr) / (3 * np.bincount(y_tr, minlength=3).clip(min=1)), dtype=torch.float, device=DEVICE)
loss_fn = torch.nn.CrossEntropyLoss(weight=w)
EPOCHS, BS, MAXLEN = (2, 32, 256) if DEVICE == "cuda" else (1, 16, 128)
opt = torch.optim.AdamW(clf.parameters(), lr=3e-5, weight_decay=0.01)
steps = EPOCHS * math.ceil(len(tr) / BS)
sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

def batches(texts, bs):
    for i in range(0, len(texts), bs):
        yield i, tok(texts[i:i + bs], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(DEVICE)

t0 = time.time()
texts_tr = tr["clf_text"].tolist()
for ep in range(EPOCHS):
    clf.train()
    perm = np.random.RandomState(SEED + ep).permutation(len(texts_tr))
    tt, yy = [texts_tr[i] for i in perm], y_tr[perm]
    for i, enc in batches(tt, BS):
        with torch.autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
            loss = loss_fn(clf(**enc).logits.float(), torch.tensor(yy[i:i + BS], device=DEVICE))
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(clf.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
    print(f"epoch {ep + 1}/{EPOCHS} done ({time.time() - t0:.0f}s, last loss {loss.item():.3f})")

@torch.no_grad()
def predict(texts, bs=128):
    clf.eval(); out = []
    for i, enc in batches(texts, bs):
        with torch.autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
            out.append(torch.softmax(clf(**enc).logits.float(), -1).cpu().numpy())
    return np.concatenate(out)

pv = predict(va["clf_text"].tolist())
yv = va["level_regex"].map({l: i for i, l in enumerate(LABELS)}).values
val = {"val_n": len(va), "val_acc_vs_weak": accuracy_score(yv, pv.argmax(1)),
       "val_macroF1_vs_weak": f1_score(yv, pv.argmax(1), average="macro")}
pd.DataFrame([val]).to_csv(f"{OUT}/seniority_clf_val.csv", index=False)
print("validation against (masked) weak labels:", {k: round(v, 3) for k, v in val.items()})
print(pd.DataFrame(confusion_matrix(yv, pv.argmax(1)), index=LABELS, columns=LABELS))

P = predict(roles["clf_text"].tolist())
roles["level_clf"] = np.array(LABELS)[P.argmax(1)]
roles["clf_conf"] = P.max(1)
for i, l in enumerate(LABELS):
    roles[f"p_{l}"] = P[:, i]
''')
    nb.code(r'''
# Cell 5 — final level + hand-check export / evaluation
title_ok = (roles["level_source"] == "title") & (roles["level_regex"] != "mixed")
roles["level_final"] = np.select(
    [title_ok, roles["clf_conf"] >= CONF_MIN, roles["level_source"] == "years"],
    [roles["level_regex"], roles["level_clf"], roles["level_regex"]], default="mid")
roles["final_source"] = np.select([title_ok, roles["clf_conf"] >= CONF_MIN, roles["level_source"] == "years"],
                                  ["title", "classifier", "years"], default="default")
print(roles.groupby(["final_source", "level_final"]).size().unstack(fill_value=0))

# blind hand-check sample: half from roles without level words (where the classifier matters)
def take(src_, n):
    g = roles[roles["level_source"] == src_]
    return g.sample(min(n, len(g)), random_state=SEED)
parts = [take("default", HANDCHECK_N // 2), take("title", HANDCHECK_N // 4), take("years", HANDCHECK_N // 4)]
hc = pd.concat(parts).sample(frac=1, random_state=SEED)
blind = hc[["role_id", "comment_id", "month", "role_raw", "header"]].copy()
blind["text_start"] = hc["text"].str[:1200].values
blind["true_level"] = ""          # entry / mid / senior
blind["notes"] = ""
blind.to_csv(f"{OUT}/seniority_handcheck_sample.csv", index=False)
hc[["role_id", "level_source", "level_regex", "level_clf", "clf_conf", "level_final"]].to_csv(
    f"{OUT}/seniority_handcheck_key.csv", index=False)

lab_path = find_file("seniority_handcheck_labeled.csv", required=False)
if lab_path:
    gl = pd.read_csv(lab_path)[["role_id", "true_level"]].dropna()
    gl["true_level"] = gl["true_level"].str.strip().str.lower()
    gl = gl[gl["true_level"].isin(LABELS)].merge(roles, on="role_id")
    rows = []
    for name, col, sub in [("rule (title → years → default mid)", "level_regex", gl[gl["level_regex"] != "mixed"]),
                           ("rule, title-labelled roles only", "level_regex", gl[gl["level_source"] == "title"]),
                           ("classifier only", "level_clf", gl),
                           ("final (title, else classifier, else years)", "level_final", gl),
                           ("rule on roles with no level words", "level_regex", gl[gl["level_source"] == "default"]),
                           ("classifier on roles with no level words", "level_clf", gl[gl["level_source"] == "default"])]:
        if len(sub):
            rows.append({"labeller": name, "n": len(sub), "accuracy": accuracy_score(sub["true_level"], sub[col]),
                         "macro_f1": f1_score(sub["true_level"], sub[col], average="macro")})
    ev = pd.DataFrame(rows); ev.to_csv(f"{OUT}/seniority_eval.csv", index=False)
    print(ev.round(3).to_string(index=False))
else:
    print("No hand labels yet: fill true_level in seniority_handcheck_sample.csv, upload as 'jobai-seniority-labels'.")
roles.drop(columns=["text", "clf_text"]).to_csv(f"{OUT}/role_levels.csv", index=False)
''')
    nb.md("## Fig 4a — entry-level share of roles by month, high vs low exposure")
    nb.code(r'''
# Cell 6 — Fig 4a
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, col, title in [(axes[0], RQ3_EXP, "Within-HN exposure split"), (axes[1], "high_exposure", "Population split")]:
    r = roles[roles["level_final"].isin(["entry", "senior"])].dropna(subset=[col])
    s = (r.assign(entry=(r["level_final"] == "entry").astype(int))
          .groupby(["month", col])["entry"].mean().unstack())
    s.index = pd.to_datetime(s.index + "-01")
    s = s.rolling(3, min_periods=1).mean()
    for g, lab_, ls in [(1.0, "High exposure", "-"), (0.0, "Low exposure", "--")]:
        if g in s:
            ax.plot(s.index, 100 * s[g], ls, label=lab_)
    ax.axvline(pd.Timestamp("2022-11-30"), color="k", lw=0.8)
    ax.set_title(title, fontsize=9); ax.set_ylabel("entry roles, % of entry + senior (3-mo avg)")
axes[0].legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4a_entry_share.png", bbox_inches="tight"); plt.show()
''')
    nb.md(r'''
## Step 3 — main RQ3 design: junior vs senior within the same company and month (PPML)
Cells = company × month × level (entry/senior) × exposure group, **zero-filled** within every company-month that posts. Only companies that post both before and after Dec 2022. `exp(β) − 1` is the % change.
''')
    nb.code(r'''
# Cell 7 — company × month PPML
def make_cells(df, exp_col):
    lv = df[df["level_final"].isin(["entry", "senior"]) & (df["company"] != "") & df[exp_col].notna()]
    cell = lv.groupby(["company", "month", "level_final", exp_col]).size().rename("ads").reset_index()
    cm = cell[["company", "month"]].drop_duplicates()
    grid = (cm.merge(pd.DataFrame({"level_final": ["entry", "senior"]}), how="cross")
              .merge(pd.DataFrame({exp_col: [0.0, 1.0]}), how="cross"))
    cell = grid.merge(cell, how="left", on=["company", "month", "level_final", exp_col]).fillna({"ads": 0})
    cell["junior"] = (cell["level_final"] == "entry").astype(int)
    cell["high"] = cell[exp_col].astype(int)
    cell["post"] = (cell["month"] >= "2022-12").astype(int)
    cell["year"] = cell["month"].str[:4].astype(int)
    cell["company_month"] = cell["company"] + "|" + cell["month"]
    cell["level_exp"] = cell["level_final"] + "_" + cell["high"].astype(str)
    active = cell[cell["ads"] > 0].groupby("company")["post"].nunique()
    return cell[cell["company"].isin(active[active == 2].index)].copy()

cell = make_cells(roles, RQ3_EXP)
print(f"{cell['company'].nunique():,} companies posting before and after; {cell['company_month'].nunique():,} company-months; "
      f"{int(cell['ads'].sum()):,} entry/senior roles")
CV = {"CRV1": "company"}
res = []
TRIPLE = "ads ~ junior:post + post:high + junior:post:high | company_month + level_exp"
def add(label, fn):
    try:
        res.append(coef_rows(fn(), label))
    except Exception as e:
        print(f"  {label} failed: {type(e).__name__} {str(e)[:120]}")

add(f"PPML main ({RQ3_EXP})", lambda: pf.fepois(TRIPLE, data=cell, vcov=CV))
add("PPML junior × post only", lambda: pf.fepois("ads ~ junior:post | company_month + level_final", data=cell, vcov=CV))
add("PPML population exposure split", lambda: pf.fepois(TRIPLE, data=make_cells(roles, "high_exposure"), vcov=CV))
add("PPML title-labelled roles only",
    lambda: pf.fepois(TRIPLE, data=make_cells(roles[roles["final_source"] == "title"], RQ3_EXP), vcov=CV))
cell_g = cell.copy(); cell_g["post"] = (cell_g["month"] >= "2023-03").astype(int)
add("PPML treatment Mar 2023 (GPT-4)", lambda: pf.fepois(TRIPLE, data=cell_g, vcov=CV))
add("PPML drop COVID months (Mar 2020–Dec 2021)",
    lambda: pf.fepois(TRIPLE, data=cell[~cell["month"].between("2020-03", "2021-12")], vcov=CV))
# role-level linear probability: is a posted role junior? (within company-month)
rl = roles[roles["level_final"].isin(["entry", "senior"]) & (roles["company"] != "")].dropna(subset=[RQ3_EXP]).copy()
rl["junior"] = (rl["level_final"] == "entry").astype(int)
rl["high"] = rl[RQ3_EXP].astype(int)
rl["company_month"] = rl["company"] + "|" + rl["month"]
add("LPM P(junior), company × month FE", lambda: pf.feols("junior ~ high + post:high | company_month", data=rl, vcov=CV))
rq3 = pd.concat(res, ignore_index=True)
rq3["pct_effect"] = np.where(rq3["model"].str.startswith("PPML"), 100 * (np.exp(rq3["Estimate"]) - 1), np.nan)
rq3.to_csv(f"{OUT}/rq3_ppml_results.csv", index=False)
rq3[["model", "term", "Estimate", "Std. Error", "Pr(>|t|)", "pct_effect", "nobs"]].round(4)
''')
    nb.code(r'''
# Cell 8 — Fig 4b: junior × year effects within company-month, by exposure group (ref. 2022)
ev_rows = []
for g, name in [(1, "high exposure"), (0, "low exposure")]:
    sub = cell[cell["high"] == g]
    try:
        f = pf.fepois("ads ~ i(year, junior, ref=2022) | company_month + level_final", data=sub, vcov=CV)
        t = coef_rows(f, name, keep="year")
        t["year"] = t["term"].str.extract(r"(20\d\d)").astype(float)
        ev_rows.append(t)
    except Exception as e:
        print(name, "failed:", e)
rq3_ev = pd.concat(ev_rows, ignore_index=True) if ev_rows else pd.DataFrame(columns=["model", "year", "Estimate", "Std. Error"])
rq3_ev.to_csv(f"{OUT}/rq3_event_coefs.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 3.5))
for (name, g), dx in zip(rq3_ev.groupby("model"), [-0.08, 0.08]):
    ax.errorbar(g["year"] + dx, g["Estimate"], yerr=1.96 * g["Std. Error"], fmt="o-", ms=3, capsize=2, label=name)
ax.axhline(0, color="k", lw=0.6); ax.axvline(2022.5, color="k", lw=0.8, ls=":")
ax.set_ylabel("junior vs senior, log points (ref. 2022)"); ax.legend(frameon=False)
ax.set_title("Junior postings relative to senior, within company × month", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4b_junior_event.png", bbox_inches="tight"); plt.show()
''')
    nb.md("## Step 4 — other RQ3 outcomes: required years and skill content (from NB4)")
    nb.code(r'''
# Cell 9 — required years and skill content
out_rows = []
yr = roles[roles["min_years"].notna() & (roles["company"] != "")].dropna(subset=[RQ3_EXP]).copy()
yr["high"] = yr[RQ3_EXP].astype(int)
yr["junior"] = (yr["level_final"] == "entry").astype(int)
for lvl in ["entry", "senior", None]:
    s = yr if lvl is None else yr[yr["level_final"] == lvl]
    try:
        f = pf.feols("min_years ~ post:high + remote | company + month", data=s, vcov=CV)
        out_rows.append(coef_rows(f, f"min years required ({lvl or 'all levels'})", keep="post"))
    except Exception as e:
        print("min_years", lvl, "failed:", e)
if SKL:
    ss = pd.read_csv(f"{SKL}/ad_skill_summary.csv", dtype={"month": str})
    prim = roles[roles["role_idx"] == 0][["comment_id", "level_final", RQ3_EXP, "remote", "post", "company"]]
    ad = ss.drop(columns=["company"], errors="ignore").merge(prim, on="comment_id")
    ad = ad[ad["level_final"].isin(["entry", "senior"]) & (ad["company"] != "")].dropna(subset=[RQ3_EXP])
    ad["junior"] = (ad["level_final"] == "entry").astype(int); ad["high"] = ad[RQ3_EXP].astype(int)
    for y in ["n_skills", "mean_skill_beta", "share_high_beta", "auto_share", "has_ai_skill"]:
        if y in ad and ad[y].notna().sum() > 500:
            try:
                f = pf.feols(f"{y} ~ junior:post + post:high + junior:post:high + junior + remote | company + month",
                             data=ad.dropna(subset=[y]), vcov=CV)
                out_rows.append(coef_rows(f, f"skill content: {y}", keep="post"))
            except Exception as e:
                print(y, "failed:", e)
else:
    print("jobai-skills not attached → skill-content outcomes skipped (attach it after NB4 has run)")
rq3_out = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()
rq3_out.to_csv(f"{OUT}/rq3_outcomes.csv", index=False)
rq3_out[["model", "term", "Estimate", "Std. Error", "Pr(>|t|)", "nobs"]].round(4) if len(rq3_out) else None
''')
    nb.code(r'''
# Cell 10 — NB3's occupation DiD re-run separately for entry and senior roles
panel = pd.read_csv(f"{PROC}/panel_occ_month.csv", dtype={"soc6": str, "month": str})
occs = panel["soc6"].unique(); months = sorted(panel["month"].unique())
rows = []
for lvl in ["entry", "senior"]:
    c = roles[(roles["level_final"] == lvl) & roles["soc6"].isin(occs)].groupby(["soc6", "month"]).size()
    full = c.reindex(pd.MultiIndex.from_product([occs, months], names=["soc6", "month"]), fill_value=0).rename("ads").reset_index()
    full = full.merge(panel.drop_duplicates("soc6")[["soc6", "high_exposure", "high_exp_hn"]], on="soc6")
    full["log_ads"] = np.log1p(full["ads"]); full["post"] = (full["month"] >= "2022-12").astype(int)
    full["level"] = lvl
    rows.append(full)
bl = pd.concat(rows, ignore_index=True)
res = []
def safe_rows(fml, data, label):
    """Fit and return coefficient rows; if the split has no variation (e.g. population split: ~0.2% control ads),
    record the model as not estimable instead of stopping the notebook."""
    try:
        return coef_rows(pf.feols(fml, data=data, vcov={"CRV1": "soc6"}), label, keep="post")
    except Exception as e:
        print(f"⚠ {label}: not estimable ({type(e).__name__}); rows = {len(data):,}")
        return pd.DataFrame([{"term": "not estimable", "model": label, "nobs": len(data), "note": str(e).strip()[:120]}])
for col in ["high_exp_hn", "high_exposure"]:
    for lvl in ["entry", "senior"]:
        s = bl[(bl["level"] == lvl)].dropna(subset=[col])
        res.append(safe_rows(f"log_ads ~ post:{col} | soc6 + month", s, f"{lvl} roles, {col}"))
    s = bl.dropna(subset=[col]).copy(); s["entry"] = (s["level"] == "entry").astype(int)
    # post:{col} = senior-role effect; post:{col}:entry = entry minus senior (the triple difference)
    res.append(safe_rows(f"log_ads ~ post:{col} + post:{col}:entry | soc6^level + month^level", s,
                         f"triple difference entry vs senior, {col}"))
did_lvl = pd.concat(res, ignore_index=True)
did_lvl.to_csv(f"{OUT}/did_by_level.csv", index=False)
did_lvl.reindex(columns=["model", "term", "Estimate", "Std. Error", "Pr(>|t|)"]).round(4)
''')
    nb.md(r'''
## Step 5 — countries (RQ4): change in Indeed postings, 2025–26 vs Feb 2020–Nov 2022
Sector exposure comes from NB2 (US example titles), applied to the same Indeed sector names in every country. The pre-period includes the 2020 dip and the 2021–22 boom, so the quantity of interest is the **gap** between high- and low-exposure sectors, not the raw change.
''')
    nb.code(r'''
# Cell 11 — country × sector changes, heatmap, gaps, panel regression
sec = pd.read_csv(f"{RAW}/indeed_sector.csv", parse_dates=["date"])
agg = pd.read_csv(f"{RAW}/indeed_aggregate.csv", parse_dates=["date"])
# aggregate file has _SA / _NSA index columns (sector file has one plain column): use seasonally adjusted
if "indeed_job_postings_index" not in agg:
    agg["indeed_job_postings_index"] = agg.get("indeed_job_postings_index_SA", agg.get("indeed_job_postings_index_NSA"))
sx = pd.read_csv(f"{PROC}/sector_exposure.csv")
VAR = "total postings" if (sec["variable"] == "total postings").any() else sec["variable"].iloc[0]
m = (sec[sec["variable"] == VAR].groupby(["jobcountry", "display_name", pd.Grouper(key="date", freq="MS")])
        ["indeed_job_postings_index"].mean().rename("idx").reset_index())
LAST = m["date"].max()
m = m.merge(sx[["display_name", "exp_main", "high_sector"]], on="display_name", how="left")
for c, g in m.groupby("jobcountry"):
    miss = sorted(g.loc[g["exp_main"].isna(), "display_name"].unique())
    if miss:
        print(f"{c}: sectors without exposure → {miss[:8]}{'…' if len(miss) > 8 else ''}")
pre = m[m["date"].between(*PRE_IDX)].groupby(["jobcountry", "display_name"])["idx"].mean()
late = m[m["date"] >= LATE_FROM].groupby(["jobcountry", "display_name"])["idx"].mean()
chg = pd.DataFrame({"pre": pre, "late": late}).dropna()
chg["pct_change"] = 100 * (chg["late"] / chg["pre"] - 1)
chg = chg.reset_index().merge(sx[["display_name", "exp_main", "high_sector"]], on="display_name", how="left")
chg.to_csv(f"{OUT}/country_change.csv", index=False)

gaps = []
rs = np.random.RandomState(SEED)
for c, g in chg.dropna(subset=["high_sector"]).groupby("jobcountry"):
    hi, lo = g.loc[g["high_sector"] == 1, "pct_change"].values, g.loc[g["high_sector"] == 0, "pct_change"].values
    if len(hi) and len(lo):
        b = [rs.choice(hi, len(hi)).mean() - rs.choice(lo, len(lo)).mean() for _ in range(5000)]
        gaps.append({"country": c, "high_mean": hi.mean(), "low_mean": lo.mean(), "gap_pp": hi.mean() - lo.mean(),
                     "ci_low": np.percentile(b, 2.5), "ci_high": np.percentile(b, 97.5), "n_high": len(hi), "n_low": len(lo)})
gaps = pd.DataFrame(gaps); gaps.to_csv(f"{OUT}/country_gap.csv", index=False)
print(gaps.round(2).to_string(index=False))

hm = chg.pivot_table(index="jobcountry", columns="display_name", values="pct_change")
order = sx.sort_values("exp_main", ascending=False)["display_name"]
hm = hm.reindex(index=[c for c in SECTOR_COUNTRIES if c in hm.index], columns=[s for s in order if s in hm.columns])
fig, ax = plt.subplots(figsize=(max(10, 0.32 * hm.shape[1]), 3.6))
lim = np.nanpercentile(np.abs(hm.values), 95)
im = ax.imshow(hm.values, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
ax.set_yticks(range(len(hm.index))); ax.set_yticklabels(hm.index)
ax.set_xticks(range(len(hm.columns))); ax.set_xticklabels(hm.columns, rotation=75, ha="right", fontsize=6)
ax.set_title(f"% change in Indeed postings, {LATE_FROM[:7]}–{LAST:%Y-%m} vs Feb 2020–Nov 2022 "
             "(sectors ordered from most to least AI-exposed)", fontsize=8)
fig.colorbar(im, ax=ax, fraction=0.02)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5a_country_heatmap.png", bbox_inches="tight"); plt.show()

p = m.dropna(subset=["high_sector"]).copy()
p["log_idx"] = np.log(p["idx"]); p["post"] = (p["date"] >= "2022-12-01").astype(int)
p["post_high"] = p["post"] * p["high_sector"]
p["cs"] = p["jobcountry"] + "|" + p["display_name"]; p["cm"] = p["jobcountry"] + "|" + p["date"].dt.strftime("%Y-%m")
regs = [coef_rows(pf.feols("log_idx ~ post_high | cs + cm", data=p, vcov={"CRV1": "cs"}), "pooled: post × high sector"),
        coef_rows(pf.feols("log_idx ~ i(jobcountry, post_high) | cs + cm", data=p, vcov={"CRV1": "cs"}),
                  "by country: post × high sector")]
pd.concat(regs).to_csv(f"{OUT}/country_panel_regs.csv", index=False)

ag = agg[agg["variable"] == VAR] if "variable" in agg else agg
ag = ag.groupby(["jobcountry", pd.Grouper(key="date", freq="MS")])["indeed_job_postings_index"].mean().unstack(0)
fig, ax = plt.subplots(figsize=(9, 3.6))
for c in ag.columns:
    ax.plot(ag.index, ag[c], lw=1.4 if c == "US" else 0.8, label=c)
ax.axvline(pd.Timestamp("2022-11-30"), color="k", lw=0.8)
ax.set_ylabel("Indeed postings index, all jobs (Feb 2020 = 100)"); ax.legend(ncol=6, fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/figA_country_totals.png", bbox_inches="tight"); plt.show()
''')
    nb.md(r'''
## Step 6 — Section 174 test: US software vs the same sector abroad
The 2022 US change to R&D amortisation (software salaries deductible over 5 years from tax years starting in 2022) applies only in the US. If software postings fell as much in GB, CA, DE, FR and AU, neither Section 174 nor a purely US rate story explains the drop. The 2025 restoration of domestic R&E expensing (OBBBA, July 2025 — **verify the date and scope before citing**) gives a second, reverse test. With 6 countries, use the Webb wild-bootstrap p-values.
''')
    nb.code(r'''
# Cell 12 — Section 174: plots, DiD, triple difference
sw = p[p["display_name"].str.contains("Software Development", case=False)].copy()
if sw.empty:
    print("⚠ no 'Software Development' sector found — check display_name values:", sorted(p["display_name"].unique())[:20])
s174 = sw.pivot_table(index="date", columns="jobcountry", values="idx")
s174.to_csv(f"{OUT}/section174_series.csv")
fig, ax = plt.subplots(figsize=(9, 3.8))
for c in s174.columns:
    ax.plot(s174.index, s174[c], lw=2 if c == "US" else 0.9, label=c, color="k" if c == "US" else None)
for (lab_, dte), ls in zip(EVENTS.items(), [":", "-.", "-", "--"]):
    ax.axvline(pd.Timestamp(dte), color="grey", lw=0.8, ls=ls, label=lab_)
ax.set_ylabel("Indeed 'Software Development' postings (Feb 2020 = 100)")
ax.legend(fontsize=6, ncol=2, frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5b_section174.png", bbox_inches="tight"); plt.show()

tests = []
sw["us"] = (sw["jobcountry"] == "US").astype(int)
sw["month"] = sw["date"].dt.strftime("%Y-%m")
for lab_, dte in [("post Jan 2022 (Section 174)", "2022-01-01"), ("post Dec 2022 (ChatGPT)", "2022-12-01")]:
    sw["after"] = (sw["date"] >= dte).astype(int)
    f = pf.feols("log_idx ~ us:after | jobcountry + month", data=sw, vcov={"CRV1": "jobcountry"})
    r = coef_rows(f, f"software, US vs 5 countries, {lab_}", keep="us")
    r["p_wild_webb"] = boot_p(f, r["term"].iloc[0], weights="webb")
    tests.append(r)
late = sw[sw["date"] >= "2023-01-01"].copy(); late["after"] = (late["date"] >= EVENTS[list(EVENTS)[-1]]).astype(int)
f = pf.feols("log_idx ~ us:after | jobcountry + month", data=late, vcov={"CRV1": "jobcountry"})
r = coef_rows(f, "software, US vs others, after Jul 2025 expensing restored (2023+ sample)", keep="us")
r["p_wild_webb"] = boot_p(f, r["term"].iloc[0], weights="webb"); tests.append(r)
tri = p[p["jobcountry"].isin(SECTOR_COUNTRIES)].copy()
tri["us"] = (tri["jobcountry"] == "US").astype(int)
tri["software"] = tri["display_name"].str.contains("Software Development", case=False).astype(int)
tri["sm"] = tri["display_name"] + "|" + tri["date"].dt.strftime("%Y-%m")
tri["after"] = (tri["date"] >= "2022-01-01").astype(int)
f = pf.feols("log_idx ~ us:software:after | cs + cm + sm", data=tri, vcov={"CRV1": "cs"})
tests.append(coef_rows(f, "triple difference US × software × post-Jan 2022", keep="software"))
s174_tests = pd.concat(tests, ignore_index=True)
s174_tests.to_csv(f"{OUT}/section174_tests.csv", index=False)
s174_tests[["model", "term", "Estimate", "Std. Error", "Pr(>|t|)"] +
           (["p_wild_webb"] if "p_wild_webb" in s174_tests else [])].round(4)
''')
    nb.md(r'''
## Step 7 — synthetic control for the US software index
Donor pool A (plan): other countries' software indices + other US sectors. Donor pool B: non-US sectors only (no US unit can be hit by Section 174). Weights ≥ 0 summing to 1, fitted on monthly log index over the pre-period; two treatment dates (Jan 2022, Dec 2022). Inference: in-space placebos (each donor treated in turn), p = rank of the US post/pre RMSPE ratio. `pysyncon` is run as a cross-check of the weights.
''')
    nb.code(r'''
# Cell 13 — synthetic control (own implementation + pysyncon cross-check)
W = p.copy(); W["unit"] = W["jobcountry"] + "|" + W["display_name"]
Y = W.pivot_table(index="date", columns="unit", values="log_idx").dropna(axis=1)
TREATED = next((u for u in Y.columns if u.startswith("US|") and "Software Development" in u), None)
assert TREATED, "US software unit not found"
MAX_DONORS, MAX_PLACEBOS = 60, 40
pre_corr = Y[Y.index < "2022-01-01"].corr()[TREATED]
nonus = [u for u in Y.columns if not u.startswith("US|")]
pools = {"A: other countries' software + other US sectors":
             [u for u in Y.columns if u != TREATED and (u.startswith("US|") or "Software Development" in u)],
         # pool B: the non-US sectors whose pre-2022 path is closest to US software (keeps it tractable)
         "B: non-US sectors only": list(pre_corr[nonus].sort_values(ascending=False).index[:MAX_DONORS])}

def synth(y, X, pre):
    """Non-negative weights summing to 1 minimising pre-period squared error (analytic gradient)."""
    n = X.shape[1]; Xp, yp = X[pre], y[pre]
    obj = lambda w: np.sum((yp - Xp @ w) ** 2)
    jac = lambda w: -2 * Xp.T @ (yp - Xp @ w)
    r = optimize.minimize(obj, np.full(n, 1 / n), jac=jac, method="SLSQP", bounds=[(0, 1)] * n,
                          constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}],
                          options={"maxiter": 500})
    return r.x

def rmspe_ratio(y, yhat, pre):
    return np.sqrt(np.mean((y[~pre] - yhat[~pre]) ** 2)) / max(np.sqrt(np.mean((y[pre] - yhat[pre]) ** 2)), 1e-9)

paths, weights, placebos = [], [], []
for pool_name, donors in pools.items():
    for t_name, t0 in [("Jan 2022", "2022-01-01"), ("Dec 2022", "2022-12-01")]:
        pre = (Y.index < t0)
        y, X = Y[TREATED].values, Y[donors].values
        w = synth(y, X, pre); yhat = X @ w
        paths.append(pd.DataFrame({"date": Y.index, "us_software": y, "synthetic": yhat, "gap": y - yhat,
                                   "pool": pool_name, "treatment": t_name}))
        weights.append(pd.DataFrame({"donor": donors, "weight": w, "pool": pool_name, "treatment": t_name})
                       .query("weight > 0.01").sort_values("weight", ascending=False))
        ratios = {TREATED: rmspe_ratio(y, yhat, pre)}
        plac_units = donors if len(donors) <= MAX_PLACEBOS else \
            list(np.random.RandomState(SEED).choice(donors, MAX_PLACEBOS, replace=False))
        for d_ in plac_units:
            others = [u for u in donors if u != d_]
            wd = synth(Y[d_].values, Y[others].values, pre)
            ratios[d_] = rmspe_ratio(Y[d_].values, Y[others].values @ wd, pre)
        rank = sorted(ratios.values(), reverse=True).index(ratios[TREATED]) + 1
        placebos.append({"pool": pool_name, "treatment": t_name, "donors": len(donors),
                         "us_rmspe_ratio": ratios[TREATED], "rank": rank, "p_value": rank / len(ratios),
                         "mean_gap_post": float(np.mean((y - yhat)[~pre])),
                         "pre_rmspe": float(np.sqrt(np.mean((y - yhat)[pre] ** 2)))})
paths = pd.concat(paths); paths.to_csv(f"{OUT}/synth_paths.csv", index=False)
pd.concat(weights).to_csv(f"{OUT}/synth_weights.csv", index=False)
placebos = pd.DataFrame(placebos); placebos.to_csv(f"{OUT}/synth_placebos.csv", index=False)
print(placebos.round(4).to_string(index=False))

try:
    from pysyncon import Dataprep, Synth
    long = Y.reset_index().melt(id_vars="date", var_name="unit", value_name="log_idx")
    long["t"] = (long["date"].dt.year - 2020) * 12 + long["date"].dt.month
    pre_t = sorted(long.loc[long["date"] < "2022-12-01", "t"].unique())
    dp = Dataprep(foo=long, predictors=["log_idx"], predictors_op="mean", time_predictors_prior=pre_t,
                  special_predictors=[("log_idx", [t], "mean") for t in pre_t[::3]], dependent="log_idx",
                  unit_variable="unit", time_variable="t", treatment_identifier=TREATED,
                  controls_identifier=pools[list(pools)[0]], time_optimize_ssr=pre_t)
    sy = Synth(); sy.fit(dataprep=dp)
    print("pysyncon weights (pool A, Dec 2022):\n", sy.weights().sort_values(ascending=False).head(8).round(3))
except Exception as e:
    print("pysyncon cross-check skipped:", type(e).__name__, str(e)[:150])

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, t_name in zip(axes, ["Jan 2022", "Dec 2022"]):
    g = paths[(paths["treatment"] == t_name) & (paths["pool"] == list(pools)[0])]
    ax.plot(g["date"], np.exp(g["us_software"]), "k", lw=1.6, label="US software")
    ax.plot(g["date"], np.exp(g["synthetic"]), "--", label="synthetic US software")
    ax.axvline(pd.Timestamp("2022-01-01" if t_name == "Jan 2022" else "2022-12-01"), color="grey", lw=0.8)
    ax.set_title(f"Treatment {t_name} (donor pool A)", fontsize=9)
axes[0].set_ylabel("index (Feb 2020 = 100)"); axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5c_synth.png", bbox_inches="tight"); plt.show()
''')
    nb.code(OUTPUT_LIST)
    return nb
