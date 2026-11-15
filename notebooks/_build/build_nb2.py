"""NB2 — 02-clean-and-link."""
from common import NB, PIP, FIND, SEEDS, OUTPUT_LIST, VERSIONS
from hn_rules import HN_RULES_SRC

# Pure-stdlib helpers: inserted into the notebook AND unit-tested locally (test_helpers.py).
HELPERS = r'''
import re

# Words that mark a job role. \b...\b = whole words only ("ai" must not match "maintain").
ROLE_WORDS = (r"\b(?:engineer|engineering|developer|dev|programmer|scientist|designer|manager|analyst|architect|"
              r"administrator|admin|writer|support|marketing|marketer|sales|devops|sre|researcher|product|"
              r"recruiter|data|ml|ai|consultant|specialist|technician|accountant|counsel|lawyer|attorney|editor|"
              r"operations|ops|qa|tester|coordinator|representative|success|officer|cto|cfo|coo|vp|head|director|"
              r"lead|intern|internship|founder|generalist|associate|executive|strategist|economist|statistician|"
              r"teacher|instructor|nurse|physician|clinician|pharmacist|copywriter|illustrator|animator|producer|"
              r"planner|bookkeeper|controller|auditor|paralegal|assistant|evangelist|hacker|full[- ]?stack|"
              r"front[- ]?end|back[- ]?end|mobile|ios|android|security|infrastructure|platform)s?\b")
ROLE_RE = re.compile(ROLE_WORDS, re.I)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
HIRING_RE = re.compile(
    r"\b(?:hiring|looking for|seeking|searching for)\s+(?:an?\s+|our\s+(?:first\s+)?|(?:a\s+)?(?:few|several|multiple|two|three|\d+)\s+)?"
    r"((?:[A-Za-z0-9+#./-]+\s+){0,4}?(?:engineer|developer|programmer|scientist|designer|manager|analyst|architect|"
    r"administrator|writer|marketer|researcher|recruiter|consultant|specialist|technician|accountant|lead|intern|director)s?)\b",
    re.I)
SPLIT_RE = re.compile(r"\s*(?:,|;|\s&\s|\sand\s|\s/\s|\s\+\s|\sor\s)\s*", re.I)
KEEP_S = {"sales", "operations", "devops", "mlops", "analytics", "logistics", "economics", "graphics", "robotics",
          "physics", "electronics", "systems", "ops", "relations", "communications", "services", "ios", "aws",
          "kubernetes", "news", "affairs", "business", "success"}

def header_line(text):
    """First non-empty line of an ad (HN ads usually start 'Company | Role | Location | ...')."""
    for line in (text or "").split("\n"):
        line = line.strip()
        if line:
            if "|" not in line and len(line) > 200:          # one-paragraph ad: cut at first sentence end
                m = re.search(r"[.!?](\s|$)", line[:300])
                line = line[:m.end()].strip() if m else line[:200]
            return line[:400]
    return ""

def split_header(h):
    """Split a header into fields on '|'; fall back to ' - ' / ' — ' / ' – ' separators."""
    for sep in ["|", "｜", " — ", " – ", " - ", " • "]:
        parts = [p.strip() for p in h.split(sep)]
        parts = [p for p in parts if p]
        if len(parts) >= 2:
            return parts
    return [h.strip()] if h.strip() else []

def split_roles(part):
    """'Backend Engineer, Data Scientist' -> two roles; keeps 'Senior and Staff Engineers' whole."""
    base = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", part)
    base = re.sub(r"\s+", " ", base).strip(" -:")
    pieces = [p.strip(" -:") for p in SPLIT_RE.split(base) if p and p.strip(" -:")]
    if len(pieces) > 1 and all(ROLE_RE.search(p) for p in pieces):
        return pieces
    return [base] if base else []

def extract_roles(text, max_roles=6):
    """Return (header, company_field, roles, source). source = 'header' | 'text' | 'none'."""
    h = header_line(text)
    parts = split_header(h)
    company = parts[0] if parts else ""
    roles = []
    for p in parts[1:]:
        if URL_RE.search(p) or not ROLE_RE.search(p):
            continue
        roles.extend(split_roles(p))
    if roles:
        return h, company, roles[:max_roles], "header"
    m = HIRING_RE.search((text or "")[:600])
    if m:
        return h, company, [m.group(1).strip()], "text"
    if len(parts) == 1 and ROLE_RE.search(parts[0]) and len(parts[0]) < 80:   # header is just a role
        return h, "", [parts[0]], "header"
    return h, company, [], "none"

def clean_role(r):
    """Normalise a role for SOC matching: drop level words, locations-in-brackets, work-mode words, plurals.
    (Level words are removed only here; NB5 reads seniority from the raw role text.)"""
    r = (r or "").lower()
    r = URL_RE.sub(" ", r)
    r = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", r)
    r = re.sub(r"\b(?:senior|sr\.?|junior|jr\.?|staff|principal|intern(?:ship)?|entry[- ]level|mid[- ]level|"
               r"new[- ]grad(?:uate)?|i{1,3}|iv|level\s*\d|l\d)\b", " ", r)
    r = re.sub(r"\blead\b\s+(?=[a-z])", " ", r)              # 'lead engineer' -> 'engineer'; 'team lead' stays
    r = re.sub(r"\b(?:remote|onsite|on-site|hybrid|full[- ]?time|part[- ]?time|contract|multiple|several|various|"
               r"positions?|roles?|openings?|opportunit(?:y|ies)|wanted|needed|hiring|m/f/d|f/m/d|w/m/d)\b", " ", r)
    r = re.sub(r"[^a-z0-9+#./ -]", " ", r)
    r = re.sub(r"\s+", " ", r).strip(" -/.")
    words = r.split()
    if words:
        w = words[-1]
        if w.endswith("s") and not w.endswith("ss") and w not in KEEP_S and len(w) > 3:
            words[-1] = w[:-1]
    return " ".join(words)

def norm_title(t):
    """Normalisation used for exact title lookups (ad role vs O*NET titles)."""
    t = re.sub(r"[^a-z0-9+# ]", " ", (t or "").lower())
    words = t.split()
    if words:
        w = words[-1]
        if w.endswith("s") and not w.endswith("ss") and w not in KEEP_S and len(w) > 3:
            words[-1] = w[:-1]
    return " ".join(words)

def company_key(field):
    """Normalised company name from the first header field ('' if it does not look like a company)."""
    c = (field or "").lower()
    c = URL_RE.sub(" ", c)
    c = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", c)                 # '(YC S19)', '[remote]'
    c = re.sub(r"\b(?:is hiring|are hiring|hiring)\b.*$", " ", c)
    c = re.sub(r"\.(?:com|io|ai|co|org|net|dev|app|xyz|tech|so|sh)\b", " ", c)
    c = re.sub(r"\b(?:inc|llc|ltd|limited|gmbh|corp|corporation|co|plc|s\.?a|b\.?v|ag|pty|technologies|labs?)\b\.?", " ", c)
    c = re.sub(r"[^a-z0-9 ]", " ", c)
    c = re.sub(r"\s+", " ", c).strip()
    if not c or len(c.split()) > 6 or re.fullmatch(r"(?:remote|onsite|hybrid|full time|we re|we are)", c):
        return ""
    return c[:60]

def ad_flags(text):
    t = (text or "").lower()
    return {
        "remote": int(bool(re.search(r"\bremote\b", t))),
        "onsite": int(bool(re.search(r"\b(?:onsite|on-site|on site|in[- ]office)\b", t))),
        "hybrid": int(bool(re.search(r"\bhybrid\b", t))),
        "visa": int(bool(re.search(r"\bvisa\b", t))),
        "salary": int(bool(re.search(r"[$€£]\s?\d|\b\d{2,3}\s?k\b|\bsalary\b", t))),
    }

def rel_month(ym):
    """Months since Dec 2022 (0 = Dec 2022, first full month after ChatGPT; Nov 2022 = -1)."""
    y, m = int(ym[:4]), int(ym[5:7])
    return (y - 2022) * 12 + m - 12
'''


def build():
    nb = NB(gpu=True, internet=True)
    nb.md(r'''
# NB2 — Clean ads and link them to occupations (`02-clean-and-link`)
**Project:** *Is AI Changing Who Gets Hired?* — `RESEARCH_PLAN.md` §4 and `KAGGLE_IMPLEMENTATION.md` §3.

| | |
|---|---|
| **Input** | dataset **`jobai-raw`** (from NB1): `hn_jobs.csv`, `fred_series.csv`, O\*NET tables (`onet_Occupation_Data.csv`, `onet_Task_Statements.csv`, `onet_Job_Titles.csv` + `onet_Sample_of_Reported_Titles.csv` — O\*NET 31; falls back to `onet_Alternate_Titles.csv` for O\*NET ≤ 30), `exposure_eloundou.csv`, `aioe/aioe_*.csv`, `anthropic/release_*/data/aei_*.csv`, `indeed_sector_job_title_examples.csv` |
| **Optional input** | dataset **`jobai-handcheck`** with your labelled `soc_handcheck_labeled.csv` (second run only) |
| **Output** → dataset **`jobai-processed`** | `hn_clean.csv` (one row per ad), `ad_roles.csv` (one row per role in an ad), `panel_occ_month.csv`, `exposure_table.csv`, `sector_exposure.csv`, `aei_task_table.csv`, `handcheck_soc_sample.csv`, `cleaning_log.csv`, `control_group_check.csv`, `soc_match_summary.csv`, `onet_soc_crosswalk_2010_2019.csv` |
| **Kaggle settings** | Accelerator **GPU T4** (CPU works, ~4× slower) · **Internet On** (pip, Hugging Face models, O\*NET crosswalk) |
| **Run time** | ~25–40 min on T4 (most of it reading the 300 MB Anthropic files and encoding ~100k titles) |

**Steps:** 1 clean ads → 2 extract company, roles and flags → 3 load O\*NET titles → 4 match each role to an O\*NET occupation with a **job-title encoder** (JobBERT-v2, MiniLM fallback; exact title matches first) → 5 hand-check export (+ accuracy if labels are uploaded) → 6 exposure merge (Eloundou β main, AIOE, Anthropic usage, ensemble = first principal component) → 7 control-group size check → 8 occupation × month panel with zero-fill (`rel_month` 0 = Dec 2022) → 9 Indeed sector exposure.

**Two-pass workflow for the hand-check:** run once → download `handcheck_soc_sample.csv` → fill `correct_soc6` (1/0) and, when wrong, `true_soc6` → upload as dataset `jobai-handcheck` → re-run. The second run reports accuracy and sets the high-confidence threshold from your labels.

Nothing in this notebook has been run yet; every number is produced when you run it.
''')
    nb.code(PIP + r'''
# Cell 1 — installs (all here so "Run All" works), imports, settings
try:
    import sentence_transformers  # preinstalled on Kaggle; install only if missing
except ImportError:
    pip("sentence-transformers")
''' + FIND + r'''
import re, io, json, math, time, warnings
import numpy as np, pandas as pd
import torch
''' + SEEDS + VERSIONS + r'''
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 200, "display.max_columns", 40)

OUT = "/kaggle/working"
RAW = find_dir("jobai-raw", "hn_jobs.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("raw data:", RAW, "| device:", DEVICE)

START_MONTH = "2019-01"
MIN_CHARS = 100                 # drop very short ads
MIN_ADS_OCC = 20                # occupations with fewer ads in the whole period are left out of the panel
TITLE_MODEL = "TechWolf/JobBERT-v2"                       # job-title encoder (Decorte et al. 2025)
FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # generic encoder (fallback + agreement check)
EXPOSURE_MAIN = "dv_rating_beta"   # Eloundou et al. GPT-4-rated beta; 'human_rating_beta' is the robustness version
HANDCHECK_N = 300
TARGET_ACC = 0.85               # accuracy the high-confidence subset should reach (from the hand-check)
save_versions(["pandas", "numpy", "torch", "sentence_transformers", "transformers"], f"{OUT}/versions_nb2.txt")
''')
    nb.md("## Helpers (plain Python, unit-tested outside Kaggle on synthetic examples)")
    nb.code("# Cell 2 — text helpers" + HELPERS + r'''
for ex in ["Acme (YC S19) | Senior Backend Engineer, Data Scientist | Remote (US) | $150k",
           "We are hiring a Machine Learning Engineer in Berlin.\nMore text..."]:
    print(extract_roles(ex), "->", [clean_role(r) for r in extract_roles(ex)[2]])
''')
    nb.md(r'''
## Step 1 — clean the ads
Keeps complete months only (NB1 flags the still-filling current month), 2019-01 onward, ads ≥ 100 characters; drops repeated comment ids and the same text posted twice in one month. Reposts of an identical ad in a later month are **kept** (they are new postings) but flagged `repost = 1`.
''')
    nb.code(r'''
# Cell 3 — clean
hn = pd.read_csv(f"{RAW}/hn_jobs.csv", dtype={"thread_id": str, "month": str})
log = [("raw top-level ads", len(hn))]
hn["complete_month"] = hn["complete_month"].astype(str).str.lower().isin(["true", "1"])
hn = hn[hn["complete_month"]];                         log.append(("complete months only", len(hn)))
hn = hn[hn["month"] >= START_MONTH];                   log.append((f"month >= {START_MONTH}", len(hn)))
hn["text"] = hn["text"].fillna("").astype(str)
hn = hn[hn["text"].str.len() >= MIN_CHARS];            log.append((f">= {MIN_CHARS} characters", len(hn)))
hn = hn.drop_duplicates("comment_id");                 log.append(("unique comment id", len(hn)))
hn["text_norm"] = hn["text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
hn = hn.drop_duplicates(["month", "text_norm"]);       log.append(("no same-text double post in a month", len(hn)))
hn = hn.sort_values(["month", "posted_at"]).reset_index(drop=True)

LAST_MONTH = hn["month"].max()
months = pd.period_range(START_MONTH, LAST_MONTH, freq="M").strftime("%Y-%m")
missing = sorted(set(months) - set(hn["month"]))
print("window:", START_MONTH, "→", LAST_MONTH, "| missing months:", missing or "none")
cleaning_log = pd.DataFrame(log, columns=["step", "ads_left"])
cleaning_log.to_csv(f"{OUT}/cleaning_log.csv", index=False)
print(cleaning_log.to_string(index=False))
''')
    nb.md(r'''
## Step 2 — company, roles and ad flags
The header line usually reads `Company | Role(s) | Location | REMOTE | Salary`. The first field is the company; role fields are those containing role words (whole-word match). Several roles in one field are split (`Backend Engineer, Data Scientist`). Ads without a usable header fall back to "hiring a … engineer" in the first 600 characters (`role_source = text`).
''')
    nb.code(r'''
# Cell 4 — roles, company, flags
ext = hn["text"].apply(extract_roles)
hn["header"] = ext.str[0]
hn["company_field"] = ext.str[1]
hn["roles"] = ext.str[2]
hn["role_source"] = ext.str[3]
hn["n_roles"] = hn["roles"].str.len()
hn["role_text"] = hn["roles"].apply(lambda rs: " ; ".join(rs))
hn["company"] = hn["company_field"].apply(company_key)
flags = pd.DataFrame(hn["text"].apply(ad_flags).tolist(), index=hn.index)
hn = pd.concat([hn, flags], axis=1)
hn["year"] = hn["month"].str[:4].astype(int)
hn["rel_month"] = hn["month"].apply(rel_month)
hn["post"] = (hn["rel_month"] >= 0).astype(int)
hn["text_len"] = hn["text"].str.len()

# repost = identical ad from the same company already seen in an earlier month
hn["text_hash"] = pd.util.hash_pandas_object(hn["text_norm"], index=False)
hn["repost"] = hn.duplicated(["company", "text_hash"], keep="first").astype(int)

print(hn["role_source"].value_counts(normalize=True).round(3).to_string())
print("ads with a company name:", f"{(hn['company'] != '').mean():.1%}",
      "| distinct companies:", hn.loc[hn["company"] != "", "company"].nunique())
print("roles per ad:", hn["n_roles"].value_counts().sort_index().to_dict())
print("remote share by year:", hn.groupby("year")["remote"].mean().round(2).to_dict())
hn[["month", "header", "company", "role_text", "role_source"]].sample(10, random_state=SEED)
''')
    nb.code(r'''
# Cell 5 — one row per role (an ad can list several roles)
ad_roles = (hn[["comment_id", "month", "company", "roles"]].explode("roles")
              .rename(columns={"roles": "role_raw"}).dropna(subset=["role_raw"]))
ad_roles["role_idx"] = ad_roles.groupby("comment_id").cumcount()
ad_roles["role_id"] = ad_roles["comment_id"].astype(str) + "_" + ad_roles["role_idx"].astype(str)
ad_roles["role_clean"] = ad_roles["role_raw"].apply(clean_role)
ad_roles = ad_roles[ad_roles["role_clean"].str.len() >= 2].reset_index(drop=True)
print(len(ad_roles), "roles from", ad_roles["comment_id"].nunique(), "ads;",
      ad_roles["role_clean"].nunique(), "distinct cleaned role strings")
ad_roles["role_clean"].value_counts().head(25)
''')
    nb.md(r'''
## Step 3 — O\*NET title list
Titles = occupation titles + O\*NET 31 **Job Titles** (`Job Title`, plus non-empty `Short Title`) + **Sample of Reported Titles**. O\*NET ≤ 30 had `Alternate_Titles` instead; the loader falls back to it. A missing table raises a clear error listing what *is* there.
''')
    nb.code(r'''
# Cell 6 — load O*NET defensively
def load_onet(table, need_cols):
    """Load onet_<table>.csv; match columns case-insensitively; explain clearly what is missing."""
    want = f"onet_{table}.csv"
    path = os.path.join(RAW, want)
    if not os.path.exists(path):
        avail = sorted(os.path.basename(p) for p in glob.glob(os.path.join(RAW, "onet_*.csv")))
        key = table.lower().replace("_", "")
        close = [a for a in avail if key[:8] in a.lower().replace("_", "")]
        if len(close) == 1:
            print(f"  note: {want} not found, using {close[0]}")
            path = os.path.join(RAW, close[0])
        else:
            raise FileNotFoundError(f"O*NET table {want} not in jobai-raw. Close matches: {close}. "
                                    f"All O*NET files: {avail}")
    df = pd.read_csv(path, dtype=str)
    lower = {c.lower().strip(): c for c in df.columns}
    ren = {}
    for c in need_cols:
        if c in df.columns:
            continue
        hit = lower.get(c.lower())
        if hit is None:
            raise KeyError(f"{os.path.basename(path)}: column '{c}' missing; columns are {list(df.columns)}")
        ren[hit] = c
    return df.rename(columns=ren)

occ = load_onet("Occupation_Data", ["O*NET-SOC Code", "Title"])
tasks = load_onet("Task_Statements", ["O*NET-SOC Code", "Task ID", "Task"])
parts = [occ[["O*NET-SOC Code", "Title"]].assign(kind="occupation")]
if os.path.exists(os.path.join(RAW, "onet_Job_Titles.csv")):          # O*NET 31+
    alt = load_onet("Job_Titles", ["O*NET-SOC Code", "Job Title"])
    parts.append(alt[["O*NET-SOC Code", "Job Title"]].rename(columns={"Job Title": "Title"}).assign(kind="job_title"))
else:                                                                  # O*NET <= 30
    alt = load_onet("Alternate_Titles", ["O*NET-SOC Code", "Alternate Title"])
    parts.append(alt[["O*NET-SOC Code", "Alternate Title"]].rename(columns={"Alternate Title": "Title"})
                 .assign(kind="alternate"))
if "Short Title" in alt.columns:
    parts.append(alt[["O*NET-SOC Code", "Short Title"]].rename(columns={"Short Title": "Title"}).assign(kind="short"))
try:
    rep = load_onet("Sample_of_Reported_Titles", ["O*NET-SOC Code", "Reported Job Title"])
    parts.append(rep[["O*NET-SOC Code", "Reported Job Title"]].rename(columns={"Reported Job Title": "Title"})
                 .assign(kind="reported"))
except (FileNotFoundError, KeyError) as e:
    print("  (optional table skipped)", str(e)[:150])

titles = pd.concat(parts, ignore_index=True).dropna(subset=["Title"])
titles = titles[titles["Title"].str.strip() != ""]                     # empty Short Title cells
titles = titles.rename(columns={"O*NET-SOC Code": "soc"})
titles["soc6"] = titles["soc"].str[:7]
titles["title_norm"] = titles["Title"].apply(norm_title)
titles = titles[titles["title_norm"].str.len() >= 2].drop_duplicates(["soc6", "title_norm"]).reset_index(drop=True)
soc_title = (occ.assign(soc6=occ["O*NET-SOC Code"].str[:7])
                .sort_values("O*NET-SOC Code").drop_duplicates("soc6")
                .set_index("soc6")["Title"])
print(len(occ), "O*NET occupations |", titles["soc6"].nunique(), "SOC-6 codes |", len(titles), "titles")
titles["kind"].value_counts()
''')
    nb.md(r'''
## Step 4 — match roles to occupations
1. **Exact match** of the cleaned role to a normalised O\*NET title (score 1.0).
2. Otherwise the **job-title encoder** (JobBERT-v2, encoded through its `anchor` branch as the model card shows); if it cannot be loaded or fails a sanity check, MiniLM is used and the notebook says so.
3. Scores are kept per SOC-6 code: `match_score` = best cosine, `match_margin` = gap to the second-best occupation (low margin = ambiguous).
4. MiniLM is also run on the same roles to report **agreement between the two encoders** (a robustness number for the paper).
''')
    nb.code(r'''
# Cell 7 — encoders
from sentence_transformers import SentenceTransformer

def _encode_anchor(m, texts, bs):
    """JobBERT-v2 / v3: route texts through the 'anchor' (job-title) branch of the asymmetric model."""
    from sentence_transformers.util import batch_to_device
    order = np.argsort([len(t) for t in texts])
    chunks = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            batch = [texts[j] for j in order[i:i + bs]]
            f = batch_to_device(m.tokenize(batch), m.device)
            f["text_keys"] = ["anchor"]
            chunks.append(m.forward(f)["sentence_embedding"].float())
    E = torch.cat(chunks)
    out = torch.empty_like(E)
    out[torch.as_tensor(order, device=E.device)] = E
    return torch.nn.functional.normalize(out, dim=1)

def make_encoder(name):
    m = SentenceTransformer(name, device=DEVICE)
    asym = any(type(mod).__name__ in ("Asym", "Router") for mod in m.modules())
    def enc(texts, bs=256):
        texts = [t if t else " " for t in texts]
        if asym:
            for attempt in ("anchor_forward", "task_kw", "encode_query"):
                try:
                    if attempt == "anchor_forward":
                        return _encode_anchor(m, texts, bs)
                    if attempt == "task_kw":
                        E = m.encode(texts, batch_size=bs, convert_to_tensor=True, task="anchor")
                    else:
                        E = m.encode_query(texts, batch_size=bs, convert_to_tensor=True)
                    return torch.nn.functional.normalize(E.float(), dim=1)
                except Exception as e:
                    print(f"  {name}: {attempt} failed ({type(e).__name__}: {str(e)[:120]})")
            raise RuntimeError(f"could not encode with {name}")
        E = m.encode(texts, batch_size=bs, convert_to_tensor=True, normalize_embeddings=True,
                     show_progress_bar=len(texts) > 5000)
        return E.float()
    # sanity check: two software titles should be closer than software vs nurse
    E = enc(["software engineer", "senior software developer", "registered nurse"])
    s = (E @ E.T).cpu().numpy()
    if not s[0, 1] > s[0, 2]:
        raise RuntimeError(f"{name} failed the sanity check: {s.round(3)}")
    print(f"✓ {name} loaded (asymmetric={asym}); sim(SWE, senior dev)={s[0,1]:.3f} > sim(SWE, nurse)={s[0,2]:.3f}")
    return enc

try:
    encode_main = make_encoder(TITLE_MODEL)
    MAIN_MODEL = TITLE_MODEL
except Exception as e:
    print(f"⚠ {TITLE_MODEL} unavailable ({e}); falling back to {FALLBACK_MODEL}. Say so in the paper.")
    encode_main = make_encoder(FALLBACK_MODEL)
    MAIN_MODEL = FALLBACK_MODEL
encode_minilm = encode_main if MAIN_MODEL == FALLBACK_MODEL else make_encoder(FALLBACK_MODEL)
''')
    nb.code(r'''
# Cell 8 — match every distinct cleaned role to its best SOC-6 occupations
def match_to_soc(roles, enc, title_df, k_titles=25, chunk=4096):
    """For each role string: top-3 SOC-6 codes by best title cosine. Returns a DataFrame indexed like `roles`."""
    T = enc(title_df["Title"].tolist())
    soc_arr = title_df["soc6"].values
    ttl_arr = title_df["Title"].values
    rows = []
    for i in range(0, len(roles), chunk):
        R = enc(roles[i:i + chunk])
        sims, idx = torch.topk(R @ T.T, k=k_titles, dim=1)
        sims, idx = sims.cpu().numpy(), idx.cpu().numpy()
        for r in range(len(R)):
            best = {}
            for s, j in zip(sims[r], idx[r]):
                c = soc_arr[j]
                if c not in best:
                    best[c] = (float(s), ttl_arr[j])
                if len(best) == 3:
                    break
            items = list(best.items()) + [(None, (np.nan, None))] * (3 - len(best))
            rows.append({"soc6_1": items[0][0], "score_1": items[0][1][0], "title_hit_1": items[0][1][1],
                         "soc6_2": items[1][0], "score_2": items[1][1][0], "title_hit_2": items[1][1][1],
                         "soc6_3": items[2][0], "score_3": items[2][1][0], "title_hit_3": items[2][1][1]})
    return pd.DataFrame(rows)

''' + HN_RULES_SRC + r'''
t0 = time.time()
uniq = pd.Series(sorted(ad_roles["role_clean"].unique()))
exact = (titles.sort_values("kind", key=lambda s: s.map({"occupation": 0, "job_title": 1, "alternate": 1, "short": 2, "reported": 3}))
               .drop_duplicates("title_norm").set_index("title_norm")[["soc6", "Title"]])
res = match_to_soc(uniq.tolist(), encode_main, titles)
res.insert(0, "role_clean", uniq.values)
res["match_method"] = MAIN_MODEL
is_exact = uniq.apply(norm_title).isin(exact.index).values
ex = exact.loc[uniq[is_exact].apply(norm_title).values]
res.loc[is_exact, ["soc6_1", "title_hit_1"]] = ex[["soc6", "Title"]].values
res.loc[is_exact, "score_1"] = 1.0
res.loc[is_exact, "match_method"] = "exact"
# HN title rules override both (O*NET has an exact title "Engineer" -> rail yard engineers; HN ads are tech ads)
rule = uniq.apply(hn_rule_soc).values
has_rule = pd.notna(rule)
res["soc6_model"] = res["soc6_1"]                       # keep the model pick for robustness checks
res.loc[has_rule, "soc6_1"] = rule[has_rule]
res.loc[has_rule, "title_hit_1"] = "HN title rule"
res.loc[has_rule, "score_1"] = 1.0
res.loc[has_rule, "match_method"] = "hn_rule"
print("match method shares:", res["match_method"].value_counts(normalize=True).round(3).to_dict())
res["match_margin"] = res["score_1"] - res["score_2"].fillna(0)

mini = match_to_soc(uniq.tolist(), encode_minilm, titles)
res["soc6_minilm"] = mini["soc6_1"].values
res["score_minilm"] = mini["score_1"].values
print(f"matched {len(res):,} role strings in {time.time() - t0:.0f}s; exact title matches: {is_exact.mean():.1%}")

ad_roles = ad_roles.merge(res, on="role_clean", how="left")
ad_roles = ad_roles.rename(columns={"soc6_1": "soc6", "score_1": "match_score", "title_hit_1": "matched_title"})
ad_roles["soc_title"] = ad_roles["soc6"].map(soc_title)
agree = (ad_roles["soc6"] == ad_roles["soc6_minilm"]).mean()
print(f"role-level agreement {MAIN_MODEL} vs MiniLM (top-1 SOC-6): {agree:.1%}")
ad_roles["soc6"].map(soc_title).value_counts().head(20)
''')
    nb.md(r'''
## Step 5 — hand-check sample (300 ads, stratified by year)
Fill in `correct_soc6` (1 = the top-1 occupation is right, 0 = wrong) and, if wrong, `true_soc6` (look it up on onetonline.org). Also mark `correct_major` (1 if at least the 2-digit major group, e.g. `15-`, is right). Upload the file as Kaggle dataset **`jobai-handcheck`** named `soc_handcheck_labeled.csv` and re-run this notebook.
''')
    nb.code(r'''
# Cell 9 — primary role per ad, hand-check export and (second run) accuracy
prim = ad_roles[ad_roles["role_idx"] == 0].set_index("comment_id")
for c in ["soc6", "soc_title", "match_score", "match_margin", "match_method", "role_clean", "soc6_minilm"]:
    hn[c] = hn["comment_id"].map(prim[c])

per_year = math.ceil(HANDCHECK_N / hn["year"].nunique())
hc = (hn[hn["soc6"].notna()].groupby("year", group_keys=False)
        .apply(lambda g: g.sample(min(per_year, len(g)), random_state=SEED)))
hc = hc.merge(prim[["soc6_2", "score_2", "soc6_3", "score_3", "matched_title"]], left_on="comment_id",
              right_index=True, how="left")
hc_out = hc[["comment_id", "month", "header", "role_text", "role_clean", "matched_title", "soc6", "soc_title",
             "match_score", "match_margin", "match_method", "soc6_2", "score_2", "soc6_3", "score_3"]].copy()
hc_out["soc_title_2"] = hc_out["soc6_2"].map(soc_title)
hc_out["soc_title_3"] = hc_out["soc6_3"].map(soc_title)
hc_out["text_start"] = hc["text"].str[:500].values
for c in ["correct_soc6", "correct_major", "true_soc6", "notes"]:
    hc_out[c] = ""
hc_out.to_csv(f"{OUT}/handcheck_soc_sample.csv", index=False)
print("hand-check sample written:", len(hc_out), "ads")

HIGH_CONF = float(hn["match_score"].quantile(0.25))      # provisional: top 75% of scores
hc_lab_path = find_file("soc_handcheck_labeled.csv", required=False)
match_summary = [("encoder", MAIN_MODEL), ("ads with a matched role", f"{hn['soc6'].notna().mean():.3f}"),
                 ("encoder agreement with MiniLM", f"{agree:.3f}")]
if hc_lab_path:
    lab = pd.read_csv(hc_lab_path)
    lab = lab[pd.to_numeric(lab["correct_soc6"], errors="coerce").notna()].copy()
    lab["correct_soc6"] = lab["correct_soc6"].astype(int)
    # labels judged the pick shown in the file; rebuild the true code, then score the CURRENT pick against it
    lab["truth"] = np.where(lab["correct_soc6"] == 1, lab["soc6"], lab["true_soc6"])
    lab["truth"] = lab["truth"].where(lab["truth"].astype(str).str.match(r"^\d\d-\d{4}$"), None)
    print(f"labelled pick (as exported): top-1 accuracy {lab['correct_soc6'].mean():.1%}")
    cur = res.set_index("role_clean")[["soc6_1", "score_1", "match_method"]]
    lab = lab.drop(columns=["match_score", "match_method"], errors="ignore").join(cur, on="role_clean")
    lab = lab.rename(columns={"score_1": "match_score"})
    lab["correct_soc6"] = (lab["soc6_1"] == lab["truth"]).astype(int)
    lab["correct_major"] = (lab["soc6_1"].str[:2] == lab["truth"].str[:2]).astype(int)
    acc = lab["correct_soc6"].mean()
    print(f"hand-check: {len(lab)} labelled ads, CURRENT top-1 SOC-6 accuracy {acc:.1%}; "
          f"major-group {lab['correct_major'].mean():.1%}")
    print("  accuracy by match method:", lab.groupby("match_method")["correct_soc6"].agg(["mean", "size"]).round(3).to_dict("index"))
    lab.to_csv(f"{OUT}/handcheck_scored_current.csv", index=False)
    lab["year"] = lab["month"].str[:4]
    print("  accuracy by year:", lab.groupby("year")["correct_soc6"].mean().round(2).to_dict())
    lab["score_bin"] = pd.qcut(lab["match_score"], 5, duplicates="drop")
    print(lab.groupby("score_bin", observed=True)["correct_soc6"].agg(["mean", "size"]).round(3))
    # smallest threshold whose 'score >= t' subset reaches TARGET_ACC
    best_t = None
    for t in np.sort(lab["match_score"].unique()):
        sub = lab[lab["match_score"] >= t]
        if len(sub) >= 30 and sub["correct_soc6"].mean() >= TARGET_ACC:
            best_t = float(t)
            break
    if best_t is not None:
        HIGH_CONF = best_t
    match_summary += [("hand-check n", len(lab)), ("hand-check accuracy top-1", round(acc, 3)),
                      ("high-confidence threshold source", "hand-check" if best_t is not None else "quantile")]
else:
    print("No labelled hand-check yet → HIGH_CONF is provisional (25th percentile of match scores).")
    match_summary.append(("high-confidence threshold source", "provisional quantile"))
match_summary.append(("HIGH_CONF", round(HIGH_CONF, 4)))
hn["high_conf"] = (hn["match_score"] >= HIGH_CONF).astype(int)
ad_roles["high_conf"] = (ad_roles["match_score"] >= HIGH_CONF).astype(int)
print(f"HIGH_CONF = {HIGH_CONF:.3f}; high-confidence share of ads = {hn['high_conf'].mean():.1%}")
pd.DataFrame(match_summary, columns=["item", "value"]).to_csv(f"{OUT}/soc_match_summary.csv", index=False)
''')
    nb.md(r'''
## Step 6 — exposure scores
* **Eloundou et al. β** (pre-ChatGPT, main treatment): `dv_rating_beta` (GPT-4-rated) by default; `human_rating_beta` for robustness.
* **Felten AIOE**: keyed on **SOC 2010** codes → mapped to the O\*NET 2019 taxonomy with the official O\*NET 2010→2019 crosswalk (downloaded here; averaged when one code splits into several).
* **Anthropic Economic Index** (measured *after* ChatGPT → heterogeneity / robustness only): task-level usage share and the automation vs augmentation split from the June 2026 release (long format: `category_name`, `hierarchy_level`, `metric_id`, `node_external_id`). Older release layouts are also handled; anything else is reported and skipped.
* **Ensemble** = first principal component of the standardised pre-ChatGPT measures (Eloundou GPT-4 β, human β, AIOE); `exp_ensemble_all` adds Anthropic usage.
''')
    nb.code(r'''
# Cell 10 — O*NET 2010 -> 2019 crosswalk (needed for AIOE)
XWALK_URL = "https://www.onetcenter.org/taxonomy/2019/walk/2010_to_2019_Crosswalk.csv?fmt=csv"
xw = None
try:
    import requests
    r = requests.get(XWALK_URL, headers={"User-Agent": "academic-research-notebook"}, timeout=120)
    r.raise_for_status()
    xw = pd.read_csv(io.StringIO(r.text), dtype=str)
    xw.columns = [c.strip() for c in xw.columns]
    c10 = [c for c in xw.columns if "2010" in c and "Code" in c][0]
    c19 = [c for c in xw.columns if "2019" in c and "Code" in c][0]
    xw = xw[[c10, c19]].rename(columns={c10: "soc8_2010", c19: "soc8_2019"}).dropna()
    xw["soc6_2010"], xw["soc6_2019"] = xw["soc8_2010"].str[:7], xw["soc8_2019"].str[:7]
    xw.to_csv(f"{OUT}/onet_soc_crosswalk_2010_2019.csv", index=False)
    print("crosswalk rows:", len(xw))
except Exception as e:
    print("⚠ crosswalk download failed:", e, "→ AIOE merged on raw SOC-6 codes (coverage will be lower)")

def to_2019_soc6(df, code_col, value_cols):
    """Map SOC-2010 6-digit codes to 2019 SOC-6, averaging values over split/merged codes."""
    d = df[[code_col] + value_cols].copy()
    d["soc6_2010"] = d[code_col].astype(str).str.strip().str[:7]
    if xw is None:
        return d.rename(columns={"soc6_2010": "soc6"}).groupby("soc6")[value_cols].mean().reset_index()
    m = d.merge(xw[["soc6_2010", "soc6_2019"]].drop_duplicates(), on="soc6_2010", how="left")
    m["soc6"] = m["soc6_2019"].fillna(m["soc6_2010"])
    return m.groupby("soc6")[value_cols].mean().reset_index()
''')
    nb.code(r'''
# Cell 11 — Eloundou + AIOE
el = pd.read_csv(f"{RAW}/exposure_eloundou.csv")
code_col = next(c for c in el.columns if "soc" in c.lower())
rating_cols = [c for c in el.columns if re.fullmatch(r"(dv|human)_rating_(alpha|beta|gamma)", c)]
assert EXPOSURE_MAIN in rating_cols, f"{EXPOSURE_MAIN} not in {list(el.columns)}"
in_2019 = el[code_col].astype(str).isin(set(occ["O*NET-SOC Code"])).mean()
print(f"Eloundou: {len(el)} rows; share of codes found in O*NET 2019 taxonomy: {in_2019:.1%}")
if in_2019 < 0.8 and xw is not None:
    print("  → codes look like the 2010 taxonomy; mapping through the crosswalk")
    ex_el = to_2019_soc6(el, code_col, rating_cols)
else:
    ex_el = el.assign(soc6=el[code_col].astype(str).str[:7]).groupby("soc6")[rating_cols].mean().reset_index()
exposure = ex_el.rename(columns={c: f"el_{c}" for c in rating_cols})

aioe_files = sorted(glob.glob(f"{RAW}/aioe/aioe_*.csv"))
aioe = None
for f in aioe_files:
    d = pd.read_csv(f)
    soc_c = [c for c in d.columns if "soc" in str(c).lower()]
    val_c = [c for c in d.columns if "aioe" in str(c).lower() and pd.api.types.is_numeric_dtype(d[c])]
    print(f"  {os.path.basename(f)}: {d.shape}, SOC cols {soc_c}, AIOE cols {val_c}")
    if soc_c and val_c and len(d) > 300 and aioe is None:
        d = d.rename(columns={v: re.sub(r"[^a-z0-9]+", "_", str(v).lower()).strip("_") for v in val_c})
        vals = [re.sub(r"[^a-z0-9]+", "_", str(v).lower()).strip("_") for v in val_c]
        aioe = to_2019_soc6(d, soc_c[0], vals)
        print(f"  → AIOE from {os.path.basename(f)}: {len(aioe)} SOC-6 codes, columns {vals}")
if aioe is not None:
    exposure = exposure.merge(aioe, on="soc6", how="outer")
else:
    print("⚠ no AIOE table recognised — check the aioe/ files listed above")
exposure.describe().T.round(3)
''')
    nb.code(r'''
# Cell 12 — Anthropic Economic Index (usage-based; robustness / heterogeneity only)
aei_paths = sorted(glob.glob(f"{RAW}/anthropic/**/*.csv", recursive=True))
inspect_lines = []
for p in aei_paths:
    cols = list(pd.read_csv(p, nrows=3).columns)
    inspect_lines.append(f"{os.path.relpath(p, RAW)} ({os.path.getsize(p)/1e6:.0f} MB): {cols}")
docs = glob.glob(f"{RAW}/anthropic/**/data_documentation.md", recursive=True)
open(f"{OUT}/aei_inspection.txt", "w").write("\n".join(inspect_lines))
print("\n".join(inspect_lines) or "no Anthropic CSV files found", "\n| documentation:", docs)

V6 = {"geo_id", "category_name", "hierarchy_level", "metric_id", "value", "node_name", "node_external_id"}
AEI_METRICS = ["pct", "collaboration_bucket_automation_pct", "collaboration_bucket_augmentation_pct",
               "ai_autonomy_mean"]

def read_aei_v6(path):
    keep = []
    for ch in pd.read_csv(path, usecols=lambda c: c in V6 | {"date_start"}, dtype={"node_external_id": str},
                          chunksize=1_000_000):
        ch = ch[(ch["geo_id"] == "GLOBAL") & (ch["hierarchy_level"].astype(str) == "0")
                & ch["category_name"].isin(["onet", "soc_occupation"]) & ch["metric_id"].isin(AEI_METRICS)]
        keep.append(ch)
    d = pd.concat(keep, ignore_index=True)
    # several months in one release -> average over months
    return (d.pivot_table(index=["category_name", "node_external_id", "node_name"], columns="metric_id",
                          values="value", aggfunc="mean").reset_index())

def read_aei_legacy(path):
    """Older releases: columns facet / variable / cluster_name / value."""
    d = pd.read_csv(path)
    if "geo_id" in d:
        d = d[d["geo_id"].astype(str).str.upper().isin(["GLOBAL", "ALL"])]
    t = d[(d["facet"] == "onet_task") & d["variable"].astype(str).str.endswith("_pct")]
    t = t.groupby("cluster_name")["value"].mean().rename("pct").reset_index()
    c = d[d["facet"].astype(str).str.contains("onet_task::collaboration")].copy()
    if len(c):
        c[["task", "collab"]] = c["cluster_name"].astype(str).str.split("::", n=1, expand=True)
        c = c.pivot_table(index="task", columns="collab", values="value", aggfunc="mean").fillna(0)
        auto = c.filter(regex="directive|feedback").sum(axis=1)
        aug = c.filter(regex="iteration|validation|learning").sum(axis=1)
        tot = (auto + aug).replace(0, np.nan)
        t = t.merge(pd.DataFrame({"cluster_name": c.index,
                                  "collaboration_bucket_automation_pct": (100 * auto / tot).values,
                                  "collaboration_bucket_augmentation_pct": (100 * aug / tot).values}),
                    on="cluster_name", how="left")
    return t.rename(columns={"cluster_name": "node_name"}).assign(category_name="onet", node_external_id=None)

aei_parts = []
for p in aei_paths:
    cols = set(pd.read_csv(p, nrows=3).columns)
    src = "claude_ai" if "claude_ai" in p else ("1p_api" if "1p_api" in p else os.path.basename(p))
    try:
        if V6 <= cols:
            aei_parts.append(read_aei_v6(p).assign(source=src))
        elif {"facet", "variable", "cluster_name", "value"} <= cols:
            aei_parts.append(read_aei_legacy(p).assign(source=src))
        else:
            print("  skipped (unknown layout):", os.path.basename(p))
    except Exception as e:
        print("  ⚠ could not parse", os.path.basename(p), "→", e)

aei_task_table = pd.DataFrame()
occ_aei = pd.DataFrame(columns=["soc6"])
if aei_parts:
    aei = pd.concat(aei_parts, ignore_index=True)
    MAIN_SRC = "claude_ai" if (aei["source"] == "claude_ai").any() else aei["source"].iloc[0]
    a = aei[aei["source"] == MAIN_SRC]
    # tasks: join on O*NET Task ID, fall back to exact task text
    tk = a[a["category_name"] == "onet"].copy()
    ts = tasks[["O*NET-SOC Code", "Task ID", "Task"]].rename(columns={"O*NET-SOC Code": "soc", "Task ID": "task_id"})
    ts["task_key"] = ts["Task"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
    tk["task_key"] = tk["node_name"].astype(str).str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
    by_id = tk.merge(ts.drop(columns=["task_key"]), left_on="node_external_id", right_on="task_id", how="inner")
    rest = tk[~tk["node_external_id"].isin(by_id["node_external_id"])]
    by_txt = rest.merge(ts, on="task_key", how="inner")
    aei_task_table = pd.concat([by_id, by_txt], ignore_index=True)
    aei_task_table["soc6"] = aei_task_table["soc"].str[:7]
    aei_task_table = aei_task_table.rename(columns={"pct": "usage_pct",
                                                    "collaboration_bucket_automation_pct": "automation_pct",
                                                    "collaboration_bucket_augmentation_pct": "augmentation_pct"})
    print(f"AEI tasks ({MAIN_SRC}): {len(tk)} published, {len(aei_task_table)} linked to O*NET tasks "
          f"({aei_task_table['task_id'].nunique()} task ids, {aei_task_table['soc6'].nunique()} SOC-6)")
    keep_cols = [c for c in ["task_id", "Task", "soc", "soc6", "usage_pct", "automation_pct", "augmentation_pct",
                             "ai_autonomy_mean"] if c in aei_task_table]
    aei_task_table = aei_task_table[keep_cols]
    # occupation level: prefer the direct SOC breakdown, otherwise aggregate tasks
    so = a[a["category_name"] == "soc_occupation"].copy()
    if len(so) and "pct" in so:
        so["soc6"] = so["node_external_id"].astype(str).str[:7]
        agg = {"aei_usage": ("pct", "sum")}
        if "collaboration_bucket_automation_pct" in so:
            agg["aei_automation"] = ("collaboration_bucket_automation_pct", "mean")
        occ_aei = so.groupby("soc6").agg(**agg).reset_index()
    elif len(aei_task_table):
        occ_aei = aei_task_table.groupby("soc6")["usage_pct"].sum().rename("aei_usage").reset_index()
    if "automation_pct" in aei_task_table:
        w = aei_task_table.dropna(subset=["automation_pct", "usage_pct"])
        w = w.assign(wa=w["automation_pct"] * w["usage_pct"])
        g = w.groupby("soc6")[["wa", "usage_pct"]].sum()
        task_auto = (g["wa"] / g["usage_pct"].replace(0, np.nan)).rename("aei_automation_tasks").reset_index()
        occ_aei = occ_aei.merge(task_auto, on="soc6", how="outer")
    print("AEI occupation measures:", occ_aei.shape, list(occ_aei.columns))
else:
    print("⚠ no Anthropic data parsed — AEI robustness columns will be empty. See aei_inspection.txt")
aei_task_table.to_csv(f"{OUT}/aei_task_table.csv", index=False)
exposure = exposure.merge(occ_aei, on="soc6", how="outer")
''')
    nb.code(r'''
# Cell 13 — main measure, terciles, ensemble, tech flags
def zscore(s):
    return (s - s.mean()) / s.std(ddof=0)

def first_pc(df, cols, sign_col):
    """First principal component of standardised columns (complete cases), sign aligned with sign_col."""
    d = df[cols].dropna()
    if len(d) < 30 or len(cols) < 2:
        return pd.Series(np.nan, index=df.index)
    Z = (d - d.mean()) / d.std(ddof=0)
    _, svals, vt = np.linalg.svd(Z.values, full_matrices=False)
    pc = pd.Series(Z.values @ vt[0], index=d.index)
    if pc.corr(df.loc[d.index, sign_col]) < 0:   # sign column lives in df, not in the PCA subset
        pc = -pc
    print(f"  PC1 of {cols}: explains {svals[0]**2 / (svals**2).sum():.1%}; loadings {np.round(vt[0], 2)}")
    return zscore(pc).reindex(df.index)

exposure = exposure[exposure["soc6"].str.match(r"^\d\d-\d{4}$", na=False)].copy()
exposure["exp_main"] = exposure[f"el_{EXPOSURE_MAIN}"]
exposure["exp_z"] = zscore(exposure["exp_main"])
pre_cols = [c for c in ["el_dv_rating_beta", "el_human_rating_beta", "aioe"] if c in exposure]
exposure["exp_ensemble"] = first_pc(exposure, pre_cols, "exp_main")
if "aei_usage" in exposure and exposure["aei_usage"].notna().sum() > 30:
    exposure["log_aei_usage"] = np.log1p(exposure["aei_usage"])
    exposure["exp_ensemble_all"] = first_pc(exposure, pre_cols + ["log_aei_usage"], "exp_main")

# terciles over ALL O*NET occupations (population-based, not driven by what HN posts)
q1, q2 = exposure["exp_main"].quantile([1/3, 2/3])
exposure["exp_tercile"] = np.select([exposure["exp_main"] <= q1, exposure["exp_main"] >= q2], [1, 3], 2).astype(float)
exposure.loc[exposure["exp_main"].isna(), "exp_tercile"] = np.nan
exposure["high_exposure"] = exposure["exp_tercile"].map({3.0: 1.0, 1.0: 0.0})     # middle tercile -> NaN
exposure["tech"] = exposure["soc6"].str.startswith("15-12").astype(int)       # computer occupations
exposure["software"] = exposure["soc6"].str.startswith("15-125").astype(int)
exposure["title"] = exposure["soc6"].map(soc_title)
print(f"tercile cut-offs on {EXPOSURE_MAIN}: {q1:.3f}, {q2:.3f}")
''')
    nb.md(r'''
## Step 7 — control-group size check (do this before any regression)
HN ads are mostly software jobs, which are mostly high-exposure. Rule from the plan: if the low-exposure group has **< 20 occupations or < 10 % of ads**, HN cannot answer RQ1 on its own → the headline designs stay within-HN (RQ3 company × month; RQ2 ad-level exposure), and RQ1 is reported as a check. A second split (`high_exp_hn`) uses terciles **among HN occupations** (weighted by pre-ChatGPT ads) so a within-HN comparison always exists — say clearly in the paper that it compares "more vs less exposed *tech-adjacent* jobs".
''')
    nb.code(r'''
# Cell 14 — control group check + HN-based split
hn = hn.merge(exposure[["soc6", "exp_main", "exp_z", "exp_ensemble", "high_exposure", "tech", "software"]],
              on="soc6", how="left")
pre_counts = hn[hn["post"] == 0].groupby("soc6").size()
e = exposure.set_index("soc6")["exp_main"].reindex(pre_counts.index).dropna()
order = e.sort_values()
cw = pre_counts.reindex(order.index).cumsum() / pre_counts.reindex(order.index).sum()
lo_cut = order[cw <= 1/3].max() if (cw <= 1/3).any() else order.iloc[0]
hi_cut = order[cw >= 2/3].min()
exposure["high_exp_hn"] = np.select([exposure["exp_main"] >= hi_cut, exposure["exp_main"] <= lo_cut], [1.0, 0.0], np.nan)
exposure.loc[exposure["exp_main"].isna(), "high_exp_hn"] = np.nan
hn["high_exp_hn"] = hn["soc6"].map(exposure.set_index("soc6")["high_exp_hn"])
print(f"HN-weighted cut-offs: low <= {lo_cut:.3f}, high >= {hi_cut:.3f}")

rows = []
for col in ["high_exposure", "high_exp_hn"]:
    for g, name in [(0.0, "low (control)"), (1.0, "high (treated)"), (None, "middle / no score")]:
        sel = hn[col].isna() if g is None else hn[col] == g
        rows.append({"split": col, "group": name, "ads": int(sel.sum()), "share_ads": sel.mean(),
                     "occupations": hn.loc[sel, "soc6"].nunique()})
ccheck = pd.DataFrame(rows)
ccheck.to_csv(f"{OUT}/control_group_check.csv", index=False)
print(ccheck.round(3).to_string(index=False))
low = ccheck[(ccheck["split"] == "high_exposure") & (ccheck["group"] == "low (control)")].iloc[0]
if low["occupations"] < 20 or low["share_ads"] < 0.10:
    print("\n⚠ Population-tercile control group is small "
          f"({low['occupations']} occupations, {low['share_ads']:.1%} of ads). "
          "Treat RQ1 on HN as a check; lead with the within-HN designs (NB5 company × month, NB4 ad-level exposure).")
else:
    print("\n✓ control group passes the plan's rule of thumb")
''')
    nb.md(r'''
## Step 8 — occupation × month panel
`ads` = ads whose **first** role maps to the occupation (main); `ads_roles` = every role counted once; `ads_w` = each ad split equally over its roles; `ads_hc` = high-confidence matches only. Months with no ads get **0** (otherwise `log_ads` is biased upward). `rel_month` 0 = Dec 2022.
''')
    nb.code(r'''
# Cell 15 — panel
ad_roles = ad_roles.merge(hn[["comment_id", "n_roles"]], on="comment_id", how="left")
ad_roles["w"] = 1.0 / ad_roles["n_roles"].clip(lower=1)
g_ad = hn.dropna(subset=["soc6"]).groupby(["soc6", "month"])
p = pd.DataFrame({"ads": g_ad.size(), "ads_hc": g_ad["high_conf"].sum(), "remote_share": g_ad["remote"].mean()})
g_r = ad_roles.dropna(subset=["soc6"]).groupby(["soc6", "month"])
p = p.join(pd.DataFrame({"ads_roles": g_r.size(), "ads_w": g_r["w"].sum()}), how="outer")

keep_occ = p.groupby(level="soc6")["ads"].sum()
keep_occ = keep_occ[keep_occ >= MIN_ADS_OCC].index
print(f"{len(keep_occ)} of {p.index.get_level_values('soc6').nunique()} occupations have >= {MIN_ADS_OCC} ads")
full = pd.MultiIndex.from_product([keep_occ, list(months)], names=["soc6", "month"])
panel = p.reindex(full)
for c in ["ads", "ads_hc", "ads_roles", "ads_w"]:
    panel[c] = panel[c].fillna(0)
panel = panel.reset_index()
panel["log_ads"] = np.log1p(panel["ads"])
panel["share"] = panel["ads"] / panel.groupby("month")["ads"].transform("sum")
panel["hn_total"] = panel["month"].map(hn.groupby("month").size())
panel = panel.merge(exposure.drop(columns=["title"]), on="soc6", how="left")
panel["title"] = panel["soc6"].map(soc_title)
panel["date"] = pd.to_datetime(panel["month"] + "-01")
panel["year"] = panel["date"].dt.year
panel["rel_month"] = panel["month"].apply(rel_month)
panel["post"] = (panel["rel_month"] >= 0).astype(int)
panel.to_csv(f"{OUT}/panel_occ_month.csv", index=False)
exposure.to_csv(f"{OUT}/exposure_table.csv", index=False)
print("panel:", panel.shape, "| occupations with exposure:", panel.dropna(subset=["exp_main"])["soc6"].nunique())
panel.groupby("year")["ads"].sum()
''')
    nb.md(r'''
## Step 9 — Indeed sector exposure (for NB3's whole-market check and NB5's country analysis)
Indeed sectors are broader than SOC codes. The example job titles Indeed publishes per sector are matched to O\*NET with the same encoder; the sector's exposure is the mean over its example titles. `high_sector` = top third of sectors, `0` = bottom third.
''')
    nb.code(r'''
# Cell 16 — sector exposure from Indeed's example titles
ex_path = os.path.join(RAW, "indeed_sector_job_title_examples.csv")
sector_exposure = pd.DataFrame()
if os.path.exists(ex_path):
    sx = pd.read_csv(ex_path)
    print("example-title file columns:", list(sx.columns))
    sec_col = next((c for c in sx.columns if c.lower() in ("display_name", "sector", "category", "occupation")), sx.columns[0])
    txt_cols = [c for c in sx.columns if c != sec_col and sx[c].dtype == object]
    long = sx.melt(id_vars=[sec_col], value_vars=txt_cols, value_name="example").dropna(subset=["example"])
    long["example"] = long["example"].astype(str).str.split(r"[;,|]")
    long = long.explode("example")
    long["example"] = long["example"].str.strip()
    long = long[long["example"].str.len() > 2].drop_duplicates([sec_col, "example"]).reset_index(drop=True)
    m = match_to_soc(long["example"].apply(clean_role).tolist(), encode_main, titles)
    long["soc6"] = m["soc6_1"].values
    long["score"] = m["score_1"].values
    long = long.merge(exposure[["soc6", "exp_main", "exp_ensemble"]], on="soc6", how="left")
    sector_exposure = (long.groupby(sec_col).agg(n_titles=("example", "size"), exp_main=("exp_main", "mean"),
                                                exp_ensemble=("exp_ensemble", "mean"), match_score=("score", "mean"))
                           .reset_index().rename(columns={sec_col: "display_name"}))
    a, b = sector_exposure["exp_main"].quantile([1/3, 2/3])
    sector_exposure["high_sector"] = np.select([sector_exposure["exp_main"] >= b, sector_exposure["exp_main"] <= a],
                                               [1.0, 0.0], np.nan)
    sector_exposure.loc[sector_exposure["exp_main"].isna(), "high_sector"] = np.nan
    print(sector_exposure.sort_values("exp_main", ascending=False).round(3).to_string(index=False))
else:
    print("⚠ indeed_sector_job_title_examples.csv missing — NB3/NB5 sector analyses will need it")
sector_exposure.to_csv(f"{OUT}/sector_exposure.csv", index=False)
''')
    nb.code(r'''
# Cell 17 — save ad-level tables
ad_cols = ["comment_id", "thread_id", "month", "year", "rel_month", "post", "author", "posted_at", "header",
           "company", "company_field", "role_text", "role_source", "n_roles", "role_clean", "soc6", "soc_title",
           "match_score", "match_margin", "match_method", "soc6_minilm", "high_conf", "exp_main", "exp_z",
           "exp_ensemble", "high_exposure", "high_exp_hn", "tech", "software", "remote", "onsite", "hybrid",
           "visa", "salary", "repost", "text_len", "text"]
hn[ad_cols].to_csv(f"{OUT}/hn_clean.csv", index=False)
ad_roles = ad_roles.merge(exposure[["soc6", "exp_main", "exp_z", "exp_ensemble", "high_exposure", "high_exp_hn",
                                    "tech", "software"]], on="soc6", how="left")
role_cols = ["role_id", "comment_id", "month", "company", "role_idx", "n_roles", "w", "role_raw", "role_clean",
             "soc6", "soc_title", "matched_title", "match_score", "match_margin", "match_method", "high_conf",
             "soc6_2", "score_2", "soc6_minilm", "soc6_model", "exp_main", "exp_z", "exp_ensemble", "high_exposure",
             "high_exp_hn", "tech", "software"]
ad_roles[role_cols].to_csv(f"{OUT}/ad_roles.csv", index=False)
with open(f"{OUT}/SOURCES_processed.md", "w") as fh:
    fh.write(f"# jobai-processed\n\nBuilt by 02-clean-and-link on {pd.Timestamp.today():%Y-%m-%d}.\n"
             f"Window {START_MONTH}–{LAST_MONTH}; {len(hn):,} ads; title encoder {MAIN_MODEL}; "
             f"exposure main = Eloundou {EXPOSURE_MAIN}; HIGH_CONF = {HIGH_CONF:.3f}.\n")
print("saved hn_clean.csv", hn.shape, "| ad_roles.csv", ad_roles.shape)
''')
    nb.md("**Next:** Save Version → Output → **New Dataset `jobai-processed`**. Then download `handcheck_soc_sample.csv`, label it, and re-run once the labels are uploaded.")
    nb.code(OUTPUT_LIST)
    return nb
