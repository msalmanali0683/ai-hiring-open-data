"""NB4 — 04-skill-extraction (main deep-learning notebook)."""
from common import NB, PIP, FIND, SEEDS, OUTPUT_LIST, VERSIONS

HELPERS = r'''
import re, json, math, random

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+#]*(?:[./&-][A-Za-z0-9+#]+)*")
# Skills whose names are also ordinary words: matched only with the same capitalisation as the skill name.
CASE_SENSITIVE = {"go", "r", "c", "swift", "rust", "chef", "puppet", "spark", "access", "word", "excel", "outlook",
                  "teams", "office", "project", "visio", "shell", "make", "dart", "salt", "ant", "elm", "crystal",
                  "pig", "hive", "react", "express", "node", "next", "box", "slack", "zoom", "notion", "net",
                  "basic", "logo", "scheme", "prolog", "pascal", "processing", "sage", "maple", "julia", "keras",
                  "bash", "unity", "ember", "meteor", "flow", "jest", "mocha", "storm", "beam", "airflow", "dash"}

def light_stem(w):
    """Very light plural stripping, applied identically to ad text and skill names."""
    w = w.lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("ses", "xes", "ches", "shes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w

def build_kw_index(skills):
    """skills: list of names. Returns (index: stem-tuple -> [ids], cs_index: original-token-tuple -> [ids], maxn)."""
    index, cs_index, maxn = {}, {}, 1
    for i, s in enumerate(skills):
        toks = TOKEN_RE.findall(s or "")
        if not toks:
            continue
        maxn = max(maxn, len(toks))
        if len(toks) == 1 and (toks[0].lower() in CASE_SENSITIVE or len(toks[0]) <= 2):
            cs_index.setdefault(tuple(toks), []).append(i)
        else:
            index.setdefault(tuple(light_stem(t) for t in toks), []).append(i)
    return index, cs_index, min(maxn, 6)

def kw_match(text, index, cs_index, maxn):
    """Set of skill ids whose (stemmed) token sequence appears in the text."""
    toks = TOKEN_RE.findall(text or "")
    stems = [light_stem(t) for t in toks]
    hits = set()
    for i in range(len(toks)):
        if (toks[i],) in cs_index:
            hits.update(cs_index[(toks[i],)])
        for n in range(1, maxn + 1):
            if i + n > len(toks):
                break
            key = tuple(stems[i:i + n])
            if key in index:
                hits.update(index[key])
    return hits

def split_sentences(text, min_len=20, max_len=400):
    """Lines and sentences; short bullet lines ('- Python') are merged so they are not lost."""
    out, buf = [], ""
    for line in (text or "").split("\n"):
        for s in re.split(r"(?<=[.!?;])\s+", line.strip()):
            s = s.strip(" -*•\t")
            if not s:
                continue
            buf = f"{buf}, {s}" if buf else s
            if len(buf) >= min_len:
                out.append(buf[:max_len])
                buf = ""
    if buf:
        if out and len(buf) < min_len:
            out[-1] = (out[-1] + ", " + buf)[:max_len]
        else:
            out.append(buf[:max_len])
    return out

def parse_llm_list(s, max_len=60):
    """Skills from an LLM answer: a JSON list if there is one, else bullet / numbered / comma lists."""
    s = s or ""
    items = None
    a, b = s.find("["), s.rfind("]")
    if a != -1 and b > a:
        try:
            v = json.loads(s[a:b + 1])
            if isinstance(v, list):
                items = [str(x) for x in v if isinstance(x, (str, int, float))]
        except ValueError:
            items = None
    if items is None:
        lines = [l.strip() for l in s.split("\n") if l.strip()]
        bullets = [re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", l) for l in lines if re.match(r"^(?:[-*•]|\d+[.)])\s*", l)]
        items = bullets if bullets else re.split(r"[,;]", s)
    seen, out = set(), []
    for x in items:
        x = x.strip().strip("\"'`").strip()
        k = x.lower()
        if x and len(x) <= max_len and k not in seen:
            seen.add(k)
            out.append(x)
    return out

def parse_int_list(s, n_max):
    """Candidate numbers chosen by the LLM re-ranker, e.g. '[1, 4, 7]' -> [1, 4, 7] (1-based, <= n_max)."""
    a, b = (s or "").find("["), (s or "").rfind("]")
    body = s[a:b + 1] if a != -1 and b > a else (s or "")
    return sorted({int(x) for x in re.findall(r"\d+", body) if 1 <= int(x) <= n_max})

def micro_prf(tp, npred, ngold):
    p = tp / npred if npred else 0.0
    r = tp / ngold if ngold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f

def rp_at_k(ranked, gold, k=5):
    """R-precision@k (Decorte et al.): relevant in top k / min(k, |gold|). None if no gold."""
    if not gold:
        return None
    return len(set(ranked[:k]) & set(gold)) / min(k, len(gold))

def mrr(ranked, gold):
    for i, s in enumerate(ranked, 1):
        if s in gold:
            return 1.0 / i
    return 0.0

def cohen_kappa(a, b):
    """Cohen's kappa for two equal-length label lists (any hashable labels)."""
    n = len(a)
    if n == 0 or n != len(b):
        return float("nan")
    labels = sorted(set(a) | set(b), key=str)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)

def norm_skill(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+#. ]", " ", (s or "").lower())).strip()
'''


def build():
    nb = NB(gpu=True, internet=True)
    nb.md(r'''
# NB4 — Skill extraction and skill trends (`04-skill-extraction`) → RQ2 (headline)
Main deep-learning notebook. Compares **eight extractors** on a hand-labelled, year-stratified set, then runs the best scalable one on every ad (`RESEARCH_PLAN.md` §5.2; `KAGGLE_IMPLEMENTATION.md` §5).

| Method | What it is | Runs on |
|---|---|---|
| `keyword` | O\*NET skill names, token n-gram match + light plural stripping (ambiguous names like *Go*, *R*, *Excel* need matching capitals) | all ads |
| `minilm` | `sentence-transformers/all-MiniLM-L6-v2`, sentence ↔ skill cosine | all ads |
| `bge` | `BAAI/bge-small-en-v1.5` (second generic encoder) | eval ads |
| `context` | **`TechWolf/ConTeXT-Skill-Extraction-base`** (Decorte et al. 2025 job-domain skill encoder; used as a bi-encoder, mean pooling as published) | all ads if best |
| `context_tok` | same model, **token-level attention** scoring (our implementation of the ConTeXT-match idea; check against the paper before naming it ConTeXT-match) | eval ads |
| `ce_rerank` | bi-encoder top-30 re-scored by a cross-encoder (`cross-encoder/ms-marco-MiniLM-L6-v2`) | eval ads |
| `llm_extract` | open **7–8B instruct LLM in 4-bit** (Qwen2.5-7B-Instruct; fallbacks Qwen3-8B, Mistral-7B-v0.3), few-shot "list the skills", outputs mapped to the taxonomy | eval ads |
| `llm_rerank` | retrieve (best bi-encoder top-25) → the same LLM selects the skills the ad asks for (Clavié & Soulié 2023 design) | eval ads |

(JobBERT-v2 is a *job-title* encoder; it is used for ad → occupation matching in NB2, not for skills.)

| | |
|---|---|
| **Input** | **`jobai-processed`** (`hn_clean.csv`, `exposure_table.csv`, `aei_task_table.csv`), **`jobai-raw`** (O\*NET 31: `onet_Software_Skills.csv`, `onet_Essential_Skills.csv`, `onet_Transferable_Skills.csv`; falls back to O\*NET ≤ 30 `onet_Technology_Skills.csv`, `onet_Skills.csv`) |
| **Optional input (2nd run)** | dataset **`jobai-skill-labels`** with the filled `labels_candidates.csv` and `labels_ads.csv` |
| **Output** → dataset **`jobai-skills`** | `skill_taxonomy.csv`, `ad_skills.csv` (best method), `ad_skills_keyword.csv`, `ad_skills_minilm.csv`, `ad_skill_summary.csv`, `skill_exposure.csv`, `skill_month_share.csv`, `skill_change.csv`, `skill_change_tests.csv`, `fig3_skills.png`; labelling pack `labels_ads.csv`, `labels_candidates.csv`, `LABELING_GUIDELINES.md`; evaluation `table3_skill_eval.csv`, `eval_by_year.csv`, `eval_error_analysis.csv`, `eval_drift_test.csv`, `pr_curves.csv`, `fig_pr_curves.png`, `fig_f1_by_year.png`, `kappa.csv`, `chosen_method.json`; `emerging_topics.csv` (BERTopic) |
| **Kaggle settings** | Accelerator **GPU T4** (×2 is fine) · **Internet On** (models from Hugging Face; nothing is gated) |
| **Run time** | ~1.5–2.5 h with both LLM passes (each ~20–40 min for 400 ads); ~45 min with `RUN_LLM = False` |

**Two-pass workflow.** Run 1 builds the year-stratified evaluation sample (50 ads × 8 years), runs every method on it and writes a **pooled candidate list** for annotation. Two annotators label the first 15 ads of each year (≥100 shared ads → Cohen's κ), settle disagreements, and one labels the rest. Upload the filled files as `jobai-skill-labels` and run again: run 2 tunes thresholds on the **dev** split, reports **test** P/R/F1, PR curves, RP@5/MRR, F1 by year, and picks the method for the full corpus by **dev** F1. Until labels exist, the full-corpus run uses a provisional threshold and every downstream number is provisional.

Nothing has been run; no result is written here.
''')
    nb.code(PIP + r'''
# Cell 1 — installs, imports, settings
RUN_LLM = True              # LLM extractor + LLM re-ranker on the eval sample (~40-80 min on T4)
RUN_BERTOPIC = True         # emerging-skill clusters from sentences the taxonomy misses
pip("bitsandbytes>=0.43", "accelerate")
try:
    import sentence_transformers
except ImportError:
    pip("sentence-transformers")
if RUN_BERTOPIC:
    pip("bertopic")
''' + FIND + r'''
import re, io, gc, json, math, time, warnings
import numpy as np, pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy import stats
''' + SEEDS + VERSIONS + r'''
warnings.filterwarnings("ignore")
OUT = "/kaggle/working"
PROC = find_dir("jobai-processed", "hn_clean.csv")
RAW = find_dir("jobai-raw", "hn_jobs.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("processed:", PROC, "| raw:", RAW, "| device:", DEVICE,
      torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")

ENCODER_IDS = {"minilm": "sentence-transformers/all-MiniLM-L6-v2",
               "bge": "BAAI/bge-small-en-v1.5",
               "context": "TechWolf/ConTeXT-Skill-Extraction-base"}
CROSS_ENCODER_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
LLM_CANDIDATES = ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-8B", "mistralai/Mistral-7B-Instruct-v0.3"]
N_PER_YEAR = 50             # evaluation ads per year (8 years -> 400)
N_DOUBLE_PER_YEAR = 15      # of which labelled by both annotators (-> 120 >= 100)
DEV_SHARE = 0.4             # dev (threshold tuning) vs test (reporting), by ad, stratified by year
PROVISIONAL_T = 0.60        # used only until labels exist
PRE = ("2019-01", "2022-11")          # pre-ChatGPT window for trends
MIN_MENTIONS = 100          # skills with fewer ad mentions in total are not ranked
save_versions(["pandas", "numpy", "torch", "sentence_transformers", "transformers", "bitsandbytes", "accelerate"],
              f"{OUT}/versions_nb4.txt")
plt.rcParams.update({"savefig.dpi": 300, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
''')
    nb.code("# Cell 2 — helpers (plain Python, unit-tested locally on synthetic text)" + HELPERS + r'''
print(split_sentences("We use Python and Go.\n- AWS\n- Postgres\nYou will build APIs!"))
print(parse_llm_list('Here you go: ["Python", "AWS", "python"]'), parse_int_list("[2, 5, 99]", 10))
''')
    nb.md(r'''
## Skill taxonomy
O\*NET 31 **Software Skills** (`Workplace Example` = tools, languages, platforms; `Element Name` = product category; `Hot Technology` / `In Demand` flags kept), O\*NET 31 **Essential Skills** (2.A.\*, e.g. *Reading Comprehension*) + **Transferable Skills** (2.B.\*, e.g. *Social Perceptiveness*), plus a short custom list of AI-era skills O\*NET may lag on. O\*NET ≤ 30 names (`Technology_Skills` with `Example` / `Commodity Title`, `Skills`) are used as a fallback. Internally the columns keep the old names `Example` and `Commodity Title`.
''')
    nb.code(r'''
# Cell 3 — data + taxonomy
def load_onet(table, need_cols):
    path = os.path.join(RAW, f"onet_{table}.csv")
    if not os.path.exists(path):
        avail = sorted(os.path.basename(p) for p in glob.glob(os.path.join(RAW, "onet_*.csv")))
        close = [a for a in avail if table.lower().replace("_", "")[:8] in a.lower().replace("_", "")]
        if len(close) != 1:
            raise FileNotFoundError(f"onet_{table}.csv not in jobai-raw; close matches {close}; all: {avail}")
        path = os.path.join(RAW, close[0])
    df = pd.read_csv(path, dtype=str)
    low = {c.lower(): c for c in df.columns}
    miss = [c for c in need_cols if c not in df.columns and c.lower() not in low]
    if miss:
        raise KeyError(f"{os.path.basename(path)} lacks {miss}; columns {list(df.columns)}")
    return df.rename(columns={low[c.lower()]: c for c in need_cols if c not in df.columns})

hn = pd.read_csv(f"{PROC}/hn_clean.csv", dtype={"month": str, "soc6": str})
hn = hn[hn["text"].notna()].reset_index(drop=True)
exposure = pd.read_csv(f"{PROC}/exposure_table.csv", dtype={"soc6": str})
def has_onet(t):
    return os.path.exists(os.path.join(RAW, f"onet_{t}.csv"))
SKILL_COLS = ["O*NET-SOC Code", "Element ID", "Element Name", "Scale ID", "Data Value"]
if has_onet("Software_Skills"):                                       # O*NET 31+
    tech = load_onet("Software_Skills", ["O*NET-SOC Code", "Workplace Example", "Element Name"])
    tech = tech.rename(columns={"Workplace Example": "Example", "Element Name": "Commodity Title"})
else:                                                                  # O*NET <= 30
    tech = load_onet("Technology_Skills", ["O*NET-SOC Code", "Example"])
if has_onet("Essential_Skills"):                                      # O*NET 31+: skills split in two tables
    sk = pd.concat([load_onet("Essential_Skills", SKILL_COLS).assign(skill_group="essential"),
                    load_onet("Transferable_Skills", SKILL_COLS).assign(skill_group="transferable")], ignore_index=True)
else:                                                                  # O*NET <= 30
    sk = load_onet("Skills", SKILL_COLS)
    sk["skill_group"] = np.where(sk["Element ID"].str.startswith("2.A"), "essential", "transferable")
print("O*NET software rows", len(tech), "| skill rows", len(sk), sk.drop_duplicates("Element Name")["skill_group"].value_counts().to_dict())

CUSTOM_AI = ["large language models", "LLM", "prompt engineering", "retrieval-augmented generation", "RAG",
             "generative AI", "fine-tuning", "vector database", "AI agents", "LangChain", "LlamaIndex", "MLOps",
             "LLMOps", "PyTorch", "Hugging Face", "OpenAI API", "embeddings", "transformers", "diffusion models",
             "computer vision", "natural language processing", "reinforcement learning from human feedback",
             "model evaluation", "AI safety", "GPU programming", "CUDA"]
TECH_COLS = ["Example", "Commodity Title", "Hot Technology", "In Demand"]
rows = ([{"skill": s, "source": "onet_tech", "commodity": c, "hot": h, "in_demand": d}
         for s, c, h, d in tech.reindex(columns=TECH_COLS).itertuples(index=False)]
        + [{"skill": s, "source": "onet_skill", "skill_group": g}
           for s, g in sk.drop_duplicates("Element Name")[["Element Name", "skill_group"]].dropna(subset=["Element Name"])
                         .itertuples(index=False)]
        + [{"skill": s, "source": "custom_ai"} for s in CUSTOM_AI])
tax = pd.DataFrame(rows).dropna(subset=["skill"])
tax["key"] = tax["skill"].apply(norm_skill)
tax = tax[tax["key"].str.len() >= 1].drop_duplicates("key").reset_index(drop=True)
tax["skill_id"] = [f"S{i:05d}" for i in range(len(tax))]
tax["skill_type"] = tax["source"].map({"onet_tech": "tool / technology", "onet_skill": "generic skill",
                                       "custom_ai": "AI-era (custom)"})
SKILLS = tax["skill"].tolist()
ID = tax["skill_id"].values
print(len(tax), "skills:", tax["source"].value_counts().to_dict(), "| ads:", len(hn))
''')
    nb.code(r'''
# Cell 4 — sentences of every ad + keyword baseline on every ad
sent = hn[["comment_id", "month", "text"]].copy()
sent["sentence"] = sent["text"].apply(split_sentences)
sent = sent.drop(columns="text").explode("sentence").dropna(subset=["sentence"]).reset_index(drop=True)
print(f"{len(sent):,} sentences ({len(sent) / len(hn):.1f} per ad)")

t0 = time.time()
KW = build_kw_index(SKILLS)
kw_hits = [(cid, i) for cid, txt in zip(hn["comment_id"], hn["text"]) for i in kw_match(txt, *KW)]
kw_long = pd.DataFrame(kw_hits, columns=["comment_id", "skill_idx"])
kw_long["skill_id"] = ID[kw_long["skill_idx"].values]
print(f"keyword baseline: {len(kw_long):,} (ad, skill) hits in {time.time() - t0:.0f}s; "
      f"{kw_long.groupby('comment_id').size().median():.0f} skills per ad (median)")
''')
    nb.code(r'''
# Cell 5 — year-stratified evaluation sample, dev/test split (deterministic -> identical on run 2)
rng = np.random.RandomState(SEED)
ev = (hn.groupby("year", group_keys=False)
        .apply(lambda g: g.sample(min(N_PER_YEAR, len(g)), random_state=SEED))
        .reset_index(drop=True))
ev["order"] = ev.groupby("year").cumcount()
ev["double"] = (ev["order"] < N_DOUBLE_PER_YEAR).astype(int)
ev["split"] = ev.groupby("year")["order"].transform(lambda o: np.where(
    rng.permutation(len(o)) < round(DEV_SHARE * len(o)), "dev", "test"))
EV_IDS = set(ev["comment_id"])
ev_sent = sent[sent["comment_id"].isin(EV_IDS)].reset_index(drop=True)
print(ev.groupby("year").agg(ads=("comment_id", "size"), double=("double", "sum"),
                             dev=("split", lambda s: (s == "dev").sum())).T)
''')
    nb.md("## Embedding extractors (sentence ↔ skill cosine; an ad's score for a skill = its best sentence)")
    nb.code(r'''
# Cell 6 — encoders and ad-level scoring
from sentence_transformers import SentenceTransformer
ENC, SKILL_EMB = {}, {}
for name, mid in ENCODER_IDS.items():
    try:
        ENC[name] = SentenceTransformer(mid, device=DEVICE)
        if DEVICE == "cuda":
            ENC[name].half()
        print("✓", name, mid)
    except Exception as e:
        print(f"⚠ {name} ({mid}) could not be loaded: {e} — it is skipped and the paper should say so")

def encode(name, texts, bs=256):
    return ENC[name].encode(texts, batch_size=bs, convert_to_tensor=True, normalize_embeddings=True,
                            show_progress_bar=len(texts) > 50000).float()

for name in ENC:
    SKILL_EMB[name] = encode(name, SKILLS)

def score_ads(name, sdf, k=10, chunk=16384):
    """Top-k skills per sentence, then max over each ad's sentences.
    Returns comment_id, skill_idx, skill_id, score, sent_row (row of sdf holding the best sentence)."""
    S = SKILL_EMB[name]
    parts = []
    texts = sdf["sentence"].tolist()
    for i in range(0, len(texts), chunk):
        E = encode(name, texts[i:i + chunk])
        v, idx = torch.topk(E @ S.T, k=k, dim=1)
        parts.append(pd.DataFrame({"sent_row": np.repeat(np.arange(i, i + len(E)), k),
                                   "skill_idx": idx.cpu().numpy().ravel(), "score": v.cpu().numpy().ravel()}))
    long = pd.concat(parts, ignore_index=True)
    long["comment_id"] = sdf["comment_id"].values[long["sent_row"].values]
    long = long.sort_values("score", ascending=False).drop_duplicates(["comment_id", "skill_idx"])
    long["skill_id"] = ID[long["skill_idx"].values]
    return long.reset_index(drop=True)

EV_SCORES = {}
for name in ENC:
    t0 = time.time()
    EV_SCORES[name] = score_ads(name, ev_sent, k=50)
    print(f"{name}: scored {len(ev_sent):,} eval sentences in {time.time() - t0:.0f}s")
''')
    nb.code(r'''
# Cell 7 — token-level attention scoring with the ConTeXT model (eval ads only)
# score(sentence, skill) = sum_t softmax_t(e_t·s / TAU) * (e_t·s), e_t = normalised token embeddings.
TAU = 0.05
if "context" in ENC:
    m = ENC["context"]
    S = SKILL_EMB["context"]
    parts = []
    texts = ev_sent["sentence"].tolist()
    for i in range(0, len(texts), 256):
        toks = m.encode(texts[i:i + 256], output_value="token_embeddings", batch_size=64, convert_to_tensor=False)
        for j, T in enumerate(toks):
            T = torch.nn.functional.normalize(T.float().to(DEVICE)[1:-1] if len(T) > 2 else T.float().to(DEVICE), dim=1)
            A = T @ S.T                                            # tokens x skills
            sc = (torch.softmax(A / TAU, dim=0) * A).sum(0)
            v, idx = torch.topk(sc, k=50)
            parts.append(pd.DataFrame({"sent_row": i + j, "skill_idx": idx.cpu().numpy(), "score": v.cpu().numpy()}))
    long = pd.concat(parts, ignore_index=True)
    long["comment_id"] = ev_sent["comment_id"].values[long["sent_row"].values]
    long = long.sort_values("score", ascending=False).drop_duplicates(["comment_id", "skill_idx"])
    long["skill_id"] = ID[long["skill_idx"].values]
    EV_SCORES["context_tok"] = long.reset_index(drop=True)
    print("context_tok scored", EV_SCORES["context_tok"]["comment_id"].nunique(), "ads")
''')
    nb.code(r'''
# Cell 8 — cross-encoder re-ranking of the bi-encoder's top-30 per ad (eval ads only)
BASE_RETRIEVER = "context" if "context" in EV_SCORES else "minilm"
try:
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(CROSS_ENCODER_ID, device=DEVICE, max_length=256)
    cand = EV_SCORES[BASE_RETRIEVER].groupby("comment_id").head(30).copy()
    pairs = list(zip(ev_sent["sentence"].values[cand["sent_row"].values], np.array(SKILLS)[cand["skill_idx"].values]))
    cand["score"] = 1 / (1 + np.exp(-np.asarray(ce.predict(pairs, batch_size=256, show_progress_bar=False))))
    EV_SCORES["ce_rerank"] = cand.sort_values("score", ascending=False).reset_index(drop=True)
    print(f"ce_rerank: {len(cand):,} pairs re-scored (retriever = {BASE_RETRIEVER})")
    del ce
except Exception as e:
    print("⚠ cross-encoder skipped:", e)
''')
    nb.md(r'''
## Open LLM (4-bit) — extractor and re-ranker on the evaluation ads
Greedy decoding, fixed prompt, two short few-shot examples. Extracted strings are mapped to the taxonomy with the MiniLM encoder (cosine ≥ 0.80, or an exact normalised name); unmapped strings are kept as out-of-taxonomy predictions, so they count against precision unless an annotator wrote the same skill.
''')
    nb.code(r'''
# Cell 9 — LLM passes
FEWSHOT = [
    ("Acme | Senior Backend Engineer | Remote (EU)\nWe build payment APIs in Go and PostgreSQL on AWS. "
     "You have 5+ years of backend experience, know Kubernetes, and write clear docs.",
     '["Go", "PostgreSQL", "AWS", "Kubernetes", "backend development", "technical writing"]'),
    ("Foodly | Marketing Manager | NYC\nOwn our email campaigns and SEO, analyse results in Google Analytics, "
     "and manage a small team.",
     '["email marketing", "SEO", "Google Analytics", "team management"]'),
]
INSTR = ("You extract skills from job ads. List every skill, tool, programming language, framework, method or "
         "area of knowledge that the ad asks candidates to have or will use in the job. Use short canonical names. "
         "Do not list benefits, locations, or the company's product. Answer with a JSON list of strings only.")
LLM_ID, llm, tok = None, None, None
llm_extract = pd.DataFrame(columns=["comment_id", "raw"])
llm_rr = pd.DataFrame(columns=["comment_id", "skill_idx"])

def chat(user_text, examples=FEWSHOT, instr=INSTR):
    msgs = []
    for k, (q, a) in enumerate(examples):
        msgs += [{"role": "user", "content": (instr + "\n\nJob ad:\n" if k == 0 else "Job ad:\n") + q},
                 {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": ("" if examples else instr + "\n\n") + user_text})
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def generate(prompts, max_new_tokens=200, bs=8):
    outs = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=3072,
                  add_special_tokens=False).to(llm.device)      # the chat template already added BOS
        g = llm.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        if (i // bs) % 10 == 0:
            print(f"  {i + len(prompts[i:i + bs])}/{len(prompts)}")
    return outs

# full-corpus LLM outputs from NB4b (vLLM, sharded); if present they are also used for the eval ads, so the
# reported F1 is the F1 of exactly what ran on the corpus
LLM_FULL_FILES = sorted(glob.glob("/kaggle/input/**/llm_raw_full_shard*.csv", recursive=True))
llm_full = pd.DataFrame(columns=["comment_id", "raw"])
if LLM_FULL_FILES:
    llm_full = pd.concat([pd.read_csv(f, dtype={"comment_id": str}) for f in LLM_FULL_FILES], ignore_index=True)
    llm_full = llm_full.drop_duplicates("comment_id")
    LLM_ID = str(llm_full["engine"].iloc[0]) if "engine" in llm_full else "NB4b"
    cover = hn["comment_id"].astype(str).isin(llm_full["comment_id"]).mean()
    print(f"NB4b full-corpus LLM outputs: {len(LLM_FULL_FILES)} shards, {len(llm_full):,} ads, coverage {cover:.1%}, engine {LLM_ID}")
    ev_ids = set(ev["comment_id"].astype(str))
    llm_extract = llm_full[llm_full["comment_id"].isin(ev_ids)][["comment_id", "raw"]].copy()
    llm_extract["comment_id"] = llm_extract["comment_id"].astype(ev["comment_id"].dtype)
    print(f"  eval ads covered by NB4b: {llm_extract['comment_id'].nunique()} / {len(ev_ids)}")
    RUN_LLM = False                                # no need to load the model again here (re-ranker skipped)

if RUN_LLM and DEVICE == "cuda":
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.float16)
    for cand_id in LLM_CANDIDATES:
        try:
            tok = AutoTokenizer.from_pretrained(cand_id)
            llm = AutoModelForCausalLM.from_pretrained(cand_id, quantization_config=bnb, device_map="auto",
                                                       torch_dtype=torch.float16)
            LLM_ID = cand_id
            break
        except Exception as e:
            print(f"⚠ {cand_id} failed to load: {str(e)[:200]}")
    if LLM_ID:
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        print("LLM:", LLM_ID)
        ads = ev[["comment_id", "text"]].copy()
        ads["text"] = ads["text"].str[:2000]
        t0 = time.time()
        raw = generate([chat("Job ad:\n" + t) for t in ads["text"]])
        llm_extract = pd.DataFrame({"comment_id": ads["comment_id"].values, "raw": raw})
        print(f"LLM extraction: {time.time() - t0:.0f}s")
        # re-ranker: choose among the retriever's top-25
        RR_INSTR = ("Below is a job ad and a numbered list of candidate skills. Return the numbers of the candidates "
                    "that the ad asks for or that the job clearly uses, as a JSON list of integers. Return [] if none.")
        top = EV_SCORES[BASE_RETRIEVER].groupby("comment_id").head(25)
        cand_lists = top.groupby("comment_id")["skill_idx"].apply(list)
        prompts, ids = [], []
        for cid, txt in zip(ads["comment_id"], ads["text"]):
            cl = cand_lists.get(cid, [])
            numbered = "\n".join(f"{k + 1}. {SKILLS[s]}" for k, s in enumerate(cl))
            prompts.append(chat(f"Job ad:\n{txt}\n\nCandidates:\n{numbered}", examples=[], instr=RR_INSTR))
            ids.append(cid)
        t0 = time.time()
        rr_raw = generate(prompts, max_new_tokens=80)
        rr_rows = [(cid, cand_lists.get(cid, [])[k - 1]) for cid, r in zip(ids, rr_raw)
                   for k in parse_int_list(r, len(cand_lists.get(cid, [])))]
        llm_rr = pd.DataFrame(rr_rows, columns=["comment_id", "skill_idx"])
        print(f"LLM re-ranking: {time.time() - t0:.0f}s, {len(llm_rr)} selections")
        del llm; gc.collect(); torch.cuda.empty_cache()
else:
    print("LLM passes skipped (RUN_LLM is False or no GPU)")

# map LLM strings to the taxonomy
llm_long = pd.DataFrame(columns=["comment_id", "skill_id", "text"])
if len(llm_extract):
    ll = llm_extract.assign(item=llm_extract["raw"].apply(parse_llm_list)).explode("item").dropna(subset=["item"])
    ll["key"] = ll["item"].apply(norm_skill)
    exact_map = dict(zip(tax["key"], tax["skill_id"]))
    uniq = ll["key"].drop_duplicates().tolist()
    E = encode("minilm", uniq)
    v, idx = (E @ SKILL_EMB["minilm"].T).max(dim=1)
    nn_map = {k: (ID[i] if s >= 0.80 else None) for k, s, i in zip(uniq, v.cpu().numpy(), idx.cpu().numpy())}
    ll["skill_id"] = ll["key"].map(lambda k: exact_map.get(k) or nn_map.get(k) or f"oov:{k}")
    llm_long = ll[["comment_id", "skill_id", "item"]].rename(columns={"item": "text"}).drop_duplicates(["comment_id", "skill_id"])
    print(f"LLM items: {len(ll):,}; mapped to taxonomy: {(~llm_long['skill_id'].str.startswith('oov:')).mean():.1%}")
if len(llm_rr):
    llm_rr["skill_id"] = ID[llm_rr["skill_idx"].astype(int).values]
llm_extract.to_csv(f"{OUT}/llm_raw_outputs.csv", index=False)
llm_full_long = pd.DataFrame(columns=["comment_id", "skill_id", "text"])
if len(llm_full):
    lf = llm_full.assign(item=llm_full["raw"].apply(parse_llm_list)).explode("item").dropna(subset=["item"])
    lf["key"] = lf["item"].apply(norm_skill)
    exact_map = dict(zip(tax["key"], tax["skill_id"]))
    uniq = lf["key"].drop_duplicates().tolist()
    nn_map = {}
    for i in range(0, len(uniq), 50_000):
        E = encode("minilm", uniq[i:i + 50_000])
        v, idx = (E @ SKILL_EMB["minilm"].T).max(dim=1)
        nn_map.update({k: (ID[j] if s >= 0.80 else None) for k, s, j in zip(uniq[i:i + 50_000], v.cpu().numpy(), idx.cpu().numpy())})
    lf["skill_id"] = lf["key"].map(lambda k: exact_map.get(k) or nn_map.get(k) or f"oov:{k}")
    llm_full_long = lf[["comment_id", "skill_id", "item"]].rename(columns={"item": "text"}).drop_duplicates(["comment_id", "skill_id"])
    llm_full_long["comment_id"] = llm_full_long["comment_id"].astype(hn["comment_id"].dtype)
    print(f"full-corpus LLM: {len(llm_full_long):,} (ad, skill) pairs; in taxonomy {(~llm_full_long['skill_id'].str.startswith('oov:')).mean():.1%}")
''')
    nb.md(r'''
## Labelling pack (always written)
Pooled candidates per ad = union of all methods' suggestions (keyword hits, each encoder's top 15, LLM outputs), shuffled so annotators cannot tell which method proposed what. Recall is therefore measured against the pool **plus** the skills annotators add in `extra_skills`.
''')
    nb.code(r'''
# Cell 10 — export the annotation pool + guidelines
pool = [kw_long[kw_long["comment_id"].isin(EV_IDS)][["comment_id", "skill_id"]]]
for name, sc in EV_SCORES.items():
    pool.append(sc.groupby("comment_id").head(15)[["comment_id", "skill_id"]])
if len(llm_long):
    pool.append(llm_long[["comment_id", "skill_id"]])
pool = pd.concat(pool).drop_duplicates()
name_of = dict(zip(tax["skill_id"], tax["skill"]))
type_of = dict(zip(tax["skill_id"], tax["skill_type"]))
if len(llm_long):
    name_of.update({r.skill_id: r.text for r in llm_long.itertuples() if r.skill_id.startswith("oov:")})
pool["candidate_skill"] = pool["skill_id"].map(name_of)
pool["skill_type"] = pool["skill_id"].map(type_of).fillna("out of taxonomy (LLM)")
pool = pool.sample(frac=1, random_state=SEED).sort_values("comment_id", kind="stable")
pool = pool.merge(ev[["comment_id", "year", "split", "double"]], on="comment_id")
for c in ["label_A", "label_B", "label"]:
    pool[c] = ""
pool.to_csv(f"{OUT}/labels_candidates.csv", index=False)
ads_out = ev[["comment_id", "year", "month", "split", "double", "text"]].copy()
for c in ["extra_skills_A", "extra_skills_B", "extra_skills", "notes"]:
    ads_out[c] = ""
ads_out.to_csv(f"{OUT}/labels_ads.csv", index=False)
print(f"pool: {len(pool):,} candidates for {pool['comment_id'].nunique()} ads "
      f"({pool.groupby('comment_id').size().median():.0f} per ad median)")

GUIDE = """# Skill labelling guidelines (NB4)

**Unit:** one candidate skill for one job ad (`labels_candidates.csv`), plus free-text additions per ad (`labels_ads.csv`).

1. Read the whole ad in `labels_ads.csv` first, then its candidates.
2. `label = 1` if the ad asks candidates for this skill, says the job uses it (including "our stack is ..."), or lists it
   as nice-to-have. Synonyms and versions count (candidate *PostgreSQL*, ad says *Postgres* -> 1).
3. `label = 0` if the skill is not in the ad, is only part of the company's product description, is a benefit,
   or the candidate is too generic to be supported by the text (e.g. *Microsoft Office* for a backend role that never says so).
4. Soft skills count only when the ad asks for them explicitly ("strong written communication" -> *Writing* = 1).
5. Implicit skills: label 1 only if a reasonable recruiter would agree the ad requires it (e.g. "build iOS apps" -> *Swift*
   is NOT implied; "build iOS apps in SwiftUI" -> *Swift* = 1). Write the reason in `notes` when unsure.
6. `extra_skills`: skills the ad clearly asks for that no candidate covers, separated by `;`.
7. **Double-labelled ads** (`double = 1`): annotator A fills `label_A` / `extra_skills_A`, annotator B fills `label_B` /
   `extra_skills_B` **without looking at A's labels**. Afterwards discuss each disagreement and write the agreed value in
   `label` / `extra_skills`. For all other ads one annotator fills `label_A` / `extra_skills_A` only.
8. Do not change `comment_id`, `skill_id`, `split` or `double`. Upload both files as Kaggle dataset `jobai-skill-labels`.
"""
open(f"{OUT}/LABELING_GUIDELINES.md", "w").write(GUIDE)
''')
    nb.md("## Run 2 only — agreement (κ), gold sets, Table 3")
    nb.code(r'''
# Cell 11 — read labels if they exist
def find_labels(name):
    hits = sorted(glob.glob(f"/kaggle/input/**/jobai-skill-labels/**/{name}", recursive=True) +
                  glob.glob(f"/kaggle/input/jobai-skill-labels/{name}"), key=len)
    if not hits:                                   # labels shipped inside another input dataset (e.g. jobai-nb4-inputs)
        hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True), key=len)
    return hits[0] if hits else None

LAB_C, LAB_A = find_labels("labels_candidates.csv"), find_labels("labels_ads.csv")
HAVE_LABELS = bool(LAB_C and LAB_A)
gold_long = pd.DataFrame(columns=["comment_id", "skill_id"])
if HAVE_LABELS:
    lc = pd.read_csv(LAB_C, dtype={"skill_id": str})
    la = pd.read_csv(LAB_A)
    to01 = lambda s: pd.to_numeric(s, errors="coerce")
    for c in ["label_A", "label_B", "label"]:
        lc[c] = to01(lc[c]) if c in lc else np.nan
    dbl = lc[(lc["double"] == 1) & lc["label_A"].notna() & lc["label_B"].notna()]
    kap = {"pairs": len(dbl), "ads": dbl["comment_id"].nunique(),
           "kappa": cohen_kappa(dbl["label_A"].astype(int).tolist(), dbl["label_B"].astype(int).tolist()),
           "raw_agreement": float((dbl["label_A"] == dbl["label_B"]).mean()) if len(dbl) else np.nan}
    by_type = {t: cohen_kappa(g["label_A"].astype(int).tolist(), g["label_B"].astype(int).tolist())
               for t, g in dbl.groupby("skill_type") if len(g) >= 20}
    pd.DataFrame([kap] + [{"skill_type": t, "kappa": k} for t, k in by_type.items()]).to_csv(f"{OUT}/kappa.csv", index=False)
    print("Cohen's kappa:", {k: round(v, 3) if isinstance(v, float) else v for k, v in kap.items()}, by_type)
    lc["final"] = lc["label"].fillna(lc["label_A"])
    pos = lc[lc["final"] == 1][["comment_id", "skill_id"]]
    # extra skills -> taxonomy ids (exact normalised name or MiniLM >= 0.80), else kept as oov:<name>
    la["final_extra"] = la.get("extra_skills", pd.Series(np.nan, index=la.index)).fillna(la.get("extra_skills_A"))
    ex = la.assign(item=la["final_extra"].fillna("").astype(str).str.split(";")).explode("item")
    ex["key"] = ex["item"].fillna("").apply(norm_skill)
    ex = ex[ex["key"].str.len() > 0]
    if len(ex):
        exact_map = dict(zip(tax["key"], tax["skill_id"]))
        uniq = ex["key"].drop_duplicates().tolist()
        E = encode("minilm", uniq)
        v, idx = (E @ SKILL_EMB["minilm"].T).max(dim=1)
        nn = {k: (ID[i] if s >= 0.80 else f"oov:{k}") for k, s, i in zip(uniq, v.cpu().numpy(), idx.cpu().numpy())}
        ex["skill_id"] = ex["key"].map(lambda k: exact_map.get(k) or nn[k])
        pos = pd.concat([pos, ex[["comment_id", "skill_id"]]])
    gold_long = pos.drop_duplicates()
    labelled = set(lc.loc[lc["final"].notna(), "comment_id"])
    ev = ev[ev["comment_id"].isin(labelled)]
    print(f"gold: {len(gold_long):,} (ad, skill) pairs over {gold_long['comment_id'].nunique()} ads; "
          f"{len(ev)} labelled eval ads ({(ev['split'] == 'dev').sum()} dev / {(ev['split'] == 'test').sum()} test)")
else:
    print("No labels yet (dataset 'jobai-skill-labels' not attached) → evaluation skipped; "
          f"full-corpus threshold = provisional {PROVISIONAL_T}.")
''')
    nb.code(r'''
# Cell 12 — evaluation: dev-tuned thresholds, test P/R/F1 with bootstrap CIs, PR curves, RP@5 / MRR, F1 by year
def counts(pred, gold, ads):
    """Per-ad tp / n_pred / n_gold for the given ads."""
    p = pred[pred["comment_id"].isin(ads)][["comment_id", "skill_id"]].drop_duplicates()
    g = gold[gold["comment_id"].isin(ads)][["comment_id", "skill_id"]].drop_duplicates()
    tp = p.merge(g, on=["comment_id", "skill_id"]).groupby("comment_id").size()
    c = pd.DataFrame({"tp": tp, "npred": p.groupby("comment_id").size(), "ngold": g.groupby("comment_id").size()})
    return c.reindex(list(ads)).fillna(0)

def boot_f1(c, n=2000):
    rs = np.random.RandomState(SEED)
    a = c[["tp", "npred", "ngold"]].values
    f = []
    for _ in range(n):
        s = a[rs.randint(0, len(a), len(a))].sum(0)
        f.append(micro_prf(*s)[2])
    return np.percentile(f, [2.5, 97.5])

PRED_SETS = {"keyword": kw_long[kw_long["comment_id"].isin(EV_IDS)]}
if len(llm_long):
    PRED_SETS["llm_extract"] = llm_long
if len(llm_rr):
    PRED_SETS["llm_rerank"] = llm_rr
THRESH = {name: PROVISIONAL_T for name in EV_SCORES}
CHOSEN = {"method": "context" if "context" in ENC else "minilm", "threshold": PROVISIONAL_T, "source": "provisional"}

if HAVE_LABELS:
    dev_ads = ev.loc[ev["split"] == "dev", "comment_id"].tolist()
    test_ads = ev.loc[ev["split"] == "test", "comment_id"].tolist()
    table, pr_rows, year_rows, err_rows, drift_rows = [], [], [], [], []
    for name, sc in EV_SCORES.items():
        grid = np.unique(np.round(np.quantile(sc["score"], np.linspace(0.05, 0.995, 120)), 4))
        dev_f1 = []
        for t in grid:
            c = counts(sc[sc["score"] >= t], gold_long, dev_ads).sum()
            dev_f1.append(micro_prf(*c.values)[2])
            ct = counts(sc[sc["score"] >= t], gold_long, test_ads).sum()
            P, R, F = micro_prf(*ct.values)
            pr_rows.append({"method": name, "threshold": t, "precision": P, "recall": R, "f1": F})
        THRESH[name] = float(grid[int(np.argmax(dev_f1))])
        PRED_SETS[name] = sc[sc["score"] >= THRESH[name]]
    gold_sets = gold_long.groupby("comment_id")["skill_id"].apply(set)
    for name, pred in PRED_SETS.items():
        c = counts(pred, gold_long, test_ads)
        P, R, F = micro_prf(*c[["tp", "npred", "ngold"]].sum().values)
        lo, hi = boot_f1(c)
        dev_F = micro_prf(*counts(pred, gold_long, dev_ads)[["tp", "npred", "ngold"]].sum().values)[2]
        row = {"method": name, "threshold": THRESH.get(name, np.nan), "precision": P, "recall": R, "f1": F,
               "f1_ci_low": lo, "f1_ci_high": hi, "dev_f1": dev_F, "test_ads": len(test_ads),
               "skills_per_ad": c["npred"].mean()}
        if name in EV_SCORES:                         # ranking metrics before thresholding
            ranked = EV_SCORES[name].groupby("comment_id")["skill_id"].apply(list)
            rp = [rp_at_k(ranked.get(a, []), gold_sets.get(a, set()), 5) for a in test_ads]
            row["RP@5"] = np.nanmean([x for x in rp if x is not None])
            row["MRR"] = np.mean([mrr(ranked.get(a, []), gold_sets.get(a, set())) for a in test_ads
                                  if gold_sets.get(a)])
        table.append(row)
        # F1 by year (test ads) + drift test
        tyr = ev[ev["split"] == "test"].set_index("comment_id")["year"]
        c["year"] = tyr.reindex(c.index).values
        for y, g in c.groupby("year"):
            lo_y, hi_y = boot_f1(g, n=500)
            year_rows.append({"method": name, "year": int(y), "f1": micro_prf(*g[["tp", "npred", "ngold"]].sum().values)[2],
                              "ci_low": lo_y, "ci_high": hi_y, "ads": len(g)})
        per_ad = c.apply(lambda r: micro_prf(r["tp"], r["npred"], r["ngold"])[2] if r["ngold"] + r["npred"] else 1.0, axis=1)
        sl = stats.linregress(c["year"].astype(float), per_ad)
        early = c[c["year"] <= 2022][["tp", "npred", "ngold"]].sum()
        late = c[c["year"] >= 2023][["tp", "npred", "ngold"]].sum()
        drift_rows.append({"method": name, "slope_per_year": sl.slope, "slope_p": sl.pvalue,
                           "f1_2019_22": micro_prf(*early.values)[2], "f1_2023_26": micro_prf(*late.values)[2]})
        # error analysis: recall by skill type and explicit vs implicit gold skills
        g = gold_long[gold_long["comment_id"].isin(test_ads)].copy()
        g["type"] = g["skill_id"].map(type_of).fillna("out of taxonomy")
        kw_set = set(zip(kw_long["comment_id"], kw_long["skill_id"]))
        g["explicit"] = [(a, s) in kw_set for a, s in zip(g["comment_id"], g["skill_id"])]
        hit = set(zip(pred["comment_id"], pred["skill_id"]))
        g["found"] = [(a, s) in hit for a, s in zip(g["comment_id"], g["skill_id"])]
        for key, grp in list(g.groupby("type")) + [("explicit (name appears in text)", g[g["explicit"]]),
                                                   ("implicit (name not in text)", g[~g["explicit"]])]:
            err_rows.append({"method": name, "slice": key, "gold": len(grp), "recall": grp["found"].mean()})
    table3 = pd.DataFrame(table).sort_values("f1", ascending=False)
    table3.to_csv(f"{OUT}/table3_skill_eval.csv", index=False)
    pd.DataFrame(pr_rows).to_csv(f"{OUT}/pr_curves.csv", index=False)
    byyear = pd.DataFrame(year_rows); byyear.to_csv(f"{OUT}/eval_by_year.csv", index=False)
    pd.DataFrame(err_rows).to_csv(f"{OUT}/eval_error_analysis.csv", index=False)
    pd.DataFrame(drift_rows).to_csv(f"{OUT}/eval_drift_test.csv", index=False)
    print(table3.round(3).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    prc = pd.DataFrame(pr_rows)
    for name, g in prc.groupby("method"):
        ax[0].plot(g["recall"], g["precision"], label=name, lw=1)
    for name in PRED_SETS:
        if name not in EV_SCORES:
            r = table3.set_index("method").loc[name]
            ax[0].scatter(r["recall"], r["precision"], marker="x", s=40, label=name, zorder=3)
    ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision"); ax[0].set_title("Test set PR curves"); ax[0].legend(fontsize=7)
    for name, g in byyear.groupby("method"):
        ax[1].errorbar(g["year"], g["f1"], yerr=[g["f1"] - g["ci_low"], g["ci_high"] - g["f1"]], label=name,
                       capsize=2, lw=1, marker="o", ms=3)
    ax[1].set_title("Test F1 by ad year (stability check)"); ax[1].set_ylabel("micro F1"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_pr_curves.png", bbox_inches="tight"); plt.show()

    scalable = [m for m in ["context", "minilm", "bge", "keyword"] + (["llm_extract"] if len(llm_full) else [])
                if m in table3["method"].values]
    best = table3[table3["method"].isin(scalable)].sort_values("dev_f1", ascending=False).iloc[0]
    CHOSEN = {"method": best["method"], "threshold": float(best["threshold"]) if best["method"] not in ("keyword", "llm_extract") else None,
              "source": "best dev F1 among scalable methods", "test_f1": float(best["f1"])}
json.dump({"chosen": CHOSEN, "thresholds": THRESH, "llm": LLM_ID, "retriever_for_rerank": BASE_RETRIEVER},
          open(f"{OUT}/chosen_method.json", "w"), indent=1, default=str)
print("full-corpus method:", CHOSEN)
''')
    nb.md("## Full corpus: extract skills from every ad")
    nb.code(r'''
# Cell 13 — run the chosen method (+ MiniLM and keyword as robustness) on all ads
kw_long[["comment_id", "skill_id"]].merge(hn[["comment_id", "month"]], on="comment_id") \
       .to_csv(f"{OUT}/ad_skills_keyword.csv", index=False)

def full_run(name, t):
    t0 = time.time()
    sc = score_ads(name, sent, k=10)
    sc = sc[sc["score"] >= t][["comment_id", "skill_id", "score"]]
    print(f"{name}: {len(sc):,} (ad, skill) pairs at t={t:.3f} in {time.time() - t0:.0f}s")
    return sc.merge(hn[["comment_id", "month"]], on="comment_id")

if CHOSEN["method"] == "llm_extract":
    ad_skills = llm_full_long[["comment_id", "skill_id"]].assign(score=1.0).merge(hn[["comment_id", "month"]], on="comment_id")
elif CHOSEN["method"] == "keyword":
    ad_skills = kw_long[["comment_id", "skill_id"]].assign(score=1.0).merge(hn[["comment_id", "month"]], on="comment_id")
else:
    ad_skills = full_run(CHOSEN["method"], CHOSEN["threshold"])
if "minilm" in ENC:
    (ad_skills if CHOSEN["method"] == "minilm" else full_run("minilm", THRESH.get("minilm", PROVISIONAL_T))) \
        .to_csv(f"{OUT}/ad_skills_minilm.csv", index=False)
ad_skills["method"] = CHOSEN["method"]
ad_skills.to_csv(f"{OUT}/ad_skills.csv", index=False)
print("ads with ≥1 skill:", f"{ad_skills['comment_id'].nunique() / len(hn):.1%}")
''')
    nb.md(r'''
## Skill exposure and automation vs augmentation
* **Skill exposure (β):** mean Eloundou β of the occupations that list the skill (technology skills: every listing occupation; generic O\*NET skills: occupations in the top quartile of that skill's importance). `high_beta_skill` = top third of skills.
* **Automation share (Anthropic Economic Index):** the same occupation average of the AEI occupation automation share (`aei_automation`, else the task-based version). A second version maps each skill directly to its 5 most similar AEI tasks (MiniLM) and takes their usage-weighted automation share. A skill is tagged **automation-type** if its share is above the median of tagged skills, else **augmentation-type**. AEI is measured after ChatGPT → it describes *how* AI is used, not treatment.
''')
    nb.code(r'''
# Cell 14 — skill-level exposure and automation tags
occ_x = exposure.set_index("soc6")
auto_col = next((c for c in ["aei_automation", "aei_automation_tasks"] if c in occ_x and occ_x[c].notna().any()), None)
t_occ = tech.assign(soc6=tech["O*NET-SOC Code"].str[:7], key=tech["Example"].apply(norm_skill))[["key", "soc6"]]
imp = sk[sk["Scale ID"] == "IM"].assign(soc6=lambda x: x["O*NET-SOC Code"].str[:7],
                                         key=lambda x: x["Element Name"].apply(norm_skill),
                                         v=lambda x: pd.to_numeric(x["Data Value"], errors="coerce"))
imp = imp[imp["v"] >= imp.groupby("key")["v"].transform(lambda s: s.quantile(0.75))][["key", "soc6"]]
links = pd.concat([t_occ, imp]).drop_duplicates()
links = links.join(occ_x[["exp_main"] + ([auto_col] if auto_col else [])], on="soc6")
agg = {"skill_beta": ("exp_main", "mean"), "n_occ": ("soc6", "nunique")}
if auto_col:
    agg["skill_auto_occ"] = (auto_col, "mean")
sx = links.groupby("key").agg(**agg).reset_index()
skill_exposure = tax.merge(sx, on="key", how="left")
b_hi = skill_exposure["skill_beta"].quantile(2 / 3)
skill_exposure["high_beta_skill"] = np.where(skill_exposure["skill_beta"].isna(), np.nan,
                                             (skill_exposure["skill_beta"] >= b_hi).astype(float))

aei_t = pd.read_csv(f"{PROC}/aei_task_table.csv") if os.path.getsize(f"{PROC}/aei_task_table.csv") > 5 else pd.DataFrame()
if len(aei_t) and "automation_pct" in aei_t and "minilm" in ENC:
    aei_t = aei_t.dropna(subset=["automation_pct", "usage_pct"]).drop_duplicates("task_id")
    T = encode("minilm", aei_t["Task"].astype(str).tolist())
    v, idx = torch.topk(SKILL_EMB["minilm"] @ T.T, k=5, dim=1)
    idx = idx.cpu().numpy(); w = aei_t["usage_pct"].values[idx]; a = aei_t["automation_pct"].values[idx]
    skill_exposure["skill_auto_task"] = (w * a).sum(1) / np.where(w.sum(1) > 0, w.sum(1), np.nan)
for col in ["skill_auto_occ", "skill_auto_task"]:
    if col in skill_exposure:
        med = skill_exposure[col].median()
        skill_exposure[col.replace("skill_auto", "tag")] = np.where(skill_exposure[col].isna(), "no data",
            np.where(skill_exposure[col] > med, "automation", "augmentation"))
TAG = "tag_occ" if "tag_occ" in skill_exposure else ("tag_task" if "tag_task" in skill_exposure else None)
if TAG is None:
    skill_exposure["tag_occ"] = "no data"; TAG = "tag_occ"
    print("⚠ no AEI automation data → automation/augmentation split unavailable (check NB2 aei_task_table.csv)")
skill_exposure.to_csv(f"{OUT}/skill_exposure.csv", index=False)
tax.to_csv(f"{OUT}/skill_taxonomy.csv", index=False)
print(skill_exposure[["skill_beta", "skill_auto_occ" if "skill_auto_occ" in skill_exposure else "skill_beta"]].describe().round(3))
print(skill_exposure[TAG].value_counts().to_dict())
''')
    nb.code(r'''
# Cell 15 — ad-level skill summary (used by NB5 for skill-content outcomes)
s = ad_skills.merge(skill_exposure[["skill_id", "skill_beta", "high_beta_skill", "skill_type", TAG]], on="skill_id", how="left")
s["is_auto"] = (s[TAG] == "automation").astype(float).where(s[TAG] != "no data")
s["is_ai"] = (s["skill_type"] == "AI-era (custom)").astype(int)
summ = s.groupby("comment_id").agg(n_skills=("skill_id", "nunique"), mean_skill_beta=("skill_beta", "mean"),
                                   share_high_beta=("high_beta_skill", "mean"), auto_share=("is_auto", "mean"),
                                   n_ai_skills=("is_ai", "sum"),
                                   n_generic=("skill_type", lambda x: (x == "generic skill").sum()))
summ = hn[["comment_id", "month", "year", "company", "soc6", "exp_main"]].merge(summ, on="comment_id", how="left")
summ["n_skills"] = summ["n_skills"].fillna(0)
summ["has_ai_skill"] = (summ["n_ai_skills"].fillna(0) > 0).astype(int)
summ.to_csv(f"{OUT}/ad_skill_summary.csv", index=False)
print(summ.groupby("year")[["n_skills", "mean_skill_beta", "share_high_beta", "auto_share", "has_ai_skill"]].mean().round(3))
''')
    nb.md(r'''
## Fig 3 — rising and falling skills (2019-01–2022-11 vs 2025-01–last month)
Share of ads mentioning each skill; skills with < 100 ad mentions are not ranked. Two-proportion z-tests with Benjamini–Hochberg correction. **H2 test:** do falls concentrate in automation-type skills? (difference in mean log-ratio by tag, bootstrap CI over skills; Spearman correlation with skill β and automation share). The same ranking with the keyword baseline is reported as a robustness check.
''')
    nb.code(r'''
# Cell 16 — skill trends, tests, Fig 3
def skill_trends(asd, label):
    n_month = hn.groupby("month").size()
    sm = asd.drop_duplicates(["comment_id", "skill_id"]).groupby(["skill_id", "month"]).size().rename("ads").reset_index()
    sm["share"] = sm["ads"] / sm["month"].map(n_month)
    pre_m = (hn["month"] >= PRE[0]) & (hn["month"] <= PRE[1])
    late_m = hn["month"] >= "2025-01"
    n_pre, n_late = pre_m.sum(), late_m.sum()
    a = asd.drop_duplicates(["comment_id", "skill_id"]).merge(hn[["comment_id", "month"]], on=["comment_id", "month"])
    c_pre = a[a["month"].between(*PRE)].groupby("skill_id").size()
    c_late = a[a["month"] >= "2025-01"].groupby("skill_id").size()
    ch = pd.DataFrame({"ads_pre": c_pre, "ads_late": c_late}).fillna(0)
    ch["total"] = a.groupby("skill_id").size()
    ch = ch[ch["total"] >= MIN_MENTIONS].copy()
    ch["share_pre"], ch["share_late"] = ch["ads_pre"] / n_pre, ch["ads_late"] / n_late
    ch["diff_pp"] = 100 * (ch["share_late"] - ch["share_pre"])
    ch["log_ratio"] = np.log((ch["ads_late"] + 0.5) / n_late) - np.log((ch["ads_pre"] + 0.5) / n_pre)
    pp = (ch["ads_pre"] + ch["ads_late"]) / (n_pre + n_late)
    zz = (ch["share_late"] - ch["share_pre"]) / np.sqrt(pp * (1 - pp) * (1 / n_pre + 1 / n_late))
    ch["p"] = 2 * stats.norm.sf(np.abs(zz))
    o = np.argsort(ch["p"].values); m = len(o)
    q = ch["p"].values[o] * m / np.arange(1, m + 1)
    ch["q_bh"] = np.empty(m); ch.iloc[o, ch.columns.get_loc("q_bh")] = np.minimum.accumulate(q[::-1])[::-1].clip(max=1)
    ch["method"] = label
    return sm, ch.reset_index().rename(columns={"index": "skill_id"})

sm, change = skill_trends(ad_skills, CHOSEN["method"])
_, change_kw = skill_trends(kw_long[["comment_id", "skill_id"]].merge(hn[["comment_id", "month"]], on="comment_id"), "keyword")
cols = ["skill_id", "skill", "skill_type", "skill_beta", "high_beta_skill", TAG] + \
       [c for c in ["skill_auto_occ", "skill_auto_task"] if c in skill_exposure]
change = change.merge(skill_exposure[cols], on="skill_id", how="left")
# LLM skills outside the taxonomy ("oov:<text>") have no taxonomy name: label them with the model's own wording
def _name_oov(ids):
    return ids.astype(str).str.replace(r"^oov:", "", regex=True) + " (not in taxonomy)"
change["skill"] = change["skill"].fillna(_name_oov(change["skill_id"]))
change["skill_type"] = change["skill_type"].fillna("out of taxonomy (LLM)")
sm = sm.merge(tax[["skill_id", "skill"]], on="skill_id", how="left")
sm["skill"] = sm["skill"].fillna(_name_oov(sm["skill_id"]))
sm.to_csv(f"{OUT}/skill_month_share.csv", index=False)
change.sort_values("diff_pp").to_csv(f"{OUT}/skill_change.csv", index=False)

both = change.merge(change_kw[["skill_id", "log_ratio"]], on="skill_id", suffixes=("", "_kw"))
tests = [{"test": f"Spearman log-ratio {CHOSEN['method']} vs keyword", "n": len(both),
          "stat": stats.spearmanr(both["log_ratio"], both["log_ratio_kw"]).correlation if len(both) > 5 else np.nan}]
for col in ["skill_beta", "skill_auto_occ", "skill_auto_task"]:
    if col in change and change[col].notna().sum() > 10:
        r = stats.spearmanr(change["log_ratio"], change[col], nan_policy="omit")
        tests.append({"test": f"Spearman log-ratio vs {col}", "n": int(change[col].notna().sum()),
                      "stat": r.correlation, "p": r.pvalue})
g_auto = change.loc[change[TAG] == "automation", "log_ratio"].values
g_aug = change.loc[change[TAG] == "augmentation", "log_ratio"].values
if len(g_auto) > 5 and len(g_aug) > 5:
    rs = np.random.RandomState(SEED)
    bd = [rs.choice(g_auto, len(g_auto)).mean() - rs.choice(g_aug, len(g_aug)).mean() for _ in range(5000)]
    tests.append({"test": "mean log-ratio: automation − augmentation skills (H2 expects < 0)",
                  "n": len(g_auto) + len(g_aug), "stat": g_auto.mean() - g_aug.mean(),
                  "ci_low": np.percentile(bd, 2.5), "ci_high": np.percentile(bd, 97.5),
                  "p": stats.mannwhitneyu(g_auto, g_aug).pvalue})
pd.DataFrame(tests).to_csv(f"{OUT}/skill_change_tests.csv", index=False)
print(pd.DataFrame(tests).round(4).to_string(index=False))

rise = change.sort_values("diff_pp", ascending=False).head(20)
fall = change.sort_values("diff_pp").head(20)
colors = {"automation": "#D55E00", "augmentation": "#0072B2", "no data": "#999999"}
fig, ax = plt.subplots(1, 2, figsize=(11, 6))
for a, dfp, title in [(ax[0], fall[::-1], "Falling"), (ax[1], rise[::-1], "Rising")]:
    a.barh(dfp["skill"].str[:35], dfp["diff_pp"], color=dfp[TAG].map(colors).fillna("#999999"))
    a.set_title(f"{title} skills: share of ads, 2025–26 minus 2019–Nov 2022 (pp)", fontsize=9)
    a.axvline(0, color="k", lw=0.6)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
ax[1].legend(handles, colors.keys(), frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_skills.png", bbox_inches="tight"); plt.show()
''')
    nb.md(r'''
## Emerging skill clusters the taxonomy misses (BERTopic, optional)
Sentences from 2023 onward whose best skill score is below the threshold (i.e. the taxonomy found nothing) are clustered; each topic's share of these sentences is compared with the same share before ChatGPT. Use this to spot new skill phrases (e.g. "evals", "agentic workflows") and to decide whether the custom AI list needs additions — then re-run.
''')
    nb.code(r'''
# Cell 17 — BERTopic on unmatched sentences
if RUN_BERTOPIC and "minilm" in ENC:
    try:
        from bertopic import BERTopic
        best_t = CHOSEN["threshold"] or PROVISIONAL_T
        samp = sent.sample(min(len(sent), 400_000), random_state=SEED).reset_index(drop=True)
        # topics should be about work content: drop URLs, emails and hiring boilerplate (pay, equity, office, how to apply)
        samp["sentence"] = (samp["sentence"].str.replace(r"https?://\S+|www\.\S+|\S+@\S+", " ", regex=True)
                                            .str.replace(r"\s+", " ", regex=True).str.strip())
        boiler = (r"(apply|application|applicants?|resume|cv|salary|compensation|equity|benefits?|401k|pto|vacation|"
                  r"visa|sponsor|relocation|hybrid|onsite|on-site|remote|office|days a week|full[- ]?time|part[- ]?time|"
                  r"email|e-mail|contact|interview|hiring process|usd|eur|gbp|\$\d)")
        samp = samp[(samp["sentence"].str.len() >= 30) & ~samp["sentence"].str.lower().str.contains(boiler, regex=True)]
        samp = samp.reset_index(drop=True)
        E = encode("minilm", samp["sentence"].tolist())
        mx = []
        for i in range(0, len(E), 32768):
            mx.append((E[i:i + 32768] @ SKILL_EMB["minilm"].T).max(dim=1).values.cpu().numpy())
        samp["best"] = np.concatenate(mx)
        un = samp[samp["best"] < best_t].copy()
        un["post"] = (un["month"] >= "2022-12").astype(int)       # treatment month as in the plan
        un = un.groupby("post", group_keys=False).apply(lambda g: g.sample(min(len(g), 40_000), random_state=SEED))
        emb = encode("minilm", un["sentence"].tolist()).cpu().numpy()
        from sklearn.feature_extraction.text import CountVectorizer
        topic_model = BERTopic(min_topic_size=40, calculate_probabilities=False, verbose=False,
                               vectorizer_model=CountVectorizer(stop_words="english", min_df=5, ngram_range=(1, 2)))
        un["topic"], _ = topic_model.fit_transform(un["sentence"].tolist(), emb)
        info = topic_model.get_topic_info().set_index("Topic")
        sh = un[un["topic"] >= 0].groupby(["topic", "post"]).size().unstack(fill_value=0)
        sh = sh.div(sh.sum(0), axis=1)
        sh["log_ratio"] = np.log((sh.get(1, 0) + 1e-4) / (sh.get(0, 0) + 1e-4))
        sh["name"] = info.reindex(sh.index)["Name"]
        sh.sort_values("log_ratio", ascending=False).to_csv(f"{OUT}/emerging_topics.csv")
        print(sh.sort_values("log_ratio", ascending=False).head(20)[["name", "log_ratio"]].to_string())
    except Exception as e:
        print("BERTopic step skipped:", type(e).__name__, str(e)[:200])
else:
    print("BERTopic skipped")
''')
    nb.md("**Next:** Save Version → Output → **New Dataset `jobai-skills`**. After run 1, download `labels_candidates.csv`, `labels_ads.csv` and `LABELING_GUIDELINES.md`, label, upload as `jobai-skill-labels`, and run again.")
    nb.code(OUTPUT_LIST)
    return nb
