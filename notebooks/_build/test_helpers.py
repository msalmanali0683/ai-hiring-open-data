"""Stdlib-only checks of the plain-Python helpers embedded in the notebooks. Run: python -I test_helpers.py"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_nb2, build_nb3, build_nb4, build_nb5, build_nb6  # noqa: E402


def load(mod):
    ns = {}
    exec(mod.HELPERS, ns)
    return ns


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


# NB2
h = load(build_nb2)
hdr, comp, roles, src = h["extract_roles"]("Acme Corp | Senior Backend Engineer, Data Scientist | Remote (US) | https://acme.com\nWe build things.")
check(comp and "acme" in comp.lower(), f"company {comp!r}")
check(any("backend engineer" in r.lower() for r in roles), f"roles {roles}")
check(src == "header", src)
check(h["rel_month"]("2022-12") == 0 and h["rel_month"]("2022-11") == -1 and h["rel_month"]("2023-12") == 12, "rel_month")
check(h["company_key"]("Acme, Inc.") == h["company_key"]("acme inc"), "company_key")
check(isinstance(h["clean_role"]("Sr. Software Engineer (Remote)"), str), "clean_role")
f = h["ad_flags"]("Remote OK, visa sponsorship, $150k-$180k")
check(f["remote"] and f["visa"] and f["salary"], f"ad_flags {f}")

# NB3
h = load(build_nb3)
check(h["event_label"](-3) == "Mar–May 2022", h["event_label"](-3))
check(h["event_label"](0) == "Dec 2022–Feb 2023", h["event_label"](0))

# NB4
h = load(build_nb4)
SK = ["Python", "Go", "R", "machine learning", "Kubernetes"]
idx, cs, maxn = h["build_kw_index"](SK)
got = {SK[i] for i in h["kw_match"]("We use Python and Go for machine learning; go to R&D.", idx, cs, maxn)}
check({"Python", "Go", "machine learning"} <= got, got)
check("R" not in got, f"R matched in R&D: {got}")
check(h["parse_int_list"]("Answer: 3, 1, 9, 3", 5) == [1, 3], h["parse_int_list"]("Answer: 3, 1, 9, 3", 5))
lst = h["parse_llm_list"]('["Python", "SQL"]')
check([s.lower() for s in lst] == ["python", "sql"], lst)
p, r, f1 = h["micro_prf"](8, 10, 16)
check(abs(p - 0.8) < 1e-9 and abs(r - 0.5) < 1e-9 and abs(f1 - 2 * 0.8 * 0.5 / 1.3) < 1e-9, (p, r, f1))
check(h["rp_at_k"](["a", "x", "b"], {"a", "b"}, k=5) == 1.0, "rp@k")
check(h["mrr"](["x", "a"], {"a"}) == 0.5, "mrr")
check(abs(h["cohen_kappa"]([1, 1, 0, 0], [1, 1, 0, 0]) - 1.0) < 1e-9, "kappa")
sents = h["split_sentences"]("Short. This sentence is long enough to keep. Another fine sentence follows here.")
check(len(sents) >= 1 and all(len(s) >= 20 for s in sents), sents)

# NB5
h = load(build_nb5)
check(h["level_from_title"]("Lead Generation Specialist") is None, h["level_from_title"]("Lead Generation Specialist"))
check(h["level_from_title"]("Senior Software Engineer") == "senior", "senior")
check(h["level_from_title"]("Junior Developer") == "entry", "entry")
check(h["level_from_title"]("Software Engineer II") == "mid", h["level_from_title"]("Software Engineer II"))
check(h["years_required"]("5+ years of experience with Python") == 5, h["years_required"]("5+ years of experience with Python"))
check(h["years_required"]("we have been around for 20 years") is None, "20 years ignored")
check("senior" not in h["mask_level_words"]("Senior Engineer").lower(), h["mask_level_words"]("Senior Engineer"))

# NB6
h = load(build_nb6)
stat, df = h["dm_test"]([0.1, 0.1, 0.1, 0.1, 0.2], [0.5, 0.4, 0.6, 0.5, 0.45])
check(stat < 0 and df == 4, (stat, df))
stat2, _ = h["dm_test"]([0.5, 0.4, 0.6, 0.5, 0.45], [0.1, 0.1, 0.1, 0.1, 0.2])
check(abs(stat + stat2) < 1e-9, "dm antisymmetric")
m = h["forecast_metrics"]([0.0, 1.0], [0.0, 1.0])
check(m["MAE"] == 0 and m["sMAPE"] == 0, m)

print("all helper tests passed")
