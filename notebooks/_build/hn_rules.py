"""HN title rules for SOC matching. Plain-stdlib module: imported by build_nb2.py (pasted into the notebook)
and by the local test script. Rules run on the cleaned role string, before the embedding match.

Why: in a 304-ad hand check the embedding-only match got ~43% of detailed SOC codes right; most errors were
generic software titles ("engineer", "full stack", "founding engineer") matched to non-computer engineers.
HN 'Who is hiring' ads are tech-company ads, so a bare engineering/developer title defaults to 15-1252 unless
a physical-engineering qualifier is present."""

HN_RULES_SRC = r'''
PHYSICAL_ENG = (r"mechanical|electrical|electronics?|civil|chemical|structural|aerospace|manufacturing|industrial|"
                r"hardware|rf|analog|asic|fpga|pcb|firmware|embedded|robotics?|mechatronics?|optical|photonics?|"
                r"process|materials?|biomedical|nuclear|petroleum|mining|environmental|agricultural|marine|"
                r"automotive|construction")
HN_RULES = [
    # managers / leadership
    (r"\b(cto|vp\b.*\bengineering|vice president\b.*\bengineering|head of (engineering|technology|infrastructure|platform)|"
     r"engineering manager|director of engineering|eng manager|manager,? engineering|engineering director)\b", "11-3021"),
    (r"\b(ceo|chief executive|co-?founder)\b(?!.*engineer)", "11-1011"),
    (r"\b(product manager|product owner|head of product|vp\b.*\bproduct|director of product|product lead|"
     r"technical product manager)\b", "11-2021"),
    (r"\b(project manager|program manager|scrum master|delivery manager)\b", "15-1299"),
    # design
    (r"\bdesigner\b|\bux\b|\bui/ux\b", "15-1255"),
    # data
    (r"\bdata scien|\bmachine learning scien|\bml scien|\bapplied scien|\bresearch scien", "15-2051"),
    (r"\bdata engineer|\betl\b|\banalytics engineer|\bdata platform|\bdata infrastructure|\bdatabase architect", "15-1243"),
    (r"\b(dba|database administrator|database engineer)\b", "15-1242"),
    (r"\b(data analyst|business intelligence|bi analyst|bi developer|business analyst)\b", "15-2051"),
    # security
    (r"\b(security|appsec|infosec|penetration|pentest|red team|cyber)\b", "15-1212"),
    # QA
    (r"\b(qa|quality assurance|test automation|sdet|software tester|tester|test engineer)\b", "15-1253"),
    # ops / infra
    (r"\b(sre|site reliability|devops|dev ops|sysadmin|system administrator|systems administrator|"
     r"infrastructure engineer|cloud engineer|network engineer|it administrator|production engineer)\b", "15-1244"),
    (r"\b(it support|help ?desk|desktop support|technical support|support engineer|customer engineer|"
     r"solutions? engineer|sales engineer|implementation engineer)\b", "15-1232"),
    # web / front end (only when explicitly front-end/web)
    (r"\b(front[- ]?end|frontend|web developer|web engineer|ui engineer|ui developer|wordpress)\b", "15-1254"),
    # everything software
    (r"(\b(software|swe|developer|programmer|full[- ]?stack|fullstack|back[- ]?end|backend|mobile|ios|android|"
     r"react|node|python|java|golang|ruby|rails|django|php|scala|rust|elixir|clojure|haskell|"
     r"kotlin|swift|typescript|javascript|js|ml engineer|machine learning engineer|ai engineer|mlops|"
     r"platform engineer|blockchain|smart contract|game (developer|engineer|programmer)|graphics engineer|"
     r"compiler|kernel|distributed systems|architect|tech lead|hacker|coder)\b|c\+\+|c#|\.net)", "15-1252"),
]
_HN_RULES = [(re.compile(p), c) for p, c in HN_RULES]
_PHYS = re.compile(r"\b(" + PHYSICAL_ENG + r")\b")
_ENG = re.compile(r"\b(engineers?|engineering|member of technical staff|mts)\b")

def hn_rule_soc(role):
    """SOC-6 from HN title rules, or None. A bare 'engineer' (no physical-engineering qualifier) -> 15-1252."""
    r = (role or "").lower()
    if not r or len(r) > 80:            # long strings are parse errors / company blurbs: leave to the model
        return None
    phys = bool(_PHYS.search(r))
    for rx, code in _HN_RULES:
        if rx.search(r):
            if code in ("15-1252", "15-1244") and phys and not re.search(r"software|full[- ]?stack|backend|web", r):
                return None              # e.g. 'hardware systems engineer' -> let the model decide
            return code
    if _ENG.search(r) and not phys:
        return "15-1252"
    return None
'''
