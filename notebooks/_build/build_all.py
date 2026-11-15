"""Write NB2–NB6 as Kaggle .ipynb files next to this folder. Run: python -I build_all.py (stdlib only)."""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_nb2, build_nb3, build_nb4, build_nb5, build_nb6  # noqa: E402

TARGETS = [(build_nb2, "02-clean-and-link.ipynb"), (build_nb3, "03-event-study.ipynb"),
           (build_nb4, "04-skill-extraction.ipynb"), (build_nb5, "05-seniority-countries.ipynb"),
           (build_nb6, "06-forecast-and-figures.ipynb")]

for mod, name in TARGETS:
    path = os.path.join(os.path.dirname(HERE), name)
    n_cells, n_code = mod.build().write(path)       # write() runs ast.parse on every code cell first
    with open(path, encoding="utf-8") as f:
        json.load(f)                                 # the file is valid JSON
    print(f"{name}: {n_cells} cells ({n_code} code) OK")
