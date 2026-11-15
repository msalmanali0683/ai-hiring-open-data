# Is AI Changing Who Gets Hired? Code

Code for the paper *Is AI Changing Who Gets Hired? An Open Data Analysis of Technology Job Postings and Skill Demand, 2019–2026* (under review at *Information Processing & Management*).

The study uses only open data. It combines 48,157 job advertisements from the monthly Hacker News *Who is hiring?* threads (January 2019 to October 2026) with Indeed Hiring Lab posting indices for six countries and four measures of how exposed each occupation is to AI. It asks four questions:

1. **RQ1** Did postings for highly exposed occupations fall relative to less exposed ones after ChatGPT?
2. **RQ2** Which skills did employers ask for more or less often?
3. **RQ3** Did junior postings fall relative to senior postings within the same companies?
4. **RQ4** Was the fall in US software postings larger than in other countries, and how much of it fits the 2022 change to Section 174 of the US tax code?

## Repository contents

| Path | What it does |
|---|---|
| `notebooks/01-collect-data.ipynb` | Downloads all raw inputs from public sources (see the list below). |
| `notebooks/02-clean-and-link.ipynb` | Cleans the advertisements, splits them into roles and links each role to an O\*NET occupation with a job title encoder. Builds the exposure table and the occupation × month panel. |
| `notebooks/03-event-study.ipynb` | RQ1. Event study and difference in differences, placebo tests, wild bootstrap and the Indeed sector checks. |
| `notebooks/04-skill-extraction.ipynb` | RQ2. Compares eight skill extraction methods on a labelled evaluation set and estimates skill trends. |
| `notebooks/04b-llm-extract-shard0..3.ipynb` | Runs the chosen LLM extractor (Qwen2.5-7B-Instruct, AWQ 4 bit, vLLM) on the full corpus in four parallel shards. |
| `notebooks/05-seniority-countries.ipynb` | RQ3 (seniority within companies, PPML) and RQ4 (country comparison, Section 174 test, synthetic control). |
| `notebooks/06-forecast-and-figures.ipynb` | Forecasting appendix and the final figures and tables. |
| `notebooks/_build/` | Plain Python scripts that generate notebooks 02 to 06 (`python build_all.py`) and test the helper functions (`python test_helpers.py`). Standard library only. |
| `scripts/redraw_figure6.py` | Draws Figure 6 from the notebook 05 outputs with room for all occupation labels. Standard library only. |

## How to run

The notebooks were written for and run on [Kaggle](https://www.kaggle.com/) notebooks. Each one reads its inputs from `/kaggle/input/` and writes to `/kaggle/working/`. The output of one notebook is saved as a Kaggle dataset and added as input to the next.

| Order | Notebook | Hardware | Output saved as |
|---|---|---|---|
| 1 | `01-collect-data` | CPU, Internet on | dataset `jobai-raw` |
| 2 | `02-clean-and-link` | GPU T4, Internet on | dataset `jobai-processed` |
| 3 | `03-event-study` | CPU | notebook output |
| 4 | `04-skill-extraction` and `04b-llm-extract-shard0..3` | GPU T4 | dataset `jobai-skills` |
| 5 | `05-seniority-countries` | GPU T4 | notebook output |
| 6 | `06-forecast-and-figures` | CPU | notebook output (figures and tables) |

Notebooks 03, 04 and 05 can run in parallel once `jobai-processed` exists. Notebook 06 needs the outputs of 03 and 05. Each notebook installs the packages it needs with `pip`; `requirements.txt` lists them for local use. To run outside Kaggle, copy the input files under a folder and change the search path in the `find_file` helper at the top of each notebook.

Random seeds are fixed (`SEED = 42`). Results that use a GPU or a quantised LLM can still differ slightly between runs and hardware.

## Data sources

All inputs are public and are downloaded by notebook 01. No data are stored in this repository.

| Source | Link |
|---|---|
| Hacker News *Who is hiring?* threads, through the Algolia Hacker News Search API | https://hn.algolia.com/api |
| Indeed Hiring Lab job postings indices | https://github.com/hiring-lab/job_postings_tracker |
| FRED (Federal Reserve Bank of St. Louis) | https://fred.stlouisfed.org |
| O\*NET 31.0 | https://www.onetcenter.org/database.html |
| GPTs are GPTs exposure scores (Eloundou et al.) | https://github.com/openai/GPTs-are-GPTs |
| AIOE exposure scores (Felten et al.) | https://github.com/AIOE-Data/AIOE |
| Anthropic Economic Index | https://huggingface.co/datasets/Anthropic/EconomicIndex |

## What is not included

- **Raw advertisement text.** Hacker News advertisements contain names and email addresses, so they are not redistributed. Notebook 01 downloads them again from the public API.
- **Labelled evaluation sets and processed panels.** These are not part of the repository.
- **Planning documents.** Some notebook headers refer to internal planning files (`RESEARCH_PLAN.md`, `KAGGLE_IMPLEMENTATION.md`) that are not included. The notebooks run without them.

## Licence

The code is released under the MIT Licence (see `LICENSE`). The data sources above have their own licences and terms of use.

## Citation

If you use this code, please cite the paper. Full citation details will be added after publication.
