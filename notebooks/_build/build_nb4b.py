"""NB4b — full-corpus LLM skill extraction, sharded across Kaggle GPU sessions.
Writes ../04b-llm-extract-shard{K}.ipynb for K in 0..N_SHARDS-1. Run: python -I build_nb4b.py
The prompt (instruction + few-shot examples + chat template) is copied verbatim from build_nb4.py, so NB4's
evaluation of `llm_extract` measures exactly this pipeline once NB4 reads these outputs."""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import NB, PIP, FIND

N_SHARDS = 4
src4 = open(os.path.join(HERE, "build_nb4.py"), encoding="utf-8").read()
m = re.search(r"^FEWSHOT = \[.*?^INSTR = \(.*?\)\n", src4, re.S | re.M)
assert m, "FEWSHOT/INSTR block not found in build_nb4.py"
PROMPT_BLOCK = m.group(0)


def build(shard):
    nb = NB(gpu=True, internet=True)
    nb.md(f"""# 04b — LLM skill extraction on the full corpus (shard {shard} of {N_SHARDS})
Runs the NB4 extraction prompt (Qwen2.5-7B-Instruct) on every ad whose position in `hn_clean.csv` (sorted by
`comment_id`) is ≡ {shard} mod {N_SHARDS}. Engine: **vLLM** with the official AWQ 4-bit checkpoint; if vLLM can't
be installed or loaded, it falls back to transformers + bitsandbytes 4-bit (the NB4 setup), within a time budget.
Output: `llm_raw_full_shard{shard}.csv` (`comment_id`, `raw`, `engine`), saved after every chunk.
NB4 reads all shards, re-scores the 400 labelled ads with these outputs, and uses them for the full corpus.
""")
    nb.code(PIP + r'''
import os, sys, time, json, glob
import numpy as np, pandas as pd
SHARD, N_SHARDS = ''' + f"{shard}, {N_SHARDS}" + r'''
OUT = "/kaggle/working"
OUT_FILE = f"{OUT}/llm_raw_full_shard{SHARD}.csv"
TIME_BUDGET_H = 11.0                     # Kaggle sessions stop at 12 h
T_START = time.time()
VLLM_OK = False
try:
    pip("vllm")
    import vllm  # noqa
    VLLM_OK = True
except Exception as e:
    print("vLLM not available:", str(e)[:200])
''' + FIND + r'''
hn = pd.read_csv(find_file("hn_clean.csv"), dtype={"comment_id": str}, usecols=["comment_id", "text"])
hn = hn[hn["text"].notna()].sort_values("comment_id").reset_index(drop=True)
mine = hn[hn.index % N_SHARDS == SHARD].copy()
mine["text"] = mine["text"].str[:2000]
print(f"shard {SHARD}/{N_SHARDS}: {len(mine):,} of {len(hn):,} ads")
''')
    nb.code(PROMPT_BLOCK + r'''
def chat(tok, user_text, examples=FEWSHOT, instr=INSTR):
    msgs = []
    for k, (q, a) in enumerate(examples):
        msgs += [{"role": "user", "content": (instr + "\n\nJob ad:\n" if k == 0 else "Job ad:\n") + q},
                 {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": ("" if examples else instr + "\n\n") + user_text})
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

done = pd.DataFrame(columns=["comment_id", "raw", "engine"])
if os.path.exists(OUT_FILE):
    done = pd.read_csv(OUT_FILE, dtype={"comment_id": str})
todo = mine[~mine["comment_id"].isin(done["comment_id"])]
print("to do:", len(todo))
''')
    nb.code(r'''
ENGINE = None
if VLLM_OK:
    try:
        from vllm import LLM, SamplingParams
        llm = LLM(model="Qwen/Qwen2.5-7B-Instruct-AWQ", quantization="awq", dtype="float16",
                  max_model_len=4096, gpu_memory_utilization=0.90, seed=42)
        tok = llm.get_tokenizer()
        sp = SamplingParams(temperature=0.0, max_tokens=200)
        ENGINE = "vllm:Qwen/Qwen2.5-7B-Instruct-AWQ"
        def gen(prompts):
            return [o.outputs[0].text for o in llm.generate(prompts, sp, use_tqdm=False)]
    except Exception as e:
        print("vLLM load failed:", str(e)[:300])
if ENGINE is None:
    pip("bitsandbytes>=0.43", "accelerate")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    MID = "Qwen/Qwen2.5-7B-Instruct"
    tok = AutoTokenizer.from_pretrained(MID); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(MID, quantization_config=bnb, device_map="auto", dtype=torch.float16)
    ENGINE = "transformers-nf4:" + MID
    @torch.no_grad()
    def gen(prompts, bs=16):
        outs = []
        for i in range(0, len(prompts), bs):
            enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=3072,
                      add_special_tokens=False).to(model.device)
            g = model.generate(**enc, max_new_tokens=200, do_sample=False, pad_token_id=tok.pad_token_id)
            outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return outs
print("engine:", ENGINE)
''')
    nb.code(r'''
CHUNK = 1000
for i in range(0, len(todo), CHUNK):
    if (time.time() - T_START) / 3600 > TIME_BUDGET_H:
        print("time budget reached; stopping (re-run this notebook to continue from the saved file)")
        break
    part = todo.iloc[i:i + CHUNK]
    t0 = time.time()
    raw = gen([chat(tok, "Job ad:\n" + t) for t in part["text"]])
    done = pd.concat([done, pd.DataFrame({"comment_id": part["comment_id"].values, "raw": raw, "engine": ENGINE})],
                     ignore_index=True)
    done.to_csv(OUT_FILE, index=False)
    print(f"{len(done):,}/{len(mine):,} ads | chunk {time.time() - t0:.0f}s | total {(time.time() - T_START) / 60:.0f} min")
print("finished:", len(done), "of", len(mine), "| empty answers:", (done["raw"].fillna("").str.strip() == "").sum())
''')
    return nb


if __name__ == "__main__":
    for k in range(N_SHARDS):
        path = os.path.join(HERE, "..", f"04b-llm-extract-shard{k}.ipynb")
        print(os.path.basename(path), build(k).write(path))
