#!/usr/bin/env python3
import json
import sys
from pathlib import Path

if len(sys.argv) != 3:
    raise SystemExit(f"使い方: {sys.argv[0]} warodecode.json llama.json")

wd_text = Path(sys.argv[1]).read_text().strip().splitlines()
wd = json.loads(wd_text[-1])
ll = json.loads(Path(sys.argv[2]).read_text())
if isinstance(ll, dict):
    rows = [ll]
else:
    rows = ll
rows = [r for r in rows if int(r.get("n_prompt", -1)) == 0 and int(r.get("n_gen", -1)) > 0]
if not rows:
    raise SystemExit("llama-bench JSONにtext-generation行がありません")
# ab.shは-n 128を1条件だけ流す。将来複数行になっても同じn_genを優先。
target = next((r for r in rows if int(r.get("n_gen", 0)) == int(wd["tokens"])), rows[0])
wd_tps = float(wd["tok_s"])
ll_tps = float(target["avg_ts"])
ratio = wd_tps / ll_tps
pct = (ratio - 1.0) * 100.0
verdict = "PASS" if pct >= 5.0 else ("TIE" if pct > -5.0 else "FAIL")
print(json.dumps({
    "warodecode_tok_s": wd_tps,
    "llama_cpp_tok_s": ll_tps,
    "speedup_x": round(ratio, 4),
    "difference_percent": round(pct, 2),
    "gate": verdict,
}, ensure_ascii=False, indent=2))
