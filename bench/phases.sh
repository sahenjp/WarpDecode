#!/usr/bin/env bash
# Prefill (プロンプト処理) と decode (1トークン生成) を同一プロセス内で分離計測する。
#
# 計測はエンジン側の CUDA event (prefill_seconds / decode_seconds) で行うため、
# モデルのロード・autotune・CUDA初期化は数値に含まれない。expert cache を変えて
# 複数回走らせても、それぞれ別プロセスとして同じ条件で比較できる。
#
# 使い方:
#   bench/phases.sh ./build-sm89/warodecode /content/models/model.gguf
#
# 実行結果は bench-out/phases-<label>.records.jsonl (生レコード) と
# bench-out/phases-<label>.json (集計) に保存する。
set -euo pipefail

BIN=""
MODEL=""
PREFILL=64
GENERATE=64
CTX=""
REPEATS=3
WARMUP=1
CACHE_LIST="default,8000"
REF=""
SEED=1
VOCAB_LIMIT=100000
LABEL=""
OUT=${OUT:-bench-out}

usage() {
  cat >&2 <<'EOF'
使い方: bench/phases.sh <warodecode> <model.gguf> [オプション]

  --prefill N     プロンプト長 (既定 64, 0で prefill なし = decode のみ)
  --generate N    生成トークン数 (既定 64)
  --ctx N         コンテキスト長 (既定 prefill + generate)
  --repeats N     同一プロセス内の反復回数。最速の回を採用 (既定 3)
  --warmup N      計測前に捨てる予備実行の反復数 (既定 1、エンジンへは
                  --bench-repeat N として渡す)。VMのディスクキャッシュが冷えていると
                  同一構成でも2倍近く遅い値が出るため
  --cache-mib L   expert cache MiB。カンマ区切りで複数指定、"default" はエンジン既定
                  (既定 "default,8000")
  --ref PATH      参照トークンJSON。先頭をプロンプトに使い、生成結果の parity も検証する
                  (prefill + generate <= 参照トークン数 が必要。
                   --prefill 0 なら decode単体を ref[0:generate] と比較する)
  --seed N        合成プロンプトの乱数シード (既定 1)
  --vocab-limit N 合成プロンプトのトークンID上限 (既定 100000)
  --label NAME    出力ファイルの接尾辞 (既定 p<prefill>d<generate>)
  --out DIR       出力先 (既定 bench-out)
EOF
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefill) PREFILL=$2; shift 2 ;;
    --generate) GENERATE=$2; shift 2 ;;
    --ctx) CTX=$2; shift 2 ;;
    --repeats) REPEATS=$2; shift 2 ;;
    --warmup) WARMUP=$2; shift 2 ;;
    --cache-mib) CACHE_LIST=$2; shift 2 ;;
    --ref) REF=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --vocab-limit) VOCAB_LIMIT=$2; shift 2 ;;
    --label) LABEL=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    -h|--help) usage ;;
    -*) echo "不明なオプション: $1" >&2; usage ;;
    *) if [[ -z "$BIN" ]]; then BIN=$1
       elif [[ -z "$MODEL" ]]; then MODEL=$1
       else echo "不明な引数: $1" >&2; usage
       fi
       shift ;;
  esac
done

if [[ -z "$BIN" || -z "$MODEL" ]]; then usage; fi
if [[ ! -x "$BIN" ]]; then echo "エンジンが見つかりません: $BIN" >&2; exit 2; fi
if [[ ! -r "$MODEL" ]]; then echo "モデルが見つかりません: $MODEL" >&2; exit 2; fi
if (( PREFILL < 0 || GENERATE < 1 || REPEATS < 1 || WARMUP < 0 )); then
  echo "--prefill >= 0, --generate >= 1, --repeats >= 1, --warmup >= 0 が必要です" >&2
  exit 2
fi
if [[ -z "$CTX" ]]; then
  # エンジン側の下限 (--ctx > --generate + 2) も満たすようにする
  CTX=$(( PREFILL + GENERATE ))
  if (( CTX < GENERATE + 3 )); then CTX=$(( GENERATE + 3 )); fi
fi
if (( CTX < PREFILL + GENERATE )); then
  echo "--ctx は prefill + generate 以上が必要です" >&2
  exit 2
fi
LABEL=${LABEL:-p${PREFILL}d${GENERATE}}

mkdir -p "$OUT"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
PROMPT_FILE="$TMP/prompt.txt"
RECORDS="$OUT/phases-${LABEL}.records.jsonl"
: > "$RECORDS"

# --- プロンプト ---
if (( PREFILL == 0 )); then
  : > "$PROMPT_FILE"
elif [[ -n "$REF" ]]; then
  # 参照トークン列の先頭をプロンプトにすると、生成結果の期待値も決まる
  # (generated == ref[PREFILL : PREFILL+GENERATE])。
  python3 - "$REF" "$PREFILL" "$GENERATE" > "$PROMPT_FILE" <<'PY'
import json
import sys

ref = json.load(open(sys.argv[1]))
ids = ref["token_ids"] if isinstance(ref, dict) else ref
prefill, generate = int(sys.argv[2]), int(sys.argv[3])
if prefill + generate > len(ids):
    raise SystemExit("--ref が短すぎます: prefill + generate = %d > %d" % (prefill + generate, len(ids)))
print(" ".join(str(t) for t in [1] + list(ids[:prefill - 1])))
PY
else
  # 決定的な合成プロンプト。同じ --seed なら常に同じトークン列になる。
  python3 - "$PREFILL" "$SEED" "$VOCAB_LIMIT" > "$PROMPT_FILE" <<'PY'
import random
import sys

prefill, seed, limit = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
rng = random.Random(seed)
print(" ".join(str(rng.randrange(1, limit)) for _ in range(prefill)))
PY
fi
PROMPT_TOKENS=$(wc -w < "$PROMPT_FILE" | tr -d ' ')
PROMPT_SHA=$(sha256sum "$PROMPT_FILE" | cut -d' ' -f1)

echo "# engine   : $BIN"
echo "# model    : $MODEL"
echo "# ctx=$CTX prefill=$PREFILL generate=$GENERATE repeats=$REPEATS warmup=$WARMUP seed=$SEED"
echo "# prompt   : $PROMPT_TOKENS tokens sha256=$PROMPT_SHA${REF:+ (ref=$REF)}"
echo "# cache    : $CACHE_LIST"

printf '{"kind":"header","label":"%s","model":"%s","engine":"%s","ctx":%s,"prefill":%s,"generate":%s,"repeats":%s,"warmup":%s,"seed":%s,"vocab_limit":%s,"prompt_tokens":%s,"prompt_sha256":"%s","ref":"%s"}\n' \
  "$LABEL" "$MODEL" "$BIN" "$CTX" "$PREFILL" "$GENERATE" "$REPEATS" "$WARMUP" "$SEED" "$VOCAB_LIMIT" \
  "$PROMPT_TOKENS" "$PROMPT_SHA" "$REF" >> "$RECORDS"

# 1 つの構成を 1 プロセスで走らせ、wall 秒を返す。
run_engine() {
  local cache=$1 repeats=$2 raw=$3 log=$4
  local envargs=()
  if [[ "$cache" != "default" ]]; then envargs+=("WARODECODE_EXPERT_CACHE_MIB=$cache"); fi

  local cmd=("$BIN" --model "$MODEL" --ctx "$CTX" --device 0 --no-graph --no-autotune
             --start-token 1 --generate "$GENERATE" --bench-repeat "$repeats")
  if (( PREFILL > 0 )); then cmd+=(--prefill-tokens-file "$PROMPT_FILE"); fi

  # stdout は wall 秒の受け渡しに使うので、ログは stderr へ出す
  echo "# --- cache=$cache repeats=$repeats: ${cmd[*]}" >&2
  local started ended wall
  started=$(date +%s.%N)
  if ! env "${envargs[@]}" "${cmd[@]}" > "$raw" 2> >(tee -a "$log" >&2); then
    echo "エンジンが失敗しました (cache=$cache)。ログ: $log" >&2
    tail -n 20 "$log" >&2 || true
    exit 1
  fi
  ended=$(date +%s.%N)
  wall=$(awk -v a="$started" -v b="$ended" 'BEGIN { printf "%.2f", b - a }')
  printf '%s' "$wall"
}

run_config() {
  local cache=$1
  local raw="$TMP/raw-${cache}.json"
  local log="$TMP/log-${cache}.txt"

  # VM のページキャッシュが冷えていると同一構成でも 2 倍近く遅い値が出るので、
  # 計測前に捨て実行を挟む (WARMUP=0 で無効)。
  if (( WARMUP > 0 )); then
    echo "# warmup cache=$cache (捨て実行 ${WARMUP}回)"
    run_engine "$cache" "$WARMUP" "$TMP/warmup-${cache}.json" "$TMP/warmup-${cache}.log" >/dev/null
  fi

  local wall
  wall=$(run_engine "$cache" "$REPEATS" "$raw" "$log")

  local engine
  engine=$(grep -a '^{"engine":"warodecode-qwen38"' "$raw" | tail -n 1 || true)
  if [[ -z "$engine" ]]; then
    echo "engine JSONが見つかりません (cache=$cache)。ログ: $log" >&2
    tail -n 20 "$log" >&2 || true
    exit 1
  fi
  printf '{"kind":"run","config":"cache=%s","cache_mib":"%s","wall_seconds":%s,"engine":%s}\n' \
    "$cache" "$cache" "$wall" "$engine" >> "$RECORDS"
}

IFS=',' read -r -a CACHES <<< "$CACHE_LIST"
for cache in "${CACHES[@]}"; do
  [[ -z "$cache" ]] && continue
  run_config "$cache"
done

echo
python3 "$(dirname "$0")/phases_report.py" --records "$RECORDS" \
  --out "$OUT/phases-${LABEL}.json" ${REF:+--ref "$REF"}
echo "# 集計: $OUT/phases-${LABEL}.json"
