# Prefill / Decode 分離ベンチマーク (Task A)

対象: `warodecode` qwen4exp (Qwen3.8-Flash-Next-IQ3E-Q8D-MTP, IQ3E-Q8D-MTP) / batch=1 / greedy。
Prefill (プロンプト処理) と decode (1トークン生成) を**同一プロセス内で分離計測**し、
改善の前後比較に使えるベースラインを固定する。計測値はエンジンが CUDA event で取るので、
GGUF の mmap・モデルロード・autotune・CUDA 初期化は数値に含まれない。

## 1. 追加したもの

| 追加 / 変更 | 内容 |
|---|---|
| `bench/phases.sh` (新規) | prefill + decode を同一プロセスで測るドライバ |
| `bench/phases_report.py` (新規) | 生レコードを表と JSON に集計する |
| `tools/colab_run_cmd.py` (新規) | Colab VM 上で任意コマンドを回すランナー (build + 実行 + ログ回収) |
| `src/warodecode.cu` | `--prefill-tokens` / `--prefill-tokens-file` / `--bench-repeat` と JSON 4項目 |
| `src/qwen38_forward.cuh` | `Qwen38RunTiming` と prefill フェーズ (**prefill 指定時のみ通る分岐**) |
| `tools/colab_run_bg.py` | `WARODECODE_SOURCE_DIR` で凍結ツリーを送れるようにした (既定は従来どおり) |

推論ロジックは変更していない。prefill 分岐は `--prefill-*` を渡したときだけ通り、
**prefill なしの decode 経路は既存コードそのまま**(else 側に残してある)。

## 2. 使い方

```bash
bash bench/phases.sh ./build-sm89/warodecode /content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf \
  --prefill 64 --generate 64 --repeats 3 --cache-mib default,8000
```

| オプション | 既定 | 意味 |
|---|---|---|
| `--prefill N` | 64 | プロンプト長。0 で prefill なし (decode 単体) |
| `--generate N` | 64 | 生成トークン数 |
| `--ctx N` | prefill + generate (decode 単体なら generate+3) | コンテキスト長 |
| `--repeats N` | 3 | 同一プロセス内の反復回数。エンジンが最速の回を報告する |
| `--warmup N` | 1 | 計測前に捨てる予備実行の反復数 (エンジンへ `--bench-repeat N` として渡す) |
| `--cache-mib L` | default,8000 | `WARODECODE_EXPERT_CACHE_MIB`。`default` はエンジン既定 |
| `--ref PATH` | - | 参照トークンJSON。先頭をプロンプトに使い、parity も検証する |
| `--seed N` | 1 | 合成プロンプトの乱数シード (決定的) |

出力は `bench-out/phases-<label>.records.jsonl` (生レコード) と
`bench-out/phases-<label>.json` (集計 + markdown 表)。exit コードは全構成の成功で 0。

Colab (L4) 経由:

```bash
export PATH=/home/sahen/.local/bin:$PATH
export WARODECODE_SOURCE_DIR=/tmp/q38-task-a   # 凍結ツリー (任意)
python3 tools/colab_run_cmd.py --label task_a_matrix --source-dir /tmp/q38-task-a \
  --cmd "bash bench/phases.sh build-sm89/warodecode /content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf \
         --prefill 64 --generate 64 --repeats 3 --cache-mib default,8000" \
  --upload issue1_artifacts/reference_tokens_64.json:/content/reference_tokens_64.json \
  --download /content/warodecode/bench-out/phases-p64d64.json:issue1_artifacts/phases-p64d64.json
```

## 3. 計測の定義

エンジン JSON に追加した 4 項目:

| フィールド | 意味 |
|---|---|
| `prefill_tokens` | プロンプトとして forward したトークン数 N (未指定なら 0) |
| `prefill_seconds` | その N forward の実時間 (最後の forward のサンプリング含む) |
| `decode_tokens` | 生成トークン数 M (= `--generate`) |
| `decode_seconds` | その M forward の実時間 |

* prefill は**1トークンずつ** forward する (このエンジンにバッチ prefill はまだ無い)。
  N 個の forward が positions 0..N-1 を進み、最後の位置でサンプルしたトークンが
  最初の decode forward の入力になる。decode フェーズは
  「そのサンプル → 生成トークン」を M 回 forward し、**M forward で M トークン**を出す。
* 1 回の `--prefill N --generate M` は N + M forward を行う (`--prefill 0` は M forward)。
* `--bench-repeat R` は同じモデルを載せたまま prefill+decode を R 回繰り返し、
  **最速の回**を報告する。`token_ids` / `top_logits` / trace は 1 回目の値
  (greedy なので各回で同一)。expert cache は 2 回目以降温まる。
* `wall_seconds` (プロセス起動〜終了) も記録するが、ロード込みなので性能値には使わない。

### parity との関係

`--ref` を渡すと参照トークン列の先頭をプロンプトに使う。参照列 `ref` に対して
`--prefill N --generate M` の期待出力は `ref[N : N+M]` になる
(prefill が positions 0..N-1 を進めるので最初の decode は ref[N] を出す)。
`--prefill 0` なら期待値は `ref[0:M]` = 既存の 64/64 と同条件。

## 4. 実測 (Colab L4 / SM89, CUDA 13.0)

測定条件:
- エンジン: コミット `85405c9` + Task A 差分のみの凍結ツリー
  (worktree `/tmp/q38-task-a`; 他タスクの未コミット変更を含まない)
- モデル: Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf (85.9GB), ctx=128 (参照32の行のみ ctx=64)
- 数値は 3 反復の最速値。probe は合成プロンプト (seed=1, id<100000)、
  参照32の行だけ参照トークン列の先頭 32 個を使用
- 1 パス目は VM のディスクキャッシュが冷えており遅い値が出るため、
  **下表は 2 パス目 (=温まった状態) の値**。1 パス目の値は §5 に示す
- 計測に使った `bench/phases.sh` は warmup 無しで実行し、直前のパスで温めた
  (`--warmup` はその後の改善なので、下表の再現は「同じ構成リストを 2 回回して 2 パス目を採る」で行う)

| 構成 | expert cache | prefill tok/s | prefill ms/tok | decode tok/s | decode ms/tok | 合計 | parity |
|---|---|---|---|---|---|---|---|
| decode 64 (prefill なし) | 既定 (1536) | - | - | 3.305 | 302.6 | 19.36 s | **64/64** |
| decode 64 (prefill なし) | 8000 | - | - | **4.107** | 243.5 | 15.58 s | **64/64** |
| **prefill 64 → decode 64** | 既定 (1536) | 3.410 | 293.3 | 3.250 | 307.7 | 38.46 s | - |
| **prefill 64 → decode 64** | 8000 | **4.626** | 216.2 | **4.214** | 237.3 | 29.02 s | - |
| prefill 32 → decode 32 (参照トークン) | 8000 | 4.022 | 248.6 | 4.176 | 239.5 | 15.62 s | **32/32** |

読み取れること:

1. **expert cache 8000 MiB は既定 (1536 MiB) より 1.24〜1.36 倍速い**。
   decode 単体 3.305 → 4.107 / prefill 3.410 → 4.626 / decode フェーズ 3.250 → 4.214。
   L4 24GB ではこれ以上増やすと VRAM 不足で起動しない (§33.1 の 18000 MiB 失敗と同じ)。
2. prefill と decode は**同じ forward でも速度が違う** (4.626 vs 4.214)。decode は
   positions 64-127 を回るため文脈長のぶんだけ重い。分離計測しないと見えない差。
3. decode 単体 (従来の「64トークン生成」) の 8000 MiB 値は **4.107 tok/s** で、
   `docs/ISSUE1_FINDINGS_20261012.md` §33.1 の 3.893 tok/s、既定 3.305 は同 3.195 tok/s と
   同じ水準 (VM・ディスク状態の差で ±5% 程度)。計測系は従来値と整合している。

## 5. 落とし穴: 初回パスは 2 倍近く遅い

同じ凍結ソースで「1 パス目 (直前まで別構成を回していない状態)」を測ると:

| 構成 | expert cache | 1 パス目 | 2 パス目 |
|---|---|---|---|
| decode 64 | 既定 | 2.233 tok/s (28.67 s) | **3.305 tok/s** (19.36 s) |
| decode 64 | 8000 | 2.881 tok/s (22.22 s) | **4.107 tok/s** (15.58 s) |
| prefill 64 → decode 64 | 既定 | 3.406 / 3.249 | 3.410 / 3.250 |
| prefill 64 → decode 64 | 8000 | 4.629 / 4.216 | 4.626 / 4.214 |

* 実行順の後ろにある構成ほど速い = 86GB のモデルを読むうちに VM の
  ページキャッシュが温まるため。1 パス目の先頭 2 構成だけが大きく遅れ、
  3 構成目以降は 1 パス目でも再現している (p64d64 の 2 パス一致は 0.2% 以内)。
* 対策として `bench/phases.sh` は既定で**捨て実行 1 回**(`--warmup 1` = 計測前に
  `--bench-repeat 1` のプロセスを 1 つ余分に走らせる)を挟む。
  厳密に比較したいときは同じ構成リストを 2 回回して 2 パス目を採用するのが確実。
* さらに、複数エージェントが同じ Colab セッションを共有すると
  `/content/run_bg.log` が上書きされ、計測そのものが壊れる (実際に 1 回衝突した)。
  1 セッション 1 実行を守ること。

## 6. parity の確認 (実機)

| 検証 | コマンド | 結果 |
|---|---|---|
| 正規ランナー (凍結ツリー) | `tools/colab_run_bg.py --label task_a_parity_frozen --generate 64` | **64/64** (`colab_run_task_a_parity_frozen.json`) |
| 正規ランナー (共有ツリー, 17:08時点) | `tools/colab_run_bg.py --label task_a_parity_l4 --generate 64` | **64/64** (`colab_run_task_a_parity_l4.json`) |
| decode 単体 in-bench | `--prefill 0 --generate 64 --ref` | **64/64** (5 回すべて) |
| prefill 経路 in-bench | `--prefill 32 --generate 32 --ref` | **32/32** |

`--prefill 32 --generate 32 --ref` は「参照トークン列の先頭 32 個を prefill し、
続く 32 トークンを decode」して `ref[32:64]` と一致することを見ている。
prefill 分岐を足しても greedy 出力が CPU オラクルと一致したままであることが確認できる。

## 7. 未検証リスク

1. **A100 では未計測**。L4 のみ。A100 は既定 cache だとディスク律速で 1.08 tok/s (§33.1) なので、
   同じ表を A100 で取る場合は `--cache-mib 30000` などで回す必要がある。
2. **prefill はバッチ化されていない**。数字は「1トークンずつ forward する現行実装」の実測で、
   バッチ prefill を実装したときの上限値ではない。
3. **MTP (投機実行) は未実装**なので、その効果は含まれない。
4. **VM の状態依存**: 上記のとおり初回パスは最大 2 倍遅い。GPU クロック・電力状態は統制していない。
   同条件で比較するには凍結ツリー + 2 パス + 同一セッションを使う。
5. **per-repetition の生値は JSON に出していない** (`--bench-repeat` は最速値のみ、
   token_ids は 1 回目)。反復のばらつきを直接見たい場合は同じ構成を複数回実行して比較する。
6. `--top-k` と prefill の併用は「prefill 後の最初の decode forward の logits」を返す
   (位置の定義は prefill なしの場合と揃えてある)。未検証。
7. `--state-trace` / `--forced-input-tokens` と prefill は併用できない (エンジンが明示的に拒否)。
   prefill 経路の内部状態トレースは未対応。
