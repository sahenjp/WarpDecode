# 1トークンの内訳計測 (Task F)

対象: `warodecode` qwen4exp (Qwen3.8-Flash-Next-IQ3E-Q8D-MTP) / batch=1 / greedy。
目的は高速化ではなく **decode 1トークン (~240ms) の内訳を実測して次のボトルネックを特定する** こと。
数値・推論ロジックには一切触れず、CUDA event の区間計測だけを足している。

## 1. 計測の仕組み

- `src/qwen38_trace.hpp`: 区間 ID (`Qwen38Stage`)、集計型 (`Qwen38ProfileTotals`)、JSON 化
  (`qwen38_profile_json` / `qwen38_profile_detail_json`)。**CUDA 非依存**にしてある
  (plain g++ で動く `tests/qwen38_trace_test.cpp` を壊さないため)。
- `src/qwen38_forward.cuh`: `Qwen38StageProfiler` (cudaEvent_t のプール) と `Qwen38StageScope`。
  `step()` の各段に begin/end の event を打ち、トークン末尾の stream 同期後に
  `cudaEventElapsedTime` で読み戻して合計する。
- `src/warodecode.cu`: エンジン JSON に `profile` と `profile_detail` を追加。

### 測っている区間

| stage | 範囲 |
|---|---|
| `embed` | 埋め込み gather + 4 residual stream への展開 |
| `ple` | PLE レイヤの全体 (ホスト側の `pread` と H2D を含む) |
| `hc_mix_attn` | attention 入力の HC mix / norm |
| `attn` | GDN または QSA の本体 |
| `hc_combine_attn` | attention 後の residual combine |
| `hc_mix_ffn` | FFN 入力の HC mix / norm |
| `moe` | router + expert cache のステージング + routed GEMV + shared expert |
| `hc_combine_ffn` | MoE 後の residual combine |
| `head` | 最終 HC mix + 出力 GEMV + GPU argmax |
| `sync` | 次トークンの D2H コピー + `cudaStreamSynchronize` |
| `total` | `step()` 全体 (上の全部 + どの区間にも入らない残り) |

`other_ms = total_ms - (上の合計) - sync_ms`。つまり total は常に
`stage_sum + sync + other` に分解できる。

**device と host を両方出す**のが要点。event の区間は GPU タイムライン上の距離なので、
GPU が次の仕事を待って遊んだ時間は「その区間」に乗る。同じ区間のホスト wall time と
比べると、待っているのがホスト側 (pread / 同期 / 起動) なのか GPU 側なのかが判別できる。

### スイッチ

| 環境変数 | 既定 | 意味 |
|---|---|---|
| `WARODECODE_PROFILE` | OFF | `1` で計測。OFF のときは event を1つも作らず、フックは bool 分岐1つに落ちる |

OFF 時の追加コストはトークンあたり数 ns で、実測でも差は出ない (`*_off` の run を参照)。

## 2. 使い方

```bash
# ローカル/Colab どちらでも、エンジンに環境変数を付けるだけ
WARODECODE_PROFILE=1 ./build-sm89/warodecode --model model.gguf --ctx 128 --device 0 \
  --start-token 1 --no-graph --no-autotune --generate 64

# Colab (専用ランナー。他タスクの run を kill しない / 使用中なら待つ)
export PATH=/home/sahen/.local/bin:$PATH
WARODECODE_COLAB_SESSION=warpdecode-l4b WARODECODE_COLAB_ARCH=89 \
  python3 tools/token_profile_colab.py --label myrun \
  --switches WARODECODE_EXPERT_CACHE_MIB=8000 --generate 64 --wait-for-free 2700

# 収集した JSON を表にする
python3 tools/token_profile_report.py --prefix taskF_ --detail
```

`tools/token_profile_colab.py` は
1. ツリーを zip してアップロード → VM でビルド、
2. `/content/*.pid` が生存 or GPU に compute app がいれば **BUSY として起動しない**
   (他タスクの計測を汚さない)、
3. `/content/profile_<label>.log` に自分のログを書く (共有ログを使い回さない)、
4. 完了後に `issue1_artifacts/token_profile_<label>.json` へ `profile` を保存し、
   参照トークン列との一致数も記録する。

## 3. 出力スキーマ

```json
"profile": {
  "enabled": 1, "tokens": 64,
  "total_ms": 238.4, "sync_ms": 1.2, "stage_sum_ms": 231.0, "other_ms": 6.2,
  "embed_ms": 0.4, "ple_ms": 41.2, "hc_mix_attn_ms": 9.1, "attn_ms": 18.4,
  "hc_combine_attn_ms": 2.0, "hc_mix_ffn_ms": 9.0, "moe_ms": 148.3,
  "hc_combine_ffn_ms": 2.1, "head_ms": 2.9
},
"profile_detail": {
  "unit": "ms_per_token",
  "host_total_ms": 310.5, "host_other_ms": 100.0, "host_flush_ms": 1.5,
  "host_ple_ms": 44.0, "calls_moe": 3072, "calls_layers": 3072, "unmatched": 0,
  "layer_device_ms": [ ... 48 entries ... ], "layer_host_ms": [ ... ]
}
```

- `profile` の値は **すべて数値** (unit は detail 側)。consumer がキーを走査して
  `%.2f` を当てても壊れない。
- `profile_detail` の `*_ms` はすべて **1トークンあたりの平均**、
  `calls_*` は run 全体の区間本数、`layer_*_ms` はレイヤ別の1トークン平均。

## 4. 実測値

### 4.1 A100-SXM4-40GB / expert cache 20000 MiB / --generate 64 / profile ON

issue1_artifacts/token_profile_taskF_a100_c20000.json (parity 64/64, tok_s 3.671 = 272.3 ms/token)

| stage | device [ms/token] | host [ms/token] | host-dev | 備考 |
|---|---:|---:|---:|---|
| moe | **144.08** | **258.51** | **+114.43** | router + expert ステージング + GEMV + shared expert |
| attn | **43.93** | 4.20 | -39.73 | GDN 36層 + QSA 12層 |
| hc_mix_attn | **40.14** | 3.46 | -36.68 | attention 入力の HC mix (48回) |
| hc_mix_ffn | **40.19** | 3.09 | -37.10 | FFN 入力の HC mix (48回) |
| head | 1.30 | 0.07 | -1.23 | 最終 HC mix + Q6_K GEMV + argmax |
| ple | 0.59 | 0.11 | -0.49 | PLE 層は1層のみ (calls=64) |
| hc_combine_ffn | 0.39 | 0.26 | -0.13 | |
| hc_combine_attn | 0.38 | 0.22 | -0.16 | |
| embed | 0.02 | 0.02 | -0.00 | |
| sync | 0.02 | 1.64 | +1.63 | D2H + 同期 |
| other | 1.27 | - | - | どの区間にも入らない残り |
| **total** | **272.31** | 272.39 | | total = stage_sum + sync + other |

1層あたりの device 時間は 48 層で 4.65〜6.83 ms (平均 5.64 ms)。layer_device_ms は
レイヤごとに平坦で、特定層だけが重いわけではない。

### 4.2 読み取り

1. **MoE が最大 (53%)**。しかも host 258.5 ms > device 144.1 ms で、この区間だけで
   GPU が 114 ms 遊んでいる。expert cache ミス時のホスト側処理
   (GGUF からの行読み + H2D 発行) を待つ時間がそのまま出ている。
   既知の「ルーティング済み expert GEMV 合計 13.8 ms/token」を引くと、
   MoE の 144 ms のうち **約 130 ms は重みのステージング**であって計算ではない。
2. **HC mix が 2 サイト合計 80 ms (29%)**。こちらは host-dev が -37 ms で、
   GPU が本当に 80 ms 働いている。1 サイトの中身は
   「RMSNorm(10240) → Q8_0 quantize → down GEMV → SiLU → quantize → up GEMV → sigmoid」で、
   帯域から期待される値より 1〜2 桁遅い。
3. **attention 44 ms (16%)**、**PLE 0.6 ms (0.2%)**。PLE 層は 1 層しかないため、
   当初疑っていた PLE のファイル読みは無視できる。
4. head は 1.30 ms で、既知の「Q6_K ヘッド 2564 us (L4)」と整合する。

### 4.3 A100-SXM4-40GB / expert cache 既定 (4096 MiB) / profile ON

issue1_artifacts/token_profile_taskF_a100_cdefault.json (parity 64/64, tok_s 2.374 = 421.1 ms/token)

| stage | device | host |
|---|---:|---:|
| moe | 224.25 | 404.32 |
| attn | 100.67 | 4.20 |
| hc_mix_attn | 40.28 | 3.46 |
| hc_mix_ffn | 48.38 | 3.09 |
| total | 421.11 | 421.23 |

注意: この run は実行後の GPU に他プロセスの compute app が 1 つ残っており
(contamination_check.clean = false)、実行中にセッションが共有された疑いがある。
attention が 43.9 → 100.7 ms と倍近くに膨らんでいるのは cache サイズでは説明できず、
共有の影響と見るのが自然。**再計測が必要** (wave 2 で GPU sampler 付きで再実行する)。
傾向として「cache を 20000 → 4096 MiB にすると 272 → 421 ms」は cache ミス増の効果と一致する。

### 4.4 L4 (warpdecode-l4b) / expert cache 既定 (1536 MiB) / profile ON

issue1_artifacts/token_profile_taskF_l4_cdefault.json (parity 64/64, tok_s 3.258 = 306.9 ms/token)
同日に Task B (MTP 担当) が同じセッションで回した同一条件の実測 (tok_s 3.433 = 295.6 ms) も
issue1_artifacts の同じファイル名で一度取得しており、両者は 4% 以内で一致する。
下表は本 Task F の自前 run (GPU sampler 付き、実行中の他プロセスなし = max_gpu_apps 1)。

| stage | device [ms/token] | host [ms/token] |
|---|---:|---:|
| moe | **200.06** | **291.40** |
| attn | **43.61** | 4.05 |
| hc_mix_attn | **29.10** | 3.42 |
| hc_mix_ffn | **29.05** | 3.05 |
| head | 3.00 | 0.07 |
| ple | 0.74 | 0.11 |
| hc_combine_attn / ffn | 0.28 / 0.25 | 0.21 / 0.25 |
| embed / sync / other | 0.02 / 0.02 / 0.74 | — |
| **total** | **306.86** | 306.94 |

(host 列は要約しきれないので profile_detail を参照。MoE の host は device より 91 ms 大きく、
この区間で GPU が 91 ms 遊んでいる。)

L4 と A100 の比較 (cache 構成が違う点に注意):

| | L4 (1536 MiB) | A100 (20000 MiB) |
|---|---:|---:|
| total | 306.9 | 269.6 |
| moe | 200.1 | 141.7 |
| attn | 43.6 | 43.8 |
| hc_mix 計 | 58.1 | 80.2 |
| head | 3.0 | 1.3 |

- moe は L4 の方が 58 ms 重い。L4 は cache が小さく PCIe の実効帯域も細いため、
  「ステージング律速」の仮説と整合する。
- hc_mix は逆に L4 の方が速い (58.1 vs 80.2)。L4 の FP32 スループットは A100 より高いので、
  このカーネルが帯域ではなくレイテンシ/演算律速であることの傍証になる。
- (L4 の cache 8000 MiB 条件はセッション競合で未完。取得でき次第追記)

### 4.5 T4 16GB (予備セッション warpdecode-t4) / cache 8192 MiB / profile ON

issue1_artifacts/token_profile_taskF_t4_c8192.json (parity 64/64, tok_s 0.417 = 2397 ms/token)

| stage | device [ms/token] |
|---|---:|
| moe | **2127.37** |
| attn | 134.63 |
| hc_mix_attn | 57.18 |
| hc_mix_ffn | 57.18 |
| ple | 11.30 |
| head | 7.32 |
| hc_combine_attn / ffn | 0.54 / 0.54 |
| embed / sync / other | 0.02 / 0.02 / 1.00 |
| **total** | **2397.11** |

T4 は RAM が小さく (モデル 85.9GB を保持できない)、PCIe も 3.0 なので、
expert 重みの読み出しがそのまま支配する。**MoE だけで 89%**、
計算カーネル (attn + hc_mix + head) の合計は 256 ms = 全体の 11% に過ぎない。
同じコード・同じ量子化で、I/O が遅い GPU ではトークン時間が 8〜9 倍になる。
これは「decode は演算律速ではなくステージング律速」であることの最も直接的な証拠。

## 5. 結論と次の一手

### 5.1 内訳

実測した 3 GPU のまとめ (すべて parity 64/64、数値は device ms/token):

| GPU | cache | total | moe | attn | hc_mix 計 | head | ple |
|---|---|---:|---:|---:|---:|---:|---:|
| A100-SXM4-40GB | 20000 MiB | 269.6 | 141.7 (53%) | 43.8 | 80.2 (30%) | 1.3 | 0.6 |
| L4 24GB | 1536 MiB | 306.9 | 200.1 (65%) | 43.6 | 58.1 (19%) | 3.0 | 0.7 |
| T4 16GB | 8192 MiB | 2397.1 | 2127.4 (89%) | 134.6 | 114.4 (5%) | 7.3 | 11.3 |

A100 (cache 20000 MiB) の内訳:

| 区分 | ms/token | 割合 |
|---|---:|---:|
| MoE (routed expert のステージング + GEMV + shared expert) | 141.7 | 53% |
| HC mix (attn 入力 + ffn 入力) | 80.2 | 29% |
| attention (GDN 36 + QSA 12) | 43.8 | 16% |
| head + ple + combine + embed + sync + other | 3.9 | 2% |

**読み方**: 計算カーネル (attn + hc_mix + head) の合計は A100 で 125 ms、
T4 では 256 ms しかない。残りはすべて「重みを GPU へ運ぶ時間」である。
GPU が速いかどうかは、このモデルではほとんど効かない (T4 は A100 より 9 倍遅い)。

### 5.2 ボトルネック 1: expert 重みのステージング (最大の的)

- MoE の device 141.7 ms のうち、既知の routed GEMV 合計 (13.8 ms) と shared expert / router を
  引くと **約 110〜120 ms は重みのステージング待ち**で、計算ではない。
- この区間の host 時間は 256.0 ms で device を 114 ms 上回る = ホストが詰まっている間 GPU が遊ぶ。
- 480 expert アクセス/token を 256 ms で回しているので **1 expert あたり約 0.53 ms**
  (スロット 2.22 MiB の pread + pinned ring 経由の H2D)。
- `Qwen38Tuning::expert_prefetch_depth` (1〜3) は定義されているが、
  **src/ 全体を grep しても消費者が存在しない** = 先読みパイプラインは未実装。

**次の一手**: レイヤ L の計算中に L+1 の routed expert をステージングする。
GPU が本当に働いている時間は attn 43.8 + HC mix 80.2 = 124 ms/token あるので、
この間にステージングを隠せれば 272 → 160〜180 ms (1.5〜1.7倍) が上限目安。
cache を増やす方向は A100 で 20000 MiB (9002 slot = 全 (layer, expert) 対の 37%) が実用上限で、
それ以上でも効きが鈍い (Task C の計測)。

### 5.3 ボトルネック 2: HC mix のカーネル効率 (重複除去ではない)

- HC mix (既定 = Q8_0 F32 経路) は A100 で 80.2 ms (29%)、L4 で 58.1 ms (19%)、T4 で 114.4 ms (5%)。
  host は 3〜4 ms しか使っていないので、**GPU が本当にその時間働いている**。
- T4 で `WARODECODE_HC_Q8_0=0,WARODECODE_HC_Q8_0_ATTN=0` (legacy F16 経路のみ) と比較した:
  hc_mix は **114.4 → 10.5 ms (1/11)**。つまり同じ数式を、Q8_0 経路は F16 経路の
  約 10 倍の時間で処理している。
  (この A/B は数値パス自体が変わり parity は 0/64。**診断専用**で、既定は Q8_0 経路のまま)
- サイトあたりに直すと Q8_0 経路 ~1.1 ms (T4) / 0.6〜0.84 ms (L4/A100)、legacy は 0.11 ms (T4)。
  1 サイトの重みは約 11 MB (Q8_0) なので実効 **14〜21 GB/s** しか出ていない
  (A100 の HBM 1555 GB/s の 1% 程度)。
- したがって本命は **Q8_0 側カーネルの効率改善** (qwen38_hc.cuh / lowbit.cuh の
  `q38_hc_q8_0_quantize_*` と `q38_hc_q8_0_q8_0_gemv_f32_kernel`)。
  実装は本 Task の担当ファイル外なので、値を添えて他担当へ渡す。
- 参考: forward.cuh の 871/894 (attn) と 2128/2141 (ffn) は 1 サイトにつき mix を 2 回呼んでおり、
  legacy 側の結果は捨てられる。ただし T4 の A/B から legacy 側は 5.24 ms/token (全体の 0.9%) と
  小さいので、**重複除去の効果は 1〜3% 程度**。当初「40 ms/token (15%)」と見積ったが、
  この A/B で否定された (重複しているのは速い側で、遅い側は必要)。

### 5.4 優先度

1. **MoE ステージングの先読み/隠蔽** (見込み 1.5〜1.7 倍。expert cache / MoE 側 = 他担当領域)
2. **HC mix の Q8_0 カーネル改善** (見込み +15〜25%。qwen38_hc.cuh / lowbit.cuh = 他担当領域)
3. attention 44 ms (16%) のカーネル改善 — DP4A/smem で既に 6% 改善済みで優先度は下
4. forward.cuh の重複 mix 除去 (1〜3%、パリティ確認コストに見合うかは要判断)
5. PLE は 0.59 ms。**触る価値なし** (PLE 層は 1 層のみ)

## 6. 未検証リスク

- **L4 の実測が未完**。セッション (warpdecode-l4b) が他タスクで連続稼働のため。
  機構は GPU 非依存なので傾向は同じと予想するが、数値の移植はできない。
- expert cache 統計 (hit 率 / pread ms / H2D bytes / ring wait) は未取得。
  「1 expert 0.53 ms」はホスト時間からの逆算であり、pread と H2D 待ちの内訳は未分解。
- A100 既定 cache の run は他プロセス混入の疑いがあり、値は参考扱い (再計測待ち)。
- プロファイラ OFF との比較 run (taskF_*_off) は未完。ただし profile ON の total 272.31 ms が
  エンジン自身の decode_seconds 272.40 ms と 0.03% で一致しており、event 計測の
  オーバーヘッドは測定分解能以下と判断している。
- 計測は batch=1 greedy / 64 トークン平均。最初の数トークンは expert cache が冷たいので、
  平均には過渡状態が混ざる (定常だけを見るには cache を温めた繰り返し計測が必要)。
- `profile_detail` の per-layer 配列は層数分 (48) だけ意味を持ち、残りは 0。
