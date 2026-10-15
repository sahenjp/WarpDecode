# Task Q: MoE expert 重み転送の分解と改善（A100 20000MiB）

対象: `src/expert_cache.cuh`（pread + pinned ring + DMA のステージング経路）。
parity は llama.cpp CPU greedy オラクルとの 64/64 を全構成で確認している（第 5 章）。

---

## 0. 結論

1. **「MoE 区間の大部分は expert 重み転送」という前提は誤りだった。**
   A100 20000MiB の 1 トークンで MoE 区間は 95〜97 ms（新しいホストでは 89.8 ms）だが、
   そのうち **expert 重み転送は 14.6 ms（15%）だけ**である。残りは
   **expert カーネル実行 64.5 ms（67%）と、expert ループの launch 16.3 ms（17%）**。
   （内訳の差分測定は grid 既定 OFF のツリーで実施。転送コストは grid の有無に
   依存しないので、この比は現行ツリーでも成り立つ。第 5.3 節参照。）

2. **「実効 5.4 GB/s（ベンチの 1/3）」も計測ミスだった。**
   543 MB/token は **1536MiB キャッシュの実測値**（34.69 GiB / 64 token）であり、
   それを **20000MiB の MoE 区間 100 ms** で割った値。20000MiB の実転送量は
   **220 MiB/token（231 MB）**で、その速度は
   **DMA 11.2 GB/s / pread 19.4 GB/s**。
   ベンチ（`pin+h2d` 12.1 GB/s、`pinpread` 17.8 GiB/s）との差は **8% 以内**で、
   エンジンはすでにリンク上限に達している。**潰すべき差は存在しなかった。**

3. **ミスは全て compulsory miss。** 20000MiB では
   `misses == distinct_pairs == 7448`、`evictions == 0`（9002 スロット）。
   LFU 影シミュレーションでも 9002/12000/16000 スロットすべて
   ヒット率 0.7576 で完全に一致した。**置換ポリシー・容量・リング本数を
   どう変えても転送バイト数は 1 バイトも減らない。**

4. **I/O を完全に消しても MoE は 80.8 ms 残る**（pread も DMA も無し）。
   launch を 60→5/層に減らした併用でも **64.5 ms**。
   **目標の 60 ms は転送側の仕事だけでは到達不可能**で、律速は
   expert GEMV カーネル実行と launch にある。
   現行既定（grid ON）の実測でも MoE は 89.8 ms で、その下限は 64.5 ms である。

5. 実装した改善は **マルチコピーストリーム DMA**（`WARODECODE_EXPERT_IO_STREAMS`、
   既定 1 = 従来と同一）。副産物として、この A/B が
   **ストリーム追加時の consumer wait 漏れ（parity 破壊）**を検出したので、
   parity ガードとして残している。

6. 転送以外で最大のレバーは既存スイッチ `WARODECODE_MOE_FUSED_GRID`
   （expert 単位バッチ grid、60→5 launch/層、bit 一致）で、**+5.2%（MoE −6.3 ms）**。
   **この既定は Task R が `true` に変更済み**で、現ツリーはすでにこの利得を含む
   （第 5.3 節）。

7. 実装した改善（`WARODECODE_EXPERT_IO_STREAMS`）は **parity 64/64 を保つが
   速度効果はゼロ**だった。理由はベンチで説明できる（H2D リンクが 12.2 GB/s で
   飽和しており、並列化しても上限は動かない）。既定 1 のまま。

---

## 1. 計測方法

専用ランナー `tools/colab_run_cmd_q.py`（作業ディレクトリを `/content/wd_q` に隔離）で
`tests/moe_transfer_ab.py` を A100 上で実行する。1 ケース = 1 プロセスで、
全ケース共通のコマンド:

    warodecode --model <gguf> --ctx 128 --device 0 --start-token 1 \
               --no-graph --no-autotune --generate 64

各ケースは自分の 64 トークンを出力し、`issue1_artifacts/reference_tokens_64.json`
と突き合わせて 64/64 を判定する。計測は 3 系統:

| 系統 | スイッチ | 内容 |
|---|---|---|
| 区間内訳 (device) | `WARODECODE_PROFILE=1` | ステージごとの CUDA event（Task F） |
| キャッシュ統計 | `WARODECODE_EXPERT_STATS=1` | hit/miss/bytes、pread ms、gap |
| コピー経路 (新規) | `WARODECODE_EXPERT_IO_TIMING=1` | DMA 本体と consumer 待ちの event |

20000MiB の計測は 3 ラウンド（q1/q2/q3）で再現性を確認した。`plain`（計測なし）は
7.604 / 7.624 / 7.601 tok/s と 0.3% 以内で一致する。

---

## 2. 現状の分解（A100 20000MiB、64 token greedy）

以下 2.1〜2.4 の差分測定は、`WARODECODE_MOE_FUSED_GRID` がまだ既定 OFF だった
ツリー（08:2x〜08:39 のラウンド）で取ったもの。現行ツリーは grid が既定 ON で
MoE 区間が 89.8 ms（第 5.3 節）だが、**転送コストは grid の有無に依存しない**ので
2.2 の「転送 14.6 ms / 15%」は現行ツリーでもそのまま成り立つ。

### 2.1 区間内訳

| 区間 | ms/token | 割合 |
|---|---|---|
| hc_mix_attn | 10.0 | 7.6% |
| attn | 14.7 | 11.2% |
| hc_mix_ffn | 10.0 | 7.6% |
| **MoE** | **95.4〜97.1** | **72.6%** |
| ple / head / hc_combine / other | 3.0 | 2.3% |
| 合計 | 131.5 | 100% |

`plain`（計測なし）の tok_s 7.601〜7.624 = 131.5 ms/token と一致する。

### 2.2 MoE 区間の分解（差分法）

| 構成 | MoE ms | 差 | 意味 |
|---|---|---|---|
| `base`（通常） | 95.4 | — | |
| `WARODECODE_EXPERT_IO_SKIP=1`（pread も DMA も無し） | 80.8 | **−14.6** | **expert 転送の全コスト** |
| `WARODECODE_EXPERT_IO_NODMA=1`（pread のみ） | 82.5 | −12.9 | DMA と consumer 待ちのコスト |
| `SKIP + MOE_FUSED_GRID=1`（5 launch/層） | 64.5 | −30.9 | **expert 計算＋launch の下限** |

| 構成要素 | ms/token | MoE 比 | 出典 |
|---|---|---|---|
| expert カーネル実行（バッチ grid、I/O 無し） | 64.5 | 67% | `skip-grid` |
| expert ループの追加 launch（60→5/層） | 16.3 | 17% | `skip-fill` − `skip-grid` |
| expert 重み転送（pread + DMA + 待ち） | 14.6 | 15% | `base` − `skip-fill` |
| **合計** | **95.4** | 100% | |

> 注: `SKIP`/`NODMA` はスロット内容が変わるため routing が発散する
> （`misses` 1917 vs 7448）。したがって「転送を消したときの 80.8 ms」は
> 同じ routing の反実仮想ではない。ただしバイト数で独立に検算できる:
> 1536MiB と 20000MiB の差分から
> **marginal 0.137 ms/miss → 20000MiB の 116.4 miss/token = 15.9 ms** となり、
> 差分法の 14.6 ms と 2 桁目の範囲で一致する。

### 2.3 転送そのものの内訳（per token）

| 量 | ms/token | 実効速度 | ベンチ上限 |
|---|---|---|---|
| pread（4 スレッド → pinned ring） | 11.7〜12.0 | **19.4 GB/s** | 17.8 GiB/s = 19.1 GB/s |
| DMA（copy stream が busy だった時間） | 20.5 | **11.2 GB/s** | `pin+h2d` 12.1 GB/s |
| うち 1 件あたり | — | 177 µs / 1.98 MB | 173 µs / 2 MiB |
| consumer stream の DMA 待ち | 12.0 | — | — |
| ring 再利用ガードの待ち | 0.27〜0.69 | — | — |

**1 バイトあたりの速度はベンチと同等。** 残っているのは「20.5 ms の DMA のうち
12.0 ms が隠しきれずに露出している」という**重なり不足だけ**である。

### 2.4 転送量の実測（キャッシュ容量別）

| キャッシュ | slots | hit率 | misses/token | 実転送量/token | MoE ms | stage ms |
|---|---|---|---|---|---|---|
| 1536 MiB | 691 | 0.3878 | 293.8 | **555 MiB（582 MB）** | 120.9 | 34.5 |
| 20000 MiB | 9002 | 0.7576 | 116.4 | **220 MiB（231 MB）** | 95.4 | 13.6 |

- 1 expert = 1,980,824 B（gate_up + down、slot は 2,329,600 B でギャップ込み）。
- **タスクにあった「543 MB/token」は 1536MiB の値**（34.69 GiB / 64）。
  20000MiB では 231 MB であり、それを MoE 区間で割った 2.4 GB/s は
  「MoE 区間の 85% は転送以外」という事実を無視した比である。

---

## 3. ベンチ（14〜17 GiB/s）と実エンジンの差の正体

**差は無い。** 内訳は次のとおり。

1. **DMA は 11.2 GB/s。** 今回 VM 上で `tests/expert_io_bandwidth.cu` を
   チャンクサイズ 1/2/4/8 MiB で回したところ `pin+h2d` は
   **11.82 / 12.10 / 12.23 / 12.29 GB/s** で、**サイズに依存しない**。
   つまりこの VM の H2D リンク上限が約 12.2 GB/s であり、
   エンジンの 11.2 GB/s は**上限の 92%**。
   「チャンクを大きくする」「複数ストリームで並列化する」余地はリンク側に無い。
2. **pread は 19.4 GB/s**（4 スレッド、ページキャッシュ常駐時）。
   これは Task H が測った 4 スレッド上限 17.8 GiB/s = 19.1 GB/s と同等。
3. 露出している 12 ms は「DMA 完了を待つ consumer stream の待ち」であり、
   帯域ではなく**重なりの不足**。expert のカーネルは 40 µs 程度、DMA は 177 µs なので、
   1 expert 先読みでは DMA を隠しきれない。expert 列は層の MoE 直前にしか
   確定しない（router の D2H + sync が層ごとに 1 回）ため、これ以上先に
   発行することはできない。

---

## 4. 実装

### 4.1 マルチコピーストリーム DMA（`WARODECODE_EXPERT_IO_STREAMS`）

1 本の DMA では H2D リンクを飽和できない可能性を潰すため、per-expert DMA を
N 本の copy stream に round robin する。既定 1（= 従来と完全に同一の経路）。

- 各 expert の DMA は自分のストリームで発行し、そのストリームに
  `done_` を record してから consumer stream が待つ。expert ごとの
  「スロットが埋まってからカーネル」の順序は変わらない。
- ring entry の再利用ガード（`entry.done`）もその DMA を発行したストリームに
  record するので、entry ごとの安全性は保たれる。
- `io_streams` は `WARODECODE_EXPERT_STATS=1` の JSON に出る。

### 4.2 A/B が検出したバグ（parity ガード）

最初の実装は「submit が copy stream に何か積んだか」を
`s == copy_stream_` で判定していた。ストリームを増やすと 2 本目以降が
false になり、`finish()` が早期 return して **consumer がその DMA を待たない**。
実測では st2/st4/st8 が **0/64**、instrumented 版では miss 数まで
7448 → 6831 に変わって routing が発散していた。
`s != consumer` に修正して 64/64 に戻している（第 5 章）。
このバグは「速度は出るが値が違う」型なので、parity を取らない A/B では
そのまま採用してしまう危険があった。

### 4.3 採用しなかった案と、その実測

| 案 | 実測 | 判定 |
|---|---|---|
| ring を 4 → 16 本 | stage 13.6 → 19.8 ms、MoE 95.4 → 100.6、tok_s 7.601 → 7.163 | **負け**（既定 4 のまま） |
| pread スレッドを 4 → 8 | `plain` 7.601 vs 7.592（計測なし同士の A/B） | 有意差なし |
| retire event を eviction 時だけに | 1536MiB で MoE 120.9 → 135.5 ms、tok_s 6.27 → 5.73 | **負け**（差し戻し済み） |
| 置換ポリシー（LFU 化） | LFU 影でもヒット率 0.7576（LRU と同一） | 全て compulsory miss なので無効 |
| 容量を増やす | 影シミュレーションで 12000/16000 スロットも同一 | 同上 |

retire の件は、eviction 時に record すると「consumer stream が現在位置に
追いつくまで DMA を開始できない」直列化を生むためで、
1536MiB のような高 churn 構成で 8.6% の退行になる。
20000MiB では eviction が 0 回なので record 自体が消えるが、
その効果は `plain` 7.604 → 7.601（誤差）で、**利得は無かった**。

---

## 5. A/B と parity

### 5.1 最終ツリーでの確認（A100 20000MiB、計測なし、64 token greedy）

| case | tok_s | parity |
|---|---|---|
| `plain`（既定） | 7.963 | **64/64** |
| `WARODECODE_MOE_FUSED_GRID=1` | 7.983 | **64/64** |
| `WARODECODE_EXPERT_IO_STREAMS=2` | 7.955 | **64/64** |
| `WARODECODE_EXPERT_IO_STREAMS=4` | 7.963 | **64/64** |
| `WARODECODE_EXPERT_IO_STREAMS=8` | 7.846 | **64/64** |

計測あり（`PROFILE+STATS+IO_TIMING`）での対比:

| case | tok_s | moe_ms | stage/tok | pread/tok | dma/tok | stall/tok | misses |
|---|---|---|---|---|---|---|---|
| `diag` (streams=1) | 7.780 | 89.8 | 14.2 | 11.2 | 20.0 | 18.9 | 7448 |
| `diag-st4` | 7.729 | 90.3 | 14.5 | 11.2 | 20.1 | 19.1 | 7448 |

**マルチストリームは効果なし**（stall は 18.9 → 19.1 ms と同一、misses も同一で
routing が変わっていないことまで確認）。ベンチの「`pin+h2d` は 1〜8 MiB で
11.8〜12.3 GB/s とサイズ非依存」＝リンク飽和という結論と整合する。

### 5.2 マルチストリームの parity（バグ修正の検証）

| 実行 | 構成 | parity |
|---|---|---|
| 修正前 | st2 / st4 / st8 / st4-c1536 | **0/64**（consumer が DMA を待たない） |
| 修正後 | st2 / st4 / st8 (c20000) | **64/64** |
| 修正後 | st4-c1536（高 churn、eviction 多用） | **64/64** |
| 修正後 | diag-c1536（streams=1） | **64/64** |

修正後の instrumented 実行で `misses` が修正前の 6831 から **7448**（＝ブロックなし実行と同値）
に戻っていることが、routing が正しくなった直接の証拠。

### 5.3 grid（60→5 launch/層）の効果と、q5 の読み替え

| ラウンド | ツリーの既定 | plain | grid=1 | 差 | MoE (diag) |
|---|---|---|---|---|---|
| q1/q2 (08:2x) | OFF | 7.34〜7.39 | 7.79 | +5.4% | 95.4 → 89.7 |
| q3 (08:39) | OFF | 7.601 | 7.998 | **+5.22%** | 96.1 → — |
| q5 (17:55) | **ON** | 7.963 | 7.983 | +0.25% | 89.8 |

**q5 の `plain` と `grid` は同じ構成だった。** 計測に使ったソースを VM 上で
確認したところ、q5 のビルドにはすでに Task R の既定変更
（`q38_moe_env_flag("WARODECODE_MOE_FUSED_GRID", true)`）が入っていた。
したがって q5 の ±0.25% は A/B ではなく同条件の再現性チェックであり、
**「ホスト依存で効果が消えた」のではない**。

むしろ 3 ラウンドが整合する: grid を OFF にしたツリー（旧ホスト）で
7.601、ON にしたツリーで 7.963〜7.998。**既定変更そのものが +4.8%** で、
ホスト差は 0.5% 以内。MoE 区間も 96.1 → 89.8 ms（−6.5%）と、q2 の
`dp4a-grid` の −5.7 ms と一致する。

**つまり Task Q が実装すべき最大の改善は、既に Task R が既定変更として
入れており、parity 64/64 も本タスクの A/B で確認済み**である。

---

## 6. 未検証リスク

1. **`SKIP`/`NODMA` は routing を変える。** スロット内容が変わるため
   `misses` は 7448 → 1917 になる。「I/O を消したときの MoE 80.8 ms」は
   同じ routing の反実仮想ではない。バイト数からの独立検算（2.2 節の注）で
   14.6 ms と整合することを確認しているが、厳密な counterfactual ではない。
2. **compulsory miss という結論は 64 token / 9002 スロットに固有。**
   より長い実行では作業集合がキャッシュを超えて eviction が始まり、
   置換ポリシーが意味を持つようになる。今回測ったのは decode 64 token の範囲。
3. **H2D リンク上限 12.2 GB/s は VM 依存。** 別ホスト・別リンク幅なら
   DMA の上限も変わる（今回の A100 VM は 1〜8 MiB でフラットだったので
   リンク律速と判断した）。
4. **grid の既定変更は未実施。** `src/qwen38_moe.cuh` は担当外なので
   推奨に留めた。+0.25〜+5.2% の幅は実機で確認が必要。
5. **計測自体のオーバーヘッド。** `PROFILE`+`STATS`+`IO_TIMING` は
   同時に付けると tok_s が 2〜4% 落ちる（例: plain 7.963 → diag 7.780）。
   本章の絶対値は「計測あり」の値であり、相対比較にのみ使っている。
6. **q4 ラウンドは Task R（同じ MoE 転送を担当する別エージェント）の
   ジョブと競合した。** 5〜7 番目のケースが
   `cudaMalloc expert cache failed: out of memory` で落ちており、
   そのラウンドの所要時間は汚染されている（parity の判定だけを採用）。
   q5 は 4 ケースのみの短い窓で取り直したもの。
7. **A100 のみ。** L4 など VRAM の小さい構成では、eviction が常態なので
   「転送 15%」の比率は変わる（1536MiB では MoE 120.9 ms 中 34.5 ms = 28.5%）。
8. **expert カーネル実行 64.5 ms の中身は未分解。** `skip-grid` は
   「I/O なし・5 launch/層」の下限を示すが、カーネル本体と launch 間の
   ギャップの分離は `src/qwen38_moe.cuh` / `src/lowbit.cuh` 側の
   計装が必要で、担当外のため行っていない。
   なお §7.3 で launch 単価は約 6 µs と測れており、層あたり 15〜20 stream 操作
   = 約 0.1 ms を差し引いても 1.2 ms/層 が実行側に残る。

## 7. Task R の追試（並行計測、A100、他 GPU プロセス無しを確認）

Task Q の解析と独立に、同じ A100 セッションで取り直した値。追加で
**前置グループ分割**を実装して A/B したが効果が無かったので、その証拠も残す。
全ケースで `nvidia-smi` の他プロセスが空であることを記録している。

### 7.1 再計測（round 1、`tests/moe_transfer_sweep.sh`）

| case | tok_s | parity | moe_ms | pread ms | DMA ms | stall ms | hit 率 |
|---|---|---|---|---|---|---|---|
| diag (serial) | 7.328 | 64/64 | 97.76 | 899.9 | 1321.1 | 867.4 | 75.8% |
| diag-grid | 7.743 | 64/64 | 90.15 | 728.9 | 1289.1 | 1214.3 | 75.8% |
| st4 | 7.625 | 64/64 | — | — | — | — | — |
| st4-grid | 7.968 | 64/64 | — | — | — | — | — |
| skip | 8.433 | 0/64 | 79.86 | 0 | 0 | 0 | 93.8% |
| skip-grid | 9.697 | 0/64 | 64.04 | 0 | 0 | 0 | 93.8% |
| diag-4096 | 6.812 | 64/64 | 107.74 | 1220.0 | 2273.8 | 1364.9 | 57.8% |
| diag-grid-4096 | 6.854 | 64/64 | 106.81 | 1208.4 | 2212.1 | 2089.6 | 57.8% |
| plain-2000 / 4096 / 8000 | 6.659 / 7.054 / 7.386 | 64/64 | — | — | — | — | — |

- 20000 MiB の実転送量 230.5 MB/token、**pread 16.4 GB/s / DMA 11.2 GB/s** で §2.3 と一致。
- 4096 MiB では 400.7 MB/token、**DMA だけで 35.5 ms/token**、MoE 107.7 ms。
  容量を落とすほど転送の比率が上がる（20000: 18%、4096: 33%）。
- `IO_STREAMS=4` は **64/64**（§5.2 の修正の独立追認）だが tok_s は変わらない。

### 7.2 前置グループ分割（`WARODECODE_MOE_BATCH_GROUPS`）— 実装したが効かない

batched pass は k expert を全部 staging してから 5 launch するため I/O が
1 バイトも隠れない。そこで層を前置グループに分け、**グループ g+1 の pread+DMA を
g のカーネル実行と重ねる**実装を入れた（加重和は j 昇順のままなので bit 一致、
全ケース parity 64/64）。

| case | tok_s | parity | moe_ms | stall ms |
|---|---|---|---|---|
| plain（grid、1 グループ） | **7.945** | 64/64 | — | — |
| serial（`MOE_FUSED_GRID=0`） | 7.634 | 64/64 | — | — |
| g2 | 7.936 | 64/64 | — | — |
| g4 | 7.814 | 64/64 | — | — |
| g2（2 本目） | **7.956** | 64/64 | — | — |
| diag-serial | 7.380 | 64/64 | 96.64 | 806.3 |
| diag-plain | 7.691 | 64/64 | 91.20 | 1209.0 |
| diag-g2 | 7.644 | 64/64 | 91.97 | 832.3 |
| diag-g4 | 7.529 | 64/64 | 93.78 | 579.0 |
| plain-c1536 / g2-c1536 | 6.390 / 6.344 | 64/64 | — | — |
| diag-g2-c1536 | 6.112 | 64/64 | 124.73 | 1541.7 |

**DMA の consumer 待ちは設計どおり 1209 → 832 → 579 ms（−31% / −52%）と削れている。
それでも MoE 区間は 91.20 → 91.97 → 93.78 ms で縮まない（4 分割は +2.8%）。**
つまり **consumer の DMA 待ちは MoE 区間のクリティカルパスに無い**。
転送の露出を消しても MoE は縮まない、というのがこの A/B の結論である。
既定は 1（OFF）。

実装中にバグを 1 件踏んだ: 全グループが 1 枚の pinned host
`Q38MoeExpertPtrTable` を共有していたため、グループの非同期 H2D が
次グループのホスト書き込みを読んでいた（**速度は正常・トークンが誤り**、
4 分割では千鳥ポインタで `illegal memory access`）。グループごとの host/device 表
（8 面）に修正。**parity を取らない A/B はこれをそのまま採用してしまう。**

### 7.3 launch 単価と、MoE 区間が反応する唯一の軸

I/O を消した状態でグループ数だけを振ると:

| case（すべて `IO_SKIP=1`） | 層あたり launch | moe_ms |
|---|---|---|
| skip-grid（1 グループ） | 5 | 64.14 |
| skip-g2 | 10 | 65.01 |
| skip-g8 | 40 | 74.89 |
| skip（serial, 参考） | 60 | 79.86 |

1 → 8 グループで +10.75 ms、60 → 5 launch/層で −15.7 ms。
どちらも **launch 1 個あたり約 6 µs** で直線的に効く。
残る 64.14 ms（= 1.34 ms/層）は launch でも転送でもなく、
**expert GEMV 本体の実行**とみるのが最も整合する
（層あたりの stream 操作は 15〜20 個＝約 0.1 ms に過ぎない）。

### 7.4 host↔device 往復の固定費（`tests/host_sync_latency.cu`）

「48 層ぶんの router D2H + `cudaStreamSynchronize` が MoE 区間を支配している」
という仮説を潰すために直接測った:

| 計測 | µs |
|---|---|
| 空の `cudaStreamSynchronize` | 1.68 |
| 40 B D2H + sync（router と同じ形） | 8.68 |
| 1 ブロック kernel + sync | 10.22 |
| 512 要素 単一スレッド top-k kernel + sync | 20.64 |
| **44 個の event record + sync（`WARODECODE_PROFILE=1` が層ごとに足す量）** | **155.63** |
| 同 event record のみ（sync なし） | 144.51 |

- router パターンは 48 層で **0.42 ms/token** しかない。**仮説は棄却された。**
- 一方で **`WARODECODE_PROFILE=1` 自体が層あたり 0.156 ms = 7.5 ms/token を足す**
  （device event は stream 上で直列に実行される）。計測ありの値が 2〜4% 遅い主因で、
  profiler 付きの絶対値を素の性能として読んではいけない。

### 7.5 60 ms への到達可能性

| 項目 | 実測 |
|---|---|
| MoE 区間（20000 MiB、既定） | 91.2 ms |
| expert I/O を完全に消した下限 | 64.0 ms |
| 転送の露出 | 27 ms（うち DMA 待ち 19 ms は分割しても縮まない） |

- **転送側の仕事だけでは 60 ms に届かない。** 60 ms を切るには 64 ms の実行下限
  そのものを下げる必要があり、それは `src/lowbit.cuh` の expert GEMV と
  MoE 実行戦略の仕事である（本タスクの担当ファイル外）。
- 転送側で正味効く改善は既に入っている: pread + pinned ring 経路（pageable 比 +38%）と
  `WARODECODE_MOE_FUSED_GRID`（+4.1%、§5.3）。本タスクで追加できたものは無い
  （prefix split は負け）。

## 8. 追加で採用しなかった案（Task R の実測）

| 案 | 実測 | 判定 |
|---|---|---|
| 前置グループ分割 `WARODECODE_MOE_BATCH_GROUPS=2` | plain 7.945 → g2 7.936（MoE 91.20 → 91.97） | **負け**（既定 1 のまま） |
| 同 `=4` | 7.814、MoE 93.78（DMA 待ちは −52%） | **負け** |
| 同 `=8`（I/O なし） | MoE 64.14 → 74.89 | **負け** |
| `WARODECODE_PROFILE=1` を付けたまま性能を語る | event record だけで 7.5 ms/token | 計測時は必ず `plain` を併走させる |

