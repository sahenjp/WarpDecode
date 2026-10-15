# Task V: 「18 tok/s の天井」の正体と、その撤去

対象: src/qwen38_hc.cuh（exact Q8_0 GEMV と F32 inject GEMV）、
src/qwen38_moe.cuh（shared expert の呼び出し経路）、src/lowbit.cuh（IQ4_NL の行ルーチン）、tests/。
batch=1 decode / A100-SXM4-40GB / Qwen3.8-Flash-Next（86 GB GGUF）/ greedy 64 トークン。
parity は全構成で llama.cpp CPU greedy オラクルとの **64/64** を確認している。

---

## 0. 結論（先に）

1. **cache 20,000 と 30,000 MiB が同じ tok/s なのは性能の飽和ではない。**
   64 トークンで触れる (layer, expert) 対は **7,448 個**で、20,000 MiB のキャッシュは
   **9,002 スロット**ある。つまり 20,000 MiB の時点で**そのランで必要な expert は全部載っている**。
   evictions は両方で **0**、misses は両方で **7,448（完全に同一）**。
   増やしても入れるものが残っていないので、速度が動かないのは当然だった。
   最終バイナリでも 20,000 MiB = 24.52〜24.88 tok/s、30,000 MiB = 24.67 tok/s で変わらない。

2. **残り 55 ms/token の本当の中身は、Q8_0 exact GEMV が
   「1 ブロック = 1 メモリ往復」で走っていたこと。** 実効 21〜43 GB/s
   （HBM 1555 GB/s の 1.4〜2.8%）。ブロック数だけループを回し、
   次のブロックのロードを現在の FMA チェーンの後ろに置いていた。

3. **対策**: (a) MoE shared expert の 3 本が scalar カーネルのままだったのを共通ランチャー経由に、
   (b) PIPE 段のソフトウェアパイプライン版カーネルを追加（行数が少ない行列だけ）、
   (c) IQ4_NL の行ループに unroll。
   エンドツーエンドで **18.11 → 24.82 tok/s（+37.0%）**、parity は全実行 64/64。

4. パイプライン版は**行数が多い行列では逆に遅い**（GDN/QSA の [8192, 2560] で
   41.1 → 55.5 us、[2560, 6144] で 29.3 → 37.8 us）。warp が既に十分あって
   レイテンシが隠れているためで、**rows <= 1024 かつ blocks_per_row >= 32 のときだけ**
   選ぶようにした。この 1 行を入れるだけで 23.7 → 24.3 tok/s に戻った。

---

## 1. 天井の正体（区間別の実測）

すべて同一 Colab ジョブ内で逐次実行し、ジョブ内の相対値だけを根拠にしている。

### 1.1 cache 20,000 / 30,000 が同じになる理由 = compulsory miss の床

WARODECODE_EXPERT_STATS=1 の実測（cache 20000 / 64 トークン / 1 反復）:

| 指標 | cache 20,000 MiB | cache 30,000 MiB |
|---|---:|---:|
| slots | 9,002 | 13,503 |
| accesses | 30,720 | 30,720 |
| hits / misses | 23,272 / 7,448 | 23,272 / 7,448 |
| hit_rate | 0.7576 | 0.7576 |
| **evictions** | **0** | **0** |
| distinct_pairs | 7,448 | 7,448 |
| bytes_copied | 14.75 GB（230 MB/token） | 同 |
| io_pread_ms（合計） | 826.6 ms（12.9 ms/token） | 同 |

distinct_pairs (7,448) < slots (9,002) なので 20,000 MiB で容量不足は起きていない。
7,448 回の miss はすべて**初回接触（compulsory）**で、30,000 に増やしても消えない。
3 反復のときは 2〜3 反復目で misses = 0 になる（hit_rate 0.9192 = 84,712/92,160）。

これは「エンジン側の天井」ではなく **I/O を除いた計算側の天井**である、というのが次の節。

### 1.2 warm 状態の区間別実測

--bench-repeat 3 の best-of-3（= 18.04 tok/s）の正体は 1 反復目
（cold, 81.7 ms/token）ではなく **2〜3 反復目（expert cache が温まった状態）**である。
区間別の warm 値は「3 反復平均のプロファイル」と「1 反復のプロファイル」から

    warm = (3 * avg - cold) / 2

で復元できる（ステージ別の total_ms はトークンごとの和で、tokens で割って出すため、
この線形分解は厳密）。cold も warm も同一ジョブ内で別プロセスとして取得した。

warm の内訳（cache 20,000 MiB、単位 ms/token）:

| 区間 | Task V 前 | Task V 後 |
|---|---:|---:|
| moe（router + acquire + batch + shared + mirror） | 26.14 | **21.16** |
| attn（GDN 36 + QSA 12） | 14.55 | 14.44 |
| hc_mix_attn | 9.89 | **5.46** |
| hc_mix_ffn | 9.94 | **5.53** |
| head | 1.30 | 1.30 |
| ple + embed + combine + other | 約 3.0 | 約 3.0 |
| （プロファイラ自身のオーバーヘッド） | 約 9 | 約 10 |
| 合計（プロファイルなし best-of-3） | **55.4** | **41.1** |

WARODECODE_MOE_PROFILE=1 のサイト別（Task V 前 / 後、3 反復平均）:

| site | 前 | 後 |
|---|---:|---:|
| moe.stage_total | 34.90 | 29.90 |
| moe.acquire | 17.88 | 17.94 |
| moe.batch_total | 8.69 | 8.71 |
| moe.batch.down（IQ4_NL） | 3.98 | 3.98 |
| moe.batch.gate_up（IQ2_S） | 2.78 | 2.78 |
| **moe.shared** | 8.21 | **3.28** |
| moe.router_select | 4.43 | 4.43 |
| moe.mirror | 1.14 | 1.14 |

WARODECODE_HC_PROFILE=1 のサイト別（3 反復平均・プロファイルあり）で HC / attention 側の上位
（Task V 前の値）:

| site | ms/token |
|---|---:|
| attn.q8_down（HC mix down [320, 10240]） | 4.54 |
| ffn.q8_down（同） | 4.48 |
| attn.q8_inject（F32 [4, 10240]） | 3.58 |
| ffn.q8_inject（同） | 3.57 |
| gdn.qkv_gemv | 2.50 |
| gdn.out_gemv | 2.39 |
| gdn.z_gemv | 1.53 |
| qsa.qkv_gemv | 1.51 |
| attn.q8_up / ffn.q8_up | 0.75 / 0.77 |

読み方: q8_down（プロファイル 94.6 us/層）はカーネル単体ではもっと遅く、
独立ベンチで scalar 150〜163 us / dp4a 37.2 us だった。
つまり HC mix と shared expert の時間は、ほぼそのまま
「exact Q8_0 GEMV が 1 往復/ブロックで走っている時間」である。

### 1.3 なぜ「1 往復/ブロック」になるのか

参照実装の加算順は 1 行につき 8 本の独立した float チェーン:

    lane c (0..7) が 4 値 chunk c を持ち、b = 0..B-1 に対して
      acc_c = __fmaf_rn(weight_scale_b * act_scale_b, dp4a(act_b, w_b), acc_c)
    最後に ((a4+a0)+(a6+a2)) + ((a5+a1)+(a7+a3))

float の加算は再結合できないので、**チェーンの長さは縮められない**。
縮められるのはロードだけなので、PIPE ブロック分のオペランドを
FMA の「前」にレジスタへ取りに行くのが唯一の合法な並列化になる。

### 1.4 moe.acquire の dev 値は信用しない

warm 状態の expert cache 統計では acquire_ms（ホスト実測）が
3 反復合計 1,040 ms（= 192 トークン）で、**1 反復だけの 1,126 ms より小さい**。
つまり 2〜3 反復目の acquire ホスト時間はほぼ 0 で、miss も 0。
一方 WARODECODE_MOE_PROFILE の moe.acquire dev は warm 換算で 9.59 ms/token と出る。
これはイベント対の間にキューされた前段の仕事を拾っているためで、
layer_gap_per_token（acquire の外側の時間、warm 換算 43.5 ms）が
トークン時間 41.1 ms とほぼ一致することのほうが正しい。
**acquire は warm では最適化対象ではない。**

---

## 2. 実装した改善（すべて bit 一致、環境変数で A/B 可）

### 2.1 WARODECODE_HC_Q8_PIPE（既定 1）
q38_hc_q8_0_q8_0_exact_gemv_f32_dp4a_pipe_kernel を新設
（src/qwen38_hc.cuh）。ロード・__dp4a・__fmaf_rn チェーン・閉じる木は
既存の dp4a カーネルと同一で、レジスタに PIPE ブロック分を先読みするだけ。
末尾の先読みは blocks_per_row - 1 にクランプするので、行の外や確保領域の外を読まない。

選択条件（q38_hc_launch_exact_q8_0_gemv_f32）:

    WARODECODE_Q8_DP4A_GEMV != 0 && cols % 32 == 0
      && cols/32 >= 32           (kQ38HcQ8PipeMinBlocks)
      && rows <= 1024            (kQ38HcQ8PipeMaxRows)
      cols/32 >= 128 ? PIPE=32 : PIPE=16   (kQ38HcQ8PipeWideBlocks)

A100 実測（参照カーネルと bit 比較して bit-ok、tests/qwen38_hc_pipe_bench.cu）:

| shape | scalar | dp4a（PIPE=0） | pipe（既定） |
|---|---:|---:|---:|
| [320, 10240] HC down | 149.7 us | 37.2 us | **23.3 us**（PIPE=32） |
| [640, 2560] shexp gate/up | 40.1 us | 11.6 us | **9.9 us** |
| [10240, 320] HC up | 26.5 us | 9.2 us | （chain < 32 で不使用） |
| [2560, 640] shexp down | 15.8 us | 6.0 us | （同上） |
| [8192, 2560] GDN qkv | 151.2 us | **41.1 us** | 不使用（rows > 1024） |
| [2560, 6144] GDN out | 117.4 us | **29.3 us** | 不使用 |
| [6144, 2560] GDN z | 121.2 us | **31.8 us** | 不使用 |
| [4096, 2560] QSA qkv | 83.1 us | **23.0 us** | 不使用 |
| [2560, 4096] QSA out | 79.5 us | **20.8 us** | 不使用 |
| [16, 2560] GDN beta | 36.0 us | 10.4 us | **8.3 us** |
| [48, 2560] GDN alpha | 36.5 us | 10.7 us | **8.6 us** |

パイプライン深さの実測（[320, 10240]）: dp4a 40.1 / pipe4 59.5 / pipe8 38.1 /
pipe16 25.9 / pipe24 24.8 / **pipe32 23.3 us**。

### 2.2 F32 inject 側も同じスイッチ
q38_hc_dense_gemv_f32_avx2_warp_pipe_kernel を新設。
HC inject は [hc, hc*n_embd] = [4, 10240] で warp が 4 本しか立たず、
5.7 GB/s しか出ていなかった（1 往復/ステップ）。パイプライン化で
**28.5 → 14.9 us（1.9 倍）**。WARODECODE_HC_Q8_PIPE=0 で従来カーネルに戻る。

### 2.3 WARODECODE_SHEXP_GEMV_FAST（既定 1）
q38_moe_gemv_q8_0_exact()（src/qwen38_moe.cuh）は shared expert の
gate / up / down を **scalar カーネル固定**で起動していた。同じ形・同じ型の射影は
HC / GDN / QSA では既に q38_hc_launch_exact_q8_0_gemv_f32() を通っている。
そこへ寄せるだけで、カーネル単体で [640, 2560] が 40.1 → 9.9 us、
[2560, 640] が 15.8 → 6.0 us。=0 で従来カーネルに戻る。

### 2.4 IQ4_NL 行ループの unroll（src/lowbit.cuh）
iq4_nl_q8_0_act_row_dp4a() のブロック走査に #pragma unroll 4 を追加。
18 バイトブロックを 2 バイトずつ拾うループで、ロードがチェーンの後ろに隠れていた。
加算順は変えていないので値は bit 同一。

---

## 3. 検証

### 3.1 ホスト（nvcc 不要）
- tests/qwen38_hc_q8_pipe_host_check.sh（新規）: スカラー順序・__dp4a・
  パイプラインのループ構造（短い行 blocks < PIPE、末尾のクランプ、スカラーテール）を
  C++ で鏡写しにして **2,080 行を bit 比較、mismatch 0**。
  positive control（ブロックを 1 個飛ばすとビットが動く）も確認。
- tests/qwen38_cuda_syntax_check.sh PASS（launch 構文書き換え 304 箇所）。
- tests/host_checks.sh に上記を登録。

### 3.2 実機
- tests/qwen38_hc_pipe_bench.cu（新規）: 本番ヘッダを include して
  **本番カーネル同士**を bit 比較。全 11 shape + F32 inject で bit-ok。
- エンジン本体の parity は下の A/B の全実行で 64/64。

---

## 4. A/B（同一ジョブ内）と parity

A100 / cache 20,000 MiB / --bench-repeat 3 / best-of-3 / 交互実行。
base は WARODECODE_SHEXP_GEMV_FAST=0, WARODECODE_HC_Q8_PIPE=0（従来動作）。

### 4.1 到着点（最終バイナリ）

| ジョブ | base tok_s | new tok_s | 比 |
|---|---|---|---|
| round 1（PIPE 追加、行数しきい値なし） | 18.133 / 18.067 / 18.204 | 23.703 / 23.714 / 23.503 | 1.307 / 1.313 / 1.291 |
| round 1b（+ rows <= 1024 しきい値） | 18.018 / 18.035 | 24.163 / 24.486 | 1.341 / 1.358 |
| **round 2（+ IQ4_NL unroll, PIPE=32）** | **18.153 / 18.061** | **24.815 / 24.522 / 24.880** | **1.367 / 1.358 / 1.378** |

**最終: 18.11 → 24.82 tok/s（中央値、+37.0%）。**
最終バイナリの parity は 5 実行すべて **64/64**（r2_new 3 回 + round 1b の 2 回）、
cache 30,000 MiB の 1 実行も **64/64**。

### 4.2 cache サイズの再確認（最終バイナリ）

| 構成 | tok_s | ms/token | misses | evictions | parity |
|---|---:|---:|---:|---:|---:|
| cache 20,000 MiB | 24.815 / 24.522 / 24.880 | 40.2〜40.8 | 7,448 | 0 | **64/64** |
| cache 30,000 MiB | 24.669 | 40.5 | 7,448 | 0 | **64/64** |

**10 GiB 増やして 0%（−0.5%）。** 理由は 1.1 節のとおり
「20,000 MiB で既に全部載っている」で、性能の飽和ではない。

### 4.3 プロファイルの変化（3 反復平均、プロファイラ込みの絶対値なので比で見る）

| stage | 従来 | 新 | 差 |
|---|---:|---:|---:|
| moe | 34.18 | 29.52 | −4.66 |
| moe.shared（サイト） | 8.23 | 3.29 | −4.94 |
| hc_mix_attn | 9.99 | 5.50 | −4.49 |
| hc_mix_ffn | 10.02 | 5.53 | −4.49 |
| attn | 14.75 | 15.79 → 14.48 | ±0〜+1（しきい値修正前の一時的な悪化） |

---

## 5. 目標に対する正直な位置づけ

- 目標は **18 → 25 tok/s**。到達は **24.82 tok/s（中央値）** で、**0.7% 届いていない**。
- 参考の llama.cpp K=24 は **23.01 tok/s（別時間帯の計測）** なので名目上は 7.9% 上回るが、
  **同一ジョブ内の A/B ではない**ので「勝った」と主張する根拠にはしない。
  同一ジョブ A/B で確定させるには tools/llamacpp_ab.py の再実行が必要。
- それでも 18.11 → 24.82（+37.0%）は、同一ジョブ内で 5 実行とも再現し、
  全実行で parity 64/64 を維持している。

---

## 6. 未検証リスク

1. **A100 40GB のみ**。L4 / T4 では warp 数と帯域の比が違うので
   kQ38HcQ8PipeMaxRows = 1024 と kQ38HcQ8PipeWideBlocks = 128 は再測定が必要。
   しきい値を外すと GDN/QSA の行列で 23〜32% 遅くなることは A100 で確認済み。
2. **PIPE の深さは 4/8/16/24/32 の 5 点だけ**。[320, 10240] では単調に改善しており、
   さらに深い点は未探索（レジスタ圧で頭打ちになるはずだが未確認）。
3. **プロファイルの絶対値は使えない**。WARODECODE_MOE_PROFILE + WARODECODE_PROFILE は
   約 10 ms/token を足す。改善量は「プロファイルなしの tok_s の比」だけを根拠にしている。
4. **1 件の OOM**（round 2 の最初の base 実行）: cudaMalloc expert cache failed: out of memory。
   同一構成の後続実行はすべて成功しており、スイッチの影響ではない（他プロセスの VRAM 残り）。
5. **まだ手を付けていない最大項目**:
   - moe.batch_total 8.7 ms（IQ2_S gate/up 2.8 + IQ4_NL down 4.0 + SiLU/Q8/accum 1.9）
   - moe.router_select 4.4 ms（gemv 0.81 + topk 1.47 + act_quant 1.12 + d2h/sync 1.0）
   - attn 14.4 ms のうち GDN/QSA の大きな Q8_0 GEMV。
     これらは warp 数が足りているので**命令発行律速**（[8192, 2560] で 83 GB/s）で、
     パイプラインでは効かない。1 命令あたりのバイト数を増やす
     （2 行/warp にして 1 ロードで 2 セクタ運ぶ）方向が次の候補。
   - hc_mix の残り 11.0 ms（down 6.5 + inject 3.7 + norm/quant/silu 1）。
6. **下限の見積り**: routed expert + shared expert + HC mix の重みは 1 トークンで約 1.9 GB、
   HBM 1555 GB/s なら 1.2 ms 相当。まだ 30 倍以上の余地がある。
