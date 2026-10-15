# Task W: routed expert GEMV の内訳 — 律速は「メモリ」ではなく「符号帳の引き方」

対象: src/lowbit.cuh（IQ2_S / IQ4_NL の行デコードと expert 一括ランチャー）、
tests/qwen38_expert_ilp_bench.cu（新規）、tests/qwen38_expert_ilp_host_test.cpp（新規）。
batch=1 decode / A100-SXM4-40GB / Qwen3.8-Flash-Next（86 GB GGUF）/ greedy 64 トークン。

---

## 0. 結論（先に）

1. **routed expert の GEMV は 2 本で 5.6 ms/token 使っていた**（48 層 × 10 expert 換算、
   カーネル単体の実測）。IQ2_S gate/up が 29.8 us/層（176 GB/s）、IQ4_NL down が
   88.3 us/層（104 GB/s）。どちらも HBM 1555 GB/s の 7〜11% しか出ていない。

2. **原因は「18 バイトブロック / 82 バイト superblock という細かい読み方」ではなかった。**
   同じロード・同じループのまま符号帳の展開だけを最も安い消費（生のニブルを重みとして
   使う）に差し替えると、down は **88.3 → 15.0 us（615 GB/s）**。つまりアクセスパターンは
   615 GB/s まで出せる。差は**符号帳の引き方**である。

3. **`iq4_nl_lut_value()` が律速だった。** `constexpr int8_t lut[16]` を
   レーンごとに違う 4 bit インデックスで引く形は、コンパイラが定数メモリ/選択木に
   落とすため、それが供給するロードより高くつく。**16 バイトを shared に置く**と
   1 値 = 1 LDS（同一 16 バイト領域なのでブロードキャスト、バンク衝突なし）になり、
   **88.3 → 18.5 us（4.8 倍、499 GB/s、bit 一致）**。

4. IQ2_S 側は grid 符号帳（1024 エントリ × uint64）の shared 引きが 1.6 倍分ある
   （no-grid 17.7 us vs 29.1 us）。ここは未改修。

---

## 1. カーネル単体の実測（A100 / 本番形状 / 10 experts 一括）

`tests/qwen38_expert_ilp_bench.cu`（本番ヘッダを include し、出力を全要素 memcmp で比較）:

| カーネル | 時間 | 実効 GB/s | 倍率 |
|---|---:|---:|---:|
| IQ2_S gate/up [640, 2560] × 10 | 29.80 us | 176 | 1.00 |
| 　+ ILP=2 | 28.97 us | 181 | 1.03 |
| 　+ ILP=4 | 25.27〜28.42 us | 185〜208 | **1.04〜1.05** |
| 　+ ILP=8 | 26.07〜29.39 us | 179〜201 | 1.01（4 より悪い） |
| **IQ2_S no-grid（値は不正、時間のみ）** | **16.2〜17.7 us** | **297〜325** | **1.63〜1.80** |
| IQ4_NL down [2560, 640] × 10 | 79.7〜90.6 us | 102〜116 | 1.00 |
| 　+ ILP=2 | 78.3〜88.9 us | 104〜118 | 1.02 |
| 　+ ILP=4 | 76.4〜86.8 us | 106〜121 | 1.04 |
| 　+ ILP=8 | 80.2 us | 115 | 1.10 |
| **+ ILP=8 + shared 符号帳** | **18.46 us** | **499** | **4.78** |
| **IQ4_NL no-codebook（値は不正、時間のみ）** | **13.7〜15.0 us** | **615〜674** | **5.4〜5.9** |

「no-grid / no-codebook」は**値が違う**（タイミング専用の対照）。同じロードをしながら
復号だけを消したときの上限を与える。

per-token 換算（48 層 × gate/up/down の 3 本）:
**5.6 → 4.7 ms/token（ILP のみ）→ down の shared 化込みで ~2.3 ms/token**。

---

## 2. 実装

### 2.1 ILP（ソフトウェアパイプライン）版 — `WARODECODE_EXPERT_GEMV_ILP`（既定 8）
`iq2_s_q8_k_row_dp4a_ilp<USE_SMEM, ILP>` / `iq4_nl_q8_0_act_row_dp4a_ilp<ILP>`。
superblock（IQ2_S）または同パリティブロック（IQ4_NL）の**ロードと整数 partial を ILP 個
先読み**し、アキュムレータの `fmaf` は sb 昇順のまま消費する。float の加算順は 1 命令も
変わらないので bit 一致する。端数はスカラーテールに落ち、`sb + ILP <= n` /
`b + 2*(ILP-1) < n` の形なので行の外を読まない。

- IQ2_S は実測で ILP=4 が最適（8 は sequential 相当まで戻る）なので、要求値が 8 でも
  このサイトは 4 に丸める。
- `=0` で完全に従来カーネルへ戻る。

### 2.2 shared 符号帳 — `iq4_nl_q8_0_act_row_dp4a_ilp_smem<ILP>`
カーネル先頭で `__shared__ int8_t lut[16]` に
`lut[i] = (int8_t)iq4_nl_lut_value((uint8_t)i)` を入れ（early return より前、
IQ2_S の grid staging と同じ作法）、`lowbit_iq4_nl_pack4_smem(nibbles, lut)` で引く。
`i < 16` なので `iq4_nl_lut_value(i) == 符号帳[i]` であり、
`pack4_smem == pack4` が**全 65536 通りの 4 ニブル組で一致**することをホストで確認している。

---

## 3. 検証

- `tests/qwen38_expert_ilp_host_check.sh`（新規, nvcc 不要）:
  両ループ構造を実バイトオフセットで鏡写しにし、**アキュムレータが消費する
  (index, scale, 整数 partial) 列そのもの**と float チェーンを bit 比較。
  1,440 チェーン（ILP=2/4/8、行長 1〜24、superblock 1〜12）＋ shared 符号帳の
  65,536 組＋陽性対照。**checks=4325 failures=0**。
- `tests/qwen38_cuda_syntax_check.sh` PASS（launch site 314）。
- 実機: ベンチが全 expert・全行を memcmp し、ILP / shared 版すべて `bit ok`。
- エンドツーエンド: 同一ジョブ内 A/B で全実行 64/64（§4）。

---

## 4. エンドツーエンド A/B（同一ジョブ内・交互実行）

ILP のみ（shared 化前、cache 20000、--bench-repeat 3、64 トークン、3 ラウンド）:

| 構成 | 1 | 2 | 3 | 中央値 | parity |
|---|---:|---:|---:|---:|---|
| ILP=0（従来） | 24.642 | 24.581 | 24.609 | 24.609 | 64/64 |
| ILP=4 | 24.620 | 24.549 | 24.641 | 24.620 | 64/64 |
| ILP=8 | 24.815 | 24.822 | 24.959 | **24.822** | 64/64 |

ILP だけでは **+0.9%**（カーネル単体の +5〜10% が全体では薄まる）。

shared 符号帳を入れた最終形（cache 14000/20000/22000、同一ジョブ・交互実行・3 ラウンド、
llama.cpp pinned 6fcaa16f を CUDA sm80 で同時ビルド）:

| 構成 | VRAM | 1 | 2 | 3 | 中央値 | parity |
|---|---|---:|---:|---:|---:|---|
| 本エンジン cache 14,000 | 20,256 MiB | 18.310 | 18.294 | 18.365 | 18.310 | 64/64 |
| llama.cpp K=16 | 20,494 MiB | 17.921 | 17.996 | 18.035 | 18.00 | 36/64 |
| 本エンジン cache 20,000 | 26,256 MiB | 26.617 | 26.402 | 26.373 | 26.40 | 64/64 |
| 本エンジン cache 22,000 | 28,256 MiB | 26.589 | 26.479 | 26.580 | **26.58** | 64/64 |
| llama.cpp K=24 | 28,196 MiB | 23.608 | 23.741 | 23.816 | 23.62 | 40/64 |

**VRAM 28.2 GiB 同士で 26.58 対 23.62 = +12.5%**（最悪/最良の取り方でも +11.2%）。
20,000 MiB（llama.cpp より 2 GiB 少ない）でも +11.8%。20.3 GiB 同士でも +1.7%。
smem 化だけで 24.82 → 26.58 tok/s（+7.1%）でした。

---

## 5. 次に効きそうな所（未着手）

1. **IQ2_S の grid 符号帳**: no-grid で 1.63〜1.80 倍。uint64 の shared 引きを
   **32 bit に分割した 2 枚の表**にすれば、バンク衝突と 2 サイクル LDS が半分になる見込み
   （値は `grid[i] >> 32` と `(uint32)grid[i]` で厳密に一致）。
2. **`iq4_nl_lut_value()` を使う他の経路**: 単一 expert 版 `launch_iq4_nl_gemv_q8_0_f32`
   も同じ符号帳を使う。このモデルでは routed expert のみが IQ4_NL だが、
   他モデルでは効く。
3. **2 行/warp 化**: 1 命令あたりのバイト数を増やす案。今回は不要だった
   （アクセスパターンは 615 GB/s 出ており、復号を直せばそちらに張り付く）。
