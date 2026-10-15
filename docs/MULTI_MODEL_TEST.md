# 複数モデルでの動作テスト（2026-10-15/16, A100-SXM4-80GB）

dense (llama / qwen2 / qwen3) パスが実際の GGUF でどこまで動くかを、llama.cpp が配る
Q4_0 の小型モデル 4 本で確かめた。同時に llama.cpp (mainline HEAD, `-ngl 99`) の tg64 も測っている。

---

## 0. 結論

**4 本すべて動くようになった。** 到達点は「Q4_0 以外のテンソルを Q8_0 に詰め直して
lowbit.cuh の実績あるカーネルで走らせる」方式。

| モデル | arch | サイズ | 本エンジン | llama.cpp tg64 |
|---|---|---:|---:|---:|
| Qwen2.5-0.5B-Instruct-Q4_0 | qwen2 | 429 MB | **248.8 tok/s** | 555.7 |
| TinyLlama-1.1B-Chat.Q4_0 | llama | 638 MB | **230.9 tok/s** | 539.6 |
| Llama-3.2-1B-Instruct-Q4_0 | llama | 773 MB | **245.8 tok/s** | 577.8 |
| Qwen2.5-1.5B-Instruct-Q4_0 | qwen2 | 1.07 GB | **157.5 tok/s** | 360.4 |

小さい dense モデルでは llama.cpp が約 2.2 倍速い。本エンジンの設計は
「VRAM に収まらない MoE(専門家)を流す」ことに寄せてあり、全重みが VRAM に載る
小型モデルは llama.cpp の得意分野である、というのが正直なところ。
qwen38 (86 GB MoE) の 64/64 一致・26.6 tok/s の方が本来の戦場。

---

## 1. 途中で見つかった本当の原因: move 代入の use-after-free

最初は「dense パスは Q4_0 しか読めない」という *明確なエラー* だったので、
Q6_K / Q8_0 / Q4_1 を F16 に展開する経路を書いた。ところが実モデルで
`CUDA: an illegal memory access` になり、compute-sanitizer が
`f16_gemv_kernel` の 2 バイト読み出しを指した。形・確保サイズ・ポインタを
すべて出力して追い込んだ結果、**カーネルではなく行列の move 代入**が原因だった:

    Q4Matrix& operator=(Q4Matrix&& o) {
        reset(); ... q8 = o.q8; planar_q4 = o.planar_q4; owns = o.owns;
        o.q = nullptr; o.d = nullptr; o.rows = o.cols = 0; o.owns = true;
        //                                       ^^^ o.q8 を消し忘れ
    }

移動元(一時オブジェクト)のデストラクタが `owns == true` のまま `cudaFree(q8)` を
呼び、**移動先が持っていたバッファを解放**していた。以降のカーネルは解放済み領域を読む。
`o.q8 = nullptr;` を足した瞬間に 4 本とも動いた。

教訓: ポインタを増やすときは move コンストラクタと move 代入の**両方**を直す。

---

## 2. 実装したこと

- `src/ggml_dequant.hpp`（新規）: GGUF の各型を F32 へ戻す式 (Q4_0 / Q4_1 / Q8_0 / Q6_K /
  F16 / F32) と、**Q8_0 へ詰め直す** `ggml_requantize_row_to_q8_0()`。
  Q6_K (6bit) より Q8_0 (8bit) の方が精度は上で、バイト数は 1.3 倍。
- `src/gguf.hpp`: `Q4_1`（Llama-3.2-1B の ffn_down）を enum / 型名 / サイズ表に追加。
- `src/warodecode.cu`: Q4_0 以外の 2D テンソルは Q8_0 raw バッファにし、
  `q8_0_gemv_kernel` / `q8_0_gather_row_kernel`（lowbit.cuh の実績あるカーネル）で実行。
  自作の F16 GEMV は削除した。
- dense パスのモデル読み込みで、対応型の一覧と回避策を出すエラー（型名付き）。

## 3. まだできないこと

- **`--prompt` / `--interactive` は qwen4exp 専用**。dense モデルは `--start-token` からの
  生成とベンチのみ。プロンプトを使うには llama / qwen2 / qwen3 のトークナイザ
  (pre 種別と chat template) を実装する必要がある。エラー文でその旨を案内する。
- `Q4_K_M` などの k-quant は依然未対応（lowbit.cuh に Q4_K が無い）。
- 各モデルで生成トークンの一致検証はしていない（qwen38 は CPU オラクルと 64/64 済み）。
