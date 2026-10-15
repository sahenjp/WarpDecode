# 複数モデルでの動作テスト（2026-10-15, A100-SXM4-80GB）

dense (llama / qwen2 / qwen3) パスが実際の GGUF でどこまで動くかを、llama.cpp が配る
Q4_0 の小型モデル 4 本で確かめた。同時に llama.cpp (mainline HEAD) の tg64 も測っている。

| モデル | アーキ | サイズ | llama.cpp tg64 | 本エンジン | 止まった理由 |
|---|---|---:|---:|---|---|
| Qwen2.5-0.5B-Instruct-Q4_0 | qwen2 | 429 MB | 555.7 tok/s | **起動不可** | `output.weight` が Q6_K |
| TinyLlama-1.1B-Chat.Q4_0 | llama | 638 MB | 539.6 tok/s | **起動不可** | `output.weight` が Q6_K |
| Llama-3.2-1B-Instruct-Q4_0 | llama | 773 MB | 577.8 tok/s | **起動不可** | `blk.*.ffn_down.weight` が Q8_0 |
| Qwen2.5-1.5B-Instruct-Q4_0 | qwen2 | 1.07 GB | 360.4 tok/s | **起動不可** | `output.weight` が Q6_K |

**llama.cpp 自身が配る「Q4_0」の GGUF でも、全テンソルが Q4_0 ではない。**
- Qwen 系・TinyLlama の Q4_0 は **出力ヘッドだけ Q6_K**（品質のため残すのが通例）。
- Llama-3.2-1B は **ffn_down が Q8_0**。

本エンジンの dense パスは元々すべての 2D テンソルが Q4_0 であることを要求していたため、
この 4 本はどれも起動できなかった。これが「いろんなモデルで使う」ときの最初の壁である。

---

## 1. 今回入れたもの

1. **何が起きたか分かるエラー**。以前は「Q4_0 2Dのみ対応: output.weight」の一行だけで、
   何をすればよいか分からなかった。今は
   - 型名を出す (Q6_K / Q8_0 / Q4_K など、GGUF の型番号ではなく名前)
   - 対応している型の一覧を出す
   - **回避策**を出す: `llama-quantize in.gguf out.gguf Q4_0` で全テンソルを
     Q4_0 にそろえれば本エンジンで動く
   という 3 点を出す。アーキテクチャ非対応のときも同様に案内する。
2. **Q8_0 / Q6_K / F16 / F32 を F16 に展開する経路**（`src/ggml_dequant.hpp`、
   ggml のブロック定義をホストへ写したもの）と F16 GEMV。ただし **既定では無効**
   （`WARODECODE_DENSE_F16=1` で有効）。理由は §2。
3. `--help` を読みやすく整理（従来はデバッグ用フラグ全部が 1 行に連結されていた）。
   全フラグは `--help-all` へ。対応モデル・よく使うオプション・例を表示する。

---

## 2. F16 展開経路が既定で無効な理由（実測）

Q6_K ヘッドを F16 に落とす経路を入れて 4 本を走らせたところ、3 本が
`CUDA: an illegal memory access` で落ちた。compute-sanitizer の結果:

    Invalid __global__ read of size 2 bytes
      at void wd::f16_gemv_kernel<(bool)1>(...) in warodecode.cu:1408
      Access to 0x...2004 is out of bounds

`f16_gemv_kernel` の重み/活性の読み出しが確保領域を越えている。ヘッドの
(rows, cols) はローダ側で検証済み (rows == vocab, cols == hidden) なので、
残る候補は「F16 バッファの確保サイズとカーネルが想定する rows*cols の不一致」だが、
原因を特定する前にセッションが不安定になり追い切れなかった。

**既定で有効にすると、これまで明確なエラーで止まっていたモデルが不正アクセスで
落ちるようになる（悪化）**ため、既定はエラーのまま、実験フラグの裏に置いた。

---

## 3. 次にやること

1. `f16_gemv_kernel` の範囲外読み出しを潰す（compute-sanitizer で当たりは付いている）。
   直れば Qwen2.5 / TinyLlama / Llama-3.2 の Q4_0 が素のまま動く。
2. 本来は **Q4_K_M**（llama.cpp の既定量子化）を扱えるのが「使いやすい」の本命。
   `lowbit.cuh` に Q4_K は無いので、そこは新規カーネルが要る。
3. 動くモデルが増えたら、この表を README の速度表に統合する。
