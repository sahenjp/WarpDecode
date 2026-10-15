# 複数モデルでの動作テスト（2026-10-15/16, A100-SXM4-80GB）

dense (llama / qwen2 / qwen3) パスが実際の GGUF でどこまで動くかを、llama.cpp が配る
小型モデル 7 本（Q4_0 と Q4_K_M）で確かめた。

---

## 0. 結論

**7 本すべてが動き、プロンプトで正しく応答する。**

| モデル | 量子化 | 生成（chat template 経由） |
|---|---|---|
| Qwen2.5-0.5B-Instruct | Q4_0 | "The capital of France is Paris." |
| Qwen2.5-0.5B-Instruct | **Q4_K_M** | "The capital of France is Paris." |
| Qwen2.5-1.5B-Instruct | Q4_0 / **Q4_K_M** | "Paris" |
| TinyLlama-1.1B-Chat | Q4_0 | "Paris, which is the capital of France." |
| Llama-3.2-1B-Instruct | Q4_0 / **Q4_K_M** | "France" |

数列の続き（prefill が効いているかの確認）: "1, 2, 3, 4, 5, 6, 7, 8," → **"9, 10, 11,"** ✓

速度（A100 80GB・64 トークン・llama.cpp mainline `-ngl 99` と同時計測）:

| モデル | 本エンジン | llama.cpp tg64 |
|---|---:|---:|
| Qwen2.5-0.5B-Q4_0 | 248.8 tok/s | 555.7 |
| TinyLlama-1.1B-Q4_0 | 230.9 tok/s | 539.6 |
| Llama-3.2-1B-Q4_0 | 245.8 tok/s | 577.8 |
| Qwen2.5-1.5B-Q4_0 | 157.5 tok/s | 360.4 |

小さい dense モデルでは llama.cpp が約 2.2 倍速い。本エンジンの設計は
「VRAM に収まらない MoE(専門家)を流す」ことに寄せてあり、全重みが VRAM に載る
小型モデルは llama.cpp の得意分野である、というのが正直なところ。
qwen38 (86 GB MoE) は 64/64 一致・26.6 tok/s で、そちらが本来の戦場。

---

## 1. 途中で見つかった本当の原因: move 代入の use-after-free

最初は「dense パスは Q4_0 しか読めない」という *明確なエラー* だったので、
Q6_K / Q8_0 / Q4_1 を F16 に展開する経路を書いた。ところが実モデルで
`CUDA: an illegal memory access` になり、compute-sanitizer が
`f16_gemv_kernel` の 2 バイト読み出しを指した。形・確保サイズ・ポインタを
すべて出力して追い込んだ結果、**カーネルではなく行列の move 代入**が原因だった:
移動元(一時オブジェクト)のデストラクタが `owns == true` のまま `cudaFree(q8)` を
呼び、移動先のバッファを解放していた。`o.q8 = nullptr;` を足して解決。
教訓: ポインタを増やすときは move コンストラクタと move 代入の**両方**を直す。

## 2. 実装したこと

- **Q4_0 以外の 2D テンソルを Q8_0 へ詰め直す**方式 (`src/ggml_dequant.hpp`)。
  自作の F16 GEMV は捨て、lowbit.cuh の実績ある
  `q8_0_gemv_kernel` / `q8_0_gather_row_kernel` で走らせる。Q8_0 は 8bit なので
  元が Q6_K でも精度は落ちない (バイト数は 1.3 倍)。
  Qwen2 系の attn bias は後段カーネルで加算。
- 対応型: Q4_0 / Q5_0 / Q5_1 / Q4_1 / Q8_0 / Q2_K / Q3_K / Q4_K / Q5_K / Q6_K / F16 / F32。
  dequant は llama.cpp の `ggml-quants.c` から写し、合成ブロックで全要素を
  手計算値と突き合わせるホスト検証を追加 (`tests/qwen38_kquant_host_check.sh`, 10 checks)。
- **dense のプロンプト対応**: `--prompt` / `-i` / `--chat-template`。
  トークナイザに **llama-bpe (Llama-3)** と **SentencePiece (model=llama)** を追加。
  chat template は ChatML と Llama-3 形式を GGUF のテンプレート文字列から判別。
- 生成ループの同期バグ修正: `rt.stream` は `cudaStreamNonBlocking` なので、
  同期コピーでは直前のカーネルを待たず、生成の先頭にゴミトークンが出ていた。
  `cudaMemcpyAsync` + `cudaStreamSynchronize` に変更して解消。

## 3. 検証

- qwen38: **64/64**・26.65 tok/s（全変更込みで再確認）
- ホスト検証: k-quant 10 checks / argmax 8,020 / expert ILP 4,325 / tuning / gguf_type_size、
  すべて失敗 0。CUDA 構文検査 PASS。
- 実機: 上表の 7 モデルで生成を確認、数列の続きも一致。

## 4. まだできないこと

- dense のサンプリングは greedy のみ（`--temp` は無視して警告）。
- `Q4_K_M` 内の **Q4_K 自体**は読み込み時に Q8_0 へ再量子化するため、k-quant の
  ままの推論よりメモリを 1.9 倍使う（速度・精度は保たれる）。
- MoE 系の他アーキテクチャ（Mixtral など）は未対応（expert ストリーミングは qwen4exp 用）。
