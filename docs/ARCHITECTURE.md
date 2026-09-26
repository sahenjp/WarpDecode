# アーキテクチャ

目的は汎用推論フレームワークではなく、NVIDIA GPU上のbatch=1 decode hot pathを短くすること。

## 層

```text
warpdecode-tui (C++17 / CUDA不要)
  ├─ GPU検出
  ├─ GGUF検出
  ├─ 世代別ビルド
  ├─ 単体起動
  └─ GPUごとの独立複数起動

warpdecode engine (C++/CUDA)
  ├─ mmap GGUF loader
  ├─ load-time Q4 prepack
  ├─ architecture dispatch
  ├─ Q4 autotune
  ├─ fused-ish decode pipeline
  └─ CUDA Graph replay
```

## 削るもの

CPU推論、AMD、Metal、Vulkan、LoRA、画像、音声、学習、量子化器、汎用演算グラフ、continuous batchingをhot path本体に入れない。

## 世代別dispatch

共通Q4 SIMTはbring-up/fallback専用。最終hot pathは別物にする。

- SM50: SIMT Q4
- SM60: HBM帯域を使うhalf2 path
- SM61: activation Q8 + DP4A
- SM70: packed low-bit weightをregisterで展開しFP16 `mma.sync.m8n8k4`
- SM75: Turing固有のINT8 MMA。Ampereと同じlaunch tableを流用しない
- SM80/86: INT4/INT8 Tensor Core。M=1 GEMVとM=4〜8 verifierを分離
- SM89: Ada専用tile/autotune
- SM90: Hopper専用Tensor Core / attention
- SM100/103/120: Blackwellのデータセンター系とGeForce系を分離

## いま入っている高速化

- Q4_0 planar prepack
- 1 warp = 2 output rowsのQ4 GEMV
- QKV連結
- Gate/Up連結
- attention score bufferなしのonline softmax
- GPU内argmax
- CUDA Graph
- 代表Q4 GEMV/logitsを実機で64/128/256/512 threadsから選ぶautotune

## 次の性能実装順

1. SM61 DP4A Q4×Q8 fused RMSNorm path
2. SM70 Volta MMA low-bit path
3. SM75専用MMQ tile table
4. SM80/86 Marlin系weight layout + small-GEMM verifier
5. MTP / speculative verification M=4〜8
6. Hopper / Blackwell専用カーネル
7. P2P/NVLink-aware multi-GPU

単にCUTLASS/BitBLAS等へ全面依存はしない。参照実装と探索器として使い、最終hot pathは依存・dispatch・メモリ配置を削れる場合に専用カーネルへ固定する。
