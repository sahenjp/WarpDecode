# Qwen3.8-Flash-Next optimization target

warodecodeでは `general.architecture=qwen4exp` を汎用dense経路へ無理に流さない。Qwen3.8-Flash-Next専用runtimeとして扱う。

## model contract

公式checkpointを基準にする。

- 125B backbone / 6B active per token
- 51B級 n-gram / PLE table（accelerator外が前提）
- 4B MTP
- hidden 2560
- 48 decoder layers
- 36 Gated DeltaNet + 12 Qwen Sparse Attention
- 4 residual streams
- 512 routed experts、tokenあたり10 routed + 1 shared
- QSA budget: 512 micro-block / 2048 token

GGUF側のarchitecture名は `qwen4exp`。

## warodecodeが狙う差

### 1. QSA decodeをcontext長に比例させない

やらない:

```text
毎token
  全K cacheを再pool
  ↓
  dense maskを作る
  ↓
  full-length attention
```

狙う:

```text
4 tokenでblock完成
  ↓ once
mean-pool + RMSNorm + RoPE
  ↓
persistent block-key cache

毎decode token
query indexer
  ↓
TOP-K 512 blocks
  ↓
selected KVをcompact gather
  ↓
2048 token以下だけattention
```

QSA indexerのtop-kもCPUへ落とさない。CUDA上に常駐させる。

### 2. PLE / n-gram tableをGPU VRAMへ置かない

PLEは巨大だが1 tokenあたり触るrowは非常に少ない。全表をVRAMへ載せるのは帯域・容量の使い方として悪い。

warodecodeのpath:

```text
token ids
  ↓ deterministic hash
16前後のrow ids
  ↓
sort + dedup
  ↓
adjacent rowをcoalesce
  ↓
direct pread / host-RAM read
  ↓
pinned staging + async H2D
  ↓
PLE projection
```

`src/ple_reader.hpp` に、mmap page faultを避けるためのdedup + contiguous-run direct readerを追加済み。次はio_uring/pread workerとhot-row cacheを足す。

### 3. MoEはexpert cacheをruntimeの中心にする

512 expertを毎token読む必要はない。router確定後に10 routed + 1 sharedだけを実行する。

VRAMが足りる場合:

```text
all experts resident
  ↓
router ids
  ↓
grouped low-bit GEMV
```

VRAMが足りない場合:

```text
persistent expert slot pool
  ├─ hot expert hit -> H2Dなし
  └─ miss -> async staging / replacement

router -> slot remap -> grouped low-bit kernel
```

単純LRUだけでなく、直近routing frequency + transition probabilityでprefetch候補を決める。

### 4. GDN stateをtoken間で再構築しない

36 GDN layerのrecurrent/conv stateをGPU上にpersistentに持つ。batch=1ではstate layoutを固定し、projection + norm + recurrent updateのlaunchを可能な限りfusionする。

### 5. 4 residual streamsを一度のread/writeへ寄せる

Gated Residualは4 streamを毎layerで読む。個別kernelを重ねず、read gate / projection / scatterを同一epilogueへ寄せ、VRAM trafficを抑える。

### 6. MTPを後付け機能にしない

MTPなしとMTPありを別runtimeとして扱う。

```text
M=1 target decode kernel
M=2 verify kernel
M=4 verify kernel
```

`draft_n`は固定しない。2〜4を基本探索範囲として、

```text
accepted target tokens / total GPU time
```

が最大になる値を選ぶ。acceptance率だけで選ばない。

Qwen3.8-Flash-NextのMTPはrecurrent stateを持つので、reject rollback時にtargetとdraftのGDN/QSA stateを混ぜない。single-streamを先に完成させ、multi-slotは後回しにする。

## quantization strategy

pentacoxian-devの `IQ3E-Q8D-MTP` のように、targetとdraftを同一量子化に固定しない。

- target: 3〜4bit級を帯域優先で探索
- MTP: Q8級/高精度寄りを含めてacceptanceとdraft速度の積で決定
- PLE: table lookupなので独立したrow quantを許可
- recurrent state / QSA index cache: 最初はFP16 correctness優先。後でFP8/Q8を別ゲートで検証

特定のIQ3E/Q8Dフォーマット自体を現時点で実装済みとはしない。GGUF block layoutを確認してから専用loader/kernelを追加する。

## benchmark gates

比較相手はllama.cppだけではない。

- upstream llama.cpp qwen4exp CUDA
- MTP対応llama.cpp派生実装
- Qwen3.8-Flash-Next専用C/CUDA runtime
- NVIDIA/vLLM系（同じGPU/量子化条件で比較可能な場合）

記録:

```text
pp tok/s
raw target tg tok/s
MTP effective tok/s
draft tok/s
verify tok/s
acceptance by draft position
mean accepted length
QSA top-k time
gathered KV bytes/token
expert cache hit rate
expert H2D bytes/token
PLE read amplification
PLE IOPS / bytes/token
HBM GB/s
kernel launches/token
```

勝ち判定は同じGPU・同じモデル品質条件で最低5%以上。出力が壊れている高速化は失格。

## RTX 3070 first target

手元で実機検証しやすいSM86を最初のQwen3.8 hot pathにする。

1. split GGUF / qwen4exp metadata planner
2. PLE direct reader + staging
3. GDN persistent state
4. QSA block cache + compact gather
5. MoE grouped Q4/IQ3-class kernel
6. expert slot cache
7. detached/shared MTP loader
8. M=2/M=4 verifier
9. adaptive draft controller
10. llama.cpp / specialized runtime A/B

V100は同じruntime構造を使い、低bit GEMV/verify部分だけSM70 MMAへ差し替える。
