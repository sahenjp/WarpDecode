# Modal benchmark

Qwen3.8-Flash-Nextの実測用。モデル取得はCPU Function、ベンチだけGPU Functionへ分離してGPU代をモデルDLに使わない。

## 1. Modal認証

```bash
python -m pip install -U modal
modal setup
# または modal token set
```

## 2. モデルをVolumeへ取得

```bash
modal run modal/bench_flash_next.py --action prefetch
```

既定:
`pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF`

## 3. A100 80GBで実測

```bash
modal run modal/bench_flash_next.py \
  --action bench \
  --gpu A100-80GB \
  --depths 0,8192,32768
```

GPU Functionのtimeoutは1500秒。llama.cppはcommit `4e7481175cbd4759df8bee2f1c1a0073effbebd7` をimage build時にSM80 CUDAでコンパイルする。同時にWarpDecodeもSM80でコンパイルし、CUDAコンパイルエラーをGPU起動前に検出する。

現時点のWarpDecodeはqwen4exp forward未配線なので、このrunの目的は:
- Modal上でWarpDecode SM80 CUDA buildを実証
- pentacoxian GGUFのplanner認識を確認
- 同一A100上のupstream llama.cpp TG baselineをdepth 0/8K/32Kで取得
- 次のQSA/MoE/MTP実装の比較基準を固定

qwen4exp forwardが通った時点で同じrunへWarpDecode本体のA/Bを追加する。
