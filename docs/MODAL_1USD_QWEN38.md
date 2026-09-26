# Modal $1 Qwen3.8-Flash-Next tuning run

目的は長時間ベンチではなく、WarpDecodeのQSA / PLE / MoE / MTPパラメータを短時間で絞ること。

## GPU時間をモデル取得に使わない

GGUFは先にCPU-only jobかローカルからModal Volumeへ保存する。GPU Function起動後にHugging Faceから100GB級ファイルを落とさない。

Modal Volumeは `modal.Volume.from_name(..., create_if_missing=True)` で永続化できる。

## 第一候補

**A100 80GB**。H100より安く、80GB VRAMで約1.6倍長く試せるため、最初の探索向き。最終的な絶対速度記録はH100/H200で別に取る。

料金は変わるので実行直前にModal Pricingを確認する。2026-09-26時点では:

- A100 80GB: $0.000694/s → $1でGPU約1441秒（約24分）
- H100: $0.001097/s → $1で約912秒（約15.2分）
- H200: $0.001261/s → $1で約793秒（約13.2分）
- L40S: $0.000542/s → $1で約1845秒（約30.8分、ただし48GB）

CPU/RAM料金も別にかかるので、GPU時間はこれより少なめに切る。

## 24分の使い方

A100 80GBを使う場合:

1. 0-4分: load + correctness smoke
2. 4-9分: MTP draft 2 / 3 / 4を同じpromptで短く測る
3. 9-14分: PLE prefetch 32 / 48 / 64 rows
4. 14-19分: expert cache 4 / 8 / 12 GiB
5. 19-23分: 勝った組み合わせを3回反復
6. 最後1分: JSON/NVMLログ保存

QSAの512 blocks / 2048 tokensは意味論なので探索で変更しない。変えるのはblock-key cache、compact gather、stagingだけ。

## MTP score

acceptance率ではなく:

```text
reward = accepted_target_tokens / gpu_microseconds
```

を最大化する。WarpDecodeの `AdaptiveMtpController` は最初に2/3/4を1回ずつ測り、その後はEWMA reward最大のdraft長を選ぶ。

## 記録必須

```text
effective tok/s
raw target tok/s
draft n
accepted tokens / drafted tokens
verify us
QSA index us
QSA gather us + bytes/token
PLE rows requested / unique / coalesced reads
PLE bytes/token
expert cache hit rate
expert H2D bytes/token
GPU power / clocks / memory use
```

同じprompt、同じseed、同じcontextで比較する。
