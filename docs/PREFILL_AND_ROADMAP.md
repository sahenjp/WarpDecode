# Prefill 設計と実装ロードマップ

## 1. 現状

| 項目 | 状態 |
|---|---|
| Decode (batch=1, 1トークン生成) | 実装済み。64/64 で CPU オラクルと一致 |
| Prefill (プロンプト処理) | **専用経路なし**。--forced-input-tokens で1トークンずつ処理する = decode の繰り返し |
| カーネル | すべて batch=1 前提 (GEMM ではなく GEMV) |
| MoE | トークンごとに expert を選択し、重みを expert cache から引く |
| 計測 | --generate N の合計時間のみ。prefill と decode の分離なし |

## 2. Prefill が遅い理由

1. **GEMV しか無い**: プロンプト N トークンを1つずつ処理するので重み行列を N 回読む。本来は1回の GEMM で済む。
2. **MoE の重み転送がトークンごと**: 各トークンで別の expert 集合を引くため、cache ミスがトークン数だけ発生する。
3. **LM head を毎トークン計算**: prefill 中は最後の1トークン以外 logits 不要だが、現状は全トークンで argmax まで計算する。
4. **attention も逐次**: causal attention をトークンごとに計算している。

## 3. 段階的な実装計画

### 段階 1 (小さい、1日で可能): prefill 専用の無駄取り
- **LM head のスキップ**: prefill 中は最終トークン以外 head/argmax を回さない (語彙 151k x 2560 の Q6_K GEMV で1ステップの数%〜10%)
- **層ごとの expert prefetch**: 次層で使う expert を先に読む (I/O と計算のオーバーラップ)
- **期待効果**: 1.2〜1.4倍 (2〜4倍には届かない)

### 段階 2 (中、数日): バッチ化した prefill カーネル
- 射影 (QKV / gate-up / down / HC / PLE) を **バッチ GEMM** に置き換え (N=64〜512 トークンをまとめる。重み読み出しが 1/N になる)
- MoE はトークンごとに expert が違うため、**トークンを expert ごとにグループ化**して1回の GEMM で処理する (sorted grouping)
- attention をブロック単位の batched attention に置き換え
- **期待効果**: 2〜4倍 (重み読み出しが支配的なのでこれが本命)

### 段階 3 (大、1〜2週): prefill の本格最適化
- FlashAttention 相当の batched attention
- MoE の grouped GEMM + 重みの常駐化
- テンソルコア (sm80/sm89 の INT8/INT4 MMA) でのバッチ GEMM
- chunked prefill (prefill と decode の混在)

## 4. 実装上の制約

- **64/64 の parity を維持すること**。prefill のカーネル数値が変わると decode の初期状態が変わり parity が壊れる。
  段階 2 のバッチ GEMM は「整数部は同一、スケール適用順も同一」にすれば bit 一致を保てる (int32 の加算は結合則が厳密)。
- 段階 1 は数値に触れないので安全。

## 5. 作業見積もり

| 実装内容 | 推定期間 | 優先度 | 今回の実施 |
|---|---|---|---|
| Prefill・Decode の個別ベンチマーク | 0.5日 | 高 | 実施 |
| A100 向け Decode カーネル最適化 (DP4A/Tensor Core) | 2〜4日 | 高 | DP4A 経路を試作・計測 |
| MoE キャッシュ・転送最適化 | 1〜2日 | 高 | 実施 |
| MTP 投機的デコードの実装 | 3〜5日 | 中〜高 | 実装・計測 |
| Prefill 専用の並列演算最適化 | 1〜2週 | 中 | 設計 + 段階1 |
| 回帰テスト・比較検証・公開準備 | 1〜2日 | 高 | 実施 |

## 6. 目標と確度

| 対象 | 目標 | 確度 | 根拠 |
|---|---|---|---|
| Decode | 現状比 1.5〜2倍 | 要検証 | expert cache を 1536→8000MiB にするだけで L4 が 3.195→3.893 (1.22倍)。DP4A と合わせれば届く見込み |
| MTP | Decode を追加で 1.3〜2倍 | 採用率次第 | draft k=2〜4 で受理率 2トークン/ステップ前後なら 1.5〜2.5倍。未検証 |
| Prefill | 現状比 2〜4倍 | 段階2 の実装次第 | 段階1 は 1.2〜1.4倍。2〜4倍はバッチ GEMM が必要 |
| CPU オラクル一致 | 64/64 を維持 | 必須 | 各変更後に Colab ランナーで確認 |
