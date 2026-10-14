# Colab セッションの割り当てと排他ルール

## セッション一覧（L4 は容量制限で3つまで）

| セッション | GPU | 割り当て | モデル |
|---|---|---|---|
| warpdecode-l4 | L4 24GB | Task A (prefill/decode ベンチ) | ダウンロード済み |
| warpdecode-a100 | A100-SXM4-40GB | Task C (expert cache A/B) | ダウンロード済み |
| warpdecode-l4b | L4 24GB | Task F (時間内訳の計測) → その後 Task G | ダウンロード済み |

※ warpdecode-t4 は停止済み（L4 の追加作成は容量制限で拒否された）

## 排他ルール (重要)

1. **1セッションにつき同時1ジョブ**。複数タスクが同じセッションを使うと GPU とディスクが競合し、
   `tok_s` が汚染される（実測で 1.08 tok/s のような誤った値が出た）。
2. **ログは自分専用のファイル名にする**。`/content/run_bg.log` を使い回すと他タスクに上書きされる。
   例: `/content/run_<label>.log`
3. 実行前に `/content/run.lock` などで排他する（存在したら待つ）。
4. 計測は **1セッションで1回にまとめて**流し切る。
5. セッションが埋まっている場合は **待つ**。無理に同じセッションで並行実行しない。

## 既知の落とし穴

- bash では zsh 構文 `${(SECONDS - t0)}` は使えない（`bad substitution` でその行以降が実行されない）。
  `$(($SECONDS - $t0))` を使うこと。RUN_EXIT=0 でも途中終了に気づきにくいので各ケースの完了をログで確認する。
- expert cache の既定 1536 MiB は A100 では小さすぎる（I/O 律速で 1.08 tok/s）。
  20000〜30000 MiB で 3.5 tok/s 前後、L4 は 8000 MiB が上限近辺（3.89 tok/s）。
- 他プロセスが VRAM を使っていると計測が汚染される（A100 20GiB 構成の 1.926 tok/s はこれが原因）。
  計測前に `nvidia-smi` で自分以外のプロセスが無いことを確認する。

## 実行コマンドの形

```bash
export PATH=/home/sahen/.local/bin:$PATH
# L4
WARODECODE_COLAB_SESSION=warpdecode-l4 python3 tools/colab_run_bg.py --label <name> --generate 64
# A100
WARODECODE_COLAB_SESSION=warpdecode-a100 WARODECODE_COLAB_ARCH=80 python3 tools/colab_run_bg.py --label <name> --generate 64
# スイッチ
  --switches "WARODECODE_EXPERT_CACHE_MIB=8000,WARODECODE_DP4A=1"
```

## 現在の速度（実測・parity はすべて 64/64）

| 構成 | 速度 |
|---|---|
| L4 1536MiB (既定) | 3.195 tok/s |
| L4 8000MiB | 3.893 tok/s |
| L4 8000MiB + DP4A+smem | **4.128 tok/s** |
| A100 30000MiB | 3.517 tok/s |
| A100 30000MiB + DP4A+smem | **3.739 tok/s** |

## セッションが消える件（重要）

Colab は**使っていないセッションを自動回収する**ことがある。実際に 17:47 時点で `warpdecode-l4b` が
一覧から消えていた（停止コマンドは実行していない。`warpdecode-t4` は容量確保のため意図的に停止）。
再作成すると **VM が作り直されるため `/content/models` の 85.9GB を再ダウンロードする必要がある**（約5分）。

対策:
- 計測は**1回のセッションでまとめて流し切る**（途中で回収されるとモデルDLからやり直し）
- 長時間アイドルにしない（`colab exec` で定期的に触る）
- モデルDLは `hf download ... > /content/download.log 2>&1 &` を nohup で開始し、`du -sh /content/models` で進捗確認

## 現在のセッション（2026-10-14 17:50 時点）

| セッション | GPU | 割り当て |
|---|---|---|
| warpdecode-l4 | L4 24GB | Task A (bench) |
| warpdecode-a100 | A100 40GB | 空き（Task C 完了後） |
| warpdecode-l4b | L4 24GB | Task F（再作成済み・モデルDL中） |

未着手の担当: G (expert 融合), H (expert I/O 帯域) → いずれ l4b / a100 を分けて使うこと。

## 正式な割り当て（2026-10-14 18:05 更新・これが最新）

| セッション | GPU | 担当 | 状態 |
|---|---|---|---|
| warpdecode-l4 | L4 24GB | **Task G (expert 融合)** | Task A 完了により空き |
| warpdecode-a100 | A100 40GB | **Task H (expert I/O 帯域)** | 空き（Lead 統合検証は完了） |
| warpdecode-l4b | L4 24GB | **Task B (MTP)** | 使用中 |
| warpdecode-l4f | L4 24GB | **Task F (時間内訳)** | 使用中 |

**完了した担当**: Task A (ベンチ)、Task C (expert cache)、Task D (DP4A)。

### ルール（再掲・厳守）
1. 上表の担当以外は**そのセッションを使わない**。
2. ログと作業ディレクトリは自分専用にする（例: Task H は `/content/wd_h/...`）。
   `/content/run_bg.log` と `/content/warodecode` の共有は上書き事故のもと。
3. VRAM 競合を避ける: 計測前に `nvidia-smi` で他プロセスが消えるのを待つ。
4. セッションは Colab に自動回収されることがある。回収されたら**モデル 85.9GB の再DLが必要**（約5分）。
   長い計測は1回で流し切る。

### 判明している競合の実例
- 17:13 Task A と Task C が L4/A100 を同時使用 → 両者の run_bg.log が上書きされ計測が無効化。
- 17:45-18:00 3セッションすべてが別担当に使用され、Task H の smoke が失敗。
- Task C の行列スクリプトが zsh 構文 `${(SECONDS - t0)}` で bash では 1 ケース目で落ちていた。

### 参考: 確定した実測値（すべて parity 64/64）

| 構成 | decode | prefill |
|---|---|---|
| L4 expert cache 1536MiB | 3.305 tok/s | 3.410 tok/s |
| L4 expert cache 8000MiB | **4.107 tok/s** | **4.626 tok/s** |
| L4 8000MiB + DP4A+smem | **4.128 tok/s** | — |
| A100 20000-30000MiB + DP4A+smem | 3.739 tok/s | — |
| **統合ツリー（全担当の変更込み）/ A100 20000MiB** | **64/64 一致を確認** | — |

## 割り当て更新（2026-10-14 18:45・これが最新）

| セッション | GPU | 担当 |
|---|---|---|
| warpdecode-l4 | L4 24GB | **Task J (HC mix 最適化)** ← A/G 完了で空き |
| warpdecode-l4b | L4 24GB | **Task I (M トークン同時 verify)** ← B 完了で空き |
| warpdecode-l4f | L4 24GB | Task F (時間内訳) |
| warpdecode-a100 | A100 40GB | Task H (expert I/O 帯域) |

**完了**: A (ベンチ), B (MTP 実装), C (expert cache), D (DP4A), G (expert 融合)

### 確定したボトルネック（A100 1トークン ≈272ms）

| 区間 | 時間 | 割合 | 中身 |
|---|---|---|---|
| MoE | 141〜192 ms | 52〜64% | GEMV は 17ms。残りは expert 重みの staging 転送 |
| HC mix (attn+ffn) | 80 ms | 30% | 帯域期待値より1〜2桁遅い |
| attention | 44 ms | 16% | GDN36 + QSA12 |
| head | 1.3 ms | 0.5% | 帯域律速で最適済み |

実効帯域 16 GB/s（HBM 1.9 TB/s）= 起動・同期・転送のレイテンシ律速。

### 否定的結果（重要）
- MTP は batch=1 verify では -3.8〜6.2%（原理的にプラスにならない）。M トークン同時 verify が必須。
- expert 融合（60→5 launch/層）は +0.55%。launch 削減は上限近く。
- 出力ヘッドは 189 GiB/s で帯域律速。最適化対象ではない。

## 割り当て更新（2026-10-14 20:35・Task L）

| セッション | GPU | 担当 |
|---|---|---|
| warpdecode-a100 | A100 40GB | **Task L (llama.cpp 直接 A/B)** |
| warpdecode-l4b | L4 24GB | 空き確認のみ（未使用） |

Task L は a100 で llama.cpp を CUDA sm80 ビルドし、同一モデル・同一条件で
本エンジンと交互に decode を計測する。作業ディレクトリは `/content/wd_taskl`（専用）。
他担当は a100 を使わないこと。

## 2026-10-14 19:50 時点の状態

Colab のセッション回収が頻繁に起きている。現在の生存セッション:

| セッション | GPU | 状態 |
|---|---|---|
| warpdecode-a100 | A100 40GB | 再作成済み・モデルDL中 |
| warpdecode-l4b | L4 24GB | 生存 |

warpdecode-l4 / l4f / t4 は回収された。再作成のたびに 85.9GB のモデルDLが必要（約5分）。

### 共有ツリーのビルドが壊れた件（修正済み）
- `src/qwen38_forward.cuh:2175` が `q38_hc_f32_inject_enabled`（正しくは `qwen38_hc_f32_inject_enabled`）を
  参照しており nvcc が停止していた。typo を修正し、構文検査 PASS（307 launch 箇所）。
- 教訓: **サブエージェントは編集前に必ず共有ツリーがビルドできるか確認する**こと。
  壊れている場合は `git stash` ではなく、自分の変更だけを凍結ツリー（`WARODECODE_SOURCE_DIR`）で試す。

### Task I の parity リスク（要注意）
- `WARODECODE_MTP=1 + MTP_BATCH_VERIFY=1 + MOE_BATCH_K=4` の実行で **matched_prefix=41**（64/64 でない）。
  バッチ verify が MoE の数値を変えている疑い。既定スイッチは OFF のままだが、
  ON にする前に必ず原因を特定して 64/64 に戻すこと。

## 2026-10-14 20:25 割り当て（公開後の追加タスク）

**リポジトリは公開済み**（`git push origin main` → 5933825）。

| セッション | GPU | 担当 |
|---|---|---|
| warpdecode-a100 | A100 40GB | **Task L（llama.cpp との A/B）** |
| warpdecode-l4b | L4 24GB | **Task M（トークナイザ + ストリーミング CLI）** |

※ セッションは2つしか生存していない。回収されたら `colab new -s <name> --gpu A100|L4` で再作成し、
  `hf download pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf --local-dir /content/models`
  でモデルを再取得する（85.9GB・約5分）。

### 完了済みタスク（この日の全成果）
| タスク | 内容 | 主な結果 |
|---|---|---|
| Issue #1 | CPU オラクルと 64/64 一致 | **達成**（修正10系統） |
| A | prefill/decode 分離ベンチ | decode 4.107 / prefill 4.626 tok/s |
| B | MTP 投機的デコード実装 | 受理率62.5%、batch=1 verify では -3.8〜6.2% |
| C | expert cache 分析 + copy stream | 混雑時 +33%、レバーは容量 |
| D | DP4A + コードブック smem | +6%（IQ2_S カーネル 5.1倍） |
| F | トークン時間の内訳 | MoE 44% / hc_mix 30% / attn 23% |
| G | expert 融合（60→5 launch/層） | +0.55%（launch は支配的でない） |
| H | expert 転送の pread+ring+DMA | **3.7 → 14〜17 GiB/s、+21〜47%** |
| J | HC mix のカーネル改善 | **80.5 → 37.1 ms（2.17倍）、A100 +25%** |
| K | expert 先読みパイプライン | 実装・45テスト PASS、効果なし（予測不能）→ 既定OFF |

### 最終速度（すべて parity 64/64）
| 構成 | decode |
|---|---|
| L4 8000MiB | 5.287 tok/s |
| **A100 16000MiB** | **5.607 tok/s** |
