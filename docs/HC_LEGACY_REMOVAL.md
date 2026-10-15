# Task R: HC mix の legacy 重複を attn サイトからも除去

対象: `src/qwen38_forward.cuh` / `src/qwen38_hc.cuh` (+ tests / docs)。
絶対条件は parity 64/64、目標は attn + hc_mix 56.2 → 44 ms 以下 (Task P と同じ
プロファイラ付き計測の慣習)。

---

## 1. 結論

| | base (SKIP_LEGACY=0) | 出荷既定 (両サイト skip) | 差 |
|---|---:|---:|---:|
| attn + hc_mix (profiler 付き) | **56.07 ms** | **46.89 ms** | **-9.18 (-16.4%)** |
| total (profiler 付き) | 166.48 ms | 157.42 ms | -9.06 |
| tok_s (profiler 付き) | 5.98 | 6.33 | +5.8% |
| total (profiler なし) | 150.02 ms | 144.30 ms | -5.72 |
| attn + hc_mix (profiler なし) | 40.15 ms | 34.40 ms | -5.75 |
| tok_s (profiler なし) | 6.67 | 6.92 | +3.8% |
| parity | 64/64 | **64/64** | — |

- 重複していた legacy F16 mix は **両サイト 48 層ずつ、96 calls/token → 0**。
- **目標 44 ms には未達 (46.89)**。理由は 5.4 で定量化したとおり、
  消したパスが exact リプレイの L2 ウォーマーも兼ねていたため
  (名目 12.4 ms のうち回収できたのは 9.2 ms)。
- 既定は ON (`WARODECODE_HC_SKIP_LEGACY=1`)。per-site スイッチで片側ずつ A/B できる。

---

## 2. 重複の確認（まずここから）

Task P が残した A100 の profile (`issue1_artifacts/taskp_both.log`, 64 token,
`WARODECODE_HC_PROFILE=1`) の `[hc-profile]` は、**同じ層で両経路が走っている**
ことを呼び出し回数で示していた。

| site | ms/token | calls/token | 経路 |
|---|---:|---:|---|
| attn.f16_norm / down / silu / up / mix / inject | 6.20 (合計) | **48.00** | legacy F16 mix |
| attn.q8_norm / quant_norm / down / down_copy / silu / quant_low / up / mix | 4.85 (合計) | 48〜49.00 | exact Q8_0 mix |
| attn.q8_inject + attn.inject_copy | 3.86 | **48.00** | exact の F32 inject |
| ffn.f16_* | 6.22 | **48.00** | legacy F16 mix |
| ffn.q8_* | 4.83 | 48.00 | exact Q8_0 mix |

層数は 48 (GDN 36 + QSA 12)。attn 側の q8_* が 49 calls/token なのは、ヘッドの
HC mix (`run_qwen38_hc_mix_f32`) が `base_site` を省略していて attn バンクに
計上されるため (head は legacy を走らせないので f16_* は 48 のまま)。

決定的なのは **`attn.inject_copy` がちょうど 48** であること。
`kQ38HcMixInjectCopy` を出すのは `run_qwen38_hc_mix_f32_with_inject()` だけなので、
48 層すべてで **exact の F32-inject 経路が成功している** = `attn_f32_mix_active` が
48 層で true。したがって attn サイトの流れは次のとおりだった:

1. `attn_f32_mix_active == true` なので `run_qwen38_hc_mix_f32()` は呼ばれない。
2. しかしこの時点で `cur == nullptr` なので、**legacy `run_qwen38_hc_mix()` が走る**
   (`if (cur == nullptr)` にガードが無い)。
3. 直後に `run_qwen38_hc_mix_f32_with_inject()` が走り、
   `st.hc_attn.mixed()` と `st.hc_attn.inject()` を**上書き**する。

2 の結果は誰も読まない。combine は `run_qwen38_hc_combine_f32_inject()`
(F32 scratch の unrounded inject) を使うので legacy の half inject も使われない。
**attn 6.2 ms/token は完全な捨て仕事**だった。

### 2.1 Task P doc の仮説は外れていた

`docs/TASK_P_ATTENTION.md` 8.1 は「`run_qwen38_hc_mix_f32()` が inject 量子化で
全計算後に nullptr を返す」と推測していたが、実測は逆で **inject は F32 で device に
常駐している** (`attn.q8_inject` が毎層動いている)。捨てられる側が違うだけで、
「同じ行を 2 回計算している」という結論は同じ。

### 2.2 ffn サイト: Task J のガードは一度も発火していなかった

`ffn.f16_*` が 48 calls/token 残っているのに Task J の `ffn_exact_mix_active` が
入っている。理由は条件の向き:

```cpp
debug_hc_q8_0_ffn = (CLI range) || (q38_hc_q8_0_enabled() && q38_hc_q8_0_compatible(layer.hc_ffn));
ffn_exact_mix_active = q38_hc_q8_0_enabled() && !debug_hc_q8_0_ffn && ...;
```

`debug_hc_q8_0_ffn` は **exact 経路を選ぶ述語** (既定 ON の compat 項を含む) なので、
exact が有効なモデルでは常に true → `!debug_hc_q8_0_ffn` は常に false。
ガードが発火するのは「exact を走らせない層」だけだった。

---

## 3. 実装

### 3.1 共通の述語 (`src/qwen38_hc.cuh`)

```cpp
inline bool q38_hc_legacy_mix_skippable(
        bool skip_switch_enabled,
        bool exact_mix_active,      // exact Q8_0 replay がこの層で走る
        bool exact_writes_inject,   // その replay が非 nullptr 分岐 (half inject も書く)
        bool half_mirror_consumed); // trace/debug が xn / gate / lo を読む
```

skip できるのは 4 つすべてが揃ったときだけ。とくに 3 番目を要求するのは、
exact 側が nullptr を返したときのフォールバック
(`run_qwen38_hc_q8_0_f32_mix_from_residual()` + F32→half コピー) が
**half inject を書かない**ため。そこで skip すると combine が前の層の値を読む。

### 3.2 呼び出し側 (`src/qwen38_forward.cuh`)

- attn サイト: `attn_f32_mix_active` を exact の活動条件として渡し、
  legacy 呼び出しを `if (!attn_skip_legacy && cur == nullptr)` に変更。
- ffn サイト: Task J の `!debug_hc_q8_0_ffn` を **`debug_hc_q8_0_ffn` に直し**、
  同じ述語に通した (`if (!ffn_used_f32 && !ffn_skip_legacy)`)。
- half mirror を読む消費者が有効なときは skip しない。判定は
  `trace != nullptr || debug_f32_hc_attn` (ffn は `debug_f32_hc_ffn`)。
  legacy だけが書く `st.hc_*.xn / gate / lo` の読み手は
  `trace->state_trace` / `legacy_trace` / `trace->extended_parity` と
  `debug_f32_hc_*{,_0_8,_layer8}` / `debug_hc_q8_1_*` / `debug_hc_gate_q8_0` /
  `debug_hc_full_q8_0` / `debug_hc_norm_f32` だけで、前者 3 つは
  `trace != nullptr` に、後者は `debug_f32_hc_*` に含まれる。
  decoder は `capture ? trace : nullptr` を渡すので、**trace を読む位置だけ**
  legacy が残る (5.3 の実測がそのとおりになっている)。
- fail-closed: `skip == true` なのに `*_f32_inject == false` になったら例外を投げる。
  「速くなったが数値が違う」を silent に起こさないための保険。

### 3.3 環境変数

| 変数 | 既定 | 意味 |
|---|---|---|
| `WARODECODE_HC_SKIP_LEGACY` | ON | マスター (両サイト) |
| `WARODECODE_HC_SKIP_LEGACY_ATTN` | マスターに従う | attn バンクだけ A/B |
| `WARODECODE_HC_SKIP_LEGACY_FFN` | マスターに従う | ffn バンクだけ A/B |

起動時に 1 回だけ stderr へ出る:
`warodecode: HC mix Task R switches: skip_legacy_attn=1 skip_legacy_ffn=1`。
tok_s だけでは「効いているつもりで stale なバイナリ」を区別できないため。

---

## 4. ホスト検証

```
bash tests/qwen38_cuda_syntax_check.sh          # PASS (launch sites rewritten: 297)
bash tests/qwen38_hc_legacy_skip_host_check.sh  # PASS (述語 + 構造)
bash tests/qwen38_hc_dp4a_host_check.sh         # PASS (既存、回帰なし)
bash tests/qwen38_gdn_f32_gemv_host_check.sh    # PASS (既存、回帰なし)
```

新しい host check は 2 部構成:

1. `tests/qwen38_hc_legacy_skip_host_test.cpp` — 述語の 16 通りを全部評価し、
   「skip なら 4 つの前提がすべて成立」という安全条件を全行で検査する。
   (skip になるのは 1 行だけ = 誤って速くなる構成が表現できない)
2. シェル側で **実サイトの構造**を検査する: 両サイトが
   `q38_hc_legacy_mix_skippable()` を通っていること、legacy 呼び出しがガードの
   後ろにあり A/B 用に到達可能なこと、fail-closed の 2 箇所が残っていること。

`tests/host_checks.sh` の末尾に組み込み済み。

> 補足: `bash tests/host_checks.sh` 全体は、このワークスペースでは
> `build-ui/CMakeCache.txt` が元の取り込み元パス (`/mnt/c/Users/danda/...`) を
> 指しているため CMake の再構成で失敗する (Task R とは無関係の既存環境要因)。
> そのため関係する host check を個別に実行して確認した。

---

## 5. 実機検証 (A100-SXM4-40GB / sm80 / warpdecode-a100, Colab)

### 5.1 測定条件

- `tests/taskp_attn_ab.py --plan taskr` (HC/ATTN プロファイラ ON) と
  `--plan taskr_tok` (両プロファイラ OFF, stage プロファイラのみ) を新設して実行。
- 64 token greedy decode、`WARODECODE_EXPERT_CACHE_MIB=4096`、ctx 128、
  `--no-graph --no-autotune`。base と both を交互に 2 回ずつ回して drift を確認。
- 実行コマンド:
  ```
  WARODECODE_COLAB_SESSION=warpdecode-a100 WARODECODE_COLAB_ARCH=80 \
    python3 tools/colab_run_cmd.py --label taskr --poll 30 --timeout 3600 \
    --cmd "python3 tests/taskp_attn_ab.py --plan taskr ... && python3 tests/taskp_attn_ab.py --plan taskr_tok ..."
  ```
- 成果物: `issue1_artifacts/taskr_ab.json`, `taskr_tok.json`,
  `colab_log_taskr.txt`, `colab_log_taskr_trace.txt`。
- ソース同一性: 実行ログの `source_sha16` は作業ツリーと 5 ファイルすべて一致
  (`qwen38_hc.cuh d24c96db87391c66`, `qwen38_forward.cuh c75a8c844ad3eae3`,
  `qwen38_gdn.cuh bd74d854c0a0808b`, `qwen38_qsa.cuh efd503f72f5268a1`,
  `lowbit.cuh aed0aa3479c9f742`)。

### 5.2 profile 付き A/B (Task P と同じ慣習, ms/token)

| case | env | tok_s | parity | attn | hc_mix_attn | hc_mix_ffn | attn+hc | total | attn.f16 calls | ffn.f16 calls |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| base-a | SKIP_LEGACY=0 | 5.984 | **64/64** | 19.44 | 18.36 | 18.26 | 56.07 | 166.48 | 48.0 | 48.0 |
| attn-only | SKIP_LEGACY_FFN=0 | 6.167 | **64/64** | 19.36 | 13.75 | 18.22 | 51.33 | 161.59 | **0.0** | 48.0 |
| ffn-only | SKIP_LEGACY_ATTN=0 | 6.164 | **64/64** | 19.46 | 18.39 | 13.69 | 51.54 | 161.65 | 48.0 | **0.0** |
| both (出荷既定) | (なし) | 6.332 | **64/64** | 19.44 | 13.78 | 13.67 | **46.89** | 157.42 | **0.0** | **0.0** |
| base-b | SKIP_LEGACY=0 | 5.990 | **64/64** | 19.46 | 18.39 | 18.29 | 56.14 | 166.32 | 48.0 | 48.0 |

- base-a / base-b は Task P の `both` (attn 19.47/19.37, hc 18.39/18.23,
  total 166.72/166.09, tok_s 5.976/5.998) と一致 = 測定環境が同じ。
- 片側ずつ止めた 2 ケースも 64/64。**混合状態で parity が落ちない**ことは、
  「skip したのに half inject が stale」という最悪ケースが起きていない証拠。
- `q8_down` の呼び出し回数は全ケース 49.0/48.0 で不変 → 触ったのは legacy だけ。

### 5.3 プロファイラなし (壁時計に近い) と trace 経路

| case | tok_s | parity | attn | hc_mix_attn | hc_mix_ffn | attn+hc | total |
|---|---:|---|---:|---:|---:|---:|---:|
| base-1 | 6.662 | 64/64 | 14.844 | 12.744 | 12.698 | 40.29 | 150.02 |
| both-1 | 6.926 | 64/64 | 14.735 | 9.987 | 9.952 | 34.67 | 144.30 |
| base-2 | 6.669 | 64/64 | 14.829 | 12.721 | 12.678 | 40.23 | 149.87 |
| both-2 | 6.916 | 64/64 | 14.718 | 9.980 | 9.941 | 34.64 | 144.52 |

**trace 経路** (`--state-trace --state-trace-position 0 --generate 8`, RUN_EXIT=0,
token ids は参照と一致):

```
warodecode: HC mix Task R switches: skip_legacy_attn=1 skip_legacy_ffn=1
[hc-profile] attn.f16_down   0.4361  calls/token=6.00
[hc-profile] ffn.f16_down    0.4067  calls/token=6.00
```

6.00 calls/token = 8 token で 48 回 = **trace を読む位置 (pos 0) だけで legacy が
走っている**。decoder は `capture ? trace : nullptr` を渡すので、これは
「trace/debug が half mirror を読むときだけ legacy を走らせる」という設計どおり。
`--state-trace-position` を省略 (全位置 trace) した場合は全位置で `trace != nullptr`
になり、従来と同一の経路になる。

### 5.4 何がどこへ消えたか (per-site, ms/token, 各バンク内 6 サイトの合計)

| site | base-a | both | 差 |
|---|---:|---:|---:|
| attn.f16_norm/down/silu/up/mix/inject | 6.193 | 0.000 | **-6.193** |
| ffn.f16_norm/down/silu/up/mix/inject | 6.205 | 0.000 | **-6.205** |
| attn.q8_down | 2.152 | 4.553 | **+2.401** |
| ffn.q8_down | 2.075 | 4.463 | **+2.388** |
| attn.q8_up / ffn.q8_up | 1.272 | 1.546 | +0.274 |
| その他 (q8_norm/quant/down_copy/silu/quant_low/mix/inject, out_copy, inject_copy) | — | — | ±0.02 |

消えた 12.40 ms のうち **4.8 ms は exact 側の q8_down が吸収している**。
なお site 表の差分合計は 1 バンクあたり -6.19 + 2.56 = -3.63 ms で、
stage の差 (-4.58 ms/バンク, 5.2) より小さい。差の分はプロファイラ自身が
1 カーネルにつき 2 個入れていたイベントの GPU 時間で、legacy の 6 サイト × 96 回
ぶんが消えたことによる。**つまり profile 付き A/B (-9.18 ms) は saving を
大きめに見せており、プロファイラなしの -5.75 ms/token が実効値**
(tok_s +3.8% と一致する)。
両パスは同じ Q8_0 重み (down/up で 1 サイト 1 層あたり約 28 MB) を読むので、
legacy パスが残した L2 ヒットがなくなると exact 側が DRAM から読み直すため、
と解釈できる (L2 カウンタは取っていないので**推定**。attn/ffn で増分が
2.40/2.39 とほぼ同一なのが傍証)。

### 5.5 目標に対する評価

- Task P と同じ慣習での目標「attn + hc_mix ≤ 44 ms」は **46.89 ms で未達**。
  `-16.4%` (56.07 → 46.89)。名目 12.4 ms のうち回収できたのは 9.2 ms。
- プロファイラなしの実測では attn + hc_mix は 40.15 → 34.40 ms で 44 ms を下回るが、
  Task P の 56.2 ms はプロファイラ付きの値なので、**同じ土俵で比較した 46.89 を
  正式な結果**として扱う。
- 残っている最大の的 (both の profile 付き値, per bank):
  `q8_inject` 3.57 ms (= 7.1 ms/token)、`q8_down` 4.55 ms (= 9.1 ms/token)。
  `q8_inject` は 8 行 × 16384 列を 1 warp/行で舐める依存 FMA チェーンで、
  重み 1.3 MB/layer に対して 74 us/call と明らかにレイテンシ律速。
  ここを並列化できれば 44 ms は届くが、**ggml_vec_dot_f32 の加算順序を
  変えると parity が壊れる**ため別タスク扱いとした。

---

## 6. 未検証リスク

1. **44 ms 目標は未達 (46.89)**。5.4 の L2 説明は実測からの推定で、L2/DRAM の
   カウンタや nsight は取っていない。
2. **speed-of-light の残差**: 回収できたのは 9.2 ms で、残り 3.2 ms は
   exact 側の増加 (q8_down/q8_up) として戻ってきている。この 3.2 ms を狙うには
   GEMV 構造そのもの (8 lane 依存チェーン / L2 再利用) に手を入れる必要がある。
3. **測定は共有 A100 上** (同じセッションで Task Q が走っていたため、Q の終了を
   待ってから実行した)。base を 2 回、both を 2 回交互に回して drift は
   ±0.03 ms と小さいが、他ジョブの完全な排除は保証できない。
4. **trace 経路の実機確認は 1 ケースのみ** (`--state-trace-position 0`,
   generate 8)。`state_trace_position < 0` (全位置 trace) と
   `--extended-parity-trace` / `--top-k` の組み合わせは未実行。
   ただし decoder が `capture ? trace : nullptr` を渡す以上、
   trace が付く位置では必ず `trace != nullptr` になり skip は発火しない
   (制御フローは変更前と同一)。
5. **`WARODECODE_HC_Q8_0=0` / `HC_Q8_0_ATTN=0` / `HC_INJECT_F32=0` の組み合わせは
   実機未実行**。いずれも skip 述語が false になる経路で、
   `if (!skip && cur == nullptr)` は skip=false のとき `if (cur == nullptr)` と
   完全に同じ制御フローになる (fail-closed の check も skip が false なので
   到達しない)。
6. **MTP (blk.48)** は今回の計測に含まれない (mtp=false)。MTP の inject は
   量子化されているので skip は発火せず従来どおり。ただしその層では
   `run_qwen38_hc_mix_f32()` が「全計算してから nullptr を返す」無駄が残る
   (Task P doc 8.1 の提案。今回はスコープ外)。
7. **共有ツリー**: 計測時のツリーには Task Q の未コミット変更
   (`src/expert_cache.cuh`) が含まれる。base/both の比較には影響しないが、
   tok_s の絶対値はそのツリーの値。
8. `tests/host_checks.sh` 全体は既存の `build-ui` CMake キャッシュ
   (旧パス `/mnt/c/Users/danda/...`) で失敗する。CI など clean な checkout では
   通常どおり動くはずだが、このワークスペースでは未確認。

---

## 7. 関連

- `docs/HC_MIX_OPTIMIZATION.md` (Task J) に Task R の追記とリンクを入れた。
- 実装の入口: `q38_hc_legacy_mix_skippable()` (src/qwen38_hc.cuh)、
  `q38_hc_skip_legacy_{attn,ffn}_enabled()` (src/qwen38_forward.cuh)。
- A/B ドライバ: `tests/taskp_attn_ab.py --plan taskr` / `--plan taskr_tok`。
