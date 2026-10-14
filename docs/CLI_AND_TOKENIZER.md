# トークナイザと CLI (Task M)

Qwen3.8-Flash-Next (qwen4exp) の GGUF 語彙を使った **文字列 → トークン → 文字列** を
CUDA エンジンに載せ、llama.cpp の `llama-cli` に近い操作感にした。

- 実装: `src/qwen38_tokenizer.cuh` (+ 生成表 `src/qwen38_tokenizer_unicode.inc`)
- CLI: `src/warodecode.cu` の追加分 (既存の推論経路と JSON 出力はそのまま)
- 検証: `tests/qwen38_tokenizer_host_check.sh` / `tools/tokenizer_colab.py`

## 1. 使い方

```bash
ENGINE=./build-sm89/warodecode
MODEL=/content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf

# 1回だけ: 文字列を与えて文字列をストリーミング表示
$ENGINE -m $MODEL -p "こんにちは。自己紹介してください。" -n 48 --chat-template

# 対話モード (Ctrl-D か /exit で終了、Ctrl-C で生成を止めて終了)
$ENGINE -m $MODEL -i -n 48 --chat-template

# トークナイザだけ (GPU 不要、メタデータしか読まない)
$ENGINE -m $MODEL --tokenize-only -p "こんにちは、世界！"

# 既存の計測・parity 経路はそのまま (JSON を stdout に 1 行)
$ENGINE --model $MODEL --ctx 128 --device 0 --start-token 1 --generate 64 --no-graph --no-autotune
```

### 追加したオプション

| オプション | 意味 |
|---|---|
| `-p, --prompt TEXT` | プロンプト文字列。指定するとテキストのストリーミングモードになる |
| `-f, --prompt-file PATH` | プロンプトをファイルからそのまま (バイト単位で) 読む |
| `-i, --interactive` | 対話モード。1 行ずつ読んで応答を出す |
| `-m`, `-n`, `-c` | `--model` / `--generate` / `--ctx` の別名 |
| `--no-display-prompt` | プロンプトを表示せず生成結果だけ出す |
| `--temp F` | 受け付けるが greedy 固定。0 以外は警告して無視する |
| `--chat-template` / `--no-chat-template` | GGUF の chat template を Qwen 定型として適用する (既定: `-i` のときだけ有効) |
| `--add-bos` / `--no-bos` | BOS の付与を上書き (既定: GGUF の `tokenizer.ggml.add_bos_token` に従う) |
| `--stop-eos` / `--no-stop-eos` | EOS / `<|im_end|>` で生成を打ち切る (既定: 打ち切る) |
| `--stream-json PATH` | 生成結果の JSON (token_ids/時間) をファイルへ。未指定なら stderr |
| `--tokenize-only` | トークン ID を JSON で出して終了。CUDA を初期化しない |
| `--no-parse-special` | 制御トークンをテキストとして扱う (llama.cpp の `parse_special=false`) |

`-p` と `--start-token` は併用しない (`-p` がプロンプトのトークン列を作る)。
`--forced-input-tokens` / `--prefill-tokens` / トレース / `--mtp` / `--bench-repeat` は
テキストモードと排他で、指定すると明確にエラーになる (parity 計測の取り違えを防ぐ)。

### 挙動

- 生成は 1 トークンごとに `std::cout << piece << std::flush` する。バッファリングしない。
- バイト単位 BPE なのでトークンが UTF-8 の途中で切れる。`Qwen38StreamDecoder` が
  不完全な末尾バイトを次トークンまで保持し、壊れたバイトを出さない。
- モデル読込・トークナイザ・autotune の進捗は stderr に 1 行ずつ出る
  (`[tokenizer] model=gpt2 pre=qwen2 vocab=... merges=... bos=... eos=...`)。
- Ctrl-C (SIGINT) はフラグを立てるだけで、現在のトークンが終わってから自然に終了する。
- JSON 出力経路 (`--start-token` + `--generate`) は 1 バイトも変えていない。

## 2. トークナイザの仕様

llama.cpp `6fcaa16f4b360649933a54d1f91ad40ed35c0e11` の BPE 実装を移植した。

| 項目 | 実装 |
|---|---|
| 語彙 | GGUF メタデータ `tokenizer.ggml.tokens` / `merges` / `token_type` を直接読む (テンソルは読まない) |
| 種別 | `tokenizer.ggml.model == "gpt2"` の byte-level BPE のみ (`spm`/`ugm` は明示的にエラー) |
| pre-tokenization | `tokenizer.ggml.pre` が `qwen2` → `unicode_regex_split_custom_qwen2`、`qwen35` → 同 qwen35 版、`gpt2` → GPT-2 規則。それ以外はエラー |
| バイト変換 | GPT-2 の byte→unicode 表 (0x21-0x7E, 0xA1-0xAC, 0xAE-0xFF はそのまま、残りは 0x100 以降へ) |
| マージ | `merges` の行番号が rank。`(rank, left)` の最小ヒープで llama.cpp と同じ順にマージ。古い bigram は text 不一致で捨てる |
| 特殊トークン | `token_type` が CONTROL / USER_DEFINED / UNKNOWN のものを長い順に本文から切り出す (`parse_special`) |
| BOS/EOS | `tokenizer.ggml.bos_token_id` / `eos_token_id` と `add_bos_token` / `add_eos_token` に従う |
| デコード | トークン文字列を GPT-2 逆変換して生バイトに戻し、UTF-8 境界を守って出力 |
| Unicode 表 | `tests/make_qwen38_tokenizer_tables.py` が llama.cpp の `unicode-data.cpp` から生成 (5604 範囲 + 1433 の小文字対応)。llama.cpp と 1 ビットも違わない |

llama.cpp と意図的に合わせた/合わせなかった点:

- **合わせた**: 正規化なし (Qwen は lower/strip_accents なし)、`clean_spaces` なし、
  MERGES の重複は先勝ち、`tokens` の重複は後勝ち、無効 UTF-8 は 1 バイトずつ U+FFFD、
  マージ不能時の 1 バイト fallback。
- **合わせなかった**: BPE 以外のトークナイザ種別、`hybriddna`/`whitespace`/gemma4 用の
  特殊セッション、`tokenizer.ggml.pre` が未知のときの汎用 `std::regex` フォールバック
  (移植したのは llama.cpp 自身が使う手書き splitter)。未知の pre は黙って別規則に
  落ちずエラーにする。
- **chat template**: GGUF の `tokenizer.chat_template` は Jinja なので一般形は評価しない。
  `<|im_start|>` / `<|im_end|>` を使う Qwen 定型のみ再現し、適用したことを stderr に明示する。
  当てはまらないテンプレートでは無効化して警告する (tools 呼び出しの分岐は未対応)。

### 3.1 llama.cpp とのトークン一致 (ローカル)

    bash tests/qwen38_tokenizer_host_check.sh /path/to/llama.cpp <model.gguf>

57 文字列 (日本語 / 英語 / 記号 / 絵文字 / コード / 空白 / 結合文字 / 特殊トークン /
空文字列 / 全角 / アラビア語 / ヘブライ語 / タイ語 / デーヴァナーガリー / 改行・タブ /
契約形 `'s 're` / ZWJ / NBSP / 4 バイト文字) を add_special × parse_special の
4 通りで比較し、往復とストリーミング出力の UTF-8 境界も検証する。

| 語彙 | pre | ケース一致 | 往復 (通常 / ストリーミング) |
|---|---|---|---|
| **Qwen3.8-Flash-Next 実物** (vocab 248320, merges 247587) | `qwen35` | **292/292** | **73/73 / 73/73** |
| Qwen2.5-3B-Instruct Q4_K_M (vocab 151936, merges 151387) | `qwen2` | **292/292** | **73/73 / 73/73** |

実物の語彙は 86 GB のテンソルを落とせば 11 MB のメタデータだけで比較できる
(`tests/make_qwen38_tokenizer_fixture.py`):

    # モデル先頭 64 MiB を取れば key/value セクション全体 (10.9 MB) が入る
    curl -sL -r 0-67108863 -o /tmp/qwen38_head.bin <model-url>
    python3 tests/make_qwen38_tokenizer_fixture.py /tmp/qwen38_head.bin /tmp/qwen38-tokenizer-fixture.gguf
    bash tests/qwen38_tokenizer_host_check.sh /path/to/llama.cpp /tmp/qwen38-tokenizer-fixture.gguf

これで **llama.cpp の `llama_tokenize()` とトークン ID が 1 つも違わない**ことを
実物の語彙で確認できる (不一致は 0 件だったので「不一致例の性質」は該当なし)。
fixture は arch=`qwen4exp` のままなので、llama.cpp 側も本番と同じ hparams 経路を通る。

チャットテンプレートの適用結果 (実物の語彙、`--chat-template` が組む文字列):

    <|im_start|>user
    こんにちは。自己紹介してください。<|im_end|>
    <|im_start|>assistant
    -> [248045, 846, 198, 85951, 1710, 96115, 151279, 172808, 1710, 248046, 198, 248045, 74455, 198]

### 3.2 実機 (L4 24GB, warpdecode-l4c) の結果

`tools/tokenizer_colab.py` が 1 セッションで通しで回した
(生ログ: `issue1_artifacts/tokenizer_colab_run.log`)。

| 項目 | 結果 |
|---|---|
| エンジンのビルド (nvcc, sm_89) | **BUILD_EXIT=0** |
| トークナイザ一致 (実物 86GB モデル) | **78/78 ケース一致** (73 文字列 + 先頭 5 件の `--no-bos`) |
| プロンプト経由の parity | **PROMPT_PARITY 64/64** (`-p "" --stream-json`、prompt_token_ids=[1]) |
| 既存 JSON 経路の parity | **LEGACY_PARITY 64/64** (token_hash `0x6684298df825c691`) |
| ストリーミング表示 | 24〜48 トークン生成、テキストが逐次出力 (下記タイムライン) |
| 対話モード | 2 ターン応答して `/exit` で終了 |
| 速度 A/B (32 トークン、同条件・連続実行) | legacy `5.679 tok/s` / stream `5.896 tok/s` |

1 点だけ補足: `llama-tokenize` 自体は qwen4exp では使えない。

    llama_init_from_model: failed to initialize the context: qwen4exp requires ctx_other to be set
    error: could not create context.

(`llama-tokenize` は必ず llama_context を作り、qwen4exp の graph は ctx_other を要求するため)
そこで同じ `llama_tokenize()` を vocab-only で叩く `tests/qwen38_tokenizer_oracle.cpp` を
VM 上でビルドして比較した。ローカルの fixture 比較 (292/292) と実物 86GB の比較 (78/78) は
同じ結果で、**プロンプトのトークン列は llama.cpp と 1 つも違わない**。

ストリーミングの実測タイムライン (エンジンの stdout を 1 バイトずつ時刻付きで記録):

    T+  6.509 bytes=  53
    T+  6.833 bytes=  68
    T+  7.288 bytes=  71
    T+  7.758 bytes=  77
    T+  8.988 bytes=  80
    ...
    T+ 16.716 bytes= 157
    T+ 16.717 bytes= 160 '\n'
    TOTAL bytes=160 elapsed=17.098 events=26

生成テキスト (チャットテンプレート適用、L4):

    $ENGINE -m Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf --device 0 --ctx 256 --no-graph \
      --no-autotune -n 24 -p 'こんにちは。自己紹介してください。' --chat-template
    [chat-template] applied: qwen <|im_start|>role\ncontent<|im_end|>\n (+ '<|im_start|>assistant\n')
    [prompt] tokens=6 generate=24 stop_eos=1
    <|im_start|>user
    こんにちは。自己紹介してください。<|im_end|>
    <|im_start|>assistant
    こんにちは、私は山田太郎と申します。
    自己紹介の続きをどうぞ。
    趣味は読書

対話モード (`-i`、標準入力から 2 行):

    > <|im_start|>user
    こんにちは<|im_end|>
    <|im_start|>assistant
    The user said "こんにちは" which is Japanese for "Hello." This is a simple greeting, so I should
    > <|im_start|>user
    1+1は？<|im_end|>
    <|im_start|>assistant
    The user is asking "1+1は？" which means "What is 1+1?" in

### 3.3 速度について

同一条件で連続実行した A/B (32 トークン) では legacy 5.68 / stream 5.90 tok/s で、
**ストリーミング経路でも速度は落ちない**。ただし別プロセスでの実測は 1.1〜5.9 tok/s と
ばらついた (expert 重みのディスク読みが支配的なため、ページキャッシュと直前の I/O に敏感)。
速度を報告するときは必ず同一条件で A/B すること。

## 4. README に足す「使い方」節 (親が反映する用)

```markdown
## 文字列で話しかける (トークナイザ + ストリーミング CLI)

GGUF の語彙から byte-level BPE トークナイザを読み、llama.cpp の `llama-cli` と同じ
操作感で使えます。トークン ID 列は llama.cpp と完全一致します (実物の Qwen3.8 語彙で
292/292、実機 86GB モデルでも 78/78)。

```bash
# 1回だけ
./build-sm89/warodecode -m /content/models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf \
  -p "こんにちは。自己紹介してください。" -n 48 --chat-template

# 対話モード (Ctrl-D か /exit、Ctrl-C で生成を止めて終了)
./build-sm89/warodecode -m model.gguf -i -n 48 --chat-template

# トークナイザだけ (GPU 不要)
./build-sm89/warodecode -m model.gguf --tokenize-only -p "こんにちは、世界！"
```

- `-m`/`-p`/`-n`/`-c` は `--model`/`--prompt`/`--generate`/`--ctx` の別名
- 生成は 1 トークンずつ flush して標準出力に流れる (UTF-8 の途中で切れるトークンは次まで保持)
- `--temp` は greedy 固定のため 0 以外は警告して無視、`--chat-template` は Qwen 定型のみ適用
- 既存の `--start-token` + `--generate` の JSON 出力はそのまま (計測・parity 用)
- 詳細と検証結果: `docs/CLI_AND_TOKENIZER.md`
```

## 5. 未検証・制限

- **parity は「トークン列を固定した経路」で確認**: `-p ""` (BOS=1 相当) の 64/64 と
  既存 `--start-token` 経路の 64/64 を確認済み。任意の日本語プロンプトに対する CPU
  オラクル比較は行っていない (CPU で 86GB モデルの prefill+decode を回す時間がないため)。
  「トークン列が一致する」(3.1/3.2) と「そのトークン列を与えたときの出力が一致する」(3.2) に
  分けて確認している。
- **chat template は Qwen 定型のみ**: tools / vision 分岐、`raise_exception`、システム
  メッセージの特殊処理は実装していない。当てはまらないテンプレートは無効化して警告する。
- **`qwen35` 以外の pre-type は限定的**: `qwen2` は Qwen2.5 で一致確認済み、
  `gpt2` 分岐は未検証 (このモデルでは使われない)。未知の pre-type はエラーにする。
- **トークナイザのメモリ**: `tokens` / `merges` を std::string で保持するため
  語彙 248320 では RSS が 1GB 前後増える。推論の VRAM には影響しない。
- **CUDA Graph / MTP はテキストモード非対応**: テキストモードは常に非グラフ経路。`--mtp` とは
  排他にしてある (指定するとエラー)。既存の計測経路では従来どおり使える。
- **対話モードは毎ターン prefill し直す**: KV を再利用しないので長い会話では遅い。発言は履歴に
  残るので文脈は正しい。
- **途中で止めた場合**: Ctrl-C は現在のトークンで止まり、`--stream-json` に生成済みトークンを
  書き出す。ストリーミング出力にプロンプトは含まれない (プロンプトは先に 1 回だけ表示)。

