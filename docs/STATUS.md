# 状況

## ホスト側で確認済み

- CMakeでCUDAなしTUIビルド
- TUIのGPU CSV解析
- V100 / RTX 3070 / RTX 5090 fixtureで世代dispatch確認
- GGUF探索
- `setup.sh --dump-state` のJSON妥当性
- GGUF parserテスト
- shell構文
- Python比較スクリプト構文

## CUDA実機で未確認

この作業環境にはNVIDIA GPU / nvccがないため、CUDAエンジンのコンパイル・正しさ・速度はまだ未確認。

特にSM50/60/61を今回ビルド対象へ広げた部分は、共通SIMTカーネルがコンパイル可能であることも実機/該当Toolkitで確認が必要。

## 現在の複数GPU

独立した複数モデルをGPUごとに1プロセスで同時起動する機能はTUIに実装済み。

1モデルのtensor parallel / layer splitは未実装。

## 性能主張を解禁する条件

同一GGUF、同一GPU、同一GPU枚数、同一KV、batch=1、同一生成長でllama.cppと比較し、5回以上反復。5%以上の差と品質ゲートの両方を通す。
