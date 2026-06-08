# Predictive Conversion Model

日本語入力における誤変換を、文脈に応じて正しい表記へ補正する実験用リポジトリです。N-gram、文字レベルBiLSTM、BERTベースの3つのアプローチを比較できる構成になっています。

## ファイル構成

| ファイル | 内容 |
| --- | --- |
| `approach1_final.py` / `approach1_final.ipynb` | 2-gram / 3-gram言語モデルで候補文をスコアリングする軽量なベースライン |
| `approach2_corrected.py` / `approach2_corrected.ipynb` | 文字レベルBiLSTMで、既知の誤変換候補を置換すべきか判定するモデル |
| `approach3_checked.py` / `approach3_checked.ipynb` | 日本語BERTでドメイン分類と誤変換判定を行う文脈依存補正モデル |

## 前提

各スクリプトは共通モジュール `ime_common.py` に依存しています。このリポジトリには現在 `ime_common.py` が含まれていないため、実行する前に同ファイルを追加または復元してください。

`ime_common.py` には、少なくとも次のようなデータや関数が必要です。

- `BASE_DATA`
- `MANUAL_CONFUSION_SETS`
- `DOMAIN_LABELS`
- `ID_TO_DOMAIN`
- `build_confusion_map`
- `flatten_confusion_map`
- `train_test_split`
- `tokenize`
- `detokenize`
- `evaluate_correction`
- `extract_replacements`

## セットアップ

Python 3.10以上を推奨します。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers fugashi ipadic unidic_lite
```

`approach1_final.py` は標準ライブラリ中心の実装ですが、`ime_common.py` の実装内容によっては追加ライブラリが必要になる場合があります。

## 実行方法

軽量な順に試す場合は、次の順番で実行します。

```bash
python3 approach1_final.py
python3 approach2_corrected.py
python3 approach3_checked.py
```

ノートブックで確認する場合は、対応する `.ipynb` ファイルをJupyter環境で開いてください。

```bash
jupyter notebook
```

## アプローチ概要

### Approach 1: N-gram

既知の誤変換候補から置換候補文を生成し、N-gram言語モデルのスコアで最も自然な文を選びます。実装が軽く、比較用のベースラインとして使いやすい一方、長い文脈や意味的な違いの扱いには限界があります。

### Approach 2: Character-level BiLSTM

文中の対象スパンにマーカーを付け、文字列全体の文脈から「そのスパンを誤変換として置換すべきか」を二値分類します。日本語形態素解析器に依存しないよう、文字レベルで処理します。

### Approach 3: BERT

`cl-tohoku/bert-base-japanese` を使い、入力文のドメイン分類と、誤変換候補スパンの置換判定を行います。辞書ベースのベースラインも同時に評価し、予測ドメイン版と既知ドメイン版の結果を比較できます。

## 注意点

- `approach3_checked.py` は初回実行時にHugging Faceからモデルをダウンロードします。
- BERTモデルの学習はCPU環境では時間がかかる場合があります。
- 評価データはスクリプト内で `SEED = 42` に固定されており、再現性を意識した分割になっています。
- 現在のノートブックには実行出力が保存されていません。結果を確認するには各スクリプトまたはノートブックを実行してください。

