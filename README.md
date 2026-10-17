# hikabooru person-tagger

ヒカマニ系画像板 **hikabooru** のための、人物・キャラクター判定モデルです。
画像を渡すと14クラスの確率を返します。既存のコミュニティタグを教師にして、SigLIP2 を LoRA で微調整した小型モデル(約2.8Mパラメータ追加)で、GPU 1枚で動きます。

- ベースモデル: [google/siglip2-base-patch16-384](https://huggingface.co/google/siglip2-base-patch16-384)(Apache-2.0)
- 追加学習: LoRA r=16 + 線形ヘッド / 15,105枚・2エポック
- ライセンス: **Apache-2.0**

| クラス | 対応タグ |
|---|---|
| `hikakin` | ヒカキン |
| `seikin` | セイキン |
| `masuo` | ゲイマスオ |
| `moshian` | モシャン |
| `smith` | スミスカラ |
| `kirby` | カービー |
| `sega` | 瀬我 |
| `reimu` | はくれいれいむ |
| `remilia` | レミリア・スカーレット |
| `bawty` | バウティー |
| `marisa` | 切亜未マリーサ |
| `miku` | 初音ミク |
| `pikachu` | ピカチュウ |
| `koumyouji` | 光明寺恋子 |

## 使い方(クイックスタート)

```bash
git clone https://github.com/maebahesioru/hikabooru-person-tagger
cd hikabooru-person-tagger
pip install -r tools/requirements.txt

python tools/classify.py photo.jpg --top 5
```

出力例(実測):

```
$ python tools/classify.py photo.jpg --top 3
photo.jpg
  hikakin    0.9751
  seikin     0.0232
  bawty      0.0154
```

- 複数画像: `python tools/classify.py a.jpg b.png`
- JSON出力: `python tools/classify.py a.jpg --json`
- GPUが無ければCPUでも動きます(遅いだけ)。

## Python API

```python
from tools.classify import load_model, classify_images

proc, vision, head, classes, dev = load_model()
results = classify_images(["photo.jpg"], proc, vision, head, classes, dev)
print(results[0]["hikakin"])   # 0.9641
```

## 精度(テスト分割・未学習データ)

| クラス | AUC | クラス | AUC |
|---|---|---|---|
| masuo | 0.986 | smith | 0.939 |
| marisa | 0.958 | remilia | 0.932 |
| koumyouji | 0.946 | moshian | 0.902 |
| reimu | 0.944 | seikin | 0.896 |
| hikakin | 0.890 | miku | 0.852 |
| kirby | 0.845 | bawty | 0.813 |
| pikachu | 0.786 | sega | 0.784 |

自動タグ付けに使う場合は、クラスごとに較正した高しきい値(例: 適合率95%となる点)で運用してください。ロジットの尺度はクラスごとに異なるため、生の0.9固定では使えません。

## 限界と注意

- **本モデルは「タグ慣習の再現」であり、厳密な顔認識ではありません。** 学習ラベルは hikabooru の既存タグで、本人が写っていない言及スクショ・制作クレジット・似顔絵にもタグが付く慣習があります。モデルもその慣習を学習しています。
- 境界帯には数%の誤爆が残ります(例: マイクラ実況の構図をモシャンと誤認するケースを確認)。自動適用する場合は高しきい値+ログ+人手レビューを推奨します。
- ドメイン(ヒカマニ文化圏の実写・ミーム・スクリーンショット)向けです。汎用的な人物認識性能はありません。
- **実在の私人の特定・追跡・監視に使用しないでください。** 本モデルの用途はコミュニティ画像板のタグ付け支援に限られます。

## 学習を再現する

`tools/train_from_manifest.py` を使います。データは再配布していません。

```bash
# manifest.jsonl: 1行 = {"path": "img.jpg", "labels": ["hikakin"]}
python tools/train_from_manifest.py --manifest manifest.jsonl --out out/ --batch 6 --epochs 2
```

## 設計メモ(実測)

- **fp16 + autocast で GradScaler を無効にすると LoRA が発散**します(重みノルム爆発 → 視覚特徴が破壊され AUC 0.49 に崩壊)。GradScaler 有効 + 勾配クリップ1.0 + lr 1e-4 が必須です。本リポジトリのスクリプトは対策済み。
- クラスごとにロジット尺度が大きく違うため、しきい値はクラス別に較正する必要があります。
- 「タグさえあれば何でも判定器にできる」方式です。200件程度のタグ付き投稿があれば新しいクラスを追加できます(例: シリーズタグで218件→AUC 0.89 を確認)。

## クレジット

- ベース: Google SigLIP 2(Apache-2.0)
- 学習データ: hikabooru コミュニティのタグ付け(画像自体は再配布していません)
- 作成: [@maebahesioru](https://github.com/maebahesioru)
