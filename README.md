# Text to Image Sample App

FastAPIベースのテキストから画像を生成するアプリケーション。
Stable Diffusion を使用して、プロンプトから画像を生成します。

## 特徴

- FastAPIによるWebインターフェース
- 複数のStable Diffusionモデルに対応
  - Stable Diffusion XL 1.0 (SDXL1)
    - GPU12GB環境下で動作確認済
  - Stable Diffusion 3 Medium (SD3)
    - GPUメモリ不足で動作確認できず
- Docker/Docker Composeによるコンテナ化
- NVIDIA GPU対応
- uvによる高速な依存関係管理

## 必要要件

- Python 3.10以上
- NVIDIA GPU (CUDA 12.2対応)
- Docker & Docker Compose (コンテナ実行の場合)
- Hugging Face アカウントとAPIトークン

## セットアップ

### 環境変数の設定

`.env.sample`を`.env`にコピーして、Hugging Face トークンを設定:

```bash
cp .env.sample .env
```

`.env`ファイルを編集:
```
HUGGING_FACE_HUB_TOKEN=hf_yourtoken
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### ローカル開発

uvを使用して依存関係をインストール:

```bash
uv sync
```

モデルの事前読み込み (推奨):

```bash
make preload_models
```

開発サーバーの起動:

```bash
make dev
```

または本番モード:

```bash
make run
```

アプリケーションは `http://localhost:8000` でアクセス可能です。

### Docker Compose

Docker Composeで起動:

```bash
docker-compose up
```

## 利用可能なコマンド

### 開発コマンド

- `make lint` - Ruffによるコードチェック
- `make fmt` - Ruffによるコードフォーマット
- `make vet` - Pyrightによる型チェック
- `make dev` - 開発サーバー起動 (ホットリロード有効)
- `make run` - 本番サーバー起動 (2ワーカー)
- `make preload_models` - モデルの事前読み込み

### Dockerコマンド

- `make build` - Dockerイメージのビルド
- `make build-no-cache` - キャッシュなしでDockerイメージをビルド
- `make push` - Dockerイメージをレジストリにプッシュ

## API エンドポイント

### `GET /`
Webインターフェースを表示

### `POST /text-to-images`
テキストから画像を生成

パラメータ:
- `prompt` (string, 必須): 生成する画像の説明
- `negative_prompt` (string, オプション): 避けたい要素の説明
- `num_inference_steps` (int, デフォルト: 30): 推論ステップ数
- `guidance_scale` (float, デフォルト: 7.0): ガイダンススケール
- `model` (string, デフォルト: "stabilityai/stable-diffusion-xl-base-1.0"): 使用するモデル
  - `stabilityai/stable-diffusion-xl-base-1.0`: Stable Diffusion XL 1.0
  - `stabilityai/stable-diffusion-3-medium-diffusers`: Stable Diffusion 3 Medium

## プロジェクト構成

```
.
├── t2i/                    # メインアプリケーションパッケージ
│   ├── app.py             # FastAPIアプリケーション
│   ├── controller.py      # コントローラーレイヤー
│   ├── generate_image_usecase.py  # ビジネスロジック
│   ├── stable_diffusion.py        # Stable Diffusion実装
│   ├── text_to_image_item.py      # データモデル
│   ├── hugging_face_hub.py        # Hugging Face統合
│   ├── logger.py          # ロギング設定
│   └── env.py             # 環境変数管理
├── templates/             # Jinja2テンプレート
├── tmp/                   # 生成画像の保存先
├── Dockerfile             # マルチステージDockerビルド
├── docker-compose.yml     # Docker Compose設定
├── Makefile              # 開発タスク
├── pyproject.toml        # Python依存関係
└── main.py               # CLIエントリーポイント

## 技術スタック

- **フレームワーク**: FastAPI
- **画像生成**: Diffusers, Transformers, Torch
- **パッケージ管理**: uv
- **コンテナ**: Docker (CUDA 12.2 runtime)
- **コード品質**: Ruff, Pyright

## ライセンス

このプロジェクトの詳細については、プロジェクトのドキュメントを参照してください。
