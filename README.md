# Text to Image Sample App

FastAPIベースのテキストから画像を生成するアプリケーション。
Stable Diffusion を使用して、プロンプトから画像を生成します。

## 特徴

- FastAPIによるWebインターフェース
- 複数のStable Diffusionモデルに対応
  - Stable Diffusion XL 1.0 (SDXL1)
    - GPU 12GB環境下で動作確認済
  - Stable Diffusion 3 Medium (SD3)
    - GPU 12GB以上推奨
- メモリ最適化機能
  - CPU offloading (モデルの一部をCPUメモリに配置)
  - Attention slicing (アテンション機構のメモリ効率化)
  - VAE slicing (VAEのメモリ最適化)
  - 自動ガベージコレクション
- Docker/Docker Composeによるコンテナ化
- NVIDIA GPU対応
- uvによる高速な依存関係管理

## 必要要件

- Python 3.10以上
- NVIDIA GPU (CUDA 12.2対応)
  - 推奨: 12GB以上のVRAM
  - 最小: 8GB VRAM (CPU offloading使用時)
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
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
```

環境変数の説明:
- `HUGGING_FACE_HUB_TOKEN`: Hugging Face APIトークン (必須)
- `PYTORCH_CUDA_ALLOC_CONF`: CUDAメモリアロケーション設定
  - `max_split_size_mb:512`: メモリ割り当ての最大サイズを512MBに制限
  - `expandable_segments:True`: メモリ断片化を回避

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
docker-compose up -d
```
アプリケーションは `http://localhost:8100` でアクセス可能です。

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

## パフォーマンス最適化

このアプリケーションは、限られたGPUメモリでも動作するように、以下の最適化を実装しています：

### メモリ最適化

1. **CPU Offloading**
   - モデルの各コンポーネントを必要に応じてGPUとCPU間で移動
   - `enable_model_cpu_offload()` または `enable_sequential_cpu_offload()` を使用
   - GPUメモリ使用量を大幅に削減

2. **Attention Slicing**
   - アテンション機構を小さなバッチに分割して処理
   - `enable_attention_slicing(1)` で最大レベルの最適化を実行
   - メモリ使用量を削減しつつ品質を維持

3. **VAE Slicing**
   - VAE（Variational Autoencoder）の処理を最適化
   - `enable_vae_slicing()` で有効化
   - 画像のエンコード/デコード時のメモリ消費を削減

4. **自動メモリ管理**
   - 画像生成前後にCUDAキャッシュをクリア
   - Python のガベージコレクションを明示的に実行
   - メモリリークを防止

### 推奨設定

- **SDXL1モデル**: 8-12GB VRAM
- **SD3 Mediumモデル**: 12GB以上のVRAM推奨
- CPU offloadingを使用することで、より少ないVRAMでも動作可能

## ライセンス

このプロジェクトの詳細については、プロジェクトのドキュメントを参照してください。
