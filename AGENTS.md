# リポジトリガイドライン

## プロジェクト構成とモジュールの整理
- `code_training/`: JAX ベースの学習コード。設定は `code_training/conf/`、エージェントは `code_training/agents/`、環境は `code_training/environment/`、ランナーは `code_training/runners/`。
- `code_equilibrium_calculations/`: GNEP ソルバと価格計算ユーティリティ（`gnep_cs.py`, `monopoly.py`, `demand_scaling.py`）および結果ファイル（`results.csv`, `results.xlsx`）。
- `code_training` と `code_equilibrium_calculations` は別プロジェクトとして扱い、各フォルダ内でコマンドを実行してください。

## ビルド・テスト・開発コマンド
- Python 3.10 の仮想環境作成と依存関係のインストール（各サブプロジェクト内で実行）:
  - `uv venv coll_venv --python 3.10`
  - `source coll_venv/bin/activate`
  - `uv pip install --no-cache-dir -r requirements.txt`
- 学習実行（`code_training/` で実行）: `python main.py -cn config_DQN` または `python main.py -cn config_PPO`。
- デバッグ実行（簡易確認）: `python main.py -cn config_PPO_debug` または `python main.py -cn config_DQN_debug`。
- 均衡計算（`code_equilibrium_calculations/` で実行）: `python gnep_cs.py --N 2 --time_horizon 5 --capacities 440 440 --solver_name bonmin`。
- GPU 学習: `requirements.txt` の代わりに `code_training/requirements-gpu.txt` を使用してください。

## コーディングスタイルと命名規則
- Python スタイル: インデントは 4 スペース、命名は PEP 8（関数・変数は snake_case、クラスは CamelCase）。
- 設定ファイルは `code_training/conf/` に `config_*.yaml` の形式（例: `config_PPO_debug.yaml`）。
- プロット用スクリプトは `code_training/` 配下に置き、図を生成する際はインポート直下の `save_dir` を更新します。

## テストガイドライン
- 専用のテストスイートはありません。デバッグ用設定で環境や学習ループを確認してください。
- 小さく再現可能な実行を優先し、`code_training/exp/` または `paper_plots/` の出力を確認します。

## コミットとプルリクエストのガイドライン
- コミットメッセージは短く、命令形の要約にします（例: "fixed imports and requirements"）。
- PR には使用した実験・設定の説明と、関連する出力パス（例: `code_training/exp/...`, `code_equilibrium_calculations/results.csv`）を含めてください。
- `amplpy` や COIN-OR ソルバに影響する変更は、ソルバや依存関係の注意点も記載します。

## 設定と環境に関する注意
- `amplpy` はライセンスが必要な場合があります。整数値需要には COIN-OR ソルバ（BONMIN/COUENNE）が必要です。
- 2 つのサブプロジェクト間で依存関係を混在させないでください。必要に応じて別々の仮想環境を使います。

## エージェント向け指示
- 今後、コードのコメントやこのチャットでの文章はすべて日本語で記載してください。
