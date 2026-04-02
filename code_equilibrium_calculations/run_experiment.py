#!/usr/bin/env python3
"""
実験自動実行スクリプト

設定ファイル（YAML）を読み込み、Nash均衡価格とMonopoly価格を自動計算し、
結果をCSVとグラフで出力する。

使用方法:
    python run_experiment.py config.yaml
    python run_experiment.py config.yaml --dry-run  # 実行せずに設定を確認
"""

import argparse
import os
import sys
import shutil
from datetime import datetime
from typing import Dict, List, Tuple, Any
from pathlib import Path

import numpy as np
import yaml
import pandas as pd

# 同一ディレクトリのモジュールをインポート
from gnep_cs import solve_gnep
from monopoly import solve_monopoly
from demand_function import build_demand_scale


def load_config(config_path: str) -> Dict[str, Any]:
    """
    YAML設定ファイルを読み込む
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def validate_price_consistency(
    prices: Dict[int, np.ndarray],
    threshold: float,
    label: str,
) -> float:
    """
    時系列価格の一貫性を検証する。
    全時刻で価格が閾値以内に収まっているかチェック。
    
    Args:
        prices: 各エージェントの価格ベクトル
        threshold: 許容誤差（相対誤差）
        label: エラーメッセージ用ラベル
    
    Returns:
        代表価格（最終時刻の価格の平均）
    
    Raises:
        ValueError: 価格が閾値を超えて異なる場合
    """
    all_prices = []
    for agent_id, price_vector in prices.items():
        all_prices.extend(price_vector)
    
    all_prices = np.array(all_prices)
    mean_price = np.mean(all_prices)
    
    if mean_price == 0:
        # 価格が0の場合は絶対誤差でチェック
        max_diff = np.max(np.abs(all_prices - mean_price))
        if max_diff > threshold:
            raise ValueError(
                f"{label}: 価格の時系列一貫性エラー。"
                f"平均={mean_price:.4f}, 最大差={max_diff:.4f}, 閾値={threshold}"
            )
    else:
        # 相対誤差でチェック
        max_relative_diff = np.max(np.abs(all_prices - mean_price)) / abs(mean_price)
        if max_relative_diff > threshold:
            raise ValueError(
                f"{label}: 価格の時系列一貫性エラー。"
                f"平均={mean_price:.4f}, 最大相対誤差={max_relative_diff:.4f}, 閾値={threshold}"
            )
    
    # 代表価格として最終時刻の価格の平均を返す
    final_prices = [price_vector[-1] for price_vector in prices.values()]
    return np.mean(final_prices)


def run_single_experiment(
    N: int,
    capacity: int,
    fixed_params: Dict[str, Any],
    threshold: float,
) -> Tuple[float, float]:
    """
    単一のcapacity設定で Nash と Monopoly 価格を計算
    
    Args:
        N: エージェント数
        capacity: 各エージェントのキャパシティ（全員同一）
        fixed_params: 固定パラメータ
        threshold: 価格一貫性チェックの閾値
    
    Returns:
        (nash_price, monopoly_price)
    """
    # パラメータ準備（型変換を含む）
    time_horizon = int(fixed_params["time_horizon"])
    mu = float(fixed_params["mu"])
    quality_factor = float(fixed_params["quality_factor"])
    marginal_cost = float(fixed_params["marginal_cost"])
    demand_scale_factor = int(fixed_params["demand_scale_factor"])
    discount_factor = float(fixed_params["discount_factor"])
    solver_name = str(fixed_params["solver_name"])
    method = str(fixed_params.get("method", "gauss-seidel"))
    initial_prices = str(fixed_params.get("initial_prices", "quality_factor"))
    epsilon = float(fixed_params.get("epsilon", 1e-4))
    regularization_tau = float(fixed_params.get("regularization_tau", 0.01))
    max_iterations = int(fixed_params.get("max_iterations", 100))
    
    # N個のエージェント用にパラメータを展開
    quality_factors = np.array([quality_factor] * N)
    marginal_costs = np.array([marginal_cost] * N)
    capacities = np.array([capacity] * N)
    discount_factors = np.array([discount_factor] * N)
    
    # demand_scale_over_time を構築（constant scaler）
    demand_scale_over_time = build_demand_scale(
        scaler=None,  # constant
        params={},
        time_horizon=time_horizon,
        default_scale=demand_scale_factor,
    )
    
    # regularization_tau をスケーリング
    regularization_tau_scaled = regularization_tau * np.mean(demand_scale_over_time)
    
    print(f"  [N={N}, capacity={capacity}] Nash均衡を計算中...")
    
    # Nash均衡価格を計算
    nash_prices, nash_demands, nash_profits = solve_gnep(
        N=N,
        time_horizon=time_horizon,
        quality_factors=quality_factors,
        mu=mu,
        marginal_costs=marginal_costs,
        capacities=capacities,
        demand_scale_over_time=demand_scale_over_time,
        discount_factors=discount_factors,
        epsilon=epsilon,
        solver_name=solver_name,
        method=method,
        regularization_tau=regularization_tau_scaled,
        debug=False,
        initial_prices=initial_prices,
        max_iterations=max_iterations,
    )
    
    # Nash価格の一貫性検証
    nash_price = validate_price_consistency(
        nash_prices, threshold, f"Nash (N={N}, cap={capacity})"
    )
    
    print(f"  [N={N}, capacity={capacity}] Monopoly価格を計算中...")
    
    # Monopoly価格を計算
    monop_prices, monop_demands, monop_total_demands = solve_monopoly(
        N=N,
        time_horizon=time_horizon,
        quality_factors=quality_factors,
        mu=mu,
        marginal_costs=marginal_costs,
        capacities=capacities,
        demand_scale_factor=demand_scale_factor,
        discount_factors=discount_factors,
        solver_name=solver_name,
        debug=False,
        initial_prices=initial_prices,
    )
    
    # Monopoly価格の一貫性検証
    monop_price = validate_price_consistency(
        monop_prices, threshold, f"Monopoly (N={N}, cap={capacity})"
    )
    
    print(f"  [N={N}, capacity={capacity}] 完了: Nash={nash_price:.4f}, Monop={monop_price:.4f}")
    
    return nash_price, monop_price


def run_single_experiment_nash_only(
    N: int,
    capacity: int,
    fixed_params: Dict[str, Any],
    threshold: float,
) -> Tuple[float, float]:
    """Nash均衡価格のみを計算（Monopoly失敗からの独立リトライ用）"""
    time_horizon = int(fixed_params["time_horizon"])
    mu = float(fixed_params["mu"])
    quality_factor = float(fixed_params["quality_factor"])
    marginal_cost = float(fixed_params["marginal_cost"])
    demand_scale_factor = int(fixed_params["demand_scale_factor"])
    discount_factor = float(fixed_params["discount_factor"])
    solver_name = str(fixed_params["solver_name"])
    method = str(fixed_params.get("method", "gauss-seidel"))
    initial_prices = str(fixed_params.get("initial_prices", "quality_factor"))
    epsilon = float(fixed_params.get("epsilon", 1e-4))
    regularization_tau = float(fixed_params.get("regularization_tau", 0.01))
    max_iterations = int(fixed_params.get("max_iterations", 100))

    quality_factors = np.array([quality_factor] * N)
    marginal_costs = np.array([marginal_cost] * N)
    capacities = np.array([capacity] * N)
    discount_factors = np.array([discount_factor] * N)
    demand_scale_over_time = build_demand_scale(None, {}, time_horizon, demand_scale_factor)
    regularization_tau_scaled = regularization_tau * np.mean(demand_scale_over_time)

    nash_prices, _, _ = solve_gnep(
        N=N, time_horizon=time_horizon, quality_factors=quality_factors,
        mu=mu, marginal_costs=marginal_costs, capacities=capacities,
        demand_scale_over_time=demand_scale_over_time,
        discount_factors=discount_factors, epsilon=epsilon,
        solver_name=solver_name, method=method,
        regularization_tau=regularization_tau_scaled,
        debug=False, initial_prices=initial_prices, max_iterations=max_iterations,
    )
    nash_price = validate_price_consistency(nash_prices, threshold, f"Nash (N={N}, cap={capacity})")
    return nash_price, float("nan")


def run_single_experiment_monopoly_only(
    N: int,
    capacity: int,
    fixed_params: Dict[str, Any],
    threshold: float,
) -> Tuple[float, float]:
    """Monopoly価格のみを計算（Nash失敗からの独立リトライ用）"""
    time_horizon = int(fixed_params["time_horizon"])
    mu = float(fixed_params["mu"])
    quality_factor = float(fixed_params["quality_factor"])
    marginal_cost = float(fixed_params["marginal_cost"])
    demand_scale_factor = int(fixed_params["demand_scale_factor"])
    discount_factor = float(fixed_params["discount_factor"])
    solver_name = str(fixed_params["solver_name"])
    initial_prices = str(fixed_params.get("initial_prices", "quality_factor"))

    quality_factors = np.array([quality_factor] * N)
    marginal_costs = np.array([marginal_cost] * N)
    capacities = np.array([capacity] * N)
    discount_factors = np.array([discount_factor] * N)

    monop_prices, _, _ = solve_monopoly(
        N=N, time_horizon=time_horizon, quality_factors=quality_factors,
        mu=mu, marginal_costs=marginal_costs, capacities=capacities,
        demand_scale_factor=demand_scale_factor,
        discount_factors=discount_factors, solver_name=solver_name,
        debug=False, initial_prices=initial_prices,
    )
    monop_price = validate_price_consistency(monop_prices, threshold, f"Monopoly (N={N}, cap={capacity})")
    return float("nan"), monop_price


def detect_boundaries(df: pd.DataFrame, tolerance: float = 0.01) -> Tuple[int, int]:
    """
    データから領域境界を自動検出する
    
    - Overconstrained→Constrained境界: nash と monop が初めて乖離するポイント
    - Constrained→Unconstrained境界: nash が連続して同じ値になるポイント
    
    Args:
        df: inventory, nash, monop 列を持つDataFrame
        tolerance: 同一判定の許容誤差（相対誤差）
    
    Returns:
        (overconstrained_boundary, unconstrained_boundary)
    """
    inventories = df["inventory"].values
    nash_prices = df["nash"].values
    monop_prices = df["monop"].values
    
    # NaN を除外したデータで検出
    valid_mask = ~(np.isnan(nash_prices) | np.isnan(monop_prices))
    inventories_valid = inventories[valid_mask]
    nash_valid = nash_prices[valid_mask]
    monop_valid = monop_prices[valid_mask]
    
    if len(inventories_valid) == 0:
        return int(inventories[0]), int(inventories[-1])
    
    # Overconstrained→Constrained境界を検出
    # nash と monop が初めて異なる（tolerance以上の差がある）ポイント
    overconstrained_boundary = inventories_valid[0]
    for i in range(len(inventories_valid)):
        if monop_valid[i] != 0:
            relative_diff = abs(nash_valid[i] - monop_valid[i]) / monop_valid[i]
        else:
            relative_diff = abs(nash_valid[i] - monop_valid[i])
        
        if relative_diff > tolerance:
            if i > 0:
                overconstrained_boundary = inventories_valid[i - 1]
            break
    else:
        # 全て同じ場合は最後の値
        overconstrained_boundary = inventories_valid[-1]
    
    # Constrained→Unconstrained境界を検出
    # nash価格が連続して同じ値（tolerance以内）になるポイント
    unconstrained_boundary = inventories_valid[-1]
    for i in range(len(inventories_valid) - 1, 0, -1):
        if nash_valid[i - 1] != 0:
            relative_diff = abs(nash_valid[i] - nash_valid[i - 1]) / nash_valid[i - 1]
        else:
            relative_diff = abs(nash_valid[i] - nash_valid[i - 1])
        
        if relative_diff > tolerance:
            unconstrained_boundary = inventories_valid[i]
            break
    else:
        unconstrained_boundary = inventories_valid[0]
    
    return overconstrained_boundary, unconstrained_boundary


def generate_plot(
    csv_path: str,
    output_path: str,
    boundaries: Tuple[int, int] = None,
    title_suffix: str = "",
):
    """
    CSVからプロットを生成
    
    Args:
        csv_path: 入力CSVファイルパス
        output_path: 出力PNGファイルパス
        boundaries: (overconstrained_boundary, unconstrained_boundary) または None（自動検出）
        title_suffix: タイトルに追加する文字列
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    
    # スタイル設定
    plt.style.use("seaborn-v0_8-whitegrid")
    rcParams["font.family"] = "DejaVu Sans"
    rcParams["font.size"] = 11
    rcParams["axes.linewidth"] = 0.8
    rcParams["axes.edgecolor"] = "#333333"
    rcParams["xtick.major.width"] = 0.8
    rcParams["ytick.major.width"] = 0.8
    rcParams["xtick.direction"] = "out"
    rcParams["ytick.direction"] = "out"
    
    # データ読み込み
    df = pd.read_csv(csv_path)
    
    # 境界を自動検出（指定がない場合）
    if boundaries is None:
        boundaries = detect_boundaries(df)
    
    overconstrained_bound, unconstrained_bound = boundaries
    
    # x軸のスケーリング関数
    def rescale_x(x):
        if x <= overconstrained_bound:
            return x * 0.8
        elif overconstrained_bound < x <= unconstrained_bound:
            scaled_start = overconstrained_bound * 0.8
            return scaled_start + (x - overconstrained_bound) * 2
        else:
            scaled_start = overconstrained_bound * 0.8 + (unconstrained_bound - overconstrained_bound) * 2
            return scaled_start + (x - unconstrained_bound) * 0.8
    
    # データを変換（NaN行を除外）
    df = df.dropna(subset=["nash", "monop"], how="all")
    df["transformed_inventory"] = df["inventory"].apply(rescale_x)
    
    # プロット作成
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300, facecolor="white")
    
    colors = {"monop": "#2ca02c", "nash": "#d62728"}
    labels = {"monop": "Collusive", "nash": "Nash"}
    
    for column in ["nash", "monop"]:
        color = colors[column]
        label = labels[column]
        ax.plot(
            df["transformed_inventory"],
            df[column],
            label=label,
            marker="o",
            markersize=4,
            linestyle="-",
            linewidth=2,
            color=color,
        )
    
    # x軸の目盛り設定
    tick_values = sorted(set([
        df["inventory"].min(),
        overconstrained_bound,
        unconstrained_bound,
        df["inventory"].max()
    ]))
    transformed_ticks = [rescale_x(t) for t in tick_values]
    ax.set_xticks(transformed_ticks)
    ax.set_xticklabels([int(t) for t in tick_values])
    
    # ラベル設定
    ax.set_xlabel("Inventory Capacity", fontsize=14, labelpad=10)
    ax.set_ylabel("Equilibrium Price", fontsize=14, labelpad=10)
    
    # 凡例
    legend = ax.legend(fontsize=12, frameon=True, loc="upper right", bbox_to_anchor=(0.98, 0.98))
    legend.get_frame().set_edgecolor("#333333")
    legend.get_frame().set_linewidth(0.8)
    
    # 上と右のスパインを非表示
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # 境界線を描画
    ax.axvline(x=rescale_x(overconstrained_bound), color="#888888", linestyle="--", linewidth=1)
    ax.axvline(x=rescale_x(unconstrained_bound), color="#888888", linestyle="--", linewidth=1)
    
    # 領域ラベル
    mid_overconstrained = (df["inventory"].min() + overconstrained_bound) / 2
    mid_constrained = (overconstrained_bound + unconstrained_bound) / 2
    mid_unconstrained = (unconstrained_bound + df["inventory"].max()) / 2
    
    y_bottom = ax.get_ylim()[0]
    ax.text(rescale_x(mid_overconstrained), y_bottom, "Overconstrained",
            ha="center", va="bottom", fontsize=12, color="#555555")
    ax.text(rescale_x(mid_constrained), y_bottom, "Constrained",
            ha="center", va="bottom", fontsize=12, color="#555555")
    ax.text(rescale_x(mid_unconstrained), y_bottom, "Unconstrained",
            ha="center", va="bottom", fontsize=12, color="#555555")
    
    # 余白調整
    ax.set_xlim(ax.get_xlim()[0] - 10, ax.get_xlim()[1] + 10)
    ax.set_ylim(ax.get_ylim()[0] - 0.1, ax.get_ylim()[1] + 0.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    
    print(f"  プロット保存: {output_path}")
    print(f"  検出された境界: Overconstrained={overconstrained_bound}, Unconstrained={unconstrained_bound}")


def run_experiment(config: Dict[str, Any], dry_run: bool = False) -> None:
    """
    実験を実行
    
    Args:
        config: 設定辞書
        dry_run: Trueの場合、実際の計算は行わず設定を表示するのみ
    """
    experiment_name = config.get("experiment_name", "experiment")
    fixed_params = config["fixed_params"]
    sweep_params = config["sweep_params"]
    validation = config.get("validation", {})
    output_config = config.get("output", {})
    
    threshold = validation.get("price_consistency_threshold", 0.01)
    base_dir = output_config.get("base_dir", "experiments")
    
    # sweep_params からNとcapacitiesを取得
    N_list = sweep_params.get("N", [2])
    if isinstance(N_list, int):
        N_list = [N_list]
    
    capacities_list = sweep_params.get("capacities", [100, 200, 300])
    
    print("=" * 60)
    print(f"実験名: {experiment_name}")
    print("=" * 60)
    print(f"固定パラメータ:")
    for key, value in fixed_params.items():
        print(f"  {key}: {value}")
    print(f"スイープパラメータ:")
    print(f"  N: {N_list}")
    print(f"  capacities: {capacities_list}")
    print(f"検証設定:")
    print(f"  price_consistency_threshold: {threshold}")
    print(f"出力先: {base_dir}")
    print("=" * 60)
    
    if dry_run:
        print("ドライラン: 実際の計算は行いません")
        return
    
    # 各Nに対して実験を実行
    for N in N_list:
        print(f"\n{'='*60}")
        print(f"N={N} の実験を開始")
        print(f"{'='*60}")
        
        # 出力ディレクトリを作成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = Path(base_dir) / f"exp_{timestamp}_N{N}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 結果を収集
        results = []
        
        for capacity in capacities_list:
            nash_price = float("nan")
            monop_price = float("nan")
            
            try:
                nash_price, monop_price = run_single_experiment(
                    N=N,
                    capacity=capacity,
                    fixed_params=fixed_params,
                    threshold=threshold,
                )
            except Exception as e:
                print(f"  エラー (N={N}, capacity={capacity}): {e}")
                # Nash/Monopoly を個別にリトライ
                try:
                    nash_price, _ = run_single_experiment_nash_only(
                        N=N, capacity=capacity,
                        fixed_params=fixed_params, threshold=threshold,
                    )
                except Exception as e2:
                    print(f"  Nash個別リトライも失敗: {e2}")
                try:
                    _, monop_price = run_single_experiment_monopoly_only(
                        N=N, capacity=capacity,
                        fixed_params=fixed_params, threshold=threshold,
                    )
                except Exception as e2:
                    print(f"  Monopoly個別リトライも失敗: {e2}")
            
            results.append({
                "inventory": capacity,
                "nash": round(nash_price, 3) if not np.isnan(nash_price) else float("nan"),
                "monop": round(monop_price, 3) if not np.isnan(monop_price) else float("nan"),
            })
        
        # 結果をCSVに保存
        df = pd.DataFrame(results)
        csv_path = exp_dir / "results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n結果CSV保存: {csv_path}")
        
        # 設定ファイルをコピー
        config_copy_path = exp_dir / "config.yaml"
        # 実行時の設定を保存（Nを固定値として記録）
        config_for_save = config.copy()
        config_for_save["sweep_params"] = {"N": N, "capacities": capacities_list}
        config_for_save["execution_timestamp"] = timestamp
        with open(config_copy_path, "w", encoding="utf-8") as f:
            yaml.dump(config_for_save, f, allow_unicode=True, default_flow_style=False)
        print(f"設定ファイル保存: {config_copy_path}")
        
        # プロット生成
        try:
            plot_path = exp_dir / "equilibria.png"
            generate_plot(str(csv_path), str(plot_path))
        except Exception as e:
            print(f"  プロット生成エラー: {e}")
        
        print(f"\nN={N} の実験完了: {exp_dir}")
    
    print("\n" + "=" * 60)
    print("全ての実験が完了しました")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="実験自動実行スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python run_experiment.py config.yaml
  python run_experiment.py config.yaml --dry-run
        """
    )
    parser.add_argument(
        "config",
        type=str,
        help="設定ファイル（YAML）のパス",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="実際の計算は行わず、設定を確認するのみ",
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"エラー: 設定ファイルが見つかりません: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    run_experiment(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
