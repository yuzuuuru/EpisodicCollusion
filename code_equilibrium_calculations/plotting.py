# %%
"""
プロット生成スクリプト

CSVファイルから均衡価格のグラフを生成する。
境界値は自動検出または手動指定が可能。

使用方法:
    python plotting.py                          # デフォルト（results.csv → equilibria.png）
    python plotting.py -i results.csv -o out.png
    python plotting.py --auto-boundaries        # 境界自動検出
    python plotting.py --boundary1 365 --boundary2 470  # 境界手動指定
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import argparse
import os
from typing import Tuple, Optional


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
    
    # Overconstrained→Constrained境界を検出
    # nash と monop が初めて異なる（tolerance以上の差がある）ポイント
    overconstrained_boundary = inventories[0]
    for i in range(len(inventories)):
        if monop_prices[i] != 0:
            relative_diff = abs(nash_prices[i] - monop_prices[i]) / monop_prices[i]
        else:
            relative_diff = abs(nash_prices[i] - monop_prices[i])
        
        if relative_diff > tolerance:
            if i > 0:
                overconstrained_boundary = inventories[i - 1]
            break
    else:
        # 全て同じ場合は最後の値
        overconstrained_boundary = inventories[-1]
    
    # Constrained→Unconstrained境界を検出
    # nash価格が連続して同じ値（tolerance以内）になるポイント
    unconstrained_boundary = inventories[-1]
    for i in range(len(inventories) - 1, 0, -1):
        if nash_prices[i - 1] != 0:
            relative_diff = abs(nash_prices[i] - nash_prices[i - 1]) / nash_prices[i - 1]
        else:
            relative_diff = abs(nash_prices[i] - nash_prices[i - 1])
        
        if relative_diff > tolerance:
            unconstrained_boundary = inventories[i]
            break
    else:
        unconstrained_boundary = inventories[0]
    
    return int(overconstrained_boundary), int(unconstrained_boundary)


def create_rescale_function(boundary1: int, boundary2: int):
    """
    境界値に基づいてx軸のリスケール関数を作成
    
    Args:
        boundary1: Overconstrained→Constrained境界
        boundary2: Constrained→Unconstrained境界
    
    Returns:
        リスケール関数
    """
    def rescale_x(x):
        if x <= boundary1:
            return x * 0.8  # Compress values below boundary1
        elif boundary1 < x <= boundary2:
            scaled_start = boundary1 * 0.8
            return scaled_start + (x - boundary1) * 2  # Stretch values between boundaries
        else:
            scaled_start = boundary1 * 0.8 + (boundary2 - boundary1) * 2
            return scaled_start + (x - boundary2) * 0.8  # Compress values above boundary2
    
    return rescale_x


def generate_plot(
    input_csv: str,
    output_png: str,
    boundary1: Optional[int] = None,
    boundary2: Optional[int] = None,
    auto_boundaries: bool = False,
    boundary_tolerance: float = 0.01,
):
    """
    CSVファイルから均衡価格グラフを生成
    
    Args:
        input_csv: 入力CSVファイルパス
        output_png: 出力PNGファイルパス
        boundary1: Overconstrained→Constrained境界（Noneの場合はデフォルト365）
        boundary2: Constrained→Unconstrained境界（Noneの場合はデフォルト470）
        auto_boundaries: Trueの場合は境界を自動検出
        boundary_tolerance: 自動検出時の許容誤差
    """
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
    rcParams["axes.prop_cycle"] = plt.cycler(color=["#1f77b4", "#ff7f0e"])
    
    # データ読み込み
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"入力ファイルが見つかりません: {input_csv}")
    
    data = pd.read_csv(input_csv)
    
    # 境界値の決定
    if auto_boundaries:
        boundary1, boundary2 = detect_boundaries(data, boundary_tolerance)
        print(f"自動検出された境界: boundary1={boundary1}, boundary2={boundary2}")
    else:
        if boundary1 is None:
            boundary1 = 365  # デフォルト値
        if boundary2 is None:
            boundary2 = 470  # デフォルト値
        print(f"使用する境界: boundary1={boundary1}, boundary2={boundary2}")
    
    # リスケール関数を作成
    rescale_x = create_rescale_function(boundary1, boundary2)
    
    # データを分割
    data_main = data[data["inventory"] <= boundary2].copy()
    data_extension = data[data["inventory"] >= boundary2].copy()
    
    # 変換列を追加
    data_main.loc[:, "transformed_inventory"] = data_main["inventory"].apply(rescale_x)
    data_extension.loc[:, "transformed_inventory"] = data_extension["inventory"].apply(rescale_x)
    
    # プロット作成
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300, facecolor="white")
    
    colors = {"monop": "#2ca02c", "nash": "#d62728"}
    labels = {"monop": "Collusive", "nash": "Nash"}
    
    for column in ["nash", "monop"]:
        color = colors[column]
        label = labels[column]
        ax.plot(
            data_main["transformed_inventory"],
            data_main[column],
            label=label,
            marker="o",
            markersize=4,
            linestyle="-",
            linewidth=2,
            color=color,
        )
        if len(data_extension) > 0:
            ax.plot(
                data_extension["transformed_inventory"],
                data_extension[column],
                linestyle="-",
                linewidth=2,
                color=color,
            )
    
    # x軸の目盛り設定
    min_inv = data["inventory"].min()
    max_inv = data["inventory"].max()
    original_ticks = sorted(set([min_inv, boundary1, boundary2, max_inv]))
    transformed_ticks = [rescale_x(tick) for tick in original_ticks]
    ax.set_xticks(transformed_ticks)
    ax.set_xticklabels([int(t) for t in original_ticks])
    
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
    ax.axvline(x=rescale_x(boundary1), color="#888888", linestyle="--", linewidth=1)
    ax.axvline(x=rescale_x(boundary2), color="#888888", linestyle="--", linewidth=1)
    
    # 領域ラベル
    mid_overconstrained = (min_inv + boundary1) / 2
    mid_constrained = (boundary1 + boundary2) / 2
    mid_unconstrained = (boundary2 + max_inv) / 2
    
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
    
    # 保存
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    
    print(f"プロット保存: {output_png}")


def main():
    parser = argparse.ArgumentParser(
        description="CSVファイルから均衡価格グラフを生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--in", "-i",
        dest="input_csv",
        default="results.csv",
        help="入力CSVファイル (default: results.csv)"
    )
    parser.add_argument(
        "--out", "-o",
        dest="output_png",
        default="equilibria.png",
        help="出力PNGファイル (default: equilibria.png)"
    )
    parser.add_argument(
        "--boundary1",
        type=int,
        default=None,
        help="Overconstrained→Constrained境界 (default: 365)"
    )
    parser.add_argument(
        "--boundary2",
        type=int,
        default=None,
        help="Constrained→Unconstrained境界 (default: 470)"
    )
    parser.add_argument(
        "--auto-boundaries",
        action="store_true",
        help="境界を自動検出する"
    )
    parser.add_argument(
        "--boundary-tolerance",
        type=float,
        default=0.01,
        help="境界自動検出の許容誤差 (default: 0.01)"
    )
    
    args = parser.parse_args()
    
    generate_plot(
        input_csv=args.input_csv,
        output_png=args.output_png,
        boundary1=args.boundary1,
        boundary2=args.boundary2,
        auto_boundaries=args.auto_boundaries,
        boundary_tolerance=args.boundary_tolerance,
    )


if __name__ == "__main__":
    main()

# %%
