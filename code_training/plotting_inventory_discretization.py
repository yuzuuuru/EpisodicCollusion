"""
Visualization of inventory discretization effects on collusion index.
Adapted from plotting_fig2_trainresult.py.

Usage:
    python plotting_inventory_discretization.py --save_dir exp/PPO
    python plotting_inventory_discretization.py --save_dir exp/PPO exp/DQN
"""
import pickle
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from plotting_utils import (
    overall_mean_stdev_from_seed_means_variances,
    apply_moving_average,
)


def load_experiment(save_dir):
    """Load all experiment data from a save_dir."""
    with open(os.path.join(save_dir, "args.pkl"), "rb") as f:
        args = pickle.load(f)
    with open(os.path.join(save_dir, "log_data.pkl"), "rb") as f:
        log_data = pickle.load(f)
    with open(os.path.join(save_dir, "update_dict.pkl"), "rb") as f:
        update_dict = pickle.load(f)
    with open(os.path.join(save_dir, "hyperparam_mapping.pkl"), "rb") as f:
        hyperparam_mapping = pickle.load(f)

    # Unpack log_data
    if isinstance(log_data, tuple):
        train_log_data = log_data[0]
    else:
        train_log_data = log_data
    if isinstance(train_log_data, (list, tuple)):
        env_stats = train_log_data[0]
    else:
        env_stats = train_log_data

    return args, env_stats, update_dict, hyperparam_mapping


def get_inv_level_label(n):
    if n == -1:
        return "Continuous (baseline)"
    elif n == 1:
        return "n=1 (hidden)"
    else:
        return f"n={n}"


def plot_collusion_index_by_discretization(save_dir, gen_mean_p=0.5):
    """
    Main figure: collusion index training curves for each discretization level,
    overlaid on a single plot.
    """
    args, env_stats, update_dict, hyperparam_mapping = load_experiment(save_dir)
    algo = args.get("agent_default", "PPO")

    num_iters = args["num_iters"]
    num_seeds = args["num_seeds"]
    num_envs = args["num_envs"]
    num_players = args.get("num_players", 2)
    log_interval = max(num_iters // 1000, 5 if num_iters > 1000 else 1)
    x_axis = np.arange(0, num_iters, log_interval)

    # Collusion index: shape (seeds, configs, timesteps)
    # Subsample by x_axis indices (data may have more timesteps than x_axis)
    ci_players = []
    for p in range(num_players):
        ci = np.array(env_stats[f"train/collusion_index/mean_player_{p + 1}"])
        ci_players.append(ci[:, :, x_axis])
    # Average over players: (seeds, configs, len(x_axis))
    ci_avg = np.mean(ci_players, axis=0)

    # Extract inventory levels from hyperparam_mapping
    inv_levels = [m["num_inventory_levels"] for m in hyperparam_mapping]

    # --- Style ---
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["axes.edgecolor"] = "#333333"

    # Color palette: from coarse (warm) to fine/continuous (cool)
    cmap = plt.get_cmap("viridis")
    n_configs = len(inv_levels)
    colors = [cmap(i / max(n_configs - 1, 1)) for i in range(n_configs)]

    # ===== Figure 1: Collusion Index Training Curves (overlaid) =====
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    for j, n_level in enumerate(inv_levels):
        # ci_avg[:, j, :] is (seeds, timesteps) for config j
        seed_data = ci_avg[:, j, :]  # (seeds, T)
        mean_ci = seed_data.mean(axis=0)
        std_ci = seed_data.std(axis=0)

        # Moving average for smoother curves
        window = max(1, len(x_axis) // 50)
        if window > 1:
            mean_ci_ma, x_ma = apply_moving_average(mean_ci, x_axis, window)
            std_ci_ma, _ = apply_moving_average(std_ci, x_axis, window)
        else:
            mean_ci_ma, x_ma, std_ci_ma = mean_ci, x_axis, std_ci

        label = get_inv_level_label(n_level)
        ax1.plot(x_ma, mean_ci_ma, linewidth=2, color=colors[j], label=label)
        ax1.fill_between(
            x_ma,
            mean_ci_ma - std_ci_ma,
            mean_ci_ma + std_ci_ma,
            alpha=0.15,
            color=colors[j],
            linewidth=0,
        )

    ax1.axhline(0, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.7, label="Nash (Δ=0)")
    ax1.axhline(1, color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.7, label="Monopoly (Δ=1)")
    ax1.set_xlabel("Episodes", fontsize=14)
    ax1.set_ylabel("Collusion Index Δ", fontsize=14)
    ax1.set_title(f"{algo}: Collusion Index by Inventory Discretization Level", fontsize=15)
    ax1.legend(fontsize=10, loc="upper left", frameon=True)

    fig1.tight_layout()

    # ===== Figure 2: Box plot of final Δ =====
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))

    last_frac = max(1, len(x_axis) // 10)
    final_deltas = []
    labels = []
    for j, n_level in enumerate(inv_levels):
        seed_data = ci_avg[:, j, :]  # (seeds, T)
        final_ci = seed_data[:, -last_frac:].mean(axis=1)  # per-seed average of last 10%
        final_deltas.append(final_ci)
        labels.append(get_inv_level_label(n_level))

    bp = ax2.boxplot(
        final_deltas,
        tick_labels=labels,
        patch_artist=True,
        widths=0.6,
        medianprops=dict(color="black", linewidth=1.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.axhline(0, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.7)
    ax2.axhline(1, color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.7)
    ax2.set_ylabel("Collusion Index Δ (last 10%)", fontsize=14)
    ax2.set_xlabel("Inventory Discretization Level", fontsize=14)
    ax2.set_title(f"{algo}: Final Collusion Index by Discretization Level", fontsize=15)
    plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")

    fig2.tight_layout()

    # ===== Figure 3: Average Action + Collusion Index (2-panel, per level) =====
    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(10, 8))

    for j, n_level in enumerate(inv_levels):
        # Actions: average over players
        action_means_players = []
        for p in range(num_players):
            am = np.array(env_stats[f"train/all_envs/mean_action/action_player_{p + 1}"])
            action_means_players.append(am[:, j, x_axis])  # (seeds, len(x_axis))
        action_avg = np.mean(action_means_players, axis=0)  # (seeds, T)
        mean_act = action_avg.mean(axis=0)

        window = max(1, len(x_axis) // 50)
        if window > 1:
            mean_act_ma, x_ma = apply_moving_average(mean_act, x_axis, window)
        else:
            mean_act_ma, x_ma = mean_act, x_axis

        label = get_inv_level_label(n_level)
        ax3a.plot(x_ma, mean_act_ma, linewidth=2, color=colors[j], label=label)

    ax3a.axhline(args["competitive_action"], color="#d62728", linestyle="--", linewidth=1.5, alpha=0.7)
    ax3a.axhline(args["collusive_action"], color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.7)
    ax3a.set_ylabel("Average Action", fontsize=14)
    ax3a.set_title(f"{algo}: Average Action per Episode", fontsize=15)
    ax3a.legend(fontsize=9, loc="upper left", frameon=True, ncol=2)

    # Bottom panel: collusion index
    for j, n_level in enumerate(inv_levels):
        seed_data = ci_avg[:, j, :]
        mean_ci = seed_data.mean(axis=0)
        window = max(1, len(x_axis) // 50)
        if window > 1:
            mean_ci_ma, x_ma = apply_moving_average(mean_ci, x_axis, window)
        else:
            mean_ci_ma, x_ma = mean_ci, x_axis
        ax3b.plot(x_ma, mean_ci_ma, linewidth=2, color=colors[j], label=get_inv_level_label(n_level))

    ax3b.axhline(0, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.7)
    ax3b.axhline(1, color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.7)
    ax3b.set_xlabel("Episodes", fontsize=14)
    ax3b.set_ylabel("Collusion Index Δ", fontsize=14)
    ax3b.set_title(f"{algo}: Collusion Index Δ", fontsize=15)

    fig3.tight_layout()

    # ===== Figure 4: Dose-response curve (n vs final Δ) =====
    fig4, ax4 = plt.subplots(1, 1, figsize=(8, 5))
    means = []
    stds = []
    x_positions = []
    for j, n_level in enumerate(inv_levels):
        seed_data = ci_avg[:, j, :]
        final_ci = seed_data[:, -last_frac:].mean(axis=1)
        means.append(final_ci.mean())
        stds.append(final_ci.std())
        x_positions.append(n_level if n_level >= 1 else 0)

    # Sort by x_positions for clean line
    sorted_idx = np.argsort(x_positions)
    x_sorted = [x_positions[i] for i in sorted_idx]
    means_sorted = [means[i] for i in sorted_idx]
    stds_sorted = [stds[i] for i in sorted_idx]
    labels_sorted = [labels[i] for i in sorted_idx]

    ax4.errorbar(
        x_sorted, means_sorted, yerr=stds_sorted,
        fmt="o-", color="#1A5F7A", linewidth=2, markersize=8,
        capsize=5, capthick=1.5,
    )
    for xi, mi, li in zip(x_sorted, means_sorted, labels_sorted):
        ax4.annotate(li, (xi, mi), textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=9, color="#555555")

    ax4.axhline(0, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.7)
    ax4.axhline(1, color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.7)
    ax4.set_xlabel("Number of Inventory Levels (0 = hidden, -1 mapped to 0)", fontsize=13)
    ax4.set_ylabel("Collusion Index Δ (last 10%)", fontsize=13)
    ax4.set_title(f"{algo}: Dose-Response — Discretization vs Collusion", fontsize=15)

    fig4.tight_layout()

    # ===== Save all figures =====
    plot_dir = os.path.join(save_dir, "paper_plots")
    os.makedirs(plot_dir, exist_ok=True)

    paths = {}
    for name, fig in [
        ("collusion_index_by_discretization", fig1),
        ("boxplot_final_delta", fig2),
        ("action_and_collusion", fig3),
        ("dose_response", fig4),
    ]:
        path = os.path.join(plot_dir, f"{name}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
        paths[name] = path
        print(f"Saved: {path}")
        plt.close(fig)

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"  {algo} — Collusion Index Summary (last 10%)")
    print(f"{'='*60}")
    print(f"  {'Level':<25s} {'Mean Δ':>8s} {'Std Δ':>8s}")
    print(f"  {'-'*41}")
    for j, n_level in enumerate(inv_levels):
        seed_data = ci_avg[:, j, :]
        final_ci = seed_data[:, -last_frac:].mean(axis=1)
        print(f"  {get_inv_level_label(n_level):<25s} {final_ci.mean():>8.3f} {final_ci.std():>8.3f}")
    print(f"{'='*60}")

    return paths


def plot_multi_algo_comparison(save_dirs, gen_mean_p=0.5):
    """Compare discretization effects across algorithms (e.g., PPO vs DQN)."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.size"] = 12

    algo_colors = {"PPO": "#8B4B8B", "DQN": "#1A5F7A"}
    algo_markers = {"PPO": "o", "DQN": "s"}

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    for save_dir in save_dirs:
        args, env_stats, _, hyperparam_mapping = load_experiment(save_dir)
        algo = args.get("agent_default", "PPO")
        num_players = args.get("num_players", 2)
        num_iters = args["num_iters"]
        log_interval = max(num_iters // 1000, 5 if num_iters > 1000 else 1)
        x_axis = np.arange(0, num_iters, log_interval)
        last_frac = max(1, len(x_axis) // 10)

        ci_players = []
        for p in range(num_players):
            ci = np.array(env_stats[f"train/collusion_index/mean_player_{p + 1}"])
            ci_players.append(ci[:, :, x_axis])
        ci_avg = np.mean(ci_players, axis=0)

        inv_levels = [m["num_inventory_levels"] for m in hyperparam_mapping]
        x_positions = []
        means = []
        stds = []
        for j, n_level in enumerate(inv_levels):
            seed_data = ci_avg[:, j, :]
            final_ci = seed_data[:, -last_frac:].mean(axis=1)
            x_positions.append(n_level if n_level >= 1 else 0)
            means.append(final_ci.mean())
            stds.append(final_ci.std())

        sorted_idx = np.argsort(x_positions)
        x_sorted = [x_positions[i] for i in sorted_idx]
        means_sorted = [means[i] for i in sorted_idx]
        stds_sorted = [stds[i] for i in sorted_idx]

        color = algo_colors.get(algo, "#333333")
        marker = algo_markers.get(algo, "o")
        ax.errorbar(
            x_sorted, means_sorted, yerr=stds_sorted,
            fmt=f"{marker}-", color=color, linewidth=2, markersize=8,
            capsize=5, capthick=1.5, label=algo,
        )

    ax.axhline(0, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axhline(1, color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Number of Inventory Levels", fontsize=14)
    ax.set_ylabel("Collusion Index Δ (last 10%)", fontsize=14)
    ax.set_title("Dose-Response: Discretization vs Collusion (PPO vs DQN)", fontsize=15)
    ax.legend(fontsize=12, frameon=True)
    fig.tight_layout()

    plot_dir = os.path.join(os.path.dirname(save_dirs[0]), "combined_plots")
    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, "dose_response_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved: {path}")
    plt.close(fig)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot inventory discretization effects on collusion."
    )
    parser.add_argument(
        "--save_dir",
        nargs="+",
        default=["exp/PPO"],
        help="One or more experiment directories (e.g., exp/PPO exp/DQN)",
    )
    parser.add_argument("--gen_mean_p", type=float, default=0.5)
    cli_args = parser.parse_args()

    for sd in cli_args.save_dir:
        print(f"\n{'#'*60}")
        print(f"  Processing: {sd}")
        print(f"{'#'*60}")
        plot_collusion_index_by_discretization(sd, cli_args.gen_mean_p)

    if len(cli_args.save_dir) > 1:
        plot_multi_algo_comparison(cli_args.save_dir, cli_args.gen_mean_p)
