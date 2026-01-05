import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_run(save_dir: str):
    with open(os.path.join(save_dir, "args.pkl"), "rb") as f:
        args = pickle.load(f)

    with open(os.path.join(save_dir, "log_data.pkl"), "rb") as f:
        log_data = pickle.load(f)

    # log_data is (train_log_data, eval_log_data, forced_deviation_log_data)
    if isinstance(log_data, tuple):
        if len(log_data) == 3:
            train_log_data, eval_log_data, forced_dev_log_data = log_data
        elif len(log_data) == 2:
            train_log_data, eval_log_data = log_data
            forced_dev_log_data = None
        else:
            raise ValueError("Unexpected log_data tuple length")
    else:
        # Backward compatibility: some dumps may store only train_log_data
        train_log_data = log_data
        eval_log_data = None
        forced_dev_log_data = None

    all_env_stats, all_a1_metrics, all_a2_metrics = train_log_data
    return args, all_env_stats, eval_log_data


def ensure_plot_dir(save_dir: str) -> str:
    plot_dir = os.path.join(save_dir, "paper_plots")
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


def plot_training_prices(save_dir: str, args, all_env_stats):
    """
    Plot mean price per episode for both players across seeds.
    Uses env metric keys:
      - train/all_envs/mean_action/price_player_1
      - train/all_envs/mean_action/price_player_2
    """
    # X 軸の間引き間隔は既存スクリプトに合わせる
    num_iters = args["num_iters"]
    log_interval = max(num_iters // 1000, 5 if num_iters > 1000 else 1)
    x_axis = np.arange(0, num_iters, log_interval)

    # 形状: [seeds, configs, T] を想定（グリッドサーチなしなら configs=1 相当）
    def slice_env_metric(key: str):
        v = np.array(all_env_stats[key])
        # 次元数に応じて素直にスライス
        if v.ndim == 1:
            # [T]
            return v[x_axis][None, :]  # [1, T]
        elif v.ndim == 2:
            # [seeds, T]
            return v[:, x_axis]
        elif v.ndim >= 3:
            # [seeds, configs, T, ...] → config=0, 以降は先頭を使用
            return v[:, 0, x_axis, ...].reshape(v.shape[0], -1)
        else:
            raise ValueError(f"Unexpected shape for {key}: {v.shape}")

    p1_seeds = slice_env_metric("train/all_envs/mean_action/price_player_1").astype(np.float32)
    p2_seeds = slice_env_metric("train/all_envs/mean_action/price_player_2").astype(np.float32)

    p1_mean = p1_seeds.mean(axis=0)
    p1_std = p1_seeds.std(axis=0)
    p2_mean = p2_seeds.mean(axis=0)
    p2_std = p2_seeds.std(axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    ax.plot(x_axis, p1_mean, label="Player 1", color="#1f77b4", linewidth=1.5)
    ax.fill_between(x_axis, p1_mean - p1_std, p1_mean + p1_std, color="#1f77b4", alpha=0.25)

    ax.plot(x_axis, p2_mean, label="Player 2", color="#ff7f0e", linewidth=1.5)
    ax.fill_between(x_axis, p2_mean - p2_std, p2_mean + p2_std, color="#ff7f0e", alpha=0.25)

    # 競争(ナッシュ)・共謀(独占)価格の目安線
    comp_price = args.get("competitive_price", None)
    coll_price = args.get("collusive_price", None)
    if comp_price is not None:
        ax.axhline(comp_price, color="#d62728", linestyle="--", linewidth=1, label="Competitive")
    if coll_price is not None:
        ax.axhline(coll_price, color="#2ca02c", linestyle="--", linewidth=1, label="Collusive")

    ax.set_title("Mean Price per Episode (±1 SD across seeds)")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Price")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plot_dir = ensure_plot_dir(save_dir)
    out_path = os.path.join(plot_dir, "prices_training.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_eval_prices(save_dir: str, args, eval_log_data):
    """
    Plot per-timestep eval prices if available (deterministic policy rollout).
    Keys: prices_1, prices_2 (shape ~ [T] or [seeds, configs, T])
    """
    if eval_log_data is None:
        print("No eval_log_data found; skipping eval price plot.")
        return

    def slice_eval(key: str):
        v = np.array(eval_log_data[key])
        if v.ndim == 1:
            # [T]
            return v[None, :]  # [1, T]
        elif v.ndim == 2:
            # [seeds, T] もありえる
            return v
        elif v.ndim >= 3:
            # [seeds, configs, T, ...] → config=0 を選択
            return v[:, 0, :].reshape(v.shape[0], -1)
        else:
            raise ValueError(f"Unexpected eval shape for {key}: {v.shape}")

    try:
        p1 = slice_eval("prices_1").astype(np.float32)
        p2 = slice_eval("prices_2").astype(np.float32)
    except KeyError:
        print("Eval prices not present; skipping eval plot.")
        return

    t = np.arange(p1.shape[1])
    p1_mean = p1.mean(axis=0)
    p1_std = p1.std(axis=0)
    p2_mean = p2.mean(axis=0)
    p2_std = p2.std(axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    ax.plot(t, p1_mean, label="Player 1", color="#1f77b4", linewidth=1.5)
    ax.fill_between(t, p1_mean - p1_std, p1_mean + p1_std, color="#1f77b4", alpha=0.25)

    ax.plot(t, p2_mean, label="Player 2", color="#ff7f0e", linewidth=1.5)
    ax.fill_between(t, p2_mean - p2_std, p2_mean + p2_std, color="#ff7f0e", alpha=0.25)

    comp_price = args.get("competitive_price", None)
    coll_price = args.get("collusive_price", None)
    if comp_price is not None:
        ax.axhline(comp_price, color="#d62728", linestyle="--", linewidth=1, label="Competitive")
    if coll_price is not None:
        ax.axhline(coll_price, color="#2ca02c", linestyle="--", linewidth=1, label="Collusive")

    ax.set_title("Eval Price Trajectory per Timestep (±1 SD across seeds)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Price")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plot_dir = ensure_plot_dir(save_dir)
    out_path = os.path.join(plot_dir, "prices_eval.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    # デフォルトはスクリプト配置ディレクトリ基準の exp/DQN（config_DQN_debug の既定出力）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_save_dir = os.path.join(script_dir, "exp", "DQN")
    save_dir = os.environ.get("EC_SAVE_DIR", default_save_dir)
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"save_dir not found: {save_dir}")

    args, all_env_stats, eval_log_data = load_run(save_dir)
    plot_training_prices(save_dir, args, all_env_stats)
    plot_eval_prices(save_dir, args, eval_log_data)


if __name__ == "__main__":
    main()
