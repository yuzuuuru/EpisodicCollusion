# 実装計画（改訂版）：在庫情報の離散化による共謀指数への影響の実証

## 1. 研究目的

在庫情報をエージェントに対して $n$ 段階に離散化（情報粒度を削減）することで、共謀指数 $\Delta$ の増加を抑えられることを実証する。

$$
\Delta_i = \frac{\pi_i^{actual} - \pi_i^{Nash}}{\pi_i^{Monopoly} - \pi_i^{Nash}}
$$

本研究の介入点は「環境ダイナミクス」ではなく「観測情報の解像度」である。

---

## 2. 現状の在庫情報フロー

```
EnvState.inventories (整数)
    ↓ env.step() が obs["inventories"] にコピー
    ↓ runner が env.reset/env.step を vmap して batch_reset/batch_step を生成
    ↓ PPO/DQN の rescale_observations() で [0,1] へスケーリング
    ↓ MLP 入力
```

要点：現行コードでは在庫は整数だが、学習時にはスケーリングされた連続floatとして利用される。

---

## 3. 改訂実装方針（重要修正反映）

## 3-1. `code_training/environment/wrappers.py` に新ラッパー追加（最重要）

### 方針
- **`batch_reset` / `batch_step` ではなく `reset` / `step` をラップする。**
- 理由：runner 側で `env.reset` / `env.step` から `batch_*` を再構築しており、`batch_*` だけ実装したラッパーは効かない可能性がある。

### 仕様
- 観測のみ離散化し、`state.inventories`（実在庫）と報酬計算は変更しない。
- 量子化式は以下を採用：

$$
\text{bucket} = \left\lfloor \frac{inv \cdot (n-1)}{inv_{max}} \right\rfloor
$$

- 実装上は `clip(0, n-1)` を併用。
- **`n=1` は特別扱い**：常に `bucket=0`（完全情報隠蔽）。

### 疑似コード

```python
class InventoryDiscretizationWrapper(GymnaxWrapper):
    def __init__(self, env, num_inventory_levels, initial_inventories):
        super().__init__(env)
        self.n = int(num_inventory_levels)
        self.inv_max = jnp.array(initial_inventories, dtype=jnp.float32)

    def _discretize_obs(self, obs):
        new_obs = dict(obs)
        inv = obs["inventories"].astype(jnp.float32)

        if self.n <= 1:
            bucket = jnp.zeros_like(inv, dtype=jnp.int32)
        else:
            bucket = jnp.floor(inv * (self.n - 1) / self.inv_max).astype(jnp.int32)
            bucket = jnp.clip(bucket, 0, self.n - 1)

        new_obs["inventories"] = bucket
        return new_obs

    def reset(self, key, params=None):
        all_obs, state = self._env.reset(key, params)
        all_obs = tuple(self._discretize_obs(o) for o in all_obs)
        return all_obs, state

    def step(self, key, state, action, params=None):
        all_obs, state, rewards, done, info = self._env.step(key, state, action, params)
        all_obs = tuple(self._discretize_obs(o) for o in all_obs)
        return all_obs, state, rewards, done, info
```

---

## 3-2. `code_training/experiment.py` の `env_setup()` でラッパー適用

### 方針
- `env` を構築した直後〜`return env, env_params` 前で適用する。
- 番兵値を使うため、`n >= 1` のときのみ有効化。

```python
num_inv_levels = args.get("num_inventory_levels", -1)
if num_inv_levels >= 1:
    from environment.wrappers import InventoryDiscretizationWrapper
    env = InventoryDiscretizationWrapper(
        env,
        num_inventory_levels=num_inv_levels,
        initial_inventories=omegaconf.OmegaConf.to_container(args.initial_inventories, resolve=True),
    )
```

---

## 3-3. `code_training/agents/ppo/ppo.py` の正規化修正

`rescale_observations()` で在庫上限を切り替える。

```python
inv_levels = obs_limits.get("inventory_discretization_levels", -1)
if inv_levels >= 1:
    if inv_levels == 1:
        new_observation["inventories"] = jnp.zeros_like(observation["inventories"], dtype=jnp.float32)
    else:
        new_observation["inventories"] = rescale_to_zero_one(
            observation["inventories"], 0, inv_levels - 1
        )
else:
    new_observation["inventories"] = rescale_to_zero_one(
        observation["inventories"], 0, obs_limits["inventory_uppers"]
    )
```

---

## 3-4. `code_training/agents/dqn/dqn.py` も同等修正（必須）

比較実験の整合性のため、DQN 側にも同じ `inventory_discretization_levels` 分岐を入れる。

- `rescale_observations()` の在庫処理を PPO と同等に修正。
- `obs_limits` に `inventory_discretization_levels` を追加。

---

## 3-5. `obs_limits` の追加位置（修正）

`main.py` ではなく、以下の **agent factory** 内で追加する。

- `code_training/agents/ppo/ppo.py` の `make_agent()` で作る `obs_limits`
- `code_training/agents/dqn/dqn.py` の `make_DQN_agent()` で作る `obs_limits`

追加キー：

```python
"inventory_discretization_levels": args.get("num_inventory_levels", -1)
```

---

## 3-6. `code_training/conf/config_PPO.yaml` の設定値（修正）

### 重要
`gridsearch` は内部で list を `jnp.array` 化するため、`null/None` 混在は避ける。

```yaml
# -1: 離散化なし（現状）
#  1: 完全情報隠蔽
#  2,4,8,16: n段階
num_inventory_levels: -1

gridsearch:
  num_inventory_levels: [-1, 1, 2, 4, 8, 16]
```

必要に応じて `config_DQN.yaml` にも同キーを追加する。

---

## 4. 変更不要な箇所

- `code_training/environment/market_env.py` の在庫遷移（実在庫更新）
- 報酬計算 `(price - cost) * quantity_sold`
- Nash/Monopoly 利益計算ロジック
- ネットワーク層構造（入力次元は不変）
- `code_equilibrium_calculations/` 以下

---

## 5. 実験条件の注意

- 効果検証は `env_id: MarketEnv-v1` を使用する。
- `MarketEnv-InfiniteInventoryInfiniteEpisode` は在庫が減らないため、本介入の評価には不適。

---

## 6. 期待される比較軸

- 主効果：`num_inventory_levels` を小さくするほど共謀指数 $\Delta$ が低下するか。
- 比較系列：`-1 (連続)` vs `16` vs `8` vs `4` vs `2` vs `1`。
- 出力：平均 $\Delta$、分散、価格系列、報酬系列、残在庫率。

---

## 7. 実装順序（改訂）

1. `wrappers.py` に `InventoryDiscretizationWrapper(reset/step 版)` を追加
2. `experiment.py` の `env_setup()` で `n>=1` のとき適用
3. `ppo.py` の在庫正規化に `inv_levels` 分岐追加（`n=1` 例外処理含む）
4. `dqn.py` に同等修正
5. PPO/DQN の `obs_limits` に `inventory_discretization_levels` を追加
6. `config_PPO.yaml`（必要なら `config_DQN.yaml`）で `-1` 番兵方式に変更
7. スモークテスト：`n=-1` と `n=2` で短時間実行し、クラッシュなくログが出ることを確認

---

## 8. リスクと回避策

- **0除算リスク**：`n=1` で `n-1=0` → 分岐で回避。
- **型混在リスク**：`null` 混在gridsearch → `-1` 番兵で回避。
- **ラッパー無効化リスク**：`batch_*` だけ実装 → `reset/step` をラップして回避。
- **比較不整合リスク**：PPOのみ対応 → DQNも同仕様に統一。
