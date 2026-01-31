import time
import logging
import shutil
import os
import pickle
from typing import Dict, Any
from copy import deepcopy
import collections.abc
from functools import partial
from hydra.core.hydra_config import HydraConfig

# Set XLA flags before importing JAX
logical_cores = os.cpu_count() or 1
physical_cores_estimate = logical_cores // 2
num_cores_to_use = max(physical_cores_estimate // 2, 1)
os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={num_cores_to_use}'
print(f"Logical CPU cores: {logical_cores}")
print(f"Estimated physical cores: {physical_cores_estimate}")
print(f"Using {num_cores_to_use} cores for JAX parallel execution")

import wandb
import omegaconf
import hydra
import logging

import jax
import jax.numpy as jnp
import numpy as np

from experiment_reformulated import global_setup, env_setup, agent_setup, watcher_setup, runner_setup
from utils import get_unique_run_name, get_size_in_megabytes, flatten_dict

config_path = "conf"
omegaconf.OmegaConf.register_new_resolver("product", lambda x, y: x * y)


def possible_prices_func(
    p_N=1.471, p_M=1.925, num_price_steps=15, xi=0.1, which_price_grid="constrained"
):
    """Uses Calvano's discretized interval of possible prices."""
    if which_price_grid == "unconstrained":
        lower = 1.471 - xi * (1.925 - 1.471)
        upper = 1.925 + xi * (1.925 - 1.471)
    else:
        lower = p_N - xi * (p_M - p_N)
        upper = p_M + xi * (p_M - p_N)
    res = np.linspace(lower, upper, num_price_steps)
    res = np.round(res, 3)

    possible_price_nash = min(res, key=lambda x: abs(x - p_N))
    possible_action_nash: int = res.tolist().index(possible_price_nash)
    possible_price_monopolistic = min(res, key=lambda x: abs(x - p_M))
    possible_action_monopolistic: int = res.tolist().index(possible_price_monopolistic)

    return (
        res.tolist(),
        lower,
        upper,
        possible_action_nash,
        possible_action_monopolistic,
        float(possible_price_nash),
        float(possible_price_monopolistic),
    )


def equilibrium_profits(
    episode_length: int,
    price_nash: float,
    quantity_nash: int,
    price_monopolistic: float,
    quantity_monopolistic: int,
    marginal_costs: list,
):
    """Calculates the profits of the Nash equilibrium and the monopolistic price."""
    profits_nash = []
    profits_nash_episode = []
    profits_monopolistic = []
    profits_monopolistic_episode = []

    for cost_i in marginal_costs:
        profits_nash.append((price_nash - cost_i) * quantity_nash)
        profits_nash_episode.append(profits_nash[-1] * episode_length)
        profits_monopolistic.append((price_monopolistic - cost_i) * quantity_monopolistic)
        profits_monopolistic_episode.append(profits_monopolistic[-1] * episode_length)
    return (
        profits_nash,
        profits_nash_episode,
        profits_monopolistic,
        profits_monopolistic_episode,
    )


def calc_nash_price_and_quantity(constraint=1000, num_players=2):
    """Calculate the Nash equilibrium price for the given setting."""
    if num_players == 2:
        if constraint > 470:
            price_nash = 1.471
        elif constraint == 455:
            price_nash = 1.617
        elif constraint == 440:
            price_nash = 1.693
        elif constraint == 425:
            price_nash = 1.74
        elif constraint == 420:
            price_nash = 1.7588
        elif constraint == 410:
            price_nash = 1.795
        elif constraint == 395:
            price_nash = 1.843
        elif constraint == 380:
            price_nash = 1.885
        elif constraint == 365:
            price_nash = 1.925
        elif constraint == 230:
            price_nash = 2.213
        else:
            print("constraint not found")
            exit()
        quantity_nash = int(min(470, constraint))
    elif num_players == 3:
        # N=3 equilibrium data (ALIGNED WITH main.py)
        # Data points: inventory -> nash_price
        # 50 -> 2.708, 100 -> 2.486, 150 -> 2.325, 200 -> 2.173, 250 -> 2.0, 305 -> 1.681, 320 -> 1.481
        if constraint >= 320:
            price_nash = 1.481
        elif constraint >= 305:
            price_nash = 1.681
        elif constraint >= 250:
            price_nash = 2.0
        elif constraint >= 200:
            price_nash = 2.173
        elif constraint >= 150:
            price_nash = 2.325
        elif constraint >= 100:
            price_nash = 2.486
        elif constraint >= 50:
            price_nash = 2.708
        else:
            print(f"constraint {constraint} not found for N=3")
            exit()
        # For N=3, quantity_nash needs to be determined based on demand
        # Using similar logic to N=2, but adjust based on equilibrium
        quantity_nash = int(constraint)  # Placeholder: adjust as needed
    else:
        print(f"num_players={num_players} not supported")
        exit()
    
    return price_nash, quantity_nash


def calc_monopolistic_price_and_quantity(constraint=1000, num_players=2):
    """Calculate the monopolistic price.
    N=2: reward: 337
    N=3: Based on provided equilibrium calculations
    """
    if num_players == 2:
        price_monopolistic = 1.925  # Calvano setting
        quantity_monopolistic = 365
    elif num_players == 3:
        # N=3 monopoly data (ALIGNED WITH main.py)
        # For inventory <= 250, nash = monop (fully constrained)
        # For inventory > 250, monop = 2.0
        if constraint >= 305:
            price_monopolistic = 2.0
        elif constraint >= 250:
            price_monopolistic = 2.0
        elif constraint >= 200:
            price_monopolistic = 2.173  # Same as nash when constrained
        elif constraint >= 150:
            price_monopolistic = 2.325
        elif constraint >= 100:
            price_monopolistic = 2.486
        elif constraint >= 50:
            price_monopolistic = 2.708
        else:
            print(f"constraint {constraint} not found for N=3 monopoly")
            exit()
        # Quantity at monopoly price - use constraint as placeholder
        quantity_monopolistic = int(constraint)
    else:
        print(f"num_players={num_players} not supported in calc_monopolistic_price_and_quantity")
        exit()
    
    return price_monopolistic, quantity_monopolistic


def calc_rewards_range(which_price_grid, constraint, num_players=2):
    """Returns the lowest and highest reward achievable.
    Note: N=3 values are provisional and should be updated with actual calculations.
    """
    if num_players == 2:
        if which_price_grid == "unconstrained":
            if constraint > 470:
                lowest_reward = 63
                highest_reward = 445
            elif constraint == 420:
                lowest_reward = 68
                highest_reward = 379
            else:
                lowest_reward = 63
                highest_reward = 445
        elif which_price_grid == "constrained":
            if constraint > 470:
                lowest_reward = 63
                highest_reward = 445
            elif constraint == 420:
                lowest_reward = 218
                highest_reward = 368
            else:
                lowest_reward = 63
                highest_reward = 445
    elif num_players == 3:
        # N=3 reward ranges to match working 3-agent NO Reformulate experiment
        # Working setup: normalizing_rewards_min=153, normalizing_rewards_max=250
        # These values should match config_PPO_n_agents.yaml
        if which_price_grid == "unconstrained":
            lowest_reward = 0
            highest_reward = 450
        else:  # constrained
            # Match the working 3-agent setup
            lowest_reward = 153
            highest_reward = 250
    else:
        print(f"num_players={num_players} not supported in calc_rewards_range")
        lowest_reward = 50
        highest_reward = 400
    
    return lowest_reward, highest_reward


def update_dict_recursively(cfg, update):
    for k, v in update.items():
        if v is not None and isinstance(v, collections.abc.Mapping):
            cfg[k] = update_dict_recursively(cfg.get(k, {}), v)
        else:
            cfg[k] = v
    return cfg


@hydra.main(config_path=config_path, config_name="config_PPO_reformulated", version_base=None)
def main(args):
    config_name = HydraConfig.get().job.config_name
    print(f"Config name: {config_name}")
    args.num_inner_steps = args.time_horizon
    
    # Determine agent types for reward gamma
    default_agent = args.get("agent_default", None)
    agent_types = [args.get(f"agent{i+1}", default_agent) for i in range(args.num_players)]
    
    if all(a == "DQN" for a in agent_types):
        args.normalizing_rewards_gamma = args.dqn_default.discount
    elif all(a == "PPO" for a in agent_types):
        args.normalizing_rewards_gamma = args.ppo_default.gamma
    elif not hasattr(args, "normalizing_rewards_gamma") or args.normalizing_rewards_gamma is None:
        args.normalizing_rewards_gamma = args.ppo_default.gamma

    args.normalizing_rewards_min, args.normalizing_rewards_max = calc_rewards_range(
        args.which_price_grid, args.initial_inventories[0], num_players=args.num_players
    )
    price_nash, quantity_nash = calc_nash_price_and_quantity(
        args.initial_inventories[0], num_players=args.num_players
    )
    price_monopolistic, quantity_monopolistic = calc_monopolistic_price_and_quantity(
        args.initial_inventories[0], num_players=args.num_players
    )
    args.initial_inventories = [inv * args.time_horizon for inv in args.initial_inventories]

    if getattr(args.wandb, "log", False):
        args.wandb.name = get_unique_run_name(
            args.wandb.name, args.wandb.project, args.wandb.entity
        )

    if args.possible_prices == None:
        (
            args.possible_prices,
            args.min_price,
            args.max_price,
            args.competitive_action,
            args.collusive_action,
            args.competitive_price,
            args.collusive_price,
        ) = possible_prices_func(
            price_nash,
            price_monopolistic,
            num_price_steps=args.num_prices,
            xi=args.xi,
            which_price_grid=args.which_price_grid,
        )

    (
        args.competitive_profits,
        args.competitive_profits_episodetotal,
        args.collusive_profits,
        args.collusive_profits_episodetotal,
    ) = equilibrium_profits(
        args.time_horizon,
        args.competitive_price,
        quantity_nash,
        args.collusive_price,
        quantity_monopolistic,
        args.marginal_costs,
    )

    """Set up main"""
    logger = logging.getLogger()

    save_dir = global_setup(args)
    print(save_dir)
    config_save_path = os.path.join("./conf/archive/", save_dir[6:])
    os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
    shutil.copy2(f"conf/{config_name}.yaml", config_save_path)
    print(f"Config file saved to: {config_save_path}")

    env, env_params = env_setup(args, logger)
    assert args.time_horizon == args.num_inner_steps

    print(f"num actions: {env.num_actions}")

    obs_shape = jax.tree_util.tree_map(lambda x: x.shape, env.observation_space(env_params))
    obs_dtypes = jax.tree_util.tree_map(lambda x: x.dtype, env.observation_space(env_params))
    print(f"obs shape: {obs_shape}")
    print(f"obs dtypes: {obs_dtypes}")

    # Print reformulation info
    obs_reformulation = args.get("observation_reformulation", {})
    print(f"Observation reformulation enabled: {obs_reformulation.get('enabled', False)}")
    print(f"Observation reformulation type: {obs_reformulation.get('type', 'none')}")

    watchers = watcher_setup(args, logger)
    if not args.wandb.log:
        watchers = False
    print(f"Watchers: {watchers}")

    agent_list = agent_setup(args, env, env_params, logger)
    runner = runner_setup(args, env, agent_list, save_dir, logger)

    print(f"Number of Training Iterations: {args.num_iters}")

    # Load the update dict
    update_dict = omegaconf.OmegaConf.to_container(args.gridsearch)

    doing_gridsearch = (
        any(leaf is not None for leaf in jax.tree_util.tree_leaves(update_dict))
        or args.get("num_seeds") > 1
    )
    print(f"Doing gridsearch: {doing_gridsearch}")

    update_dict = jax.tree_util.tree_map(
        lambda x: jnp.array(x), update_dict, is_leaf=lambda x: isinstance(x, list)
    )
    leaves, treedef = jax.tree_util.tree_flatten(
        update_dict, is_leaf=lambda x: isinstance(x, jax.Array)
    )
    leaves_idx = [jnp.arange(len(leaf)) for leaf in leaves]
    meshgrid = jnp.meshgrid(*leaves_idx)
    update_dict = jax.tree_util.tree_map(
        lambda idx, x: x[idx.reshape(-1), ...],
        jax.tree_util.tree_unflatten(treedef, meshgrid),
        update_dict,
    )
    print(f"Update dict: {update_dict}")

    args = omegaconf.OmegaConf.to_container(args, resolve=True)

    args_path = os.path.join(save_dir, "args.pkl")
    with open(args_path, "wb") as f:
        pickle.dump(args, f)
        print(f"--- Args saved to {args_path} ---")

    def run_experiment(seed: int, update_dict: Dict[str, Any]):
        """Run a single experiment with given seed and config updates."""
        args_exp = deepcopy(args)
        args_exp = update_dict_recursively(args_exp, update_dict)
        args_exp["seed"] = seed
        args_exp["ppo_default"]["entropy_coeff_horizon"] = int(
            args_exp["ppo_default"]["entropy_anneal_duration"]
            * args_exp["num_iters"]
            * args_exp["num_envs"]
            * args_exp["num_outer_steps"]
            * args_exp["num_inner_steps"]
        )
        agent_list = agent_setup(args_exp, env, env_params, logger)
        runner = runner_setup(args_exp, env, agent_list, save_dir, logger)
        agents, log_data, init_rng = runner.run_loop(
            args_exp["seed"], env_params, agent_list, args_exp["num_iters"]
        )
        return agents, log_data, init_rng

    rng = jax.random.PRNGKey(args.get("seed"))

    print(f"--- Starting training loop ---")
    if doing_gridsearch:
        print(f"--- Running gridsearch ---")
        
        if args.get("num_seeds") > 1:
            seeds = jnp.array(
                [
                    args.get("seed") + i * args.get("num_players")
                    for i in range(args.get("num_seeds"))
                ]
            )
            print(f"seeds: {seeds}")
        else:
            seeds = jnp.array([args.get("seed")])
        
        num_devices_available = jax.local_device_count()
        num_seeds_actual = len(seeds)
        use_pmap = num_seeds_actual > 1 and num_devices_available > 1
        
        print(f"Number of JAX devices available: {num_devices_available}")
        print(f"Number of seeds: {num_seeds_actual}")
        
        if use_pmap:
            num_devices = min(num_seeds_actual, num_devices_available)
            pad_size = (num_devices - (num_seeds_actual % num_devices)) % num_devices
            if pad_size > 0:
                seeds_padded = jnp.concatenate([seeds, seeds[:pad_size]])
                print(f"Padded seeds from {num_seeds_actual} to {len(seeds_padded)}")
            else:
                seeds_padded = seeds
                pad_size = 0
            
            print(f"Using {num_devices} devices out of {num_devices_available} available")
            seeds_reshaped = seeds_padded.reshape(num_devices, -1)
            
            run_experiment_parallel = jax.pmap(
                jax.vmap(
                    jax.vmap(
                        run_experiment,
                        in_axes=(None, jax.tree.map(lambda x: 0, update_dict)),
                    ),
                    in_axes=(0, None),
                ),
                in_axes=(0, None),
            )
        else:
            run_experiment_parallel = jax.vmap(
                jax.vmap(
                    run_experiment,
                    in_axes=(None, jax.tree.map(lambda x: 0, update_dict)),
                ),
                in_axes=(0, None),
            )
            seeds_reshaped = seeds

        run_time = time.time()
        agents, log_data, init_rng = run_experiment_parallel(seeds_reshaped, update_dict)
        
        if use_pmap:
            agents = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), agents)
            log_data = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), log_data)
            init_rng = init_rng.reshape(-1, *init_rng.shape[2:])
            
            if pad_size > 0:
                agents = jax.tree.map(lambda x: x[:num_seeds_actual], agents)
                log_data = jax.tree.map(lambda x: x[:num_seeds_actual], log_data)
                init_rng = init_rng[:num_seeds_actual]
        
        jax.block_until_ready(agents)
        jax.block_until_ready(log_data)
        jax.block_until_ready(init_rng)

        update_dict_path = os.path.join(save_dir, "update_dict.pkl")
        with open(update_dict_path, "wb") as f:
            pickle.dump(update_dict, f)
            print(f"--- Update dict saved to {update_dict_path} ---")
        
        seeds_path = os.path.join(save_dir, "seeds.pkl")
        with open(seeds_path, "wb") as f:
            pickle.dump(seeds, f)
            print(f"--- Seeds saved to {seeds_path} ---")

        flattened_update_dict = flatten_dict(update_dict)
        hyperparam_mapping = []
        num_configs = len(next(iter(flattened_update_dict.values())))
        for idx in range(num_configs):
            config = {}
            for key, values in flattened_update_dict.items():
                config[key] = values[idx]
            hyperparam_mapping.append(config)

        hyperparam_mapping_path = os.path.join(save_dir, "hyperparam_mapping.pkl")
        with open(hyperparam_mapping_path, "wb") as f:
            pickle.dump(hyperparam_mapping, f)
            print(f"--- Hyperparameter mapping saved to {hyperparam_mapping_path} ---")

        for i, agent in enumerate(agents):
            agent.save_state(os.path.join(save_dir, f"agent_{i+1}_state.pkl"))
        print(f"--- {len(agents)} Agents saved to {save_dir} ---")

    else:
        print(f"--- Running single training ---")
        run_time = time.time()
        seed = args.get("seed")
        agents, log_data, init_rng = runner.run_loop(
            seed, env_params, agent_list, args.get("num_iters")
        )
    print(f"--- Train time: {time.time() - run_time:.3f} seconds ---")

    log_data_path = os.path.join(save_dir, "log_data.pkl")
    with open(log_data_path, "wb") as f:
        pickle.dump(log_data, f)
        print(f"--- Log data saved to {log_data_path} ---")

    if os.path.exists(log_data_path):
        with open(log_data_path, "rb") as f:
            log_data = pickle.load(f)
            print(f"--- Log data loaded from {log_data_path} ---")


if __name__ == "__main__":
    main()
