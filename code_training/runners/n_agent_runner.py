"""
N-Agent Runner for symmetric multi-agent reinforcement learning.
All agents are treated equally and update simultaneously after each rollout.
This runner maintains compatible batch dimensions with the TwoAgentGridsearchRunner
by using num_opps=1 internally.
"""
from functools import partial
import jax
import time

from jaxtyping import Array
from typing import NamedTuple, Any, List, Tuple
import jax.numpy as jnp
import numpy as np

from watchers import losses_ppo, losses_dqn
from utils import TrainingState, MemoryState
import wandb
import optax
from environment.wrappers import (
    DummyDoubleVecObsWrapper,
    DummyDoubleVecRewWrapper,
)

MAX_WANDB_CALLS = 1000


class Sample(NamedTuple):
    """Object containing a batch of data for one agent.
    Shape: [num_inner, num_opps, num_envs, ...]"""
    observations: Array
    actions: Array
    rewards: Array
    behavior_log_probs: Array
    behavior_values: Array
    dones: Array
    hiddens: Array
    unnormalized_rewards: Array


class EvalSample(NamedTuple):
    """Object containing evaluation data for one agent.
    Shape: [T, num_opps, num_envs]"""
    observations: Array
    actions: Array
    rewards_rescaled: Array
    rewards_unnormalized: Array
    extras: Array


class NAgentRunner:
    """
    Runner for N agents in symmetric multi-agent environment.
    All agents are treated equally: each has independent TrainingState and MemoryState,
    and all update after each rollout.
    
    This runner maintains compatibility with existing agent implementations by using
    the same batch dimensions as TwoAgentGridsearchRunner (with num_opps=1).
    
    Args:
        agents (List[Agent]): List of N agents.
        env: Environment instance.
        save_dir (str): Directory to save models.
        args (dict): Configuration arguments.
    """

    def __init__(self, agents: List, env, save_dir: str, args: dict):
        self.train_steps = 0
        self.train_episodes = 0
        self.start_time = time.time()
        self.args = args
        self.num_agents = args["num_players"]
        self.num_opps = 1  # Fixed to 1 for symmetric n-agent case
        self.random_key = jax.random.PRNGKey(args["seed"])
        self.save_dir = save_dir
        self.competitive_profits = jnp.array(args["competitive_profits"])
        self.competitive_profits_episodetotal = jnp.array(args["competitive_profits_episodetotal"])
        self.collusive_profits = jnp.array(args["collusive_profits"])
        self.collusive_profits_episodetotal = jnp.array(args["collusive_profits_episodetotal"])

        # Helper function to reshape batch dimensions
        def _reshape_opp_dim(x):
            """Flatten num_opps*num_envs to batch dimension."""
            batch_size = args["num_envs"] * self.num_opps
            return jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), x)

        self.reduce_opp_dim = jax.jit(_reshape_opp_dim)

        # VMAP the environment twice: for num_opps and num_envs
        # This maintains same structure as TwoAgentGridsearchRunner
        env.batch_reset = jax.vmap(env.reset, in_axes=(0, None), out_axes=0)
        env.batch_reset = jax.jit(jax.vmap(env.batch_reset, in_axes=(0, None), out_axes=0))
        
        env.batch_step = jax.vmap(env.step, in_axes=(0, 0, 0, None), out_axes=0)
        env.batch_step = jax.jit(jax.vmap(env.batch_step, in_axes=(0, 0, 0, None), out_axes=0))

        # Apply wrappers (same as TwoAgentGridsearchRunner)
        env = DummyDoubleVecObsWrapper(env)
        env = DummyDoubleVecRewWrapper(env)

        # VMAP split for RNG
        self.split = jax.vmap(jax.vmap(jax.random.split, (0, None)), (0, None))

        num_outer_steps = self.args["num_outer_steps"]

        # Set up each agent's batched functions (same structure as agent1 in TwoAgentGridsearchRunner)
        for i, agent in enumerate(agents):
            # For symmetric treatment, all agents have same batching as agent1
            agent.batch_init = jax.vmap(
                agent.make_initial_state, in_axes=(None, 0), out_axes=(None, 0)
            )
            agent.batch_reset = jax.jit(
                jax.vmap(agent.reset_memory, (0, None), 0), static_argnums=1
            )
            agent.batch_policy = jax.jit(
                jax.vmap(agent._policy, in_axes=(None, 0, 0), out_axes=(0, None, 0))
            )
            agent.batch_eval_policy = jax.jit(
                jax.vmap(agent._eval_policy, in_axes=(None, 0, 0), out_axes=(0, None, 0, 0))
            )

        # Initialize agent states and memories
        for agent in agents:
            init_hidden = jnp.tile(
                agent._mem.hidden, (self.num_opps, 1, 1)
            )  # [num_opps, num_envs, 1]
            agent._state, agent._mem = agent.batch_init(
                agent._state.random_key, init_hidden
            )

        def _inner_rollout(carry, unused):
            """Plays 1 step, gets scanned over to produce 1 episode.
            carry: rngs, obs_list, agent_states, agent_mems, env_state, env_params
            All obs, mems have shape [num_opps, num_envs, ...]"""
            (
                rngs,
                obs_list,
                agent_states,
                agent_mems,
                env_state,
                env_params,
            ) = carry

            # Split RNG
            rngs = self.split(rngs, 4)  # [num_opps, num_envs, 4, keydim]
            env_rng = rngs[:, :, 0, :]
            passthrough_rngs = rngs[:, :, 3, :]

            # Get actions from all agents
            actions_list = []
            new_states = []
            new_mems = []
            for i, agent in enumerate(agents):
                a, state, mem = agent.batch_policy(agent_states[i], obs_list[i], agent_mems[i])
                actions_list.append(a)
                new_states.append(state)
                new_mems.append(mem)

            # Stack actions for environment: need shape [num_opps, num_envs, num_agents]
            actions_stacked = jnp.stack(actions_list, axis=-1)

            # Environment step
            (
                next_obs_tuple,
                env_state,
                rewards,  # [num_opps, num_envs, num_agents]
                unnormalized_rewards,
                done,
                info,
            ) = env.batch_step(env_rng, env_state, actions_stacked, env_params)

            # Convert obs tuple to list
            next_obs_list = list(next_obs_tuple)

            # Squeeze done
            done = jnp.squeeze(done, axis=-1)  # [num_opps, num_envs]

            # Process rewards for each agent
            rewards_list = [rewards[:, :, i] for i in range(self.num_agents)]
            
            if args["normalize_rewards_wrapper"]:
                unnorm_rewards_list = [unnormalized_rewards[:, :, i] for i in range(self.num_agents)]
            else:
                unnorm_rewards_list = rewards_list.copy()

            if args["normalize_rewards_manually"]:
                rewards_rescaled_list = [
                    (r - args["normalizing_rewards_min"]) / 
                    (args["normalizing_rewards_max"] - args["normalizing_rewards_min"])
                    for r in rewards_list
                ]
            else:
                rewards_rescaled_list = rewards_list.copy()

            # Create trajectories for each agent
            trajs = []
            for i in range(self.num_agents):
                traj = Sample(
                    obs_list[i],
                    actions_list[i],
                    rewards_rescaled_list[i],
                    new_mems[i].extras["log_probs"],
                    new_mems[i].extras["values"],
                    done,
                    agent_mems[i].hidden,
                    unnorm_rewards_list[i],
                )
                trajs.append(traj)

            return (
                passthrough_rngs,
                next_obs_list,
                new_states,
                new_mems,
                env_state,
                env_params,
            ), (trajs, env_state, info)

        def _outer_rollout(carry, unused):
            """Plays 1 episode, trains all agents, gets scanned over."""
            vals, trajectories = jax.lax.scan(_inner_rollout, carry, None, args["num_inner_steps"])

            (
                rngs,
                obs_list,
                agent_states,
                agent_mems,
                env_state,
                env_params,
            ) = vals

            trajs_stacked, env_traj, info_traj = trajectories
            # trajs_stacked is a list of num_agents Samples, each with arrays of shape
            # [num_inner, num_opps, num_envs, ...]

            # Update all agents
            new_states = []
            new_mems = []
            metrics_list = []
            for i, agent in enumerate(agents):
                traj_i = trajs_stacked[i]
                
                # Reduce opp dim for update (similar to agent1 in TwoAgentGridsearchRunner)
                traj_reduced = jax.tree_util.tree_map(
                    lambda x: x.reshape((x.shape[0], -1) + x.shape[3:]),  # [num_inner, num_opps*num_envs, ...]
                    traj_i
                )
                obs_reduced = self.reduce_opp_dim(obs_list[i])
                mem_reduced = self.reduce_opp_dim(agent_mems[i])
                
                state, _, metrics = agent.update(
                    traj_reduced,
                    obs_reduced,
                    agent_states[i],
                    mem_reduced,
                )
                new_states.append(state)
                # Reset memory after update
                new_mem = agent.batch_reset(agent_mems[i], False)
                new_mems.append(new_mem)
                metrics_list.append(metrics)

            return (
                rngs,
                obs_list,
                new_states,
                new_mems,
                env_state,
                env_params,
            ), (trajs_stacked, env_traj, info_traj, metrics_list)

        def _rollout(
            _rng_run: jnp.ndarray,
            _agent_states: List[TrainingState],
            _agent_mems: List[MemoryState],
            _env_params: Any,
        ):
            """Run one complete rollout with all agents."""
            # Generate RNGs: [num_opps, num_envs, keydim]
            rngs_split_envs = jax.random.split(_rng_run, args["num_envs"])
            rngs_split_opps_envs = jnp.concatenate([rngs_split_envs] * self.num_opps)
            rngs = rngs_split_opps_envs.reshape((self.num_opps, args["num_envs"], -1))

            # Reset environment
            obs_tuple, env_state = env.batch_reset(rngs, _env_params)
            obs_list = list(obs_tuple)

            # Reset agent memories
            reset_mems = [agent.batch_reset(_agent_mems[i], False) for i, agent in enumerate(agents)]

            vals, stack = jax.lax.scan(
                _outer_rollout,
                (
                    rngs,
                    obs_list,
                    _agent_states,
                    reset_mems,
                    env_state,
                    _env_params,
                ),
                None,
                length=num_outer_steps,
            )

            (
                rngs,
                obs_list,
                agent_states,
                agent_mems,
                env_state,
                env_params,
            ) = vals

            trajs_stacked, env_traj, info_traj, metrics_list = stack

            # Compute environment statistics
            env_stats = self._compute_env_stats(
                env_state, trajs_stacked, env_traj, info_traj,
                args["num_envs"], self.num_opps, args["num_outer_steps"],
                _env_params.initial_inventories
            )

            return (
                env_stats,
                agent_states,
                agent_mems,
                metrics_list,
            )

        self.rollout = jax.jit(_rollout)

        def _eval_ep(carry, unused):
            """Run one evaluation step."""
            (
                rngs,
                obs_list,
                agent_states,
                agent_mems,
                env_state,
                env_params,
            ) = carry

            # Get actions from all agents (deterministic)
            actions_list = []
            new_states = []
            new_mems = []
            extras_list = []
            for i, agent in enumerate(agents):
                a, state, mem, extras = agent.batch_eval_policy(agent_states[i], obs_list[i], agent_mems[i])
                actions_list.append(a)
                new_states.append(state)
                new_mems.append(mem)
                extras_list.append(extras)

            actions_stacked = jnp.stack(actions_list, axis=-1)

            (
                next_obs_tuple,
                env_state,
                rewards,
                unnormalized_rewards,
                _,
                info,
            ) = env.batch_step(rngs, env_state, actions_stacked, env_params)

            next_obs_list = list(next_obs_tuple)

            rewards_list = [rewards[:, :, i] for i in range(self.num_agents)]
            if args["normalize_rewards_wrapper"]:
                unnorm_rewards_list = [unnormalized_rewards[:, :, i] for i in range(self.num_agents)]
            else:
                unnorm_rewards_list = rewards_list.copy()

            if args["normalize_rewards_manually"]:
                rewards_rescaled_list = [
                    (r - args["normalizing_rewards_min"]) / 
                    (args["normalizing_rewards_max"] - args["normalizing_rewards_min"])
                    for r in rewards_list
                ]
            else:
                rewards_rescaled_list = rewards_list.copy()

            eval_trajs = []
            for i in range(self.num_agents):
                traj = EvalSample(
                    obs_list[i], actions_list[i],
                    rewards_rescaled_list[i], unnorm_rewards_list[i],
                    extras_list[i]
                )
                eval_trajs.append(traj)

            return (
                rngs,
                next_obs_list,
                new_states,
                new_mems,
                env_state,
                env_params,
            ), (eval_trajs, env_state, info)

        def _eval_rollout(rng_eval, agent_states, agent_mems, env_params_eval):
            """Run evaluation rollout."""
            # [num_opps, 1 (num_envs), keydim]
            rngs_eval = jax.random.split(rng_eval, 1)
            rngs_eval = jnp.concatenate([rngs_eval] * self.num_opps).reshape(self.num_opps, 1, -1)

            obs_tuple, env_state = env.batch_reset(rngs_eval, env_params_eval)
            obs_list = list(obs_tuple)

            # Reset memories for eval
            eval_mems = [agent.batch_reset(agent_mems[i], True) for i, agent in enumerate(agents)]

            initial_carry = (
                rngs_eval,
                obs_list,
                agent_states,
                eval_mems,
                env_state,
                env_params_eval,
            )

            _, eval_trajectories = jax.lax.scan(
                _eval_ep, initial_carry, None, args["num_inner_steps"]
            )

            eval_trajs_list, eval_env_traj, eval_info_traj = eval_trajectories

            eval_log_data = self._compute_eval_stats(
                eval_env_traj, eval_trajs_list, eval_info_traj, env_params_eval.initial_inventories
            )

            return eval_log_data

        self.eval_rollout = jax.jit(_eval_rollout)

    def _compute_env_stats(self, state, trajs_stacked, env_traj, info_traj, 
                           num_envs, num_opps, num_outer, initial_inventories):
        """Compute training statistics for n agents."""
        episode_length = self.args["num_inner_steps"]
        competitive_profits_episode = self.competitive_profits * episode_length
        collusive_profits_episode = self.collusive_profits * episode_length
        num_eps = num_envs * num_opps * num_outer
        
        stats = {}
        
        for i in range(self.num_agents):
            # trajs_stacked[i] is Sample with shape [num_outer, num_inner, num_opps, num_envs, ...]
            agent_rewards = trajs_stacked[i].unnormalized_rewards  # [n_outer, n_inner, n_opp, n_env]
            
            # Episodic profits: sum over inner steps
            episodic_profits = agent_rewards.sum(axis=1)  # [num_outer, num_opps, num_envs]
            
            # Collusion index
            if i < len(self.competitive_profits) and i < len(self.collusive_profits):
                comp_prof = competitive_profits_episode[i]
                coll_prof = collusive_profits_episode[i]
                collusion_index = (episodic_profits - comp_prof) / (coll_prof - comp_prof + 1e-8)
                stats[f"train/collusion_index/mean_player_{i+1}"] = collusion_index.mean()
            
            # Mean rewards
            stats[f"train/all_envs/rewards/mean_player_{i+1}"] = trajs_stacked[i].rewards.mean()
            
            # Mean actions
            stats[f"train/all_envs/mean_action/action_player_{i+1}"] = trajs_stacked[i].actions.mean().astype(jnp.float32)
            
            # Greedy actions (behavior values for DQN)
            stats[f"vmap_metrics/greedy_action_mean_player_{i+1}"] = trajs_stacked[i].behavior_values.mean()
            
            # Profits
            stats[f"vmap_metrics/total_profit_mean_player_{i+1}"] = episodic_profits.mean()
            stats[f"vmap_metrics/total_profit_var_player_{i+1}"] = episodic_profits.var()

        # Prices from env trajectory
        for i in range(self.num_agents):
            prices = env_traj.env_state.env_state.last_prices[..., i]
            stats[f"train/all_envs/mean_action/price_player_{i+1}"] = prices.mean()
            
            # Inventory at end
            inv = env_traj.env_state.env_state.inventories[:, -1, :, :, i]
            inv_pct = inv / initial_inventories[i]
            stats[f"train/all_envs/mean_quantity/inv_player_{i+1}"] = inv_pct.mean()

        return stats

    def _compute_eval_stats(self, env_traj, trajs_list, info_traj, initial_inventories):
        """Compute evaluation statistics for n agents."""
        stats = {}
        
        for i in range(self.num_agents):
            traj = trajs_list[i]  # [num_inner, num_opps, num_envs]
            
            # Rewards
            rewards = traj.rewards_unnormalized.squeeze()
            stats[f"rewards_{i+1}"] = rewards
            stats[f"rewards_rescaled_{i+1}"] = traj.rewards_rescaled.squeeze()
            
            # Actions
            stats[f"actions_{i+1}"] = traj.actions.squeeze()
            
            # Prices
            prices = env_traj.env_state.env_state.last_prices[..., i].squeeze()
            stats[f"prices_{i+1}"] = prices
            
            # Demands and quantities
            stats[f"demands_{i+1}"] = info_traj["demands"][..., i].squeeze()
            stats[f"quantities_sold_{i+1}"] = info_traj["quantity_sold"][..., i].squeeze()
            
            # Inventory
            inv = env_traj.env_state.env_state.inventories[..., i].squeeze()
            stats[f"inventories_{i+1}"] = inv / initial_inventories[i]
            
            # Collusion index
            if i < len(self.competitive_profits):
                collusion_index = (rewards - self.competitive_profits[i]) / (
                    self.collusive_profits[i] - self.competitive_profits[i] + 1e-8
                )
                stats[f"collusion_index_{i+1}"] = collusion_index
            
            # Extras
            stats[f"extras_{i+1}"] = jax.tree.map(
                lambda x: x.squeeze() if isinstance(x, jnp.ndarray) else x,
                traj.extras
            )

        return stats

    @partial(jax.jit, static_argnums=(0, 4))
    def run_loop(self, seed, env_params, agents, num_iters):
        """Run training loop for all agents."""
        rng = jax.random.PRNGKey(seed)
        init_rng = rng

        # Get initial states and memories
        agent_states = [agent._state for agent in agents]
        agent_mems = [agent._mem for agent in agents]

        def scan_body(carry, i):
            rng, agent_states, agent_mems = carry
            rng, rng_run = jax.random.split(rng)

            (
                env_stats,
                agent_states,
                agent_mems,
                metrics_list,
            ) = self.rollout(rng_run, agent_states, agent_mems, env_params)

            log_data = (env_stats, metrics_list)

            return (rng, agent_states, agent_mems), log_data

        initial_carry = (rng, agent_states, agent_mems)
        (rng, agent_states, agent_mems), log_data = jax.lax.scan(
            scan_body, initial_carry, jnp.arange(num_iters)
        )

        train_log_data = log_data

        # Update agents with final states
        for i, agent in enumerate(agents):
            agent._state = agent_states[i]
            agent._mem = agent_mems[i]

        # Evaluation rollout
        eval_log_data = self.eval_rollout(rng, agent_states, agent_mems, env_params)

        log_data = train_log_data, eval_log_data

        return agents, log_data, init_rng

    def log_data(self, log_data, agents, num_iters, watchers):
        """Process and log training data."""
        train_log_data, eval_log_data = log_data
        all_env_stats, all_metrics_list = train_log_data

        log_interval = max(num_iters // MAX_WANDB_CALLS, 5 if num_iters > 1000 else 1)

        for i in range(num_iters):
            if i % log_interval == 0:
                env_stats = jax.tree.map(lambda x: x[i], all_env_stats)

                if watchers:
                    # Log agent metrics
                    for j, (watcher, agent) in enumerate(zip(watchers, agents)):
                        agent_metrics = jax.tree.map(
                            lambda x: x[i] if hasattr(x, '__getitem__') else x,
                            all_metrics_list[j]
                        )
                        agent._logger.metrics = agent._logger.metrics | agent_metrics
                        if agent_metrics.get("trained", True):
                            watcher(agent)

                    env_stats = jax.tree_util.tree_map(
                        lambda x: x.item() if hasattr(x, 'item') else x, env_stats
                    )
                    wandb.log({"train_iteration": i, **env_stats}, step=i, commit=True)

        print(f"Logging complete for {num_iters} iterations")

    def tree_flatten(self):
        children = (self.random_key,)
        aux_data = {
            "args": self.args,
            "num_agents": self.num_agents,
            "num_opps": self.num_opps,
            "competitive_profits": self.competitive_profits,
            "collusive_profits": self.collusive_profits,
            "competitive_profits_episodetotal": self.competitive_profits_episodetotal,
            "collusive_profits_episodetotal": self.collusive_profits_episodetotal,
            "start_time": self.start_time,
            "save_dir": self.save_dir,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.random_key = children[0]
        for key, value in aux_data.items():
            setattr(obj, key, value)
        return obj


jax.tree_util.register_pytree_node(
    NAgentRunner,
    NAgentRunner.tree_flatten,
    NAgentRunner.tree_unflatten,
)
