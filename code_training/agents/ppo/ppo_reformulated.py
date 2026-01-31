import jax
from jaxtyping import Array
from typing import NamedTuple, Tuple, Dict, Any
import optax
import haiku as hk
import jax.numpy as jnp
from utils import (
    MemoryState,
    TrainingState,
    Logger,
    add_batch_dim,
    get_advantages,
    float_precision,
)
from agents.ppo.networks_reformulated import make_marketenv_network_reformulated
import pickle


class Batch(NamedTuple):
    """A batch of data; all shapes are expected to be [B, ...]."""

    observations: Array
    actions: Array
    advantages: Array
    target_values: Array
    behavior_values: Array
    behavior_log_probs: Array


class PPOReformulated:
    """A PPO agent for reformulated observations"""

    def __init__(
        self,
        network: NamedTuple,
        optimizer: optax.GradientTransformation,
        random_key: jnp.ndarray,
        obs_spec: Tuple,
        obs_limits: dict,
        obs_type: str = "original",  # New: "original", "min_max_mean", "top_k"
        num_envs: int = 4,
        num_minibatches: int = 16,
        num_epochs: int = 4,
        clip_value: bool = True,
        value_coeff: float = 0.5,
        anneal_entropy: bool = False,
        entropy_coeff_start: float = 0.1,
        entropy_coeff_end: float = 0.01,
        entropy_coeff_horizon: int = 3_000_000,
        entropy_clipping: bool = False,
        ppo_clipping_epsilon: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        tabular: bool = False,
        player_id: int = 0,
        num_iterations: int = 1,
        initial_learning_rate: float = 1e-3,
        lr_scheduling: bool = False,
    ):
        self._initial_learning_rate = initial_learning_rate
        self._lr_scheduling = lr_scheduling
        self._total_num_transitions = num_iterations * num_epochs * num_minibatches
        self._obs_type = obs_type

        @jax.jit
        def rescale_to_zero_one(x, lower, upper):
            return (x - lower) / (upper - lower)

        @jax.jit
        def rescale_observations(observation: Dict, obs_limits: dict):
            """Rescale observations based on observation type."""
            new_observation = observation.copy()
            
            if obs_type == "min_max_mean":
                # Rescale reformulated min/max/mean observations
                new_observation["price_min"] = rescale_to_zero_one(
                    observation["price_min"],
                    obs_limits["last_prices_lower"],
                    obs_limits["last_prices_upper"],
                )
                new_observation["price_max"] = rescale_to_zero_one(
                    observation["price_max"],
                    obs_limits["last_prices_lower"],
                    obs_limits["last_prices_upper"],
                )
                new_observation["price_mean"] = rescale_to_zero_one(
                    observation["price_mean"],
                    obs_limits["last_prices_lower"],
                    obs_limits["last_prices_upper"],
                )
                new_observation["inventory_min"] = rescale_to_zero_one(
                    observation["inventory_min"], 0, jnp.max(obs_limits["inventory_uppers"])
                )
                new_observation["inventory_max"] = rescale_to_zero_one(
                    observation["inventory_max"], 0, jnp.max(obs_limits["inventory_uppers"])
                )
                new_observation["inventory_mean"] = rescale_to_zero_one(
                    observation["inventory_mean"], 0, jnp.max(obs_limits["inventory_uppers"])
                )
                new_observation["t"] = rescale_to_zero_one(observation["t"], 0, obs_limits["t_upper"])
                
            elif obs_type == "top_k" or obs_type == "set_top_k":
                # Rescale reformulated top-k or set_top_k observations
                new_observation["prices_sorted"] = rescale_to_zero_one(
                    observation["prices_sorted"],
                    obs_limits["last_prices_lower"],
                    obs_limits["last_prices_upper"],
                )
                new_observation["inventories_sorted"] = rescale_to_zero_one(
                    observation["inventories_sorted"], 0, jnp.max(obs_limits["inventory_uppers"])
                )
                new_observation["t"] = rescale_to_zero_one(observation["t"], 0, obs_limits["t_upper"])
                
            else:
                # Original observation format
                new_observation["inventories"] = rescale_to_zero_one(
                    observation["inventories"], 0, obs_limits["inventory_uppers"]
                )
                new_observation["last_actions"] = rescale_to_zero_one(
                    observation["last_actions"], 0, obs_limits["last_actions_upper"]
                )
                new_observation["last_prices"] = rescale_to_zero_one(
                    observation["last_prices"],
                    obs_limits["last_prices_lower"],
                    obs_limits["last_prices_upper"],
                )
                new_observation["t"] = rescale_to_zero_one(observation["t"], 0, obs_limits["t_upper"])
                
            return new_observation

        @jax.jit
        def policy(state: TrainingState, observation: Dict, mem: MemoryState):
            """Agent policy to select actions"""
            key, subkey = jax.random.split(state.random_key)
            normalized_observation = rescale_observations(observation, obs_limits)
            dist, values = network.apply(state.params, normalized_observation)
            actions, log_prob = dist.sample_and_log_prob(seed=subkey)
            mem.extras["values"] = values
            mem.extras["log_probs"] = log_prob
            mem = mem._replace(extras=mem.extras)
            state = state._replace(random_key=key)
            return actions, state, mem

        @jax.jit
        def eval_policy(state: TrainingState, observation: Dict, mem: MemoryState):
            """Evaluation policy"""
            actions, state, mem = policy(state, observation, mem)
            extras = {}
            return actions, state, mem, extras

        @jax.jit
        def gae_advantages(
            rewards: jnp.ndarray, values: jnp.ndarray, dones: jnp.ndarray
        ) -> jnp.ndarray:
            """Calculates GAE advantages"""
            discounts = gamma * jnp.logical_not(dones)
            reverse_batch = (
                jnp.flip(values[:-1], axis=0),
                jnp.flip(rewards, axis=0),
                jnp.flip(discounts, axis=0),
            )
            _, advantages = jax.lax.scan(
                get_advantages,
                (
                    jnp.zeros_like(values[-1]),
                    values[-1],
                    jnp.ones_like(values[-1]) * gae_lambda,
                ),
                reverse_batch,
            )
            advantages = jnp.flip(advantages, axis=0)
            target_values = values[:-1] + advantages
            target_values = jax.lax.stop_gradient(target_values)
            return advantages, target_values

        def loss(
            params: hk.Params,
            timesteps: int,
            observations: jnp.ndarray,
            actions: jnp.array,
            behavior_log_probs: jnp.array,
            target_values: jnp.array,
            advantages: jnp.array,
            behavior_values: jnp.array,
        ):
            """Surrogate loss using clipped probability ratios."""
            normalized_observations = rescale_observations(observations, obs_limits)
            distribution, values = network.apply(params, normalized_observations)
            log_prob = distribution.log_prob(actions)
            entropy = distribution.entropy()

            ratios = jnp.exp(log_prob - behavior_log_probs)
            clipped_ratios_t = jnp.clip(
                ratios, 1.0 - ppo_clipping_epsilon, 1.0 + ppo_clipping_epsilon
            )
            clipped_objective = jnp.fmin(ratios * advantages, clipped_ratios_t * advantages)
            policy_loss = -jnp.mean(clipped_objective)

            # Value loss
            value_cost = value_coeff
            unclipped_value_error = target_values - values
            unclipped_value_loss = unclipped_value_error**2

            if clip_value:
                clipped_values = behavior_values + jnp.clip(
                    values - behavior_values,
                    -ppo_clipping_epsilon,
                    ppo_clipping_epsilon,
                )
                clipped_value_error = target_values - clipped_values
                clipped_value_loss = clipped_value_error**2
                value_loss = jnp.mean(jnp.fmax(unclipped_value_loss, clipped_value_loss))
            else:
                value_loss = jnp.mean(unclipped_value_loss)

            # Entropy loss
            if anneal_entropy == "linear":
                fraction = jnp.fmax(1 - timesteps / entropy_coeff_horizon, 0)
                entropy_cost = fraction * entropy_coeff_start + (1 - fraction) * entropy_coeff_end
            elif anneal_entropy == "exponential":
                decay_rate = (entropy_coeff_end / entropy_coeff_start) ** (1 / entropy_coeff_horizon)
                entropy_cost = entropy_coeff_start * (decay_rate**timesteps)
                if entropy_clipping:
                    entropy_cost = jnp.clip(entropy_cost, entropy_coeff_end)
            else:
                entropy_cost = entropy_coeff_start
            entropy_loss = -jnp.mean(entropy)

            total_loss = policy_loss + entropy_cost * entropy_loss + value_loss * value_cost

            return total_loss, {
                "loss_total": total_loss,
                "loss_policy": policy_loss,
                "loss_value": value_loss,
                "loss_entropy": entropy_loss,
                "entropy_cost": entropy_cost,
            }

        @jax.jit
        def sgd_step(
            state: TrainingState, sample: NamedTuple
        ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
            """Performs a minibatch SGD step"""
            (
                observations,
                actions,
                rewards,
                behavior_log_probs,
                behavior_values,
                dones,
            ) = (
                sample.observations,
                sample.actions,
                sample.rewards,
                sample.behavior_log_probs,
                sample.behavior_values,
                sample.dones,
            )

            advantages, target_values = gae_advantages(
                rewards=rewards, values=behavior_values, dones=dones
            )

            behavior_values = behavior_values[:-1, :]
            trajectories = Batch(
                observations=observations,
                actions=actions,
                advantages=advantages,
                behavior_log_probs=behavior_log_probs,
                target_values=target_values,
                behavior_values=behavior_values,
            )

            assert len(target_values.shape) > 1
            num_envs = target_values.shape[1]
            num_steps = target_values.shape[0]
            batch_size = num_envs * num_steps
            assert batch_size % num_minibatches == 0

            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), trajectories
            )

            grad_fn = jax.jit(jax.grad(loss, has_aux=True))

            @jax.jit
            def model_update_minibatch(
                carry: Tuple[hk.Params, optax.OptState, int],
                minibatch: Batch,
            ) -> Tuple[Tuple[hk.Params, optax.OptState, int], Dict[str, jnp.ndarray]]:
                params, opt_state, timesteps = carry
                advantages = (minibatch.advantages - jnp.mean(minibatch.advantages, axis=0)) / (
                    jnp.std(minibatch.advantages, axis=0) + 1e-8
                )
                gradients, metrics = grad_fn(
                    params,
                    timesteps,
                    minibatch.observations,
                    minibatch.actions,
                    minibatch.behavior_log_probs,
                    minibatch.target_values,
                    advantages,
                    minibatch.behavior_values,
                )
                updates, opt_state = optimizer.update(gradients, opt_state)
                params = optax.apply_updates(params, updates)
                metrics["norm_grad"] = optax.global_norm(gradients)
                metrics["norm_updates"] = optax.global_norm(updates)
                return (params, opt_state, timesteps), metrics

            @jax.jit
            def model_update_epoch(
                carry: Tuple[jnp.ndarray, hk.Params, optax.OptState, int, Batch],
                unused_t: Tuple[()],
            ) -> Tuple[
                Tuple[jnp.ndarray, hk.Params, optax.OptState, Batch],
                Dict[str, jnp.ndarray],
            ]:
                key, params, opt_state, timesteps, batch = carry
                key, subkey = jax.random.split(key)
                permutation = jax.random.permutation(subkey, batch_size)
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [num_minibatches, -1] + list(x.shape[1:])),
                    shuffled_batch,
                )
                (params, opt_state, timesteps), metrics = jax.lax.scan(
                    model_update_minibatch,
                    (params, opt_state, timesteps),
                    minibatches,
                    length=num_minibatches,
                )
                return (key, params, opt_state, timesteps, batch), metrics

            params = state.params
            opt_state = state.opt_state
            timesteps = state.timesteps

            (key, params, opt_state, timesteps, _), metrics = jax.lax.scan(
                model_update_epoch,
                (state.random_key, params, opt_state, timesteps, batch),
                (),
                length=num_epochs,
            )

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics["sgd_steps"] = opt_state[1][0]
            metrics["scheduler_steps"] = opt_state[2][0]
            metrics["rewards_mean"] = jnp.mean(jnp.abs(jnp.mean(rewards, axis=(0, 1))))
            metrics["rewards_std"] = jnp.std(rewards, axis=(0, 1))
            metrics["trained"] = True
            metrics["mean_advantages"] = jnp.mean(advantages)
            metrics["var_advantages"] = jnp.var(advantages)

            new_state = TrainingState(
                params=params,
                opt_state=opt_state,
                random_key=key,
                timesteps=timesteps + batch_size,
            )

            new_memory = MemoryState(
                hidden=jnp.zeros((num_envs, 1)),
                extras={
                    "log_probs": jnp.zeros(num_envs),
                    "values": jnp.zeros(num_envs),
                },
            )

            return new_state, new_memory, metrics

        def make_initial_state(key: Any, hidden: jnp.ndarray) -> Tuple[TrainingState, MemoryState]:
            """Initialises the training state"""
            key, subkey = jax.random.split(key)

            if isinstance(obs_spec, dict):
                dummy_obs = {}
                for k, v in obs_spec.items():
                    if v == ():
                        dummy_obs[k] = jnp.zeros(1)
                    else:
                        dummy_obs[k] = jnp.zeros(shape=v)
            elif not tabular:
                dummy_obs = jnp.zeros(shape=obs_spec, dtype=float_precision)
            else:
                dummy_obs = jnp.zeros(shape=obs_spec)

            dummy_obs = add_batch_dim(dummy_obs)
            initial_params = network.init(subkey, dummy_obs)
            initial_opt_state = optimizer.init(initial_params)
            self.optimizer = optimizer
            return TrainingState(
                random_key=key,
                params=initial_params,
                opt_state=initial_opt_state,
                timesteps=0,
            ), MemoryState(
                hidden=jnp.zeros((num_envs, 1)),
                extras={
                    "values": jnp.zeros(num_envs),
                    "log_probs": jnp.zeros(num_envs),
                },
            )

        def prepare_batch(traj_batch: NamedTuple, done: Any, action_extras: dict):
            """Prepare a batch of data for training."""
            _value = jax.lax.select(
                done,
                jnp.zeros_like(action_extras["values"]),
                action_extras["values"],
            )
            _value = jax.lax.expand_dims(_value, [0])
            traj_batch = traj_batch._replace(
                behavior_values=jnp.concatenate([traj_batch.behavior_values, _value], axis=0)
            )
            return traj_batch

        # Initialize
        self.make_initial_state = make_initial_state
        self._state, self._mem = make_initial_state(random_key, jnp.zeros(1))
        self._prepare_batch = jax.jit(prepare_batch)
        self._sgd_step = jax.jit(sgd_step)

        self._logger = Logger()
        self._until_sgd = 0
        self._logger.metrics = {
            "trained": True,
            "sgd_steps": 0,
            "scheduler_steps": 0,
            "loss_total": 0,
            "loss_policy": 0,
            "loss_value": 0,
            "loss_entropy": 0,
            "entropy_cost": entropy_coeff_start,
            "norm_grad": 0,
            "norm_updates": 0,
            "rewards_mean": 0,
            "rewards_std": 0,
            "mean_advantages": 0,
            "var_advantages": 0,
            "learning_rate": 0,
        }

        self._policy = policy
        self.player_id = player_id
        self.network = network
        self._eval_policy = eval_policy

        self._num_envs = num_envs
        self._num_minibatches = num_minibatches
        self._num_epochs = num_epochs

        jax.debug.print("reset agent {} metrics (reformulated obs_type={})", self.player_id, obs_type)

    def reset_memory(self, memory, eval=False) -> MemoryState:
        num_envs = 1 if eval else self._num_envs
        memory = memory._replace(
            extras={
                "values": jnp.zeros(num_envs),
                "log_probs": jnp.zeros(num_envs),
            },
        )
        return memory

    def update(
        self,
        traj_batch,
        obs: jnp.ndarray,
        state: TrainingState,
        mem: MemoryState,
    ):
        """Update the agent"""
        _, _, mem = self._policy(state, obs, mem)
        traj_batch = self._prepare_batch(traj_batch, traj_batch.dones[-1, ...], mem.extras)
        state, mem, metrics = self._sgd_step(state, traj_batch)
        return state, mem, metrics

    def save_state(self, filepath: str):
        state_data = {
            "training_state": self._state,
            "memory_state": self._mem,
            "hyperparameters": {
                "player_id": self.player_id,
                "num_minibatches": self._num_minibatches,
                "num_epochs": self._num_epochs,
                "num_envs": self._num_envs,
                "obs_type": self._obs_type,
            },
            "network_params": self._state.params,
            "optimizer_state": self._state.opt_state,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state_data, f)

    @classmethod
    def load_state(cls, filepath: str, network, optimizer: optax.GradientTransformation):
        with open(filepath, "rb") as f:
            state_data = pickle.load(f)
        print(f"Loading state from {filepath}")
        obj = cls.__new__(cls)
        obj._state = state_data["training_state"]
        obj._mem = state_data["memory_state"]
        obj.player_id = state_data["hyperparameters"]["player_id"]
        obj._num_envs = state_data["hyperparameters"]["num_envs"]
        obj._num_minibatches = state_data["hyperparameters"]["num_minibatches"]
        obj._num_epochs = state_data["hyperparameters"]["num_epochs"]
        obj._obs_type = state_data["hyperparameters"].get("obs_type", "original")
        obj.network = network
        return obj

    def tree_flatten(self):
        children = (self._state, self._mem)
        aux_data = {
            "player_id": self.player_id,
            "_lr_scheduling": self._lr_scheduling,
            "_initial_learning_rate": self._initial_learning_rate,
            "_total_num_transitions": self._total_num_transitions,
            "_num_envs": self._num_envs,
            "_num_minibatches": self._num_minibatches,
            "_num_epochs": self._num_epochs,
            "_obs_type": self._obs_type,
            "network": self.network,
            "_logger": self._logger,
            "_until_sgd": self._until_sgd,
            "_policy": self._policy,
            "_eval_policy": self._eval_policy,
            "_sgd_step": self._sgd_step,
            "make_initial_state": self.make_initial_state,
            "_prepare_batch": self._prepare_batch,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj._state, obj._mem = children
        for key, value in aux_data.items():
            setattr(obj, key, value)
        return obj


jax.tree_util.register_pytree_node(PPOReformulated, PPOReformulated.tree_flatten, PPOReformulated.tree_unflatten)


def make_agent_reformulated(
    args,
    agent_args,
    obs_spec,
    action_spec,
    seed: int,
    num_iterations: int,
    player_id: int,
    tabular=False,
):
    """Make PPO agent with reformulated observations"""
    
    # Determine observation type
    obs_type = args.get("observation_reformulation", {}).get("type", "none")
    if not args.get("observation_reformulation", {}).get("enabled", False):
        obs_type = "original"
    
    print(f"Making reformulated network for {args.get('env_id')} with obs_type={obs_type}")
    
    network = make_marketenv_network_reformulated(
        action_spec,
        agent_args.get("hidden_sizes"),
        obs_type=obs_type,
    )

    # Optimizer
    transition_steps = (
        num_iterations * agent_args.get("num_epochs") * agent_args.get("num_minibatches")
    )

    if agent_args.get("lr_scheduling"):
        scale = optax.inject_hyperparams(optax.scale)(step_size=-1.0)
        scheduler = optax.linear_schedule(
            init_value=agent_args.get("learning_rate"),
            end_value=0,
            transition_steps=transition_steps,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(agent_args.get("max_gradient_norm")),
            optax.scale_by_adam(eps=agent_args.get("adam_epsilon")),
            optax.scale_by_schedule(scheduler),
            scale,
        )
    else:
        scale = optax.inject_hyperparams(optax.scale)(step_size=-agent_args.get("learning_rate"))
        optimizer = optax.chain(
            optax.clip_by_global_norm(agent_args.get("max_gradient_norm")),
            optax.scale_by_adam(eps=agent_args.get("adam_epsilon")),
            scale,
        )

    random_key = jax.random.PRNGKey(seed=seed)

    obs_limits = {
        "inventory_uppers": jnp.array(args.get("initial_inventories")),
        "last_actions_upper": args.get("num_prices"),
        "last_prices_lower": args.get("possible_prices")[0],
        "last_prices_upper": args.get("possible_prices")[-1],
        "t_upper": args.get("time_horizon"),
    }

    agent = PPOReformulated(
        network=network,
        optimizer=optimizer,
        random_key=random_key,
        obs_spec=obs_spec,
        obs_limits=obs_limits,
        obs_type=obs_type,
        num_envs=args.get("num_envs"),
        num_minibatches=agent_args.get("num_minibatches"),
        num_epochs=agent_args.get("num_epochs"),
        clip_value=agent_args.get("clip_value"),
        value_coeff=agent_args.get("value_coeff"),
        anneal_entropy=agent_args.get("anneal_entropy"),
        entropy_coeff_start=agent_args.get("entropy_coeff_start"),
        entropy_coeff_end=agent_args.get("entropy_coeff_end"),
        entropy_coeff_horizon=agent_args.get("entropy_coeff_horizon"),
        ppo_clipping_epsilon=agent_args.get("ppo_clipping_epsilon"),
        entropy_clipping=agent_args.get("entropy_clipping"),
        gamma=agent_args.get("gamma"),
        gae_lambda=agent_args.get("gae_lambda"),
        tabular=tabular,
        player_id=player_id,
        num_iterations=num_iterations,
        initial_learning_rate=agent_args.get("learning_rate"),
        lr_scheduling=agent_args.get("lr_scheduling"),
    )
    return agent
