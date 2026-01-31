from typing import Tuple, Dict
import jax
import jax.numpy as jnp
import numpy as np
import chex
from gymnax.environments import environment, spaces
from jaxtyping import Array, Float, Integer, Num


@chex.dataclass
class EnvState:
    inventories: Integer[Array, "..."]
    last_prices: Float[Array, "..."]
    last_actions: Integer[Array, "..."]
    t: Integer[Array, "1"]


@chex.dataclass
class EnvParams:
    time_horizon: int
    min_price: float
    max_price: float
    num_prices: int
    possible_prices: Float[Array, "..."]
    qualities: Float[Array, "..."]
    marginal_costs: Float[Array, "..."]
    horizontal_diff: float
    demand_scaling_factor: Float[Array, "..."]
    initial_inventories: Integer[Array, "..."]
    initial_prices: Float[Array, "..."]
    initial_actions: Num[Array, "..."]
    # Observation reformulation parameters (JAX-compatible)
    # 0 = none/original, 1 = min_max_mean, 2 = top_k
    observation_reformulation_type: int
    observation_reformulation_k: int  # for top_k


# Reformulation type constants
OBS_REFORMULATION_NONE = 0
OBS_REFORMULATION_MIN_MAX_MEAN = 1
OBS_REFORMULATION_TOP_K = 2
OBS_REFORMULATION_SET_TOP_K = 3


def get_reformulation_type_id(type_str: str) -> int:
    """Convert reformulation type string to integer ID for JAX compatibility."""
    if type_str == "none" or not type_str:
        return OBS_REFORMULATION_NONE
    elif type_str == "min_max_mean":
        return OBS_REFORMULATION_MIN_MAX_MEAN
    elif type_str == "top_k":
        return OBS_REFORMULATION_TOP_K
    elif type_str == "set_top_k":
        return OBS_REFORMULATION_SET_TOP_K
    else:
        return OBS_REFORMULATION_NONE


# region REFORMULATION FUNCTIONS
def reformulate_min_max_mean(obs_dict: Dict, num_agents: int) -> Dict:
    """Transform observations to min/max/mean statistics.
    
    Note: All values are kept as 1D arrays with shape (1,) for JAX vmap compatibility.
    """
    prices = obs_dict["last_prices"]
    inventories = obs_dict["inventories"]
    
    return {
        "price_min": jnp.atleast_1d(jnp.min(prices)),
        "price_max": jnp.atleast_1d(jnp.max(prices)),
        "price_mean": jnp.atleast_1d(jnp.mean(prices)),
        "inventory_min": jnp.atleast_1d(jnp.min(inventories).astype(jnp.float32)),
        "inventory_max": jnp.atleast_1d(jnp.max(inventories).astype(jnp.float32)),
        "inventory_mean": jnp.atleast_1d(jnp.mean(inventories).astype(jnp.float32)),
        "t": jnp.atleast_1d(obs_dict["t"])
    }


def reformulate_top_k(obs_dict: Dict, num_agents: int, k: int) -> Dict:
    """Transform observations to top-k sorted values (ascending order, smallest first).
    
    Note: k must be a Python int (compile-time constant), not a JAX array.
    If k > num_agents, we use num_agents instead.
    """
    prices = obs_dict["last_prices"]
    inventories = obs_dict["inventories"]
    
    # Sort arrays (ascending order, smallest first)
    sorted_prices = jnp.sort(prices)
    sorted_inventories = jnp.sort(inventories).astype(jnp.float32)
    
    # Use Python-level slicing with constant k (JAX JIT compatible)
    # k is guaranteed to be a Python int, not a traced value
    actual_k = min(k, num_agents)
    sorted_prices_k = sorted_prices[:actual_k]
    sorted_inventories_k = sorted_inventories[:actual_k]
    
    return {
        "prices_sorted": sorted_prices_k,
        "inventories_sorted": sorted_inventories_k,
        "t": obs_dict["t"]
    }


def reformulate_set_top_k(obs_dict: Dict, num_agents: int, k: int) -> Dict:
    """Transform observations using price-keyed sorting (like a dictionary sort).
    
    Sorts (price, inventory) pairs by price, then takes top-k.
    This maintains the correspondence between prices and their inventories.
    
    Example:
        Before: prices=[100, 110, 90], inventories=[500, 400, 600]
        After sorting by price: prices=[90, 100, 110], inventories=[600, 500, 400]
        With k=2: prices=[90, 100], inventories=[600, 500]
    
    Note: k must be a Python int (compile-time constant), not a JAX array.
    """
    prices = obs_dict["last_prices"]
    inventories = obs_dict["inventories"]
    
    # Get indices that would sort prices in ascending order
    sorted_indices = jnp.argsort(prices)
    
    # Sort both prices and inventories using the same indices
    actual_k = min(k, num_agents)
    sorted_prices_k = prices[sorted_indices][:actual_k]
    sorted_inventories_k = inventories[sorted_indices][:actual_k].astype(jnp.float32)
    
    return {
        "prices_sorted": sorted_prices_k,
        "inventories_sorted": sorted_inventories_k,
        "t": obs_dict["t"]
    }
# endregion


# region ENVIRONMENT
class MarketEnvReformulated(environment.Environment):
    def __init__(self, num_agents: int, num_actions: int, time_horizon: int, 
                 observation_reformulation_type: int = OBS_REFORMULATION_NONE,
                 observation_reformulation_k: int = 2):
        super().__init__()
        self.num_players = num_agents
        self._num_actions = num_actions
        self._time_horizon = time_horizon
        self._observation_reformulation_type = observation_reformulation_type
        self._observation_reformulation_k = observation_reformulation_k

        def _MNL_demand(
            state: EnvState,
            actions: Float[Array, "num_agents"],
            qualities: Float[Array, "num_agents"],
            horizontal_diff: float,
            demand_scaling_factor: Float[Array, "time_horizon"],
        ) -> Integer[Array, "num_agents"]:
            nonzero_inventory_mask = state.inventories > 0
            utilities = jnp.exp((qualities - actions) / horizontal_diff)
            conditional_utilities = jnp.where(nonzero_inventory_mask, utilities, 0)
            sum_utilities = jnp.sum(conditional_utilities)
            demands = utilities / (sum_utilities + 1)
            t_idx = jnp.squeeze(state.t)
            scaling_at_t = demand_scaling_factor[t_idx]
            scaled_demands = scaling_at_t * demands
            integer_demands = jnp.floor(scaled_demands)
            return integer_demands

        # Select reformulation function at class instantiation time (Python-level branching)
        # This avoids JAX JIT issues with jax.lax.switch evaluating all branches
        if observation_reformulation_type == OBS_REFORMULATION_NONE:
            def _apply_reformulation(obs_dict: Dict, params: EnvParams) -> Dict:
                return obs_dict
        elif observation_reformulation_type == OBS_REFORMULATION_MIN_MAX_MEAN:
            def _apply_reformulation(obs_dict: Dict, params: EnvParams) -> Dict:
                return reformulate_min_max_mean(obs_dict, num_agents)
        elif observation_reformulation_type == OBS_REFORMULATION_TOP_K:
            # k is now a class-level constant, not from params
            k = observation_reformulation_k
            def _apply_reformulation(obs_dict: Dict, params: EnvParams) -> Dict:
                return reformulate_top_k(obs_dict, num_agents, k)
        elif observation_reformulation_type == OBS_REFORMULATION_SET_TOP_K:
            # k is now a class-level constant, not from params
            k = observation_reformulation_k
            def _apply_reformulation(obs_dict: Dict, params: EnvParams) -> Dict:
                return reformulate_set_top_k(obs_dict, num_agents, k)
        else:
            raise ValueError(f"Unknown observation reformulation type: {observation_reformulation_type}")

        def _step(
            key: chex.PRNGKey,
            state: EnvState,
            actions: Integer[Array, "num_agents"],
            params: EnvParams,
        ) -> Tuple[Tuple[Dict], EnvState, Float[Array, "num_agents"], bool, Dict]:
            t = state.t
            done = t + 1 >= params.time_horizon
            
            actions = jnp.array(actions).squeeze()  # make sure actions is flat
            prices = params.possible_prices[actions]

            demands = _MNL_demand(
                state,
                prices,
                params.qualities,
                params.horizontal_diff,
                params.demand_scaling_factor,
            ).astype(jnp.int32)

            # Calculate quantity each agent actually sells, which can't exceed inventory
            quantities_sold = jnp.minimum(demands, state.inventories).astype(jnp.int32)
            inventories = state.inventories - quantities_sold
            rewards = (prices - params.marginal_costs) * quantities_sold
            
            new_t = t + 1
            state = EnvState(
                inventories=inventories,
                last_prices=prices,
                last_actions=actions,
                t=new_t,
            )

            done = new_t >= params.time_horizon

            original_obs = {
                "inventories": state.inventories,
                "last_prices": state.last_prices,
                "last_actions": state.last_actions,
                "t": state.t,
            }

            obs = _apply_reformulation(original_obs, params)
            all_obs = tuple([obs for _ in range(num_agents)])
            
            # info, logging
            info = {
                "demands": demands,
                "quantity_sold": quantities_sold,
            }
            return all_obs, state, rewards, done, info

        def _reset(key: chex.PRNGKey, params: EnvParams) -> Tuple[Tuple[Dict, ...], EnvState]:
            state = EnvState(
                inventories=params.initial_inventories.astype(jnp.int32),
                last_prices=params.initial_prices.astype(jnp.float32),
                last_actions=params.initial_actions.astype(jnp.int32),
                t=jnp.zeros(1, dtype=jnp.int32),
            )

            original_obs = {
                "inventories": state.inventories,
                "last_prices": state.last_prices,
                "last_actions": state.last_actions,
                "t": state.t,
            }

            obs = _apply_reformulation(original_obs, params)
            all_obs = tuple([obs for _ in range(num_agents)])
            return all_obs, state

        self.step = jax.jit(_step)
        self.reset = jax.jit(_reset)
        self.MNL_demand = _MNL_demand

    @property
    def name(self) -> str:
        return "MarketEnvReformulated-v1"

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def time_horizon(self) -> int:
        return self._time_horizon

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(params.num_prices)

    def observation_space(self, params: EnvParams) -> Dict:
        """Returns observation space based on reformulation type."""
        int_max = float(np.iinfo(np.int32).max)
        reformulation_type = params.observation_reformulation_type
        
        if reformulation_type == OBS_REFORMULATION_MIN_MAX_MEAN:
            return {
                "price_min": spaces.Box(low=params.min_price, high=params.max_price, shape=(1,), dtype=jnp.float32),
                "price_max": spaces.Box(low=params.min_price, high=params.max_price, shape=(1,), dtype=jnp.float32),
                "price_mean": spaces.Box(low=params.min_price, high=params.max_price, shape=(1,), dtype=jnp.float32),
                "inventory_min": spaces.Box(low=0, high=int_max, shape=(1,), dtype=jnp.float32),
                "inventory_max": spaces.Box(low=0, high=int_max, shape=(1,), dtype=jnp.float32),
                "inventory_mean": spaces.Box(low=0, high=int_max, shape=(1,), dtype=jnp.float32),
                "t": spaces.Discrete(params.time_horizon),
            }
        elif reformulation_type == OBS_REFORMULATION_TOP_K or reformulation_type == OBS_REFORMULATION_SET_TOP_K:
            k = params.observation_reformulation_k
            return {
                "prices_sorted": spaces.Box(low=params.min_price, high=params.max_price, shape=(k,), dtype=jnp.float32),
                "inventories_sorted": spaces.Box(low=0, high=int_max, shape=(k,), dtype=jnp.float32),
                "t": spaces.Discrete(params.time_horizon),
            }

        # Default original observation space (OBS_REFORMULATION_NONE)
        return {
            "inventories": spaces.Box(low=0, high=int_max, shape=(self.num_players,), dtype=jnp.int32),
            "last_prices": spaces.Box(low=params.min_price, high=params.max_price, shape=(self.num_players,), dtype=jnp.float32),
            "last_actions": spaces.Box(low=0, high=params.num_prices, shape=(self.num_players,), dtype=jnp.int32),
            "t": spaces.Discrete(params.time_horizon),
        }

    def state_space(self, params: EnvParams) -> spaces.Dict:
        int_max = float(np.iinfo(np.int32).max)
        return spaces.Dict({
            "inventories": spaces.Box(low=0, high=int_max, shape=(self.num_players,), dtype=jnp.int32),
            "last_prices": spaces.Box(low=params.min_price, high=params.max_price, shape=(self.num_players,), dtype=jnp.float32),
            "last_actions": spaces.Box(low=0, high=params.num_prices, shape=(self.num_players,), dtype=jnp.int32),
            "t": spaces.Discrete(params.time_horizon),
        })
# endregion
