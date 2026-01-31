from typing import Optional, Dict, Any
import distrax
import haiku as hk
import jax
import jax.numpy as jnp
from jaxtyping import Array, Integer


class CategoricalValueHead(hk.Module):
    """Network head that produces a categorical distribution and value."""

    def __init__(self, num_values: int, name: Optional[str] = None):
        super().__init__(name=name)
        self._logit_layer = hk.Linear(num_values)
        self._value_layer = hk.Linear(1)

    def __call__(self, inputs: Array):
        logits = self._logit_layer(inputs)
        value = jnp.squeeze(self._value_layer(inputs), axis=-1)
        return distrax.Categorical(logits=logits), value


class CategoricalValueHeadSeparate_reformulated(hk.Module):
    """
    Network head for reformulated observations.
    Dynamically handles different observation formats.
    """

    def __init__(self, num_actions: int, hidden_sizes, obs_type: str = "original", name: Optional[str] = None):
        super().__init__(name=name)
        self._obs_type = obs_type
        self._action_body = hk.nets.MLP(
            hidden_sizes,
            w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
            b_init=hk.initializers.Constant(0),
            activation=jnp.tanh,
            name="A",
        )
        self._value_body = hk.nets.MLP(
            hidden_sizes,
            w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
            b_init=hk.initializers.Constant(0),
            activation=jnp.tanh,
            name="C",
        )
        self._logit_layer = hk.Linear(
            num_actions,
            w_init=hk.initializers.Orthogonal(0.01),
            b_init=hk.initializers.Constant(0),
            name="logits",
        )
        self._value_layer = hk.Linear(
            1,
            w_init=hk.initializers.Orthogonal(1.0),
            b_init=hk.initializers.Constant(0),
            name="value",
        )
        self._num_actions = num_actions

    def __call__(self, inputs):
        obs = self._flatten_observation(inputs)

        # Actor
        logits = self._action_body(obs)
        logits = self._logit_layer(logits)

        # Critic
        value = self._value_body(obs)
        value = self._value_layer(value)
        return (distrax.Categorical(logits=logits), jnp.squeeze(value, axis=-1))

    def _flatten_observation(self, inputs: Dict) -> Array:
        """Flatten observation dict to a single array based on obs_type."""
        if self._obs_type == "min_max_mean":
            # Reformulated: min/max/mean for prices and inventories
            obs = jnp.concatenate([
                jnp.atleast_1d(inputs["price_min"]),
                jnp.atleast_1d(inputs["price_max"]),
                jnp.atleast_1d(inputs["price_mean"]),
                jnp.atleast_1d(inputs["inventory_min"]),
                jnp.atleast_1d(inputs["inventory_max"]),
                jnp.atleast_1d(inputs["inventory_mean"]),
                jnp.atleast_1d(inputs["t"]).astype(jnp.float32),
            ], axis=-1)
        elif self._obs_type == "top_k" or self._obs_type == "set_top_k":
            # Reformulated: top-k sorted values (independent or price-keyed)
            obs = jnp.concatenate([
                inputs["prices_sorted"],
                inputs["inventories_sorted"],
                jnp.atleast_1d(inputs["t"]).astype(jnp.float32),
            ], axis=-1)
        else:
            # Original observation format
            obs = jnp.concatenate([
                inputs["last_prices"],
                inputs["inventories"].astype(jnp.float32),
                jnp.atleast_1d(inputs["t"]).astype(jnp.float32),
            ], axis=-1)
        return obs


def make_marketenv_network_reformulated(
    num_actions: int,
    hidden_sizes: Integer[Array, "..."],
    obs_type: str = "original",
):
    """Makes a network for the reformulated market environment.
    
    Args:
        num_actions: Number of discrete actions
        hidden_sizes: List of hidden layer sizes
        obs_type: "original", "min_max_mean", "top_k", or "set_top_k"
    """

    def forward_fn(inputs: Dict[str, Any]):
        body = CategoricalValueHeadSeparate_reformulated(
            num_actions=num_actions, 
            hidden_sizes=hidden_sizes, 
            obs_type=obs_type,
            name="market_network_reformulated"
        )
        return body(inputs)

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network
