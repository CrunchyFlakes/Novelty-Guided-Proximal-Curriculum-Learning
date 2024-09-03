from typing import Callable, Any, Iterable
import numpy as np
from gymnasium.core import ObsType
from ConfigSpace import Configuration
import torch
from torch import nn

type State = Any  # TODO: placeholder, replace it
type Number = float | int | np.number  # TODO: add more?

# +------------------------+
# | Starting State Picking |
# +------------------------+


def pick_starting_state(
    value_function: Callable[[State], Number],
    novelty_function: Callable[[State], Number],
    state_candidates: Iterable[State],
    state_to_obs: Callable[[State], ObsType],
    beta_proximal: Number,
    gamma_tradeoff: Number,
) -> State:
    state_candidates = list(state_candidates)
    # Calculate distribution over starting states based on probability of success
    state_values = np.array(
        [
            value_function(obs).detach().item()
            for obs in map(state_to_obs, state_candidates)
        ]
    )
    normalized_state_values = (state_values - np.min(state_values)) / (
        np.max(state_values) - np.min(state_values)
    )
    pos_estimates = (
        beta_proximal * normalized_state_values * (1 - normalized_state_values)
    )
    pos_dist = pos_estimates / np.sum(pos_estimates)

    # Calculate distribution over starting states based on state novelty
    state_novelty = np.array([novelty_function(state) for state in state_candidates])
    novelty_dist = state_novelty / np.sum(state_novelty)
    novelty_dist = pos_dist

    combined_dist = gamma_tradeoff * pos_dist + (1 - gamma_tradeoff) * novelty_dist

    return state_candidates[np.random.choice(len(state_candidates), p=combined_dist)]


# +--------------------------+
# | State Novelty Approaches |
# +--------------------------+

# Random Network Distillation
class RND():
    def __init__(self, layers: Iterable[int]):
        pass
