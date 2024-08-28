from typing import Callable, Any, Iterable
import numpy as np
from gymnasium.core import ObsType

type State = Any  # TODO: placeholder, replace it
type Number = float | int | np.number  # TODO: add more?

# +------------------------+
# | Starting State Picking |
# +------------------------+

def pick_starting_state(value_function: Callable[[State], Number], novelty_function: Callable[[State], Number], state_candidates: Iterable[State], state_to_obs: Callable[[State], ObsType], beta_proximal: Number, gamma_tradeoff: Number) -> State:
    state_candidates = np.array(state_candidates)
    assert len(state_candidates.shape) == 1  # only list of states
    # Calculate distribution over starting states based on probability of success
    state_pos = np.array([beta_proximal * value * (1 - value) for value in map(value_function, map(state_to_obs, state_candidates))])
    pos_dist = state_pos / np.sum(state_pos)

    # Calculate distribution over starting states based on state novelty
    state_novelty = np.array([novelty_function(state) for state in state_candidates])
    novelty_dist = state_novelty / np.sum(state_novelty)

    combined_dist = gamma_tradeoff * pos_dist + (1 - gamma_tradeoff) * novelty_dist

    return np.random.choice(state_candidates, p=combined_dist)


# +--------------------------+
# | State Novelty Approaches |
# +--------------------------+
