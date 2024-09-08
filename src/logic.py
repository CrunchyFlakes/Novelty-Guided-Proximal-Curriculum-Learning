from typing import Callable, Any, Iterable
import numpy as np
from gymnasium.core import ObsType
from ConfigSpace import Configuration
import torch
from torch import nn

import logging

logger = logging.getLogger(__name__)

type State = Any  # TODO: placeholder, replace it
type Number = float | int | np.number  # TODO: add more?

# +------------------------+
# | Starting State Picking |
# +------------------------+


def pick_starting_state(
    value_function: Callable[[State], Number],
    novelty_function: Callable[[torch.Tensor], torch.Tensor],
    state_candidates: Iterable[State],
    state_to_obs: Callable[[State], ObsType],
    beta_proximal: Number,
    beta_novelty: Number,
    gamma_tradeoff: Number,
) -> State:
    state_candidates = list(state_candidates)
    states_as_obs = list(map(state_to_obs, state_candidates))
    # Calculate distribution over starting states based on probability of success
    state_values = np.array(
        [value_function(obs).detach().item() for obs in states_as_obs]
    )
    normalized_state_values = (state_values - np.min(state_values)) / (
        np.max(state_values) - np.min(state_values)
    )
    pos_estimates = np.exp(
        beta_proximal * normalized_state_values * (1 - normalized_state_values)
    )
    pos_dist = pos_estimates / np.sum(pos_estimates)

    # Calculate distribution over starting states based on state novelty
    state_novelty = (
        novelty_function(
            torch.stack(list(map(torch.flatten, states_as_obs))).to(torch.float32)
        )
        .detach()
        .clone()
        .numpy()
    )
    state_novelty_normalized = state_novelty / np.max(state_novelty)
    state_novelty_exp = np.exp(beta_novelty * state_novelty_normalized)
    novelty_dist = state_novelty_exp / np.sum(state_novelty_exp)
    

    combined_dist = gamma_tradeoff * pos_dist + (1 - gamma_tradeoff) * novelty_dist

    return state_candidates[np.random.choice(len(state_candidates), p=combined_dist)]


# +--------------------------+
# | State Novelty Approaches |
# +--------------------------+


# Random Network Distillation
class RND:
    def __init__(
        self,
        layers_fixed: list[int],
        layers_learned: list[int],
        loss: str,
        learning_rate: float,
        activation_function: str,
        optimizer: str,
    ):
        match activation_function.lower():
            case "relu":
                activation_class = nn.ReLU
            case "leakyrelu":
                activation_class = nn.LeakyReLU
            case _:
                raise NotImplementedError(
                    f'activation function "{activation_function}" not implemented.'
                )

        match loss.lower():
            case "mse":
                self.loss_function = nn.MSELoss()
                # We need an extra loss without reduction to return
                self.loss_novelty = nn.MSELoss(reduction="none")
            case _:
                raise NotImplementedError(f'loss "{loss}" not implemented.')

        self.layers_fixed = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(in_features, out_features), activation_class())
                for in_features, out_features in zip(
                    layers_fixed[:-1], layers_fixed[1:]
                )
            ]
        )
        self.layers_learned = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(in_features, out_features), activation_class())
                for in_features, out_features in zip(
                    layers_learned[:-1], layers_learned[1:]
                )
            ]
        )

        match optimizer.lower():
            case "adam":
                self.optimizer = torch.optim.Adam(self.layers_learned.parameters(), lr=learning_rate)  # type: ignore  # unresolved import bug in pytorch, only affects pyright (lsp)

    def __call__(self, input: torch.Tensor, learn_network: bool = True) -> torch.Tensor:
        self.optimizer.zero_grad()

        with torch.no_grad():
            out_fixed = self.layers_fixed(input)
        out_learned = self.layers_learned(input)

        if learn_network:
            loss = self.loss_function(out_learned, out_fixed)
            loss.backward()
            self.optimizer.step()

        return self.loss_novelty(out_learned, out_fixed).detach().clone()
