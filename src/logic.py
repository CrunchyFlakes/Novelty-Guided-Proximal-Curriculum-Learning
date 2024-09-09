from typing import Callable, Any, Iterable
import numpy as np
from gymnasium.core import ObsType
import torch
from torch import nn

import logging

logger = logging.getLogger(__name__)

type State = Any
type Number = float | int | np.number

# +------------------------+
# | Starting State Picking |
# +------------------------+


def pick_starting_state(
    value_function: Callable[[State], Number],
    novelty_function: Callable,
    state_candidates: Iterable[State],
    state_to_obs: Callable[[State], ObsType],
    beta_proximal: Number,
    beta_novelty: Number,
    gamma_tradeoff: Number,
) -> tuple:
    """Main logic of novelty-guided proximal curriculum learning. Samples from given states
    Also implements proximal curriculum learning ([Tzannetos et al. 2023])

    Args:
        value_function: value function of the agent
        novelty_function: novelty function which maps states to novelty
        state_candidates: states that may serve as starting states
        state_to_obs: function to generate observations from states
        beta_proximal: proximal curriculum beta parameter -> smoothness/exaggeration of distribution
        beta_novelty: novelty approach beta parameter -> smoothness/exaggeration of distribution
        gamma_tradeoff: hyperparameter to combine calculated distributions for proximal curriculum and state novelty

    Returns:
        starting state
    """
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
            torch.stack(list(map(torch.flatten, states_as_obs))).to(torch.float32),
            learn_network=False,
        )
        .detach()
        .clone()
        .numpy()
    )
    state_novelty_normalized = state_novelty / np.max(state_novelty)
    state_novelty_exp = np.exp(beta_novelty * state_novelty_normalized)
    novelty_dist = state_novelty_exp / np.sum(state_novelty_exp)

    combined_dist = gamma_tradeoff * pos_dist + (1 - gamma_tradeoff) * novelty_dist
    combined_dist = combined_dist / combined_dist.sum()  # fix rounding errors

    chosen_state = np.random.choice(len(state_candidates), p=combined_dist)
    return state_candidates[chosen_state], states_as_obs[chosen_state]


# +--------------------------+
# | State Novelty Approaches |
# +--------------------------+


# Random Network Distillation
class RND:
    """Class implementing Random Network Distillation ([Burda et al., 2018])

    Attributes:
        layers_fixed: layer sizes for the fixed network
        layers_learned: layer sizes for the approximating network
    """

    def __init__(
        self,
        layers_fixed: list[int],
        layers_learned: list[int],
        loss: str,
        learning_rate: float,
        activation_function: str,
        optimizer: str,
    ):
        """Initialize random network distillation

        Args:
            layers_fixed: layer sizes of the fixed network
            layers_learned: layer sizes of the approximating network
            loss: loss to use
            learning_rate: learning rate for approximating network
            activation_function: activation function to use
            optimizer: optimizer to use

        Raises:
            NotImplementedError: Given unknown activation function
            NotImplementedError: Given unknown optimizer
            NotImplementedError: Given unknown loss
        """
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
            case _:
                raise NotImplementedError(f'optimizer "{optimizer}" not implemented')

    def __call__(self, input: torch.Tensor, learn_network: bool = True) -> torch.Tensor:
        """Calculates state novelty for each given observation.

        May also learn the network.

        Args:
            input: states/observations to calculate novelty over
            learn_network: if this should count as seen by the agent and therefore the network should learn

        Returns:
            state novelty scores
        """
        self.optimizer.zero_grad()

        with torch.no_grad():
            out_fixed = self.layers_fixed(input)
        out_learned = self.layers_learned(input)

        if learn_network:
            loss = self.loss_function(out_learned, out_fixed)
            loss.backward()
            self.optimizer.step()

        return self.loss_novelty(out_learned, out_fixed).mean(axis=1).detach().clone()
