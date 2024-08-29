import gymnasium as gym
import numpy as np
from minigrid.envs import EmptyEnv
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from gymnasium.core import ObsType
from logic import pick_starting_state

from itertools import product
from typing import Any, Callable

# +--------------------------------------------------------------------------------------------+
# | All classes here act as a wrapper to allow for curriculum learning based on starting state |
# +--------------------------------------------------------------------------------------------+


def ProxCurrEmptyEnv(EmptyEnv):
    class StateToObs():
        """ Provide somewhat efficient implementation of converting a state in Minigrid to an observation

        It may be possible to use the EmptyEnv class itself, but just to be sure to not mess anything up we create a second class here for peace of mind

        Attributes:
            env: The environment which is (ab)used to calculate observation
        """
        def __init__(self):
            self.env = EmptyEnv()
            self.env.reset()

        def __call__(self, state: tuple[tuple[int, int], int]) -> ObsType:
            self.env.agent_pos = state[0]
            self.env.agent_dir = state[1]
            return self.env.get_obs()

    def set_agent(self, agent: OnPolicyAlgorithm) -> None:
        """ Set Agent inside of env to be able to pick starting states

        Args:
            agent: Stable baselines OnPolicyAlgorithm
        """
        self.agent = agent
        return

    def generate_state_candidates(self) -> list[tuple[tuple[int, int], int]]:
        """ Generate all states that the agent could be in in this environment

        Returns:
            list of states
        """
        positions = product(range(0, self.grid.width), range(0, self.grid.height))
        valid_positions = filter(lambda pos: self.grid.get(*pos) is None, positions)
        directions = range(0, 4)

        return list(product(valid_positions, directions))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """ Reset Environment, while picking starting state based on proximal curriculum

        Args:
            seed: seed to initialize env
            options: options used by EmptyEnv

        Returns:
            observation and info dict
        """
        default_obs, info = super(EmptyEnv).reset(seed=seed)

        if self.agent == None:  # Check if the user has set an agent manually
            raise UnboundLocalError("You have to set self.agent to the agent that's trained to allow for proximal curriculum learning.")

        try:
            value_function: Callable = self.agent.policy.predict_values,  # type: ignore  # lsp gets type wrong for some reason
        except:
            raise ValueError("Bound agent does not use ActorCriticPolicy")

        # Now set starting state
        pick_starting_state(
            value_function=value_function,
            novelty_function=lambda _: 0,  # TODO: set this properly
            state_candidates=self.generate_state_candidates(),
            state_to_obs=StateToObs,  # type: ignore  # this is a Callable but LSP doesn't know
            beta_proximal=options["beta_proximal"] if options and "beta_proximal" in options else 1,
            gamma_tradeoff=options["gamma_tradeoff"] if options and "gamma_tradeoff" in options else 0.5,
        )

        # Return first observation
        obs = super(EmptyEnv).gen_obs()

        return obs, {}
