import gymnasium as gym
from minigrid.envs import EmptyEnv
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from gymnasium.core import ObsType
from logic import pick_starting_state

from typing import Optional, Any

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

        # The value function works on observations, while we can only generate the states
        # Minigrid doesnt supply a convenient way to do this, so we have to build it ourselves


        if self.agent == None:
            raise UnboundLocalError("You have to set self.agent to the agent that's trained to allow for proximal curriculum learning.")

        # Now set starting state
        pick_starting_state(
            value_function=self.agent,  # TODO: set this properly
            novelty_function=lambda _: 0,  # TODO: set this properly
            state_candidates=[],
            state_to_obs=StateToObs,  # type: ignore  # this is a Callable but LSP doesn't know
            beta_proximal=0,
            gamma_tradeoff=0,
        )

        # Return first observation
        obs = super(EmptyEnv).gen_obs()

        return obs, {}
