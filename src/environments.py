import gymnasium as gym
import numpy as np
from minigrid.envs import EmptyEnv
from gymnasium import spaces
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from minigrid.core.grid import Grid
from minigrid.core.world_object import Door
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.utils import obs_as_tensor
from gymnasium.core import ObsType
from .logic import pick_starting_state
from .util import get_novelty_function
import torch
from ConfigSpace import Configuration

from itertools import product
from typing import Any, Callable

# +--------------------------------------------------------------------------------------------+
# | All classes here act as a wrapper to allow for curriculum learning based on starting state |
# +--------------------------------------------------------------------------------------------+

class FullImgObsWrapper(FullyObsWrapper):
    # currently not used because it made the results worse
    def __init__(self, env):
        """A wrapper that returns the whole grid instead of only the part in front of the agent

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.observation_space = self.observation_space["image"]

    def observation(self, obs):
        return super().observation(obs)["image"]

class ImgObsKeyWrapper(ImgObsWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.space_to_flatten = spaces.Dict({"image": self.observation_space, "key": spaces.Discrete(2)})
        self.observation_space = spaces.flatten_space(self.space_to_flatten)


    def observation(self, obs):
        return spaces.flatten(self.space_to_flatten, {"image": obs["image"], "key": self.env.carrying != None})

def get_prox_curr_env(env_class, *args, **kwargs):
    class ProxCurrMinigridWrapper(env_class):
        class StateToObs:
            """Provide somewhat efficient implementation of converting a state in Minigrid to an observation

            It may be possible to use the EmptyEnv class itself, but just to be sure to not mess anything up we create a second class here for peace of mind

            Attributes:
                env: The environment which is (ab)used to calculate observation
            """

            def __init__(self, env: "ProxCurrMinigridWrapper"):
                self.env = env
                self.wrapped_env = ImgObsKeyWrapper(self.env)

            def __call__(self, state: tuple[tuple[int, int], int, tuple[int, int] | None, list[tuple[tuple[int, int], tuple[bool, bool]]]]) -> torch.Tensor:
                agent_pos, agent_dir, pos_item_to_carry, doors_with_states = state

                # save current state
                original_agent_pos = self.env.agent_pos
                original_agent_dir = self.env.agent_dir
                original_doors_with_states = {door: (door.is_open, door.is_locked) for door in [self.env.grid.get(*pos) for pos, _ in doors_with_states]}

                # set agent to new state
                self.env.agent_pos = agent_pos
                self.env.agent_dir = agent_dir
                for door_pos, (door_is_open, door_is_locked) in doors_with_states:
                    door = self.env.grid.get(*door_pos)
                    door.is_open = door_is_open
                    door.is_locked = door_is_locked

                ## env should be freshly reset, nothing should be carried so no backup needed
                assert self.env.carrying == None
                ## pick up item if given
                if pos_item_to_carry:
                    self.env.carrying = self.env.grid.get(*pos_item_to_carry)
                    self.env.carrying.cur_pos = np.array([-1, -1])
                    self.env.grid.set(*pos_item_to_carry, None)

                obs_tensor = (
                    obs_as_tensor(
                        self.wrapped_env.observation(self.env.gen_obs()), device="cpu"
                    )
                    .unsqueeze(0)
                )  # TODO: set device properly

                # We don't need to restore agent_pos and agent_dir, as they will be set always after calls to this
                ## restore item in env
                if pos_item_to_carry:
                    self.env.carrying.cur_pos = np.array(pos_item_to_carry)
                    self.env.grid.set(*pos_item_to_carry, self.env.carrying)
                    self.env.carrying = None
                ## restore doors
                for door, (door_is_open, door_is_locked) in original_doors_with_states.items():
                    door.is_open = door_is_open
                    door.is_locked = door_is_locked

                return obs_tensor

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.state_to_obs = ProxCurrMinigridWrapper.StateToObs(self)

        def set_agent(self, agent: OnPolicyAlgorithm) -> None:
            """Set Agent inside of env to be able to pick starting states

            Args:
                agent: Stable baselines OnPolicyAlgorithm
            """
            self.agent = agent
            return

        def generate_state_candidates(self) -> list[tuple[tuple[int, int], int, tuple[int, int] | None, list[tuple[tuple[int, int], tuple[bool, bool]]]]]:
            """Generate all states that the agent could be in in this environment

            Returns:
                list of states
            """
            positions = list(product(range(0, self.grid.width), range(0, self.grid.height)))
            valid_positions = filter(lambda pos: self.grid.get(*pos) is None, positions)
            directions = range(0, 4)
            carryable_positions = [pos for pos in positions if self.grid.get(*pos) and self.grid.get(*pos).can_pickup()] + [None]
            door_positions = [pos for pos in positions if isinstance(self.grid.get(*pos), Door)]
            door_states: list[tuple[bool, bool]] = [(True, False), (False, False), (False, True)]  # open, closed, locked
            door_positions_and_states = list(product(door_positions, door_states))

            return list(product(valid_positions, directions, carryable_positions, [door_positions_and_states]))

        def setup_start_state_picking(self, config: Configuration, novelty_function: Callable[[torch.Tensor], torch.Tensor]):
            self.beta_proximal = config["beta_proximal"]
            self.gamma_tradeoff = config["gamma_tradeoff"]
            self.novelty_function = novelty_function

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> tuple[ObsType, dict[str, Any]]:
            """Reset Environment, while picking starting state based on proximal curriculum

            Args:
                seed: seed to initialize env
                options: options used by EmptyEnv

            Returns:
                observation and info dict
            """
            _ = super().reset(seed=seed)

            if self.agent == None:  # Check if the user has set an agent manually
                raise UnboundLocalError(
                    "You have to set self.agent to the agent that's trained to allow for proximal curriculum learning."
                )

            try:
                value_function: Callable = self.agent.policy.predict_values  # type: ignore  # lsp gets type wrong for some reason
            except:
                raise ValueError("Bound agent does not use ActorCriticPolicy")

            # Now set starting state
            starting_pos, starting_dir, pos_item_to_carry, doors_with_states = pick_starting_state(
                value_function=value_function,
                novelty_function=self.novelty_function,  # TODO: set this properly
                state_candidates=self.generate_state_candidates(),
                state_to_obs=self.state_to_obs,  # type: ignore  # this is a Callable but LSP doesn't know
                beta_proximal=(
                    self.beta_proximal
                ),
                gamma_tradeoff=(
                    self.gamma_tradeoff
                ),
            )
            self.agent_pos = starting_pos
            self.agent_dir = starting_dir
            # env should be freshly reset, nothing should be carried
            assert self.carrying == None
            # pick up item if given
            if pos_item_to_carry:
                self.carrying = self.grid.get(*pos_item_to_carry)
                self.carrying.cur_pos = np.array([-1, -1])
                self.grid.set(*pos_item_to_carry, None)
            # set doors
            for door_pos, (door_is_open, door_is_locked) in doors_with_states:
                door = self.grid.get(*door_pos)
                door.is_open = door_is_open
                door.is_locked = door_is_locked

            # Return first observation
            obs = super().gen_obs()

            return obs, {}

    return ProxCurrMinigridWrapper(*args, **kwargs)
