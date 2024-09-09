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
        """A wrapper to only return the image observation flattened and append if the agent currently has a key.

        Args:
            env (): The environment to use for observation generation. Will be reset to initial state after generating an observation
        """
        super().__init__(env)
        self.space_to_flatten = spaces.Dict(
            {"image": self.observation_space, "key": spaces.Discrete(2)}
        )
        self.observation_space = spaces.flatten_space(self.space_to_flatten)

    def observation(self, obs):
        return spaces.flatten(
            self.space_to_flatten,
            {"image": obs["image"], "key": self.env.carrying != None},  # type: ignore
        )


def get_prox_curr_env(env_class, *args, **kwargs):
    """Create an env which support novelty-guided proximal curriculum learning using the given base class

    You have to set the agent manually as stable baselines already needs the environment during initialization. (set_agent)
    You also have to set parameters needed for novelty-guided proximal curriculum learning yourself. (setup_start_state_picking)

    Args:
        env_class (): base_class to wrap
        *args: arguments to pass to base class
        **kwargs: keyword arguments to pass to base class

    Raises:
        UnboundLocalError: If the agent is not set in the environment during a reset
        ValueError: If the given agent does not have a value function

    Returns:
        An environment supporting novelty-guided proximal curriculum learning using the given base class
    """

    class ProxCurrMinigridWrapper(env_class):
        class StateToObs:
            """Provide somewhat efficient implementation of converting a state in Minigrid to an observation

            Attributes:
                env: The environment which is (ab)used to calculate observation
            """

            def __init__(self, env: "ProxCurrMinigridWrapper"):
                self.env = env
                self.wrapped_env = ImgObsKeyWrapper(self.env)

            def __call__(
                self,
                state: tuple[
                    tuple[int, int],
                    int,
                    tuple[int, int] | None,
                    list[tuple[tuple[int, int], tuple[bool, bool]]],
                ],
            ) -> torch.Tensor:
                """Convert a state to an observation

                Args:
                    state: state to convert

                Returns:
                    observation as torch.Tensor
                """
                agent_pos, agent_dir, pos_item_to_carry, doors_with_states = state

                # save current state
                original_agent_pos = self.env.agent_pos
                original_agent_dir = self.env.agent_dir
                original_carrying = self.env.carrying
                original_doors_with_states = {
                    door: (door.is_open, door.is_locked)
                    for door in [
                        self.env.grid.get(*pos) for pos, _ in doors_with_states
                    ]
                }

                # set agent to new state
                self.env.agent_pos = agent_pos
                self.env.agent_dir = agent_dir
                for door_pos, (door_is_open, door_is_locked) in doors_with_states:
                    door = self.env.grid.get(*door_pos)
                    door.is_open = door_is_open
                    door.is_locked = door_is_locked

                ## pick up item if given and not already carrying
                if pos_item_to_carry and not original_carrying:
                    self.env.carrying = self.env.grid.get(*pos_item_to_carry)
                    self.env.carrying.cur_pos = np.array([-1, -1])
                    self.env.grid.set(*pos_item_to_carry, None)

                obs_tensor = obs_as_tensor(
                    self.wrapped_env.observation(self.env.gen_obs()), device="cpu"
                ).unsqueeze(0)

                self.env.agent_pos = original_agent_pos
                self.env.agent_dir = original_agent_dir
                ## restore item in env
                if original_carrying:
                    self.env.carrying = original_carrying
                elif pos_item_to_carry:
                    self.env.carrying.cur_pos = np.array(pos_item_to_carry)
                    self.env.grid.set(*pos_item_to_carry, self.env.carrying)
                    self.env.carrying = None
                ## restore doors
                for door, (
                    door_is_open,
                    door_is_locked,
                ) in original_doors_with_states.items():
                    door.is_open = door_is_open
                    door.is_locked = door_is_locked

                return obs_tensor

        def __init__(self, *args, **kwargs):
            """Initialize environment
            *args: arguments to pass to base class
            **kwargs: kwargs to pass to base class
            """
            super().__init__(*args, **kwargs)
            self.state_to_obs = ProxCurrMinigridWrapper.StateToObs(self)

        def set_agent(self, agent: OnPolicyAlgorithm) -> None:
            """Set Agent inside of env to be able to pick starting states

            Args:
                agent: Stable baselines OnPolicyAlgorithm
            """
            self.agent = agent
            return

        def generate_state_candidates(
            self,
        ) -> list[
            tuple[
                tuple[int, int],
                int,
                tuple[int, int] | None,
                list[tuple[tuple[int, int], tuple[bool, bool]]],
            ]
        ]:
            """Generate all states that the agent could be in in this environment

            Returns:
                list of states
            """
            positions = list(
                product(range(0, self.grid.width), range(0, self.grid.height))
            )
            valid_positions = filter(lambda pos: self.grid.get(*pos) is None, positions)
            directions = range(0, 4)
            carryable_positions = [
                pos
                for pos in positions
                if self.grid.get(*pos) and self.grid.get(*pos).can_pickup()
            ] + [None]
            door_positions = [
                pos for pos in positions if isinstance(self.grid.get(*pos), Door)
            ]
            door_states: list[tuple[bool, bool]] = [
                (True, False),
                (False, False),
                (False, True),
            ]  # open, closed, locked
            door_positions_and_states = list(product(door_positions, door_states))

            return list(
                product(
                    valid_positions,
                    directions,
                    carryable_positions,
                    [door_positions_and_states],
                )
            )

        def setup_start_state_picking(
            self,
            config: Configuration,
            novelty_function: Callable,
        ):
            """Setup environment for novelty-guided proximal curriculum learning

            Args:
                config: approach configuration containing the needed hyperparameters
                novelty_function: novelty function to use for novelty-guidance
            """
            self.beta_proximal = config["beta_proximal"]
            self.gamma_tradeoff = config["gamma_tradeoff"]
            self.novelty_function = novelty_function
            self.beta_novelty = config["beta_novelty"]

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
            (
                starting_pos,
                starting_dir,
                pos_item_to_carry,
                doors_with_states,
            ), start_obs = pick_starting_state(
                value_function=value_function,
                novelty_function=self.novelty_function,  # TODO: set this properly
                state_candidates=self.generate_state_candidates(),
                state_to_obs=self.state_to_obs,  # type: ignore  # this is a Callable but LSP doesn't know
                beta_proximal=self.beta_proximal,
                beta_novelty=self.beta_novelty,
                gamma_tradeoff=(self.gamma_tradeoff),
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

            # novelty learning
            self.novelty_function(start_obs.to(torch.float32), learn_network=True)

            return obs, {}

        def _get_curr_state(self):
            """Generate a state which can be mapped via StateToObs

            Returns:
                state of environment
            """
            positions = list(
                product(range(0, self.grid.width), range(0, self.grid.height))
            )
            carrying_pos = (
                tuple(self.carrying.cur_pos.tolist()) if self.carrying else None
            )
            door_positions = [
                pos for pos in positions if isinstance(self.grid.get(*pos), Door)
            ]
            doors_with_states = [
                (pos, (self.grid.get(*pos).is_open, self.grid.get(*pos).is_locked))
                for pos in door_positions
            ]
            return (self.agent_pos, self.agent_dir, carrying_pos, doors_with_states)

        def step(self, action):
            """Take a step in the environment.

            Passes most to the base class, but also updates novelty function

            Args:
                action (): action to take

            Returns:
                observation
            """
            to_return = super().step(action)
            # novelty learning
            obs = self.state_to_obs(
                (self._get_curr_state()),
            )
            self.novelty_function(obs.to(torch.float32), learn_network=True)

            return to_return

    return ProxCurrMinigridWrapper(*args, **kwargs)
