from .logic import RND
import gymnasium as gym
import torch
from ConfigSpace import Configuration
from stable_baselines3.common.monitor import Monitor
from .environments import ImgObsKeyWrapper, get_prox_curr_env
from minigrid.envs import DoorKeyEnv, UnlockEnv
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_linear_fn
from .hpo import create_sb_policy_kwargs

from typing import Callable, Any, Mapping
from functools import reduce


def dummy_novelty(input: torch.Tensor, learn_network=False) -> torch.Tensor:
    """Dummy novelty function adhering to the novelty function api

    Args:
        input (): states/observations to calculate
        learn_network (): if the network should learn, does nothing here

    Returns:
        random novelty scores
    """
    return torch.rand(input.shape[0])


def get_novelty_function(
    config: dict[str, Any], env: gym.Env
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return novelty funciton corresponding to the configuration.
    Does need the environment to calculate input shape

    Args:
        config: configuration of the novelty approach
        env: environment used for observations

    Raises:
        ValueError: if the agent doesn't have an observation space shape

    Returns:
        novelty function
    """
    match config["novelty_approach"]:
        case "rnd":
            if env.observation_space.shape is None:
                raise ValueError(
                    "Environment doesn't posess observation space with shape property!"
                )
            layers_fixed = [reduce(lambda x, y: x * y, env.observation_space.shape)] + [
                config[f"rnd_layer{i}_size"] for i in range(config["rnd_n_layers"])
            ]
            layers_learned = [
                reduce(lambda x, y: x * y, env.observation_space.shape)
            ] + [config[f"rnd_layer{i}_size"] for i in range(config["rnd_n_layers"])]
            return RND(
                layers_fixed=layers_fixed,
                layers_learned=layers_learned,
                loss=config["rnd_loss"],
                learning_rate=config["rnd_learning_rate"],
                activation_function=config["rnd_activation"],
                optimizer=config["rnd_optimizer"],
            )
        case _:
            return dummy_novelty


def get_config_for_module(cfg: Configuration, module_name: str) -> dict[str, Any]:
    """This function is used to extract a sub configuration that belongs to a certain module
    Note that this function needs to call for each level


    Args:
        cfg: configuration
        module_name: subconfiguration to get

    Returns:

    """
    cfg_module = {}
    for key, value in cfg.items():
        if key.startswith(module_name):
            new_key = key.replace(f"{module_name}:", "")
            cfg_module[new_key] = value
    return cfg_module


def make_env(
    config_approach: Mapping[str, Any], env_name: str, env_kwargs: dict
) -> tuple[gym.Env, gym.Env]:
    """Return novelty-guided proximal curriculum environment with given base class and and environment without the wrapping.

    Args:
        config_approach: configuration used. Deprecated and can be removed
        env_name: name of the environment to create
        env_kwargs: kwargs to pass to environment

    Raises:
        NotImplementedError: If the given environment is not supported

    Returns:
        environment with proximal curriculum wrapper and its basic counterpart
    """
    match env_name.lower():
        case "doorkey":
            env_class = DoorKeyEnv
        case "unlock":
            env_class = UnlockEnv
            env_kwargs = {}
        case _:
            raise NotImplementedError(
                f"Starting env with name {env_name} is not supported."
            )
    env_base = Monitor(ImgObsKeyWrapper(env_class(**env_kwargs)))
    env = ImgObsKeyWrapper(get_prox_curr_env(env_class, **env_kwargs))
    return env, env_base


def initialize_model_and_env(
    config: Configuration, env_name: str, env_size: int, seed: int = 0
) -> tuple[PPO, gym.Env]:
    """Create a PPO agent and the environment given configuration. Does setup novelty-guided proximal curriculum learning

    Args:
        config: whole hyperparameter configuration
        env_name: environment to use
        env_size: size of environment to use
        seed: seed to use for PPO

    Returns:
        ppo agent with set up environment and base environment for evaluation
    """
    config_ppo = get_config_for_module(config, "sb_ppo")
    config_ppo_lr = get_config_for_module(config, "sb_lr")
    config_policy = get_config_for_module(config, "policy")
    config_approach = get_config_for_module(config, "approach")

    env, env_base = make_env(config_approach, env_name, {"size": env_size})
    model = PPO(
        "MlpPolicy",
        env=env,
        **dict(config_ppo),
        learning_rate=get_linear_fn(
            config_ppo_lr["start_lr"],
            config_ppo_lr["end_lr"],
            config_ppo_lr["end_fraction"],
        ),
        policy_kwargs=create_sb_policy_kwargs(config_policy),
        seed=seed,
    )
    env.unwrapped.set_agent(model)  # type: ignore
    # Have to get the novelty function here and not inside "setup_start_state_picking", because it has to see the wrapper
    novelty_function = get_novelty_function(config_approach, env)
    env.unwrapped.setup_start_state_picking(config_approach, novelty_function)  # type: ignore
    env.reset()  # workaround for minigrid bug

    return model, env_base
