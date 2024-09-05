from .logic import RND, Number, State
import numpy as np
import gymnasium as gym
import torch

from typing import Callable, Any
from functools import reduce


def get_novelty_function(config: dict[str, Any], env: gym.Env) -> Callable[[torch.Tensor], torch.Tensor]:
    match config["novelty_approach"]:
        case "rnd":
            if env.observation_space.shape is None:
                raise ValueError("Environment doesn't posess observation space with shape property!")
            layers_fixed = [reduce(lambda x,y: x*y, env.observation_space.shape)] + [config[f"rnd_layer{i}_size"] for i in range(config["rnd_n_layers"])]
            layers_learned = [reduce(lambda x,y: x*y, env.observation_space.shape)] + [config[f"rnd_layer{i}_size"] for i in range(config["rnd_n_layers"])]
            return RND(layers_fixed=layers_fixed, layers_learned=layers_learned, loss=config["rnd_loss"], learning_rate=config["rnd_learning_rate"], activation_function=config["rnd_activation"], optimizer=config["rnd_optimizer"])
        case _:
            return lambda input: torch.ones(input.shape[0])
