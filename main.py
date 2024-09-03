import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from src.environments import ProxCurrEmptyEnv
from src.hpo import get_ppo_config_space
from minigrid.envs import EmptyEnv
from minigrid.wrappers import ImgObsWrapper

from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import Configuration, ConfigurationSpace

from typing import Callable, Any

def get_config_for_module(cfg: Configuration, module_name: str) -> dict[str, Any]:
    """
    This function is used to extract a sub configuration that belongs to a certain module
    Note that this function needs to call for each level
    :param cfg: a configuration
    :param module_name: the module name
    :return: cfg_module: a new dict that contains all the hp values belonging to the configuration
    """
    cfg_module = {}
    for key, value in cfg.items():
        if key.startswith(module_name):
            new_key = key.replace(f"{module_name}:", "")
            cfg_module[new_key] = value
    return cfg_module

def train(config: Configuration, seed: int = 0) -> tuple[float, float]:
    env_base = Monitor(ImgObsWrapper(EmptyEnv()))
    env = ImgObsWrapper(ProxCurrEmptyEnv())
    model = PPO("MlpPolicy", env=env, **dict(get_config_for_module(config, "sb_ppo")))
    env.unwrapped.set_agent(model)  # type: ignore
    env.unwrapped.setup_start_state_picking(get_config_for_module(config, "approach"))  # type: ignore
    env.reset()
    evaluate = lambda model: evaluate_policy(model, env_base, n_eval_episodes=10)[0]  # [0] -> only return mean score and not variance
    score, timesteps_left = learn(model, evaluate, timesteps=100_000, eval_every_n_steps=10000, early_terminate=True, early_termination_threshold=0.9)
    return score, timesteps_left  # prioritize score over timesteps


def learn(model: OnPolicyAlgorithm, evaluate: Callable[[OnPolicyAlgorithm], Any], timesteps: int, eval_every_n_steps: int, early_terminate: bool = False, early_termination_threshold: float = 0.0) -> tuple[int, float]:
    timesteps_left = timesteps
    score = 0
    while timesteps_left > 0:
        # Learn
        steps_to_learn = min(timesteps_left, eval_every_n_steps)
        model.learn(steps_to_learn)
        timesteps_left -= steps_to_learn

        score = evaluate(model)
        if score >= early_termination_threshold:
            break
    return score, timesteps_left


if __name__ == "__main__":
    configspace = get_ppo_config_space(use_state_novelty=False)
    scenario = Scenario(configspace, deterministic=True, n_trials=50, n_workers=10)
    smac = HyperparameterOptimizationFacade(scenario, train)
    incumbent = smac.optimize()
    print(incumbent)
