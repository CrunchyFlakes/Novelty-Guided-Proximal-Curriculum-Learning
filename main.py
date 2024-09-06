import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import get_linear_fn
from src.environments import get_prox_curr_env, FullImgObsWrapper, ImgObsKeyWrapper
from minigrid.wrappers import ImgObsWrapper
from src.hpo import get_ppo_config_space
from src.util import get_novelty_function
from minigrid.envs import EmptyEnv, DoorKeyEnv, UnlockEnv
import numpy as np
from pathlib import PosixPath
import torch

from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import Configuration, ConfigurationSpace

from typing import Callable, Any, Mapping
import logging
import multiprocessing
from functools import partial
from itertools import product

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(__name__)

MAX_TIMESTEPS = 500_000
MAX_ENV_TIMESTEPS = 1_000

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

def create_sb_policy_kwargs(config: Mapping[str, Any]) -> dict:
    return {  # TODO: decide if they should get different architectures
        "net_arch": {
            "pi": [config[f"policy_layer{i}_size"] for i in range(config["policy_n_layers"])],
            "vf": [config[f"policy_layer{i}_size"] for i in range(config["policy_n_layers"])],
        }
    }

def target_function_configurable(config: Configuration, env_name: str, env_size: int, seed: int = 0, n_seeds: int = 1) -> tuple[float, dict]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Generate seeds
    seeds = list(map(int, np.random.randint(low=0, high=1000, size=n_seeds)))
    results = [train(config, env_name, env_size, seed=train_seed) for train_seed in seeds]
    scores, infos = zip(*results)
    return float(np.mean(scores)), infos

def train_pickleable(input: dict):
    return train(**input)

def target_function_multiprocessing(config: Configuration, env_name: str, env_size: int, seed: int = 0, n_seeds: int = 1) -> tuple[float, dict]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Generate seeds
    seeds = list(map(int, np.random.randint(low=0, high=1000, size=n_seeds)))
    pool = multiprocessing.Pool(n_seeds)
    train_inputs = [{"config": curr_config, "env_name": curr_env_name, "env_size": curr_env_size, "seed": curr_seed} for curr_config, curr_env_name, curr_env_size, curr_seed in product([config], [env_name], [env_size], seeds)]
    results = pool.map(train_pickleable, train_inputs)
    scores, infos = zip(*results)
    return float(np.mean(scores)), infos

def make_env(config_approach: Mapping[str, Any], env_name: str, env_kwargs: dict) -> tuple[gym.Env, gym.Env]:
    match env_name.lower():
        case "doorkey":
            env_class = DoorKeyEnv
        case "unlock":
            env_class = UnlockEnv
            env_kwargs = {}
        case _:
            raise NotImplementedError(f"Starting env with name {env_name} is not supported.")
    env_base = Monitor(ImgObsKeyWrapper(env_class(**env_kwargs)))
    env = ImgObsKeyWrapper(get_prox_curr_env(env_class, **env_kwargs))
    return env, env_base

def train(config: Configuration, env_name: str, env_size: int, seed: int = 0) -> tuple[float, dict]:
    logger.debug("Training new config")
    config_ppo = get_config_for_module(config, "sb_ppo")
    config_ppo_lr = get_config_for_module(config, "sb_lr")
    config_policy= get_config_for_module(config, "policy")
    config_approach = get_config_for_module(config, "approach")

    env, env_base = make_env(config_approach, env_name, {"size": env_size})
    model = PPO("MlpPolicy", env=env, **dict(config_ppo), learning_rate=get_linear_fn(config_ppo_lr["start_lr"], config_ppo_lr["end_lr"], config_ppo_lr["end_fraction"]), policy_kwargs=create_sb_policy_kwargs(config_policy), seed=seed)

    # Setup env to use agents value network
    env.unwrapped.set_agent(model)  # type: ignore
    # Have to get the novelty function here and not inside "setup_start_state_picking", because it has to see the wrapper
    novelty_function = get_novelty_function(config_approach, env)
    env.unwrapped.setup_start_state_picking(config_approach, novelty_function)  # type: ignore
    env.reset()  # workaround for minigrid bug

    score, timesteps_left, mean_train_eval_score = learn(model, timesteps=MAX_TIMESTEPS, eval_env=env_base, eval_every_n_steps=10000, early_terminate=True, early_termination_threshold=0.9)

    # real multiobjective is not that useful here, because the end score is what really counts
    # mean_train_eval_score is to get a better signal
    # timesteps_left / MAX_TIMESTEPS is meant as a tie-breaker
    combined_score = (-1) * (10*score + mean_train_eval_score + timesteps_left / MAX_TIMESTEPS)
    return combined_score, {"reward": score, "mean_train_eval_score": mean_train_eval_score, "time_left_ratio": timesteps_left / MAX_TIMESTEPS}


def learn(model: OnPolicyAlgorithm, timesteps: int, eval_env: gym.Env, eval_every_n_steps: int, early_terminate: bool = False, early_termination_threshold: float = 0.0) -> tuple[float, int, float]:
    timesteps_left = timesteps
    score = 0
    score_history = []
    while timesteps_left > 0:
        # Learn
        steps_to_learn = min(timesteps_left, eval_every_n_steps)
        model.learn(steps_to_learn)
        timesteps_left -= steps_to_learn

        score = evaluate_policy(model, eval_env,  n_eval_episodes=10)[0]
        score_history.append(score)
        logger.debug(f"Model at {timesteps - timesteps_left}/{timesteps} with {score=}")
        if score >= early_termination_threshold and early_terminate:
            break
    mean_score_history = float(np.mean(score_history))
    logger.info(f"Model finished with {score=}, {timesteps_left=}, {mean_score_history=}")
    return float(score), timesteps_left, mean_score_history

def evaluate_config(config: Configuration, seed: int = 0):
    pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--env_size", type=int, required=True)
    parser.add_argument("--skiphpo", action="store_true")
    parser.add_argument("--n_seeds_train", type=int, required=True, help="Number of seeds to use during training")
    parser.add_argument("--n_seeds_eval", type=int, required=True, help="Number of seeds to use during evaluation after HPO")
    parser.add_argument("--smac_output_dir", type=PosixPath, required=True)
    args = parser.parse_args()

    facade_params = {
        "logging_level": logging.INFO,
    }
    scenario_params = {
        "n_trials": args.trials,
        "n_workers": args.workers,
        "use_default_config": True,
        "output_directory": args.smac_output_dir,
    }
    target_function = partial(target_function_configurable, env_name=args.env_name, env_size=args.env_size, n_seeds=args.n_seeds_train)  # only one worker so it is still pickleable
    target_function_eval = partial(target_function_multiprocessing, env_name=args.env_name, env_size=args.env_size, n_seeds=args.n_seeds_eval)


    # Train Model with Proximal Curriculum and State Novelty
    logger.info(f"Now training Combined Model")
    configspace_comb = get_ppo_config_space(use_prox_curr=True, use_state_novelty=True)
    if not args.skiphpo:
        scenario_comb = Scenario(configspace_comb, **scenario_params)
        smac_comb = HyperparameterOptimizationFacade(scenario_comb, target_function, **facade_params)
        incumbent_comb: Configuration = smac_comb.optimize()  # type: ignore  # type fixed in next two lines
        if incumbent_comb is list:
            incumbent_comb = incumbent_comb[0]
    else:
        logger.info("Skipping HPO for Combined Approach, using default configuration")
        incumbent_comb = configspace_comb.get_default_configuration()
    logger.info(f"Gotten Incumbent for Combined Approach: {incumbent_comb}")
    train_result_comb_score, train_result_comb_info = target_function_eval(incumbent_comb)
    logger.info(f"Combined Approach Results: score={train_result_comb_score}, {train_result_comb_info}")


    # Train Model with Proximal Curriculum
    logger.info(f"Now training Proximal Curriculum Model")
    configspace_prox = get_ppo_config_space(use_prox_curr=True, use_state_novelty=False)
    if not args.skiphpo:
        scenario_prox = Scenario(configspace_prox, **scenario_params)
        smac_prox = HyperparameterOptimizationFacade(scenario_prox, target_function, **facade_params)
        incumbent_prox: Configuration = smac_prox.optimize()  # type: ignore  # type fixed in next two lines
        if incumbent_prox is list:
            incumbent_prox = incumbent_prox[0]
    else:
        logger.info("Skipping HPO for Proximal Curriculum, using default configuration")
        incumbent_prox = configspace_prox.get_default_configuration()
    logger.info(f"Gotten Incumbent for Proximal Curriculum: {incumbent_prox}")
    train_result_prox_score, train_result_prox_info = target_function_eval(incumbent_prox)
    logger.info(f"Proximal Curriculum Approach Results: score={train_result_prox_score}, {train_result_prox_info}")



    # Train model with State Novelty
    logger.info(f"Now training State Novelty Model")
    configspace_nov = get_ppo_config_space(use_prox_curr=False, use_state_novelty=True)
    if not args.skiphpo:
        scenario_nov = Scenario(configspace_nov, **scenario_params)
        smac_nov = HyperparameterOptimizationFacade(scenario_nov, target_function, **facade_params)
        incumbent_nov: Configuration = smac_nov.optimize()  # type: ignore  # type fixed in next two lines
        if incumbent_nov is list:
            incumbent_nov = incumbent_nov[0]
    else:
        logger.info("Skipping HPO for State Novelty, using default configuration")
        incumbent_nov = configspace_nov.get_default_configuration()
    logger.info(f"Gotten Incumbent for State Novelty Approach: {incumbent_nov}")
    train_result_nov_score, train_result_nov_info = target_function_eval(incumbent_nov)
    logger.info(f"State Novelty Approach Results: score={train_result_nov_score}, {train_result_nov_info}")


    # Train vanilla model
    logger.info(f"Now training Vanilla Model")
    configspace_vanilla = get_ppo_config_space(use_prox_curr=False, use_state_novelty=False)
    if not args.skiphpo:
        scenario_vanilla = Scenario(configspace_vanilla, **scenario_params)
        smac_vanilla = HyperparameterOptimizationFacade(scenario_vanilla, target_function, **facade_params)
        incumbent_vanilla: Configuration = smac_vanilla.optimize()  # type: ignore  # type fixed in next two lines
        if incumbent_vanilla is list:
            incumbent_vanilla = incumbent_vanilla[0]
    else:
        logger.info("Skipping HPO for Vanilla, using default configuration")
        incumbent_vanilla = configspace_vanilla.get_default_configuration()
    logger.info(f"Gotten Incumbent for Vanilla Approach: {incumbent_vanilla}")
    train_result_vanilla_score, train_result_vanilla_info = target_function_eval(incumbent_vanilla)
    logger.info(f"Vanilla Approach Results: score={train_result_vanilla_score}, {train_result_vanilla_info}")
