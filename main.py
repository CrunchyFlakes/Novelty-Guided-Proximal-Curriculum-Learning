import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from src.environments import get_prox_curr_env
from src.hpo import get_ppo_config_space
from src.util import get_novelty_function
from minigrid.envs import EmptyEnv, DoorKeyEnv, UnlockEnv
from minigrid.wrappers import ImgObsWrapper
import numpy as np
from pathlib import PosixPath

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

def target_function_configurable(config: Configuration, env_name: str, env_size: int, seed: int = 0, n_seeds: int = 1, n_workers: int = 1) -> tuple[float, float]:  # TODO: change number of seeds to evaluate on
    np.random.seed(seed)
    # Generate seeds
    seeds = list(map(int, np.random.randint(low=0, high=1000, size=n_seeds)))
    results = [train(config, env_name, env_size, seed=train_seed) for train_seed in seeds]
    result = tuple(np.mean(results, axis=0))
    print(f"Finished evaluating configuration with {result}")
    return result

def make_env(config_approach: Mapping[str, Any], env_name: str, env_kwargs: dict) -> tuple[gym.Env, gym.Env]:
    match env_name.lower():
        case "doorkey":
            env_class = DoorKeyEnv
        case "unlock":
            env_class = UnlockEnv
            env_kwargs = {}
        case _:
            raise NotImplementedError(f"Starting env with name {env_name} is not supported.")
    env_base = Monitor(ImgObsWrapper(env_class(**env_kwargs)))
    env = ImgObsWrapper(get_prox_curr_env(env_class, **env_kwargs))
    return env, env_base

def train(config: Configuration, env_name: str, env_size: int, seed: int = 0) -> tuple[float, int, float]:
    logger.info("Training new config")
    config_ppo = get_config_for_module(config, "sb_ppo")
    config_policy= get_config_for_module(config, "policy")
    config_approach = get_config_for_module(config, "approach")

    env, env_base = make_env(config_approach, env_name, {"size": env_size})
    model = PPO("MlpPolicy", env=env, **dict(config_ppo), policy_kwargs=create_sb_policy_kwargs(config_policy), seed=seed)

    # Setup env to use agents value network
    env.unwrapped.set_agent(model)  # type: ignore
    # Have to get the novelty function here and not inside "setup_start_state_picking", because it has to see the wrapper
    novelty_function = get_novelty_function(config_approach, env)
    env.unwrapped.setup_start_state_picking(config_approach, novelty_function)  # type: ignore
    env.reset()  # workaround for minigrid bug

    score, timesteps_left, mean_train_eval_score = learn(model, timesteps=500_000, eval_env=env_base, eval_every_n_steps=10000, early_terminate=True, early_termination_threshold=0.9)

    return -score, -timesteps_left, -mean_train_eval_score  # prioritize score over timesteps


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
        logger.info(f"Model at {timesteps - timesteps_left}/{timesteps} with {score=}")
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
    target_function = partial(target_function_configurable, env_name=args.env_name, env_size=args.env_size, n_seeds=args.n_seeds_train, n_workers=1)  # only one worker so it is still pickleable
    target_function_multiprocessing = partial(target_function_configurable, env_name=args.env_name, env_size=args.env_size, n_seeds=args.n_seeds_eval, n_workers=args.workers)


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
    train_result_prox = target_function_multiprocessing(incumbent_prox)
    logger.info(f"Score: {train_result_prox[0]}, Timesteps left: {train_result_prox[1]}")


    # Train model with State Novelty
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
    train_result_nov = target_function_multiprocessing(incumbent_nov)
    logger.info(f"Score: {train_result_nov[0]}, Timesteps left: {train_result_nov[1]}")


    # Train vanilla model
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
    train_result_vanilla = target_function_multiprocessing(incumbent_vanilla)
    logger.info(f"Score: {train_result_vanilla[0]}, Timesteps left: {train_result_vanilla[1]}")


    # Train Model with Proximal Curriculum and State Novelty
    configspace_comb = get_ppo_config_space(use_prox_curr=True, use_state_novelty=True)
    if not args.skiphpo:
        scenario_comb = Scenario(configspace_comb, **scenario_params)
        smac_comb = HyperparameterOptimizationFacade(scenario_comb, target_function, **facade_params)
        incumbent_comb: Configuration = smac_comb.optimize()  # type: ignore  # type fixed in next two lines
        if incumbent_comb is list:
            incumbent_comb = incumbent_comb[0]
    else:
        logger.info("Skipping HPO for Vanilla, using default configuration")
        incumbent_comb = configspace_comb.get_default_configuration()
    logger.info(f"Gotten Incumbent for Combined Approach: {incumbent_comb}")
    train_result_comb = target_function_multiprocessing(incumbent_comb)
    logger.info(f"Score: {train_result_comb[0]}, Timesteps left: {train_result_comb[1]}")
