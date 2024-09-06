import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn
from src.environments import get_prox_curr_env, ImgObsKeyWrapper
from src.hpo import get_ppo_config_space
from src.util import get_novelty_function
from minigrid.envs import DoorKeyEnv, UnlockEnv
import numpy as np
from pathlib import PosixPath
import torch

from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import Configuration, ConfigurationSpace

from typing import Callable, Any, Mapping
import logging
from functools import partial
import os
import json

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(__name__)

MAX_TIMESTEPS = 500_000

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
    seeds = [1000 * seed + subseed for subseed in range(n_seeds)]
    print(seeds)
    results = [train(*initialize_model_and_env(config, env_name, env_size, seed=train_seed)) for train_seed in seeds]
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

def initialize_model_and_env(config: Configuration, env_name: str, env_size: int, seed: int = 0) -> tuple[PPO, gym.Env]:
    config_ppo = get_config_for_module(config, "sb_ppo")
    config_ppo_lr = get_config_for_module(config, "sb_lr")
    config_policy= get_config_for_module(config, "policy")
    config_approach = get_config_for_module(config, "approach")

    env, env_base = make_env(config_approach, env_name, {"size": env_size})
    model = PPO("MlpPolicy", env=env, **dict(config_ppo), learning_rate=get_linear_fn(config_ppo_lr["start_lr"], config_ppo_lr["end_lr"], config_ppo_lr["end_fraction"]), policy_kwargs=create_sb_policy_kwargs(config_policy), seed=seed)
    env.unwrapped.set_agent(model)  # type: ignore
    # Have to get the novelty function here and not inside "setup_start_state_picking", because it has to see the wrapper
    novelty_function = get_novelty_function(config_approach, env)
    env.unwrapped.setup_start_state_picking(config_approach, novelty_function)  # type: ignore
    env.reset()  # workaround for minigrid bug

    return model, env_base

def train(model: PPO, env_eval: gym.Env) -> tuple[float, dict]:
    score, timesteps_left, score_history = learn(model, timesteps=MAX_TIMESTEPS, eval_env=env_eval, eval_every_n_steps=10000, early_terminate=True, early_termination_threshold=0.95)

    # real multiobjective is not that useful here, because the end score is what really counts
    # mean_train_eval_score is to get a better signal when there is no reward at the end, but the agent found something during evaluation runs while training
    # timesteps_left / MAX_TIMESTEPS is meant as a tie-breaker, but has to be able to compensate for terminating earlier -> multiply
    mean_train_eval_score = float(np.mean(list(score_history.values())))
    combined_score = (-1) * (10*score + mean_train_eval_score + 5 * (timesteps_left / MAX_TIMESTEPS))
    return combined_score, {"reward": score, "mean_train_eval_score": mean_train_eval_score, "time_left_ratio": timesteps_left / MAX_TIMESTEPS, "score_history": score_history}


def learn(model: OnPolicyAlgorithm, timesteps: int, eval_env: gym.Env, eval_every_n_steps: int, early_terminate: bool = False, early_termination_threshold: float = 0.0) -> tuple[float, int, dict[str, float]]:
    timesteps_left = timesteps
    score = 0
    score_history = {}
    while timesteps_left > 0:
        # Learn
        steps_to_learn = min(timesteps_left, eval_every_n_steps)
        model.learn(steps_to_learn)
        timesteps_left -= steps_to_learn

        score = evaluate_policy(model, eval_env,  n_eval_episodes=10)[0]
        score_history[timesteps - timesteps_left] = float(score)
        logger.debug(f"Model at {timesteps - timesteps_left}/{timesteps} with {score=}")
        if score >= early_termination_threshold and early_terminate:
            break
    mean_score_history = float(np.mean(list(score_history.values())))
    logger.info(f"Model finished with {score=}, {timesteps_left=}, {mean_score_history=}")
    return float(score), timesteps_left, score_history

def run_hpo(name: str, configspace: ConfigurationSpace, scenario_params: dict, facade_params: dict, target_function_smac: Callable) -> Configuration:
    logger.info(f"Now training {name} Model")
    scenario = Scenario(configspace, **scenario_params)
    smac = HyperparameterOptimizationFacade(scenario, target_function_smac, **facade_params)
    incumbent: Configuration = smac.optimize()  # type: ignore  # type fixed in next two lines
    if incumbent is list:
        incumbent = incumbent[0]
    return incumbent

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, required=False)
    parser.add_argument("--workers", type=int, required=False)
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--env_size", type=int, required=True)
    parser.add_argument("--mode", choices=["hpo", "eval"], required=True)
    parser.add_argument("--n_seeds_hpo", type=int, required=False, help="Number of seeds to use during HPO of a single model to get a more stable result. Not really needed, because SMAC actually evaluates configs multiple times if they are promising.")
    parser.add_argument("--n_seeds_eval", type=int, required=False, help="Number of seeds to use during evaluation after HPO")
    parser.add_argument("--eval_seed", type=int, required=False)
    parser.add_argument("--result_dir", type=PosixPath, required=True)
    parser.add_argument("--approach_to_check", choices=["comb", "prox", "nov", "vanilla"], required=True)
    args = parser.parse_args()


    target_function = partial(target_function_configurable, env_name=args.env_name, env_size=args.env_size, n_seeds=args.n_seeds_hpo)
    eval_function = partial(target_function_configurable, env_name=args.env_name, env_size=args.env_size, n_seeds=args.n_seeds_eval, seed=args.eval_seed)
    result_dir = args.result_dir / f"{args.env_name}{args.env_size}_{args.approach_to_check}"

    match args.mode:
        case "hpo":  # We will have to create the result dir
            os.makedirs(result_dir, exist_ok=False)

            match args.approach_to_check:
                case "comb":
                    configspace = get_ppo_config_space(use_prox_curr=True, use_state_novelty=True)
                case "prox":
                    configspace = get_ppo_config_space(use_prox_curr=True, use_state_novelty=False)
                case "nov":
                    configspace = get_ppo_config_space(use_prox_curr=False, use_state_novelty=True)
                case "vanilla":
                    configspace = get_ppo_config_space(use_prox_curr=False, use_state_novelty=False)
                case _:
                    raise NotImplementedError(f"Approach {args.approach_to_check} not implemented.")

            facade_params = {
                "logging_level": logging.INFO,
            }
            scenario_params = {
                "n_trials": args.trials,
                "n_workers": args.workers,
                "use_default_config": True,
                "output_directory": result_dir / "smac",
            }
            incumbent = run_hpo(name=args.approach_to_check, configspace=configspace, scenario_params=scenario_params, facade_params=facade_params, target_function_smac=target_function)

            logger.info(f"Gotten Incumbent for {args.approach_to_check} Approach: {incumbent}")
            # Save configuration
            configspace.to_json(result_dir / "configspace.json")
            with open(result_dir / "config.json", "w") as config_file:
                incumbent_converted = {key: value.item() if type(value).__module__ == "numpy" else value for key, value in dict(incumbent).items()}
                json.dump(incumbent_converted, config_file)
            print("Incumbent saved to directory, you can now load it")

        case "eval":  # There already has to be a result dir (the given one). We will only evaluate it
            configspace = ConfigurationSpace.from_json(result_dir / "configspace.json")
            with open(result_dir / "config.json", "r") as config_file:
                incumbent = Configuration(configuration_space=configspace, values=json.load(config_file))
            # Train configuration for given number of seeds
            train_result_score, train_result_info = eval_function(incumbent)

            logger.info(f"{args.approach_to_check} Approach Results: score={train_result_score}")
            with open(result_dir / f"result_info_seed{args.eval_seed}.json", "w") as result_file:
                json.dump(train_result_info, result_file)
        case _:
            raise NotImplementedError(f"mode {args.mode} not implemented.")


