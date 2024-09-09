import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from src.util import initialize_model_and_env
from src.hpo import get_ppo_config_space
import numpy as np
from pathlib import PosixPath
import torch

from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import Configuration, ConfigurationSpace

from typing import Callable
import logging
from functools import partial
import os
import json

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(__name__)

MAX_TIMESTEPS = 1_000_000


def target_function_configurable(
    config: Configuration, env_name: str, env_size: int, seed: int = 0, n_seeds: int = 1
) -> tuple[float, dict]:
    """Create target function for SMAC, but provide additional parameters that can be set prior by partial function application.

    Args:
        config: configuration to evaluate
        env_name: environment to create
        env_size: size the environment should be created with. Does have to match what is possible with the environment
        seed: seed to evaluate
        n_seeds: how many seeds (created using the given seed) to evaluate

    Returns:
        an aggregated score over all runs (lower is better) and an information dict of the evaluations
    """
    if (
        seed is not int
    ):  # Does SMAC sometimes pass multiple seeds? A comment in the documentation seemed like it, lets be sure that nothing happens
        seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Generate seeds
    if (
        n_seeds > 0
    ):  # We are evaluating and get small enough seeds to multiply them without going out of bounds
        seeds = [seed + subseed for subseed in range(n_seeds)]
    else:  # SMAC is passing seeds and we only get one seed
        seeds = [seed]
    results = [
        train(*initialize_model_and_env(config, env_name, env_size, seed=train_seed))
        for train_seed in seeds
    ]
    scores, infos = zip(*results)
    return float(np.mean(scores)), infos


def train(model: PPO, env_eval: gym.Env) -> tuple[float, dict]:
    """Train a given model and evaluate on the given env

    Args:
        model: the model to train, has to have the environment to train on already set
        env_eval: environment to evaluate the agent on

    Returns:
        a score (lower is better) and an information dict of the evaluation
    """
    score, timesteps_left, score_history = learn(
        model,
        timesteps=MAX_TIMESTEPS,
        eval_env=env_eval,
        eval_every_n_steps=10000,
        early_terminate=True,
        early_termination_threshold=0.95,
    )

    # real multiobjective is not that useful here, because the end score is what really counts
    # mean_train_eval_score is to get a better signal when there is no reward at the end, but the agent found something during evaluation runs while training
    # timesteps_left / MAX_TIMESTEPS is meant as a tie-breaker, but has to be able to compensate for terminating earlier -> multiply
    mean_train_eval_score = float(np.mean(list(score_history.values())))
    combined_score = (-1) * (
        10 * score + mean_train_eval_score + 5 * (timesteps_left / MAX_TIMESTEPS)
    )
    return combined_score, {
        "reward": score,
        "mean_train_eval_score": mean_train_eval_score,
        "time_left_ratio": timesteps_left / MAX_TIMESTEPS,
        "score_history": score_history,
    }


def learn(
    model: OnPolicyAlgorithm,
    timesteps: int,
    eval_env: gym.Env,
    eval_every_n_steps: int,
    early_terminate: bool = False,
    early_termination_threshold: float = 0.0,
) -> tuple[float, int, dict[str, float]]:
    """Training loop for the agent and the state novelty approach. Does also generate evaluations every n runs.

    Args:
        model: model to train
        timesteps: how many timesteps to train
        eval_env: on which environment to run evaluations on
        eval_every_n_steps: how often to evaluate
        early_terminate: terminate training if threshold is reached. Does only check on every evaluation
        early_termination_threshold: the threshold for early termination

    Returns:
        The score the agent reached, how many timesteps would have been left to train (early termination), history of scores during the training
    """
    timesteps_left = timesteps
    score = 0
    score_history = {}
    while timesteps_left > 0:
        # Learn
        steps_to_learn = min(timesteps_left, eval_every_n_steps)
        model.learn(steps_to_learn)
        timesteps_left -= steps_to_learn

        score = float(evaluate_policy(model, eval_env, n_eval_episodes=10)[0])  # type: ignore
        score_history[timesteps - timesteps_left] = float(score)
        logger.debug(f"Model at {timesteps - timesteps_left}/{timesteps} with {score=}")
        if score >= early_termination_threshold and early_terminate:
            break
    mean_score_history = float(np.mean(list(score_history.values())))
    logger.info(
        f"Model finished with {score=}, {timesteps_left=}, {mean_score_history=}"
    )
    return score, timesteps_left, score_history


def run_hpo(
    name: str,
    configspace: ConfigurationSpace,
    scenario_params: dict,
    facade_params: dict,
    target_function_smac: Callable,
) -> Configuration:
    """Run hyperparameter optimization using SMAC

    Args:
        name: Name of the run. For logging purposes
        configspace: search space for the hyperparameters
        scenario_params: keyword arguments for SMACs Scenario
        facade_params: keyword arguments for SMACs HyperparameterOptimizationFacade
        target_function_smac: function that evaluates configurations and returns a score

    Returns:

    """
    logger.info(f"Now training {name} Model")
    scenario = Scenario(configspace, **scenario_params)
    smac = HyperparameterOptimizationFacade(
        scenario, target_function_smac, **facade_params
    )
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
    parser.add_argument(
        "--n_seeds_hpo",
        type=int,
        required=False,
        help="Number of seeds to use during HPO of a single model to get a more stable result. Not really needed, because SMAC actually evaluates configs multiple times if they are promising.",
    )
    parser.add_argument(
        "--n_seeds_eval",
        type=int,
        required=False,
        help="Number of seeds to use during evaluation after HPO",
    )
    parser.add_argument("--eval_seed", type=int, required=False)
    parser.add_argument("--result_dir", type=PosixPath, required=True)
    parser.add_argument(
        "--approach_to_check", choices=["comb", "prox", "nov", "vanilla"], required=True
    )
    args = parser.parse_args()

    target_function = partial(
        target_function_configurable,
        env_name=args.env_name,
        env_size=args.env_size,
        n_seeds=args.n_seeds_hpo,
    )
    eval_function = partial(
        target_function_configurable,
        env_name=args.env_name,
        env_size=args.env_size,
        n_seeds=args.n_seeds_eval,
        seed=args.eval_seed,
    )
    result_dir = (
        args.result_dir / f"{args.env_name}{args.env_size}_{args.approach_to_check}"
    )

    match args.mode:
        case "hpo":  # We will have to create the result dir
            os.makedirs(result_dir, exist_ok=False)

            match args.approach_to_check:
                case "comb":
                    configspace = get_ppo_config_space(
                        use_prox_curr=True, use_state_novelty=True
                    )
                case "prox":
                    configspace = get_ppo_config_space(
                        use_prox_curr=True, use_state_novelty=False
                    )
                case "nov":
                    configspace = get_ppo_config_space(
                        use_prox_curr=False, use_state_novelty=True
                    )
                case "vanilla":
                    configspace = get_ppo_config_space(
                        use_prox_curr=False, use_state_novelty=False
                    )
                case _:
                    raise NotImplementedError(
                        f"Approach {args.approach_to_check} not implemented."
                    )

            facade_params = {
                "logging_level": logging.INFO,
            }
            scenario_params = {
                "n_trials": args.trials,
                "n_workers": args.workers,
                "use_default_config": True,
                "output_directory": result_dir / "smac",
            }
            incumbent = run_hpo(
                name=args.approach_to_check,
                configspace=configspace,
                scenario_params=scenario_params,
                facade_params=facade_params,
                target_function_smac=target_function,
            )

            logger.info(
                f"Gotten Incumbent for {args.approach_to_check} Approach: {incumbent}"
            )
            # Save configuration
            configspace.to_json(result_dir / "configspace.json")
            with open(result_dir / "config.json", "w") as config_file:
                incumbent_converted = {
                    key: value.item() if type(value).__module__ == "numpy" else value
                    for key, value in dict(incumbent).items()
                }
                json.dump(incumbent_converted, config_file)
            print("Incumbent saved to directory, you can now load it")

        case (
            "eval"
        ):  # There already has to be a result dir (the given one). We will only evaluate it
            configspace = ConfigurationSpace.from_json(result_dir / "configspace.json")
            with open(result_dir / "config.json", "r") as config_file:
                incumbent = Configuration(
                    configuration_space=configspace, values=json.load(config_file)
                )
            # Train configuration for given number of seeds
            train_result_score, train_result_info = eval_function(incumbent)

            logger.info(
                f"{args.approach_to_check} Approach Results: score={train_result_score}"
            )
            with open(
                result_dir / f"result_info_seed{args.eval_seed}.json", "w"
            ) as result_file:
                json.dump(train_result_info, result_file)
        case _:
            raise NotImplementedError(f"mode {args.mode} not implemented.")
