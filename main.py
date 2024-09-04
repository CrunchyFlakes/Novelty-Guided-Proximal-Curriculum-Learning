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
import logging
import multiprocessing


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
    config_ppo = get_config_for_module(config, "sb_ppo")
    config_approach = get_config_for_module(config, "approach")

    env_base = Monitor(ImgObsWrapper(EmptyEnv()))
    env = ImgObsWrapper(ProxCurrEmptyEnv()) if bool(config_approach["use_prox_curr"]) or bool(config_approach["use_state_novelty"]) else ImgObsWrapper(EmptyEnv())
    model = PPO("MlpPolicy", env=env, **dict(config_ppo))
    if bool(config_approach["use_prox_curr"]) or bool(config_approach["use_state_novelty"]):
        env.unwrapped.set_agent(model)  # type: ignore
        env.unwrapped.setup_start_state_picking(get_config_for_module(config, "approach"))  # type: ignore
    env.reset()  # workaround for minigrid bug

    evaluate = lambda model: evaluate_policy(model, env_base, n_eval_episodes=10)[0]  # [0] -> only return mean score and not variance
    score, timesteps_left = learn(model, evaluate, timesteps=100_000, eval_every_n_steps=10000, early_terminate=True, early_termination_threshold=0.9)

    return -score, -timesteps_left  # prioritize score over timesteps


def learn(model: OnPolicyAlgorithm, evaluate: Callable[[OnPolicyAlgorithm], Any], timesteps: int, eval_every_n_steps: int, early_terminate: bool = False, early_termination_threshold: float = 0.0) -> tuple[int, float]:
    timesteps_left = timesteps
    score = 0
    while timesteps_left > 0:
        # Learn
        steps_to_learn = min(timesteps_left, eval_every_n_steps)
        model.learn(steps_to_learn)
        timesteps_left -= steps_to_learn

        score = evaluate(model)
        if score >= early_termination_threshold and early_terminate:
            break
    return score, timesteps_left


if __name__ == "__main__":
    logging.basicConfig()
    logging.root.setLevel(logging.INFO)
    logger = logging.getLogger(__name__)

    scenario_params = {
        "n_trials": 20,
        "n_workers": multiprocessing.cpu_count() - 1,  # -1 to have resources for other stuff
        "use_default_config": True,
    }

    # Train vanilla model
    configspace_vanilla = get_ppo_config_space(use_prox_curr=False, use_state_novelty=False)
    scenario_vanilla = Scenario(configspace_vanilla, **scenario_params)
    smac_vanilla = HyperparameterOptimizationFacade(scenario_vanilla, train)
    incumbent_vanilla: Configuration = smac_vanilla.optimize()  # type: ignore  # type fixed in next two lines
    if incumbent_vanilla is list:
        incumbent_vanilla = incumbent_vanilla[0]
    logger.info(f"Gotten Incumbent for Vanilla Approach: {incumbent_vanilla}")
    train_result_vanilla = train(incumbent_vanilla)
    logger.info(f"Score: {train_result_vanilla[0]}, Timesteps left: {train_result_vanilla[1]}")


    # Train Model with Proximal Curriculum
    logger.info(f"Now training Proximal Curriculum Model")
    configspace_prox = get_ppo_config_space(use_prox_curr=True, use_state_novelty=False)
    scenario_prox = Scenario(configspace_prox, **scenario_params)
    smac_prox = HyperparameterOptimizationFacade(scenario_prox, train)
    incumbent_prox: Configuration = smac_prox.optimize()  # type: ignore  # type fixed in next two lines
    if incumbent_prox is list:
        incumbent_prox = incumbent_prox[0]
    logger.info(f"Gotten Incumbent for Proximal Curriculum: {incumbent_prox}")
    train_result_prox = train(incumbent_prox)
    logger.info(f"Score: {train_result_prox[0]}, Timesteps left: {train_result_prox[1]}")


    # Train Model with Proximal Curriculum and State Novelty
    configspace_comb = get_ppo_config_space(use_prox_curr=True, use_state_novelty=True)
    scenario_comb = Scenario(configspace_comb, **scenario_params)
    smac_comb = HyperparameterOptimizationFacade(scenario_comb, train)
    incumbent_comb: Configuration = smac_comb.optimize()  # type: ignore  # type fixed in next two lines
    if incumbent_comb is list:
        incumbent_comb = incumbent_comb[0]
    logger.info(f"Gotten Incumbent for Combined Approach: {incumbent_comb}")
    train_result_comb = train(incumbent_comb)
    logger.info(f"Score: {train_result_comb[0]}, Timesteps left: {train_result_comb[1]}")


    # Train model with State Novelty
    configspace_nov = get_ppo_config_space(use_prox_curr=False, use_state_novelty=True)
    scenario_nov = Scenario(configspace_nov, **scenario_params)
    smac_nov = HyperparameterOptimizationFacade(scenario_nov, train)
    incumbent_nov: Configuration = smac_nov.optimize()  # type: ignore  # type fixed in next two lines
    if incumbent_nov is list:
        incumbent_nov = incumbent_nov[0]
    logger.info(f"Gotten Incumbent for State Novelty Approach: {incumbent_nov}")
    train_result_nov = train(incumbent_nov)
    logger.info(f"Score: {train_result_nov[0]}, Timesteps left: {train_result_nov[1]}")
