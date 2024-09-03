from ConfigSpace import ConfigurationSpace, Float, Integer, Constant


def get_ppo_config_space(use_prox_curr: bool = True, use_state_novelty: bool = True) -> ConfigurationSpace:
    # Default values are taken from stable baselines ppo algorithm
    # This part only covers the stable_baselines hyperparemeters
    configspace_sb_ppo =  ConfigurationSpace({
        "learning_rate": Float("learning_rate", (1.0e-6, 1.0), default=0.0003, log=True),
        # number of steps to run environment before update
        "n_steps": Integer("n_steps", (2, 8192), default=2048),
        "batch_size": Integer("batch_size", (1, 512), default=64),
        "gamma": Float("gamma", (0.9, 1.0), default=0.99),
        "gae_lambda": Float("gae_lambda", (0.5, 1.0), default=0.95),
        "clip_range": Float("clip_range", (0.0, 1.0), default=0.2),
        "ent_coef": Float("ent_coef", (0.0, 1.0), default=0.0),
        "vf_coef": Float("vf_coef", (0.0, 1.0), default=0.5),
        "max_grad_norm": Float("max_grad_norm", (0.0, 1.0), default=0.5),
        # "use_sde": ..., don't use this for now, have to read paper first. This may be cut due to time constraints
        # There may be more hyperparameters to set, but these are the ones directly specified in stable_baselines3.ppo.PPO
    })

    # Hyperparameters needed for proximal curriculum learning with state novelty
    if use_prox_curr:
        configspace_sb_ppo.add(Float("beta_proximal", (0.0, 1.0), default=0.5))  # this default is guessed
    else:  # dummy value
        configspace_sb_ppo.add(Constant("beta_proximal", 0.0))
    if use_prox_curr and use_state_novelty:
        configspace_sb_ppo.add(Float("gamma_tradeoff", (0.0, 1.0), default=0.5))
    elif use_prox_curr:
        configspace_sb_ppo.add(Constant("gamma_tradeoff", 1.0))
    elif use_state_novelty:
        configspace_sb_ppo.add(Constant("gamma_tradeoff", 0.0))

    # TOOD: There may be hyperparameters for state novelty later on
    
    return configspace_sb_ppo
