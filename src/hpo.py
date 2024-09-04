from ConfigSpace import ConfigurationSpace, Float, Integer, Constant, EqualsCondition, Categorical, AndConjunction, GreaterThanCondition


def get_ppo_config_space(use_prox_curr: bool = True, use_state_novelty: bool = True) -> ConfigurationSpace:
    # Default values are taken from stable baselines ppo algorithm
    # This part only covers the stable_baselines hyperparemeters
    configspace_sb_ppo =  ConfigurationSpace({
        "learning_rate": Float("learning_rate", (1.0e-6, 1.0e-2), default=0.0003, log=True),
        # number of steps to run environment before update
        #"n_steps": Integer("n_steps", (2, 8192), default=2048),
        #"batch_size": Integer("batch_size", (1, 512), default=64),
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
    configspace_approach = ConfigurationSpace({
        "use_prox_curr": Constant("use_prox_curr", use_prox_curr),
        "use_state_novelty": Constant("use_state_novelty", use_state_novelty)
    })
    beta_proximal = Float("beta_proximal", (0.0, 1.0), default=0.5)  # this default is guessed
    beta_proximal_cond = EqualsCondition(beta_proximal, configspace_approach["use_prox_curr"], True)
    configspace_approach.add((beta_proximal, beta_proximal_cond))
    if use_prox_curr and use_state_novelty:
        configspace_approach.add(Float("gamma_tradeoff", (0.0, 1.0), default=0.5))
    elif use_prox_curr:
        configspace_approach.add(Constant("gamma_tradeoff", 1.0))
    elif use_state_novelty:
        configspace_approach.add(Constant("gamma_tradeoff", 0.0))


    # Additional state novelty hyperparameters per algorithm
    novelty_approach = Constant("novelty_approach", "rnd" if use_state_novelty else "none")
    ## Random Network Distillation
    rnd_loss = Constant("rnd_loss", "mse")
    rnd_loss_cond = EqualsCondition(rnd_loss, novelty_approach, "rnd")
    rnd_activation = Categorical("rnd_activation", ("relu", "leakyrelu"), default="relu")
    rnd_activation_cond = EqualsCondition(rnd_activation, novelty_approach, "rnd")
    rnd_learning_rate = Float("rnd_learning_rate", (1.0e-6, 1.0), default=1e-3, log=True)
    rnd_learning_rate_cond = EqualsCondition(rnd_learning_rate, novelty_approach, "rnd")
    rnd_optimizer = Constant("rnd_optimizer", "adam")
    rnd_optimizer_cond = EqualsCondition(rnd_optimizer, novelty_approach, "rnd")
    configspace_approach.add((rnd_loss, rnd_loss_cond, rnd_activation, rnd_activation_cond, rnd_learning_rate, rnd_learning_rate_cond, rnd_optimizer, rnd_optimizer_cond,))
    ### layer sizes
    max_layers = 5
    rnd_n_layers = Integer("rnd_n_layers", (1, max_layers), default=2)
    rnd_layersizes = [Integer(f"rnd_layer{i}_size", (1, 128), default=32) for i in range(max_layers)]
    rnd_layersize_conds = [GreaterThanCondition(rnd_layersizes[i], rnd_n_layers, i) for i in range(1, max_layers)]
    configspace_approach.add((rnd_n_layers, *rnd_layersizes, *rnd_layersize_conds))
    

    # combine everything together
    cs = ConfigurationSpace({})
    cs.add_configuration_space(prefix="sb_ppo", configuration_space=configspace_sb_ppo)
    cs.add_configuration_space(prefix="approach", configuration_space=configspace_approach)
    return cs
