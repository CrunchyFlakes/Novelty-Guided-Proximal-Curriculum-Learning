from ConfigSpace import (
    ConfigurationSpace,
    Float,
    Integer,
    Constant,
    EqualsCondition,
    Categorical,
    AndConjunction,
    GreaterThanCondition,
)

from typing import Mapping, Any


def get_ppo_config_space(
    use_prox_curr: bool = True, use_state_novelty: bool = True
) -> ConfigurationSpace:
    """Create configuration space for PPO agent including hyperparameters for the approach

    Args:
        use_prox_curr: if proximal curriculum learning should get activated
        use_state_novelty: if state novelty guidance should get activated

    Returns:
        configuration/search space for the hyperparameters
    """
    # Default values are composed of stable baselines defaults and
    # an aggregation of previously found configurations by SMAC for the different approaches.
    # This may result in SMAC not finding a new better configuration.
    # Although that undermines the reproducibility of the hyperparemeter optimization, the
    # configuration is just a tool to test the actual approach and is therefore not relevant for the actual evaluation.
    # To see the configuration changes look at commit "875d1d593a272401adc96576e76217690222612f".

    # This part only covers the stable_baselines hyperparemeters
    configspace_sb_ppo = ConfigurationSpace(
        {
            # number of steps to run environment before update
            # "n_steps": Integer("n_steps", (2, 8192), default=2048),
            "batch_size": Categorical("batch_size", (16, 32, 64, 128, 256), default=64),
            "gamma": Float("gamma", (0.9, 1.0), default=0.95),
            "gae_lambda": Float("gae_lambda", (0.5, 1.0), default=0.95),
            "clip_range": Float("clip_range", (0.0, 1.0), default=0.2),
            "ent_coef": Float(
                "ent_coef", (0.0, 1.0), default=0.02
            ),  # differs from stable baselines default, more exploration
            "vf_coef": Float("vf_coef", (0.0, 1.0), default=0.5),
            "max_grad_norm": Float("max_grad_norm", (0.0, 1.0), default=0.5),
            # "use_sde": ..., don't use this for now, have to read paper first. This may be cut due to time constraints
            # There may be more hyperparameters to set, but these are the ones directly specified in stable_baselines3.ppo.PPO
        }
    )
    # Learning rate configuration
    configspace_sb_lr = ConfigurationSpace(
        {
            "start_lr": Float("start_lr", (1.0e-6, 1.0e-2), default=0.0003, log=True),
            "end_lr": Float("end_lr", (1.0e-6, 1.0e-2), default=0.0003, log=True),
            "end_fraction": Float("end_fraction", (0.0, 1.0), default=1.0),
        }
    )
    # Policy configspace
    configspace_sb_policy = ConfigurationSpace({})
    ### layer sizes
    max_layers = 5
    policy_n_layers = Integer("policy_n_layers", (1, max_layers), default=3)
    policy_layersizes = [
        Integer(f"policy_layer{i}_size", (1, 128), default=64)
        for i in range(max_layers)
    ]
    policy_layersize_conds = [
        GreaterThanCondition(policy_layersizes[i], policy_n_layers, i)
        for i in range(1, max_layers)
    ]
    configspace_sb_policy.add(
        (policy_n_layers, *policy_layersizes, *policy_layersize_conds)
    )

    # Hyperparameters needed for proximal curriculum learning with state novelty
    configspace_approach = ConfigurationSpace(
        {
            "use_prox_curr": Constant("use_prox_curr", str(use_prox_curr)),
            "use_state_novelty": Constant("use_state_novelty", str(use_state_novelty)),
        }
    )
    if use_prox_curr:
        beta_proximal = Float("beta_proximal", (0.5, 50.0), default=20.0, log=True)
    else:
        beta_proximal = Constant("beta_proximal", 0.0)
    if use_state_novelty:
        beta_novelty = Float("beta_novelty", (0.5, 50.0), default=20.0, log=True)
    else:
        beta_novelty = Constant("beta_novelty", 0.0)
    configspace_approach.add((beta_proximal, beta_novelty))
    if use_prox_curr and use_state_novelty:
        configspace_approach.add(Float("gamma_tradeoff", (0.0, 1.0), default=0.5))
    elif use_prox_curr:
        configspace_approach.add(Constant("gamma_tradeoff", 1.0))
    elif use_state_novelty:
        configspace_approach.add(Constant("gamma_tradeoff", 0.0))
    else:
        # Prox curr is disabled due to beta_proximal being 0, so this samples starting states uniformly
        configspace_approach.add(Constant("gamma_tradeoff", 1.0))

    novelty_approach = Categorical(
        "novelty_approach", ("rnd",) if use_state_novelty else ("none",)
    )
    configspace_approach.add(novelty_approach)
    if (
        use_state_novelty
    ):  # Conditions alone are not enough because ConfigSpace is quite restrictive
        # Additional state novelty hyperparameters per algorithm
        ## Random Network Distillation
        rnd_loss = Constant("rnd_loss", "mse")
        rnd_loss_cond = EqualsCondition(rnd_loss, novelty_approach, "rnd")
        rnd_activation = Categorical(
            "rnd_activation", ("relu", "leakyrelu"), default="leakyrelu"
        )
        rnd_activation_cond = EqualsCondition(rnd_activation, novelty_approach, "rnd")
        rnd_learning_rate = Float(
            "rnd_learning_rate", (1.0e-6, 1.0), default=1e-3, log=True
        )
        rnd_learning_rate_cond = EqualsCondition(
            rnd_learning_rate, novelty_approach, "rnd"
        )
        rnd_optimizer = Constant("rnd_optimizer", "adam")
        rnd_optimizer_cond = EqualsCondition(rnd_optimizer, novelty_approach, "rnd")
        configspace_approach.add(
            (
                rnd_loss,
                rnd_loss_cond,
                rnd_activation,
                rnd_activation_cond,
                rnd_learning_rate,
                rnd_learning_rate_cond,
                rnd_optimizer,
                rnd_optimizer_cond,
            )
        )
        ### layer sizes
        max_layers = 5
        rnd_n_layers = Integer("rnd_n_layers", (1, max_layers), default=3)
        rnd_layersizes = [
            Integer(f"rnd_layer{i}_size", (1, 128), default=default_size)
            for i, default_size in zip(
                range(max_layers), [128, 64, 32] + [32] * (max_layers - 3)
            )
        ]
        rnd_layersize_conds = [
            GreaterThanCondition(rnd_layersizes[i], rnd_n_layers, i)
            for i in range(1, max_layers)
        ]
        configspace_approach.add((rnd_n_layers, *rnd_layersizes, *rnd_layersize_conds))

    # combine everything together
    cs = ConfigurationSpace({})
    cs.add_configuration_space(prefix="sb_ppo", configuration_space=configspace_sb_ppo)
    cs.add_configuration_space(prefix="sb_lr", configuration_space=configspace_sb_lr)
    cs.add_configuration_space(
        prefix="policy", configuration_space=configspace_sb_policy
    )
    cs.add_configuration_space(
        prefix="approach", configuration_space=configspace_approach
    )
    return cs


def create_sb_policy_kwargs(config: Mapping[str, Any]) -> dict:
    """Create stable baseline policy configuration using given hyperparameters

    Args:
        config: configuration which contains the architecture configuration

    Returns:
        keyword arguments for stable baselines
    """
    return {
        "net_arch": {
            "pi": [
                config[f"policy_layer{i}_size"]
                for i in range(config["policy_n_layers"])
            ],
            "vf": [
                config[f"policy_layer{i}_size"]
                for i in range(config["policy_n_layers"])
            ],
        }
    }
