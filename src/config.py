"""Hyper parameter configuration helper functions"""
from networkx import omega
from omegaconf import OmegaConf, ValidationError, DictConfig
import omegaconf
import wandb

# TODO: define generalized includes, instead of just looking for model with string. Include data too.
# TODO: Need to decide how to specify data and model in the config structure.


def print_types(value, level=0):
    """Helper function to debug config dicts."""
    value_type = type(value)
    if value_type == dict:
        print(value_type, end='')
        for k, v in value.items():
            print("\n" + "-" * level + repr(k), type(k), end=': ')
            print_types(v, level=level + 1)
    else:
        print(repr(value), value_type, end='')


def convert_lists(value, level=0):
    """Helper function to debug config dicts."""
    value_type = type(value)
    if value_type == omegaconf.DictConfig:
        # print(value_type, end='')
        for k, v in value.items():
            # print("\n"+"-"*level + repr(k), type(k), end=': ')
            if type(v) == omegaconf.ListConfig:
                value[k] = list(v)
            convert_lists(v, level=level + 1)


def load_config_from_file() -> dict:
    """Load configuration from (hierarchical) yaml files and CLI."""
    main_conf = OmegaConf.load('configs/main.yaml')
    if type(main_conf.model) == str:
        model_conf = OmegaConf.load(main_conf.model)
        main_conf = OmegaConf.merge(main_conf, model_conf)
    cli_conf = OmegaConf.from_cli()
    conf = OmegaConf.merge(main_conf, cli_conf)
    conf = OmegaConf.to_object(conf)
    if not type(conf) == dict:
        raise ValidationError("Configuration was not in dict style. Got: " + repr(conf))
    return dict(conf)


# wandb.config.update(conf)
def get_current_config():
    if not wandb.config:
        raise RuntimeError("wandb.config was not initialized yet.")
    conf = OmegaConf.create(dict(wandb.config))
    convert_lists(conf)
    return conf
