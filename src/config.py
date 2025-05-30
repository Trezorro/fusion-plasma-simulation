"""Hyper parameter configuration helper functions"""
from pathlib import Path
from typing import Literal
from omegaconf import OmegaConf, ValidationError
import omegaconf
import wandb
import wandb.apis
from pprint import pformat
import logging

logger = logging.getLogger(__name__)

# TODO: define generalized includes, instead of just looking for model with string. Include data too.
# TODO: Need to decide how to specify data and model in the config structure.

PROJECT = "flowtoy"
ENTITY = "tresoor"
MAIN_CONFIG_FILE = "fm_toy"


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


def load_config_from_file(name=MAIN_CONFIG_FILE, as_omega=False) -> dict | omegaconf.DictConfig:
    """Load configuration from (hierarchical) yaml files and CLI."""
    main_conf = OmegaConf.load(f'configs/{name}.yaml')
    if type(main_conf.model) == str:
        model_conf = OmegaConf.load(main_conf.model)
        main_conf = OmegaConf.merge(main_conf, model_conf)
    cli_conf = OmegaConf.from_cli()
    conf: omegaconf.DictConfig = OmegaConf.merge(main_conf, cli_conf)
    update_model_input_channels(conf)
    if as_omega:
        return conf
    conf = OmegaConf.to_object(conf)
    logger.info("Loaded configuration from %s", f'configs/{name}.yaml')
    if not type(conf) == dict:
        raise ValidationError("Configuration was not in dict style. Got: " + repr(conf))
    return dict(conf)

def is_reeval_run() -> str | Literal[False]:
    """Check the program input arguments just to see whether we need to do training or just testing."""
    cli_conf = OmegaConf.from_cli()
    if cli_conf.get("reeval", False):
        return cli_conf.get("run_name")
    else:
        return False

def pretty_config(conf):
    if type(conf) == omegaconf.DictConfig:
        conf = OmegaConf.to_object(conf)
    return pformat(conf, compact=True, sort_dicts=False, width=160)


def update_model_input_channels(conf):
    """Force the model configuration to have the correct input channels, based on the data configuration.

    This is a hack to avoid having to match the input channels in the config file manually.
    """
    if 'data' in conf and 'cols' in conf.data:
        if 'x' in conf.data.cols:
            conf.model.params.model_params.input_channels = len(conf.data.cols.x)
        if 'c' in conf.data.cols:
            conf.model.params.model_params.c_channels = len(conf.data.cols.c)


# wandb.config.update(conf)
def get_current_config(wandb_only=False):
    if not wandb.config:
        raise RuntimeError("wandb.config was not initialized yet.")
    try:
        config_dict = dict(wandb.config)
    except wandb.Error as e:
        if wandb_only:
            raise RuntimeError("wandb.config is not available yet. Did you call wandb.init()?")
        config_dict = load_config_from_file()
        logger.info("No wandb config found. Loaded it from current yaml file.")
    conf = OmegaConf.create(config_dict)
    convert_lists(conf)
    return conf


def find_wandb_run(find_run: str, project=PROJECT, entity=ENTITY) -> wandb.apis.public.Run | None:
    """
    Find a run by ID or name and load it.
    Args:
        find_run (str): The ID or name of the run to find.
        project (str): The name of the project.
        entity (str): The entity name.
    """
    api = wandb.Api(overrides={"entity": entity, "project": project})
    try:
        run = api.run(find_run)
        logger.debug("Found run by ID=%s", find_run)
    except wandb.errors.CommError as e:
        logger.debug("Could not find run by ID (%s). Will try by name.", find_run)
        runs = api.runs(filters={"display_name": find_run},)
        if len(runs) == 1:
            run = runs[0]
            logger.debug("Found run by name (%s)", find_run)
        elif len(runs) > 1:
            logger.warning("Found multiple runs with the same name (%s)", find_run)
            for r in runs:
                logger.warning("Run ID: %s  Date: %s", r.id, r.created_at)
            return
        else:
            logger.error("No runs found with name %s", find_run)
            return
    if run is not None:
        RUN_ID = run.id
        RUN_NAME = run.name
        print(f" ✅ Found Run ID: {RUN_ID}, Run Name: {RUN_NAME}\n Created at: {run.created_at}")
        print(f"   Run URL: {run.url}")
    return run


def find_and_download_model(run):
    artifacts = [a for a in run.logged_artifacts() if a.type == "model"]
    for artifact in artifacts:
        print(
            f"{artifact.name}\n  > Type: {artifact.type}, Version: {artifact.version}, aliases: {artifact.aliases}, size: {artifact.size:_}, updated: {artifact.updated_at}, description: {artifact.description}"
        )

    if len(artifacts) == 0:
        print("No model artifacts found")
        raise ValueError("No model artifacts found")
    elif len(artifacts) == 1:
        artifact = artifacts[0]
        print(f"Single model artifact found, so using it.")
    else:
        artifact_name = input(f"Enter the name of the artifact to use (default: {artifacts[0].name}): ")
        if artifact_name == "":
            artifact_name = artifacts[0].name
        artifact = wandb.Api().artifact(
            artifact_name
        )  # TODO fix CommError: project 'uncategorized' not found under entity 'tresoor'

        print(f"Using artifact {artifact.name}")
    print("Downloading artifact...")
    artifact_dir = artifact.download()
    print("Stored model locally in", artifact_dir)
    # Log model summary
    # logger.info("Model loaded. Summary:")
    # model.log_summary(C)
    # Get the number of steps the model has trained

    # load checkpoint
    return Path(artifact_dir) / "model.ckpt"
