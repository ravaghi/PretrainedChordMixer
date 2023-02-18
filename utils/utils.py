from omegaconf import OmegaConf, DictConfig
import numpy as np
import random
import torch
import wandb
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def seed_everything(seed: int = 42) -> None:
    """
    Seeds basic parameters for reproductibility of results.

    Args:
        seed (int): Random seed.

    Returns:
        None
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def init_wandb(config: DictConfig) -> None:
    """
    Initialize a new run on Weights & Biases.

    Args:
        config (DictConfig): Configuration file.

    Returns:
        None
    """
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        config=OmegaConf.to_container(config, resolve=True),
        name=config.wandb.name,
        dir=os.path.join(BASE_DIR, 'logs')
    )


def init_run(config: DictConfig) -> str:
    """
    Initialize a new run on Weights & Biases. Set random seeds and select cuda device.

    Args:
        config (DictConfig): Configuration file.

    Returns:
        device (str): Device to use for training.
    """
    seed_everything(config.general.seed)

    # Initiate wandb run
    if config.general.log_to_wandb:
        print(f"Logging to Weights & Biases: {config.wandb.project}/{config.wandb.name}")
        init_wandb(config=config)
    else:
        print(f"Skipping Weights & Biases logging. Set log_to_wandb to True in config to enable it.")

    # Print config file content
    print("-" * 30 + " config " + "-" * 30)
    print(OmegaConf.to_yaml(config))
    print("-" * 30 + " config " + "-" * 30)

    # Select cuda device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        # Use a specific GPU if id is specified
        device = f'{device}:{config.general.device_id}'
    except AttributeError:
        pass

    print(f'Using device: {device}')

    return device


def print_model_params(model: torch.nn.Module) -> None:
    """
    Print number of model parameters.

    Args:
        model (torch.nn.Module): Model to print parameters from.

    Returns:
        None
    """
    parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {parameters:,}")
    print(f"Number of trainable parameters: {trainable_parameters:,}")
