from sklearn.utils.class_weight import compute_class_weight
from omegaconf import OmegaConf, DictConfig
from torch import nn
import pandas as pd
import numpy as np
import random
import torch
import wandb
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
    # wandb.init(project=config.wandb.project,
    #            entity=config.wandb.entity,
    #            config=OmegaConf.to_container(config, resolve=True),
    #            name=config.wandb.name,
    #            dir=BASE_DIR)

    # Print config file content
    print("-" * 30 + " config " + "-" * 30)
    print(OmegaConf.to_yaml(config))
    print("-" * 30 + " config " + "-" * 30)

    # Select cuda device
    device = f'cuda:{config.general.device_id}' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    return device


def get_class_weights(path: str, train_data: str) -> np.ndarray:
    """
    Compute class weights for the dataset.

    Args:
        path (str): Path to the dataset.
        train_data (str): Name of the training dataset.

    Returns:
        class_weights (np.ndarray): Array of class weights.
    """
    train_data_path = os.path.join(path, train_data)
    train_data = pd.read_csv(train_data_path)
    return compute_class_weight('balanced', classes=[0, 1], y=train_data["label"])


def init_weights(model: nn.Module) -> None:
    """
    Initialize the weights of a model.

    Args:
        model (nn.Module): Model to initialize.

    Returns:
        None
    """
    if type(model) == nn.Linear:
        torch.nn.init.xavier_uniform_(model.weight)
        model.bias.data.fill_(0.01)
