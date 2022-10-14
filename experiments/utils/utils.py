from sklearn.utils.class_weight import compute_class_weight
from omegaconf import OmegaConf
from torch import nn
import pandas as pd
import numpy as np
import random
import torch
import wandb
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def init_run(config):
    seed_everything(config.general.seed)

    print("-" * 30 + " config " + "-" * 30)
    print(OmegaConf.to_yaml(config))
    print("-" * 30 + " config " + "-" * 30)

    device = f'cuda:{config.general.device_id}' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    wandb.init(project=config.wandb.project,
               entity=config.wandb.entity,
               config=OmegaConf.to_container(config, resolve=True),
               name=config.wandb.name.split(".")[-1],
               dir=BASE_DIR)

    return device


def get_class_weights(dataset_name, path, train_data):
    train_data_path = os.path.join(path, train_data)
    
    if "taxonomy" in dataset_name.lower():
        train_data = pd.read_pickle(train_data_path)
    else:
        train_data = pd.read_csv(train_data_path)
        
    return compute_class_weight('balanced', classes=[0, 1], y=train_data["label"])


def init_weights(model):
    if type(model) == nn.Linear:
        torch.nn.init.xavier_uniform_(model.weight)
        model.bias.data.fill_(0.01)
