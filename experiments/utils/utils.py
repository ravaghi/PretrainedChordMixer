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

DNA_BASE_DICT = {
    'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, 'Y': 5, 'R': 6, 'M': 7,
    'W': 8, 'K': 9, 'S': 10, 'B': 11, 'H': 12, 'D': 13, 'V': 14
}
DNA_BASE_DICT_REVERSED = {
    0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N', 5: 'Y', 6: 'R', 7: 'M',
    8: 'W', 9: 'K', 10: 'S', 11: 'B', 12: 'H', 13: 'D', 14: 'V'
}


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def init_run(config):
    seed_everything(config.general.seed)

    wandb.init(project=config.wandb.project,
               entity=config.wandb.entity,
               config=OmegaConf.to_container(config, resolve=True),
               name=config.wandb.name,
               dir=BASE_DIR)

    print("-" * 30 + " config " + "-" * 30)
    print(OmegaConf.to_yaml(config))
    print("-" * 30 + " config " + "-" * 30)

    device = f'cuda:{config.general.device_id}' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    return device


def get_class_weights(dataset_name, path, train_data):
    train_data_path = os.path.join(path, train_data)
    train_data = pd.read_csv(train_data_path)
    return compute_class_weight('balanced', classes=[0, 1], y=train_data["label"])


def init_weights(model):
    if type(model) == nn.Linear:
        torch.nn.init.xavier_uniform_(model.weight)
        model.bias.data.fill_(0.01)
