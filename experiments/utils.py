from sklearn.utils.class_weight import compute_class_weight
from omegaconf import OmegaConf
from tqdm import tqdm
import pandas as pd
import torch
import wandb
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def init_run(config):
    print("-" * 20 + " config " + "-" * 20)
    print(OmegaConf.to_yaml(config))
    print("-" * 20 + " config " + "-" * 20)

    wandb.init(project=config.wandb.project,
               entity=config.wandb.entity,
               config=OmegaConf.to_container(config, resolve=True),
               name="ChordMixer")


def get_max_seq_len(data_path, train_data_file, test_data_file):
    train_data_path = os.path.join(BASE_DIR, data_path, train_data_file)
    test_data_path = os.path.join(BASE_DIR, data_path, test_data_file)
    train_data = pd.read_pickle(train_data_path)
    test_data = pd.read_pickle(test_data_path)

    return max(max(train_data['len']), max(test_data['len']))


def get_class_weights(data_path, train_data_file):
    train_data_path = os.path.join(BASE_DIR, data_path, train_data_file)
    train_data = pd.read_pickle(train_data_path)
    return compute_class_weight('balanced', classes=[0, 1], y=train_data['label'])


def train(model, train_dataloader, device, criterion, optimizer, current_epoch_nr):
    model.train()

    running_accuracy = 0.0
    running_loss = 0.0
    num_batches = len(train_dataloader)

    loop = tqdm(enumerate(train_dataloader), total=num_batches)
    for idx, (x, y, seq_len, _bin) in loop:
        x = x.to(device)
        y = y.to(device)

        y_hat = model(x, seq_len)
        loss = criterion(y_hat, y)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        _, predicted = y_hat.max(1)
        running_accuracy += (predicted == y).sum().item()
        running_loss += loss.item()

        loop.set_description(f'Epoch {current_epoch_nr + 1}')
        loop.set_postfix(train_acc=round(running_accuracy / (idx + 1), 2),
                         train_loss=round(running_loss / (idx + 1), 2))

    train_accuracy = running_accuracy / num_batches
    train_loss = running_loss / num_batches
    wandb.log({'train_loss': train_loss})
    wandb.log({'train_accuracy': train_accuracy})


def evaluate(model, test_dataloader, device, criterion, current_epoch_nr):
    model.eval()

    running_accuracy = 0.0
    running_loss = 0.0
    num_batches = len(test_dataloader)

    with torch.no_grad():
        loop = tqdm(enumerate(test_dataloader), total=num_batches, position=0, leave=True, ascii=False)
        for idx, (x, y, seq_len, _bin) in loop:
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x, seq_len)
            loss = criterion(y_hat, y)

            _, predicted = y_hat.max(1)
            running_accuracy += (predicted == y).sum().item()
            running_loss += loss.item()

            loop.set_description(f'Epoch {current_epoch_nr + 1}')
            loop.set_postfix(val_acc=round(running_accuracy / (idx + 1), 2),
                             val_loss=round(running_loss / (idx + 1), 2))

    validation_accuracy = running_accuracy / num_batches
    validation_loss = running_loss / num_batches
    wandb.log({'validation_loss': validation_loss})
    wandb.log({'validation_accuracy': validation_accuracy})
