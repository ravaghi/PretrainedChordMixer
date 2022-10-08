import os
import hydra
import torch
import wandb
from torch import nn, optim

from models.chordmixer import ChordMixer
from chordmixer.dataloader import create_dataloader
from chordmixer.utils import init_weights
from utils import init_run, get_max_seq_len, get_class_weights, train, evaluate

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra.main(config_path="configs", config_name="chordmixer", version_base=None)
def main(config):
    init_run(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_seq_len = get_max_seq_len(config.paths.data, config.files.train_data, config.files.test_data)

    model = ChordMixer(
        problem='genbank',
        vocab_size=config.hyperparams.vocab_size,
        max_seq_len=max_seq_len,
        embedding_size=config.hyperparams.embedding_size,
        track_size=config.hyperparams.track_size,
        hidden_size=config.hyperparams.hidden_size,
        mlp_dropout=config.hyperparams.mlp_dropout,
        layer_dropout=config.hyperparams.layer_dropout,
        n_class=config.hyperparams.n_classes
    )

    model = model.to(device)
    model.apply(init_weights)
    wandb.watch(model)

    class_weights = get_class_weights(config.paths.data, config.files.train_data)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device), reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=config.hyperparams.learning_rate)

    train_dataloader = create_dataloader(
        data_path=config.paths.data,
        data_file=config.files.train_data,
        batch_size=config.hyperparams.batch_size,
    )
    test_dataloader = create_dataloader(
        data_path=config.paths.data,
        data_file=config.files.test_data,
        batch_size=config.hyperparams.batch_size,
    )

    for epoch in range(config.hyperparams.epoch):
        train(model=model, train_dataloader=train_dataloader, device=device, current_epoch_nr=epoch,
              criterion=criterion, optimizer=optimizer)
        evaluate(model=model, test_dataloader=test_dataloader, device=device, current_epoch_nr=epoch,
                 criterion=criterion)


if __name__ == '__main__':
    main()
