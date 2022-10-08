from dataclasses import dataclass


@dataclass
class WanDB:
    entity: str
    project: str


@dataclass
class Paths:
    data: str


@dataclass
class Files:
    train_data: str
    test_data: str


@dataclass
class HyperParams:
    epoch: int
    batch_size: int
    vocab_size: int
    track_size: int
    embedding_size: int
    hidden_size: int
    mlp_dropout: int
    layer_dropout: int
    learning_rate: int
    n_classes: int


@dataclass
class ChordMixerConfig:
    wandb: WanDB
    paths: Paths
    files: Files
    hyperparams: HyperParams
