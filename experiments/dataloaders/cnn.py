from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import torch
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))




def pad_sequences(dataframe, max_len=1000):
    max_seq_len = dataframe["sequence"].apply(lambda x: len(x)).max()
    dataframe["sequence"] = dataframe["sequence"].str.pad(max_seq_len, side="right", fillchar="A")
    dataframe["sequence"] = dataframe["sequence"].apply(lambda x: x[:max_len].upper())
    return dataframe


def convert_base_to_index(dataframe):
    dataframe["new_sequence"] = dataframe["sequence"].apply(lambda x: [DNA_BASE_DICT[base] for base in x])
    dataframe = dataframe.drop(columns=["sequence"])
    dataframe = dataframe.rename(columns={"new_sequence": "sequence"})
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    return dataframe


def process_taxonomy_classification_dataframe(dataframe, dataset_name):
    dataframe["new_sequence"] = dataframe["sequence"].apply(
        lambda x: "".join([DNA_BASE_DICT_REVERSED[base] for base in x]))
    dataframe = dataframe.drop(columns=["sequence", "len", "bin"])
    dataframe = dataframe.rename(columns={"new_sequence": "sequence"})
    if "carassius" in dataset_name.lower():
        max_len = 25_000
    elif "sus" in dataset_name.lower():
        max_len = 400_000
    elif "danio" in dataset_name.lower():
        max_len = 50_000
    dataframe = pad_sequences(dataframe, max_len)
    dataframe = convert_base_to_index(dataframe)
    dataframe["len"] = dataframe["sequence"].apply(lambda x: len(x))
    print(dataframe)
    return dataframe[["sequence", "label"]]


class CNNDataLoader:
    def __init__(self, data_path, dataset, dataset_name, batch_size):
        self.data_path = data_path
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.batch_size = batch_size

    def create_dataloader(self):
        data_path = os.path.join(self.data_path, self.dataset)

        dataframe = pd.read_csv(data_path)[:2000]
        #dataframe = process_taxonomy_classification_dataframe(dataframe, self.dataset_name)

        print(dataframe)
        exit()

        sequences = np.array(dataframe.sequence.values.tolist(), dtype=np.float32)
        labels = dataframe.label.values.tolist()

        sequences = torch.tensor(sequences)
        labels = torch.tensor(labels)

        dataset = TensorDataset(sequences, labels)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )
