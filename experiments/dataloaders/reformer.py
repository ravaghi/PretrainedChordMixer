from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
import pandas as pd
import numpy as np
import torch
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DNA_BASE_DICT = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, 'Y': 5, 'R': 6, 'M': 7, 'W': 8, 'K': 9, 'S': 10, 'B': 11,
                 'H': 12, 'D': 13, 'V': 14}
DNA_BASE_DICT_REVERSED = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N', 5: 'Y', 6: 'R', 7: 'M', 8: 'W', 9: 'K', 10: 'S',
                          11: 'B', 12: 'H', 13: 'D', 14: 'V'}


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
        max_len = 16_000
    elif "sus" in dataset_name.lower():
        max_len = 400_000
    elif "danio" in dataset_name.lower():
        max_len = 50_000
    dataframe = pad_sequences(dataframe, max_len)
    dataframe = convert_base_to_index(dataframe)
    dataframe["len"] = dataframe["sequence"].apply(lambda x: len(x))
    return dataframe[["sequence", "label"]]


class DatasetCreator(Dataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, index):
        X, Y = self.df.iloc[index, :]
        Y = torch.tensor(Y)
        X = torch.tensor(X)
        return X, Y

    def __len__(self):
        return len(self.df)


class ReformerDataLoader:
    def __init__(self, data_path, dataset, dataset_name, batch_size):
        self.data_path = data_path
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.batch_size = batch_size

    def create_dataloader(self):
        data_path = os.path.join(self.data_path, self.dataset)

        if "taxonomy" in self.dataset_name.lower():
            dataframe = pd.read_pickle(data_path)
            dataframe = process_taxonomy_classification_dataframe(dataframe, self.dataset_name)
        else:
            raise ValueError(f"Dataset {self.dataset_name} name not recognized")

        dataset = DatasetCreator(
            df=dataframe
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )
