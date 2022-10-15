from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
import pandas as pd
import numpy as np
import torch
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DNA_BASE_DICT = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, 'Y': 5, 'R': 6, 'M': 7, 'W': 8, 'K': 9, 'S': 10, 'B': 11, 'H': 12, 'D': 13, 'V': 14}

def complete_batch(df, batch_size):
    """
    Function to make number of instances divisible by batch_size
    within each log2-bin
    """
    complete_bins = []
    bins = [bin_df for _, bin_df in df.groupby('bin')]

    for gr_id, bin in enumerate(bins):
        l = len(bin)
        remainder = l % batch_size
        integer = l // batch_size

        if remainder != 0:
            # take the first example and copy (batch_size - remainder) times
            bin = pd.concat([bin, pd.concat([bin.iloc[:1]] * (batch_size - remainder))], ignore_index=True)
            integer += 1
        batch_ids = []
        # create indices
        for i in range(integer):
            batch_ids.extend([f'{i}_bin{gr_id}'] * batch_size)
        bin['batch_id'] = batch_ids
        complete_bins.append(bin)
    return pd.concat(complete_bins, ignore_index=True)


def shuffle_batches(df):
    """
    Shuffles batches so during training
    ChordMixer sees sequences from different log2-bins
    """
    import random

    batch_bins = [df_new for _, df_new in df.groupby('batch_id')]
    random.shuffle(batch_bins)

    return pd.concat(batch_bins).reset_index(drop=True)


def concater_collate(batch):
    """
    Packs a batch into a long sequence
    """
    (xx, yy, lengths, bins) = zip(*batch)
    xx = torch.cat(xx, 0)
    yy = torch.tensor(yy)
    return xx, yy, list(lengths), list(bins)


def process_enhancer_prediction_dataframe(dataframe):
    dataframe["new_seq"] = dataframe["sequence"].apply(lambda x: np.array([DNA_BASE_DICT[base.upper()] for base in x]))
    dataframe = dataframe.drop(columns=["sequence"])
    dataframe = dataframe.rename(columns={"new_seq": "sequence"})
    
    dataframe["len"] = dataframe["sequence"].apply(lambda x: len(x))
    
    percentiles = [i * 0.1 for i in range(10)] + [.95, .99, .995]
    bins = np.quantile(dataframe['len'], percentiles)
    bin_labels = [i for i in range(len(bins) - 1)]
    dataframe['bin'] = pd.cut(dataframe['len'], bins=bins, labels=bin_labels)
    dataframe = dataframe[['sequence', 'label', 'len', 'bin']]
    
    return dataframe


def process_variant_effect_prediction_dataframe(dataframe):
    sequence_path = os.path.join(BASE_DIR, "data", "variant_effect_prediction", "hg38.fa")
    sequences = SeqIO.to_dict(SeqIO.parse(sequence_path, "fasta"))
    
    dataframe["sequence"] = dataframe.apply(lambda x: str(sequences[x.chr].seq[x.pos-20_000:x.pos+20_000]), axis=1)
    
    dataframe["new_seq"] = dataframe["sequence"].apply(lambda x: np.array([DNA_BASE_DICT[base.upper()] for base in x]))
    dataframe = dataframe.drop(columns=["sequence"])
    dataframe = dataframe.rename(columns={"new_seq": "sequence"})
    
    dataframe["len"] = dataframe["sequence"].apply(lambda x: len(x))
    dataframe = dataframe[dataframe["len"] > 1000]
    
    # percentiles = [i * 0.1 for i in range(10)] + [.95, .99, .995]
    # bins = np.quantile(dataframe['len'], percentiles)
    # bin_labels = [i for i in range(len(bins) - 1)]
    # dataframe['bin'] = pd.cut(dataframe['len'], bins=bins, labels=bin_labels)
    dataframe['bin'] = 1
    dataframe = dataframe[['sequence', 'label', 'len', 'bin']]
    
    return dataframe


class DatasetCreator(Dataset):
    def __init__(self, df, batch_size, var_len=False):
        if var_len:
            # fill in gaps to form full batches
            df = complete_batch(df=df, batch_size=batch_size)
            # shuffle batches
            self.df = shuffle_batches(df=df)[['sequence', 'label', 'len', 'bin']]
        else:
            self.df = df

    def __getitem__(self, index):
        """
        Returns: tuple (sample, target)
        """
        X, Y, length, bin = self.df.iloc[index, :]
        Y = torch.tensor(Y)
        X = torch.from_numpy(X)
        return X, Y, length, bin

    def __len__(self):
        return len(self.df)


class ChordMixerDataLoader:
    def __init__(self, data_path, dataset, dataset_name, batch_size):
        self.data_path = data_path
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.batch_size = batch_size

    def create_dataloader(self):
        data_path = os.path.join(self.data_path, self.dataset)
        
        if "taxonomy" in self.dataset_name.lower():
            dataframe = pd.read_pickle(data_path)
        elif "enhancer" in self.dataset_name.lower():
            dataframe = pd.read_csv(data_path)
            dataframe = process_enhancer_prediction_dataframe(dataframe)
        elif "variant" in self.dataset_name.lower():
            dataframe = pd.read_csv(data_path)
            dataframe = process_variant_effect_prediction_dataframe(dataframe)
        else:
            raise ValueError(f"Dataset {self.dataset_name} name not recognized")

        dataset = DatasetCreator(
            df=dataframe,
            batch_size=self.batch_size,
            var_len=True
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=concater_collate,
            drop_last=False
        )
