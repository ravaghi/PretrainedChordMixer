from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.preprocessing import LabelBinarizer
from typing import Tuple
import numpy as np
import pandas as pd
import random
import torch
import os

from .preprocessor.preprocessor import Preprocessor


class BySequenceLengthSampler(Sampler):
    def __init__(self, lengths, bucket_boundaries, batch_size=2):
        ind_n_len = []
        for i, l in enumerate(lengths):
            ind_n_len.append((i, l))
        self.lengths = lengths
        self.ind_n_len = ind_n_len
        self.bucket_boundaries = bucket_boundaries
        self.batch_size = batch_size

    def __iter__(self):
        data_buckets = dict()
        # where p is the id number and seq_len is the length of this id number. 
        for p, seq_len in self.ind_n_len:
            pid = self.element_to_bucket_id(p, seq_len)
            if pid in data_buckets.keys():
                data_buckets[pid].append(p)
            else:
                data_buckets[pid] = [p]

        for k in data_buckets.keys():
            data_buckets[k] = np.asarray(data_buckets[k])

        iter_list = []
        for k in data_buckets.keys():
            np.random.shuffle(data_buckets[k])
            iter_list += (np.array_split(data_buckets[k]
                                         , int(data_buckets[k].shape[0] / self.batch_size)))
        random.shuffle(iter_list)  # shuffle all the batches so they arent ordered by bucket
        # size
        for i in iter_list:
            yield i.tolist()  # as it was stored in an array

    def __len__(self):
        return int(len(self.lengths) / (self.batch_size + 2))

    def element_to_bucket_id(self, x, seq_length):
        boundaries = list(self.bucket_boundaries)
        buckets_min = [np.iinfo(np.int32).min] + boundaries
        buckets_max = boundaries + [np.iinfo(np.int32).max]
        conditions_c = np.logical_and(
            np.less_equal(buckets_min, seq_length),
            np.less(seq_length, buckets_max))
        bucket_id = np.min(np.where(conditions_c))
        return bucket_id


class GenbankDataset(Dataset):
    def __init__(self, dataframe, mask_ratio):
        self.dataframe = dataframe[['sequence', 'label', 'len']]
        self.mask_ratio = mask_ratio
        self.one_hot_encoder = LabelBinarizer().fit([0, 1, 2, 3])
    
    def __getitem__(self, index):
        """
        Returns: tuple (sample, target)
        """
        dataframe = self.dataframe.iloc[index, :]
        sequence = dataframe['sequence'].to_list()
        lengths = dataframe['len'].to_list()

        sequence = np.concatenate(sequence)
        sequence[sequence > 3] = 0 # Non ACGT bases are set to 0
        label = sequence.copy()

        sequence = self.one_hot_encoder.transform(sequence)

        # 1. get binary-encoded masked indexes and masked positions
        # random_masked_ratio = (1 - self.real_mask_ratio) / 2
        uniform_vec = np.random.rand(len(sequence))
        uniform_vec = uniform_vec <= self.mask_ratio
        masked_vec = uniform_vec.astype(int)

        # 2. get real and random binary-encoded masked indexes
        uniform_vec2 = np.random.rand(len(sequence))
        random_vec = np.zeros(len(sequence))
        same_vec = np.zeros(len(sequence))
        random_vec[(masked_vec == 1) & (uniform_vec2 <= 0.1)] = 1
        same_vec[(masked_vec == 1) & (uniform_vec2 >= 0.9)] = 1
        real_vec = abs(masked_vec - random_vec - same_vec)
        random_vec = np.array(random_vec).astype(bool)
        real_vec = np.array(real_vec).astype(bool)

        # 3. masking with all zeros.
        sequence[real_vec,:] = [0, 0, 0, 0]
        # 4. masking with random one-hot encode
        sequence[random_vec,:] = np.eye(4)[np.random.choice(4, 1)]

        sequence = torch.tensor(sequence, dtype=torch.float32)
        mask = torch.tensor(masked_vec, dtype=torch.bool)
        label = torch.tensor(label, dtype=torch.int64)

        return sequence.float(), mask, label.long(), lengths

    def __len__(self):
        return len(self.dataframe)


class PretrainedChordMixerDataLoader(Preprocessor):
    """DataLoader for the pretrained ChordMixer model"""

    def __init__(self,
                 batch_size: int,
                 data_path: str,
                 dataset_type: str,
                 dataset_name: str,
                 train_dataset: str,
                 val_dataset: str,
                 test_dataset: str,
                 mask_ratio: float
                 ):
        self.batch_size = batch_size
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.dataset_name = dataset_name
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.mask_ratio = mask_ratio

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Processes the dataset and creates dataloaders for the train, validation, and test sets

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: Tuple containing the train, validation, and test dataloaders
        """
        BUCKET_BOUNDARIES = [2 ** i for i in range(5, 18)]

        train_dataframe = pd.read_parquet(os.path.join(self.data_path, self.train_dataset))
        val_dataframe = pd.read_parquet(os.path.join(self.data_path, self.val_dataset))
        test_dataframe = pd.read_parquet(os.path.join(self.data_path, self.test_dataset))

        train_dataframe = self.process_taxonomy_classification_dataframe(train_dataframe, "ChordMixer")
        val_dataframe = self.process_taxonomy_classification_dataframe(val_dataframe, "ChordMixer")
        test_dataframe = self.process_taxonomy_classification_dataframe(test_dataframe, "ChordMixer")

        train_sampler = BySequenceLengthSampler(
            lengths=list(train_dataframe["len"]),
            bucket_boundaries=BUCKET_BOUNDARIES,
            batch_size=2
        )

        val_sampler = BySequenceLengthSampler(
            lengths=list(val_dataframe["len"]),
            bucket_boundaries=BUCKET_BOUNDARIES,
            batch_size=2
        )

        test_sampler = BySequenceLengthSampler(
            lengths=list(test_dataframe["len"]),
            bucket_boundaries=BUCKET_BOUNDARIES,
            batch_size=2
        )

        train_dataloader = DataLoader(
            GenbankDataset(dataframe=train_dataframe, mask_ratio=self.mask_ratio),
            batch_size=self.batch_size,
            shuffle=False,
            sampler=train_sampler,
            drop_last=False,
            num_workers=1
        )

        val_dataloader = DataLoader(
            GenbankDataset(dataframe=val_dataframe, mask_ratio=self.mask_ratio),
            batch_size=self.batch_size,
            shuffle=False,
            sampler=val_sampler,
            drop_last=False,
            num_workers=1
        )

        test_dataloader = DataLoader(
            GenbankDataset(dataframe=val_dataframe, mask_ratio=self.mask_ratio),
            batch_size=self.batch_size,
            shuffle=False,
            sampler=test_sampler,
            drop_last=False,
            num_workers=1
        )

        return train_dataloader, val_dataloader, test_dataloader
