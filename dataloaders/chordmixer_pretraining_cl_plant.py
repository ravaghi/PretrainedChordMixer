from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelBinarizer
from typing import Tuple
import pandas as pd
import numpy as np
import string
import torch
import glob

class PlantDataset(Dataset):
    """Dataset for the pretrained ChordMixer model"""
    _DNA_BASE_DICT = {
        'A': 0, 'C': 1, 'G': 2, 'T': 3
    }

    def __init__(self, combined_df, dataset_size, mask_ratio):
        self.dataframe = combined_df.sample(n=dataset_size)
        self.dataset_size = dataset_size
        self.mask_ratio = mask_ratio
        self.one_hot_encoder = LabelBinarizer().fit([0, 1, 2, 3])
    
    def _get_sequence_mask_label(self, sequence):
        def _get_base_index(base):
            return self._DNA_BASE_DICT[base]

        sequence_ids = list(map(_get_base_index, [*sequence]))
        label = sequence_ids

        sequence_ids = self.one_hot_encoder.transform(sequence_ids)

        # 1. get binary-encoded masked indexes and masked positions
        # random_masked_ratio = (1 - self.real_mask_ratio) / 2
        uniform_vec = np.random.rand(len(sequence_ids))
        uniform_vec = uniform_vec <= self.mask_ratio
        masked_vec = uniform_vec.astype(int)

        # 2. get real and random binary-encoded masked indexes
        uniform_vec2 = np.random.rand(len(sequence_ids))
        random_vec = np.zeros(len(sequence_ids))
        same_vec = np.zeros(len(sequence_ids))
        random_vec[(masked_vec == 1) & (uniform_vec2 <= 0.1)] = 1
        same_vec[(masked_vec == 1) & (uniform_vec2 >= 0.9)] = 1
        real_vec = abs(masked_vec - random_vec - same_vec)
        random_vec = np.array(random_vec).astype(bool)
        real_vec = np.array(real_vec).astype(bool)

        # 3. masking with all zeros.
        sequence_ids[real_vec,:] = [0, 0, 0, 0]
        # 4. masking with random one-hot encode
        sequence_ids[random_vec,:] = np.eye(4)[np.random.choice(4, 1)]

        return sequence_ids, masked_vec, label

    def __getitem__(self, index):
        sequence = self.dataframe.iloc[index]['sequence']

        sequence, mask, label = self._get_sequence_mask_label(sequence)

        sequence = torch.tensor(sequence, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.bool)
        label = torch.tensor(label, dtype=torch.int64)

        return sequence.float(), mask, label.long(), []

    def __len__(self):
        return self.dataset_size


class PretrainedChordMixerDataLoader:
    """DataLoader for the pretrained ChordMixer model"""

    def __init__(self,
                 batch_size: int,
                 data_path: str,
                 dataset_type: str,
                 dataset_name: str,
                 train_dataset: str,
                 val_dataset: str,
                 test_dataset: str,
                 mask_ratio: float,
                 sequence_length: int
                 ):
        self.batch_size = batch_size
        self.data_path = data_path
        self.mask_ratio = mask_ratio

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Processes the dataset and creates dataloaders for the train, validation, and test sets

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: Tuple containing the train, validation, and test dataloaders
        """
        parquet_files = glob.glob(self.data_path + "/**/*.parquet", recursive=True)
        combined_df = pd.concat([pd.read_parquet(f)["sequence"] for f in parquet_files])

        combined_df = combined_df[~combined_df['sequence'].str.contains("|".join(list(set(list(string.ascii_uppercase)) - set(["A", "C", "G", "T"]))))]

        train_dataloader = DataLoader(
            PlantDataset(combined_df, 240_000, self.mask_ratio),
            batch_size=self.batch_size,
            shuffle=True
        )

        val_dataloader = DataLoader(
            PlantDataset(combined_df, 30_000, self.mask_ratio),
            batch_size=self.batch_size,
            shuffle=True
        )

        test_dataloader = DataLoader(
            PlantDataset(combined_df, 30_000, self.mask_ratio),
            batch_size=self.batch_size,
            shuffle=True
        )

        return train_dataloader, val_dataloader, test_dataloader
