import os
import torch
from Bio import SeqIO
from tqdm import tqdm
from typing import Tuple
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import random
import pandas as pd
import copy
import numpy as np

class HG38Dataset(Dataset):
    _DNA_BASE_DICT = {
        'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4
    }

    def __init__(self, sequences, vep_data, dataset_length, sequence_length, mask_ratio):
        self.sequences = sequences
        self.vep_data = vep_data
        self.dataset_length = dataset_length
        self.sequence_length = sequence_length
        self.mask_ratio = mask_ratio

    def _get_sequence_mask_label(self, sequence):
        def _get_base_index(base):
            return self._DNA_BASE_DICT[base]

        sequence_ids = torch.tensor(list(map(_get_base_index, [*sequence])), dtype=torch.int64)
        label = sequence_ids.clone()

        # Creating masks
        rand = torch.rand(sequence_ids.shape[0])
        mask = rand < self.mask_ratio

        # One hot encoding
        sequence_ids = F.one_hot(sequence_ids, num_classes=len(self._DNA_BASE_DICT))

        # Masking
        selection = torch.flatten(mask.nonzero()).tolist()
        sequence_ids[selection] = 0

        return sequence_ids, mask, label


    def __getitem__(self, index):
        random_id = random.randint(0, len(self.vep_data)-1)
        
        position = self.vep_data.iloc[random_id]["pos"]
        chromosome = self.vep_data.iloc[random_id]["chr"]
        sequence = self.sequences[chromosome]
        
        sequence_half_length = int(self.sequence_length / 2)
        right_position = position + sequence_half_length
        left_position = position - sequence_half_length

        if right_position > len(sequence) or left_position < 0:
            return self.__getitem__(index)
        else:
            sequence = sequence[left_position:right_position]
            sequence, mask, label = self._get_sequence_mask_label(sequence)

            return sequence.float(), mask, label.long()

    def __len__(self):
        return self.dataset_length


class PretrainedChordMixerDataLoader:
    """DataLoader for the pretrained ChordMixer model"""

    _CHROMOSOMES = [
        "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9",
        "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17",
        "chr18", "chr19", "chr20", "chr21", "chr22", "chrX", "chrY"
    ]

    def __init__(self, data_path: str, dataset_filename: str, batch_size: int, mask_ratio: float, sequence_length: int):
        self.data_path = data_path
        self.dataset_filename = dataset_filename
        self.batch_size = batch_size
        self.mask_ratio = mask_ratio
        self.sequence_length = sequence_length


    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Processes the datset and creates dataloaders for the train, and test sets

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: Tuple containing the train, and test dataloaders
        """
        vep_data = pd.read_csv("/cluster/home/mahdih/PDT/data/vep.csv", index_col=0)

        hg38_dict = SeqIO.to_dict(SeqIO.parse("/cluster/home/mahdih/PDT/data/variant_effect_prediction/hg38.fa", "fasta"))
        sequences = {chromosome:hg38_dict[chromosome].seq.upper() for chromosome in tqdm(self._CHROMOSOMES, desc="Loading sequences")}

        train_dataloader = DataLoader(HG38Dataset(sequences, vep_data, 90_000, self.sequence_length, self.mask_ratio), batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=1)
        test_dataloader = DataLoader(HG38Dataset(sequences, vep_data, 10_000, self.sequence_length, self.mask_ratio), batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=1)

        return train_dataloader, test_dataloader, test_dataloader
