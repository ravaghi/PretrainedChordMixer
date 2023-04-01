from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from typing import Tuple
from tqdm import tqdm
from Bio import SeqIO
import random
import torch
import os


class HG38Dataset(Dataset):
    """Dataset for the pretrained ChordMixer model"""
    _DNA_BASE_DICT = {
        'A': 0, 'C': 1, 'G': 2, 'T': 3
    }

    def __init__(self, sequences, dataset_size, sequence_length, mask_ratio):
        self.sequences = sequences
        self.dataset_size = dataset_size
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
        random_id = random.randint(0, len(self.sequences) - 1)
        chromosome = list(self.sequences.keys())[random_id]
        sequence = self.sequences[chromosome]

        chromosome_length = len(sequence)
        left_position = random.randint(0, chromosome_length - self.sequence_length)
        right_position = left_position + self.sequence_length

        sequence = sequence[left_position:right_position]
        if "N" in sequence:
            return self.__getitem__(index)

        sequence, mask, label = self._get_sequence_mask_label(sequence)

        return sequence.float(), mask, label.long(), []

    def __len__(self):
        return self.dataset_size


class PretrainedChordMixerDataLoader:
    """DataLoader for the pretrained ChordMixer model"""

    _CHROMOSOMES = [
        "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9",
        "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17",
        "chr18", "chr19", "chr20", "chr21", "chr22", "chrX", "chrY"
    ]

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
        self.dataset_type = dataset_type
        self.dataset_name = dataset_name
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.mask_ratio = mask_ratio
        self.sequence_length = sequence_length

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Processes the dataset and creates dataloaders for the train, validation, and test sets

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: Tuple containing the train, validation, and test dataloaders
        """
        hg38_dict = SeqIO.to_dict(SeqIO.parse(os.path.join(self.data_path, self.dataset_name), "fasta"))
        sequences = {chromosome: hg38_dict[chromosome].seq.upper() for chromosome in
                     tqdm(self._CHROMOSOMES, desc="Loading sequences")}

        train_dataloader = DataLoader(
            HG38Dataset(sequences, 120_000, self.sequence_length, self.mask_ratio),
            batch_size=self.batch_size,
            shuffle=True
        )

        val_dataloader = DataLoader(
            HG38Dataset(sequences, 15_000, self.sequence_length, self.mask_ratio),
            batch_size=self.batch_size,
            shuffle=True
        )

        test_dataloader = DataLoader(
            HG38Dataset(sequences, 15_000, self.sequence_length, self.mask_ratio),
            batch_size=self.batch_size,
            shuffle=True
        )

        return train_dataloader, val_dataloader, test_dataloader
