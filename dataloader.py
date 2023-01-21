import os
import torch
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader
import random


class SequenceProcessor:
    _DNA_BASE_DICT = {
        'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, 'Y': 5, 'R': 6, 'M': 7,
        'W': 8, 'K': 9, 'S': 10, 'B': 11, 'H': 12, 'D': 13, 'V': 14
    }

    def __init__(self, mask_ratio, sequence_length):
        self.mask_ratio = mask_ratio
        self.sequence_length = sequence_length

    def tokenize(self, sequence: str) -> torch.Tensor:
        """
        Tokenizes a DNA sequence into a tensor of integers

        Args:
            sequence (str): DNA sequence to be tokenized

        Returns:
            torch.Tensor: Tensor of integers representing the DNA sequence
        """
        return torch.tensor([self._DNA_BASE_DICT[base] for base in sequence], dtype=torch.int32)

    def split(self, sequence_ids: torch.Tensor) -> torch.Tensor:
        """
        Splits a sequence into multiple sequences of length sequence_length

        Args:
            sequence_ids (torch.Tensor): Tensor of integers representing a DNA sequence

        Returns:
            torch.Tensor: Tensor of integers representing the split DNA sequences
        """
        # Shortening the array to be divisible by sequence length
        remainder = len(sequence_ids) % self.sequence_length
        if remainder > 0:
            sequence_ids = sequence_ids[:-remainder]
        return torch.stack(torch.split(sequence_ids, self.sequence_length))

    def mask(self, sequence_ids: torch.Tensor) -> dict:
        """
        Masks a sequence with a mask ratio

        Args:
            sequence_ids (torch.Tensor): Tensor of integers representing a DNA sequence

        Returns:
            dict: Dictionary containing the masked sequence and the mask
        """
        labels = sequence_ids.clone().detach()

        rand = torch.rand(sequence_ids.shape)
        masks = rand < self.mask_ratio
        masks = masks.to(torch.int32)

        for i in range(sequence_ids.shape[0]):
            selection = torch.flatten(masks[i].nonzero()).tolist()
            sequence_ids[i, selection] = 15

        return {
            "sequence_ids": sequence_ids,
            "masks": masks,
            "labels": labels
        }


class HG38Dataset(Dataset):
    def __init__(self, data):
        self.sequence_ids = data["sequence_ids"].type(torch.float32)
        self.masks = data["masks"]
        self.labels = data["labels"].type(torch.float32)

    def __getitem__(self, index):
        return self.sequence_ids[index], self.masks[index], self.labels[index]

    def __len__(self):
        return len(self.sequence_ids)


class PretrainedChordMixerDataLoader:
    _CHROMOSOMES = [
        "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9",
        "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", 
        "chr18", "chr19", "chr20", "chr21", "chr22", "chrX", "chrY"
    ]

    # _CHROMOSOMES = [
    #     "chr21"
    # ]

    def __init__(self, data_path, dataset_filename, batch_size, mask_ratio, sequence_length):
        self.data_path = data_path
        self.dataset_filename = dataset_filename
        self.batch_size = batch_size
        self.mask_ratio = mask_ratio
        self.sequence_length = sequence_length

    def _load_sequences(self) -> str:
        """
        Loads the sequences from the dataset file

        Returns:
            str: DNA sequences
        """
        data_path = os.path.join(self.data_path, self.dataset_filename)
        sequences_dict = SeqIO.to_dict(SeqIO.parse(data_path, "fasta"))
        sequences = ""
        for chromosome in self._CHROMOSOMES:
            sequences += str(sequences_dict[chromosome].seq).upper()
        return sequences

    def _process_sequences(self, sequences: str) -> dict:
        """
        Processes the sequences by tokenizing, splitting and masking

        Args:
            sequences (str): DNA sequences

        Returns:
            dict: Dictionary containing the masked sequences, masks and labels
        """
        sequence_processor = SequenceProcessor(self.mask_ratio, self.sequence_length)
        tokenized_sequences = sequence_processor.tokenize(sequences)
        splitt_sequences = sequence_processor.split(tokenized_sequences)
        masked_sequences = sequence_processor.mask(splitt_sequences)
        return masked_sequences

    @staticmethod
    def _split_sequences(masked_sequences: dict) -> tuple:
        """
        Splits the sequences into train, validation and test sets

        Args:
            masked_sequences (dict): Dictionary containing the masked sequences, masks and labels

        Returns:
            tuple: Tuple containing the train, validation and test sets
        """
        train_size = int(0.8 * len(masked_sequences["sequence_ids"]))
        val_size = int(0.1 * len(masked_sequences["sequence_ids"]))

        train = {
            "sequence_ids": masked_sequences["sequence_ids"][:train_size],
            "masks": masked_sequences["masks"][:train_size],
            "labels": masked_sequences["labels"][:train_size]
        }

        val = {
            "sequence_ids": masked_sequences["sequence_ids"][train_size:train_size + val_size],
            "masks": masked_sequences["masks"][train_size:train_size + val_size],
            "labels": masked_sequences["labels"][train_size:train_size + val_size]
        }

        test = {
            "sequence_ids": masked_sequences["sequence_ids"][train_size + val_size:],
            "masks": masked_sequences["masks"][train_size + val_size:],
            "labels": masked_sequences["labels"][train_size + val_size:]
        }

        return train, val, test

    def create_dataloaders(self):
        sequences = self._load_sequences()
        # sequences = "".join(["ACTG"[random.randint(0, 3)] for _ in range(10_000)])
        masked_sequences = self._process_sequences(sequences)
        train, val, test = self._split_sequences(masked_sequences)

        train_dataloader = DataLoader(HG38Dataset(train), batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(HG38Dataset(val), batch_size=self.batch_size, shuffle=True)
        test_dataloader = DataLoader(HG38Dataset(test), batch_size=self.batch_size, shuffle=True)

        return train_dataloader, val_dataloader, test_dataloader
