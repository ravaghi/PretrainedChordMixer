import os
import torch
from Bio import SeqIO
from tqdm import tqdm
from typing import Tuple
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random

class SequenceProcessor:
    """Processes a DNA sequence by tokenizing, splitting and masking"""

    # _DNA_BASE_DICT = {
    #     'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, 'Y': 5, 'R': 6, 'M': 7,
    #     'W': 8, 'K': 9, 'S': 10, 'B': 11, 'H': 12, 'D': 13, 'V': 14
    # }

    _DNA_BASE_DICT = {
        'A': 0, 'C': 1, 'G': 2, 'T': 3
    }

    def __init__(self, mask_ratio: float, sequence_length: int):
        self.mask_ratio = mask_ratio
        self.sequence_length = sequence_length

    def tokenize(self, sequence: str) -> torch.Tensor:
        """
        Tokenizes a string of DNA bases into a tensor of integers

        Args:
            sequence (str): DNA sequence to be tokenized

        Returns:
            torch.Tensor: 1D tensor of integers representing the DNA sequence
        """
        return torch.tensor([self._DNA_BASE_DICT[base] for base in tqdm(sequence, desc="Tokenizing sequences")], dtype=torch.int64)

    def split(self, sequence_ids: torch.Tensor) -> torch.Tensor:
        """
        Splits a 1D tensor into multiple tensors of length sequence_length

        Args:
            sequence_ids (torch.Tensor): 1D Tensor of integers representing a DNA sequence

        Returns:
            torch.Tensor: 2D Tensor of integers representing the split DNA sequences
        """
        # Shortening the tenssor to be divisible by sequence length
        remainder = len(sequence_ids) % self.sequence_length
        if remainder > 0:
            sequence_ids = sequence_ids[:-remainder]

        print(f"Splitting sequences into sequences of length {self.sequence_length}")
        return torch.stack(torch.split(sequence_ids, self.sequence_length))

    def mask(self, sequence_ids: torch.Tensor) -> dict:
        """
        Masks and one hot encodes a 2D tensor with the given mask ratio

        Args:
            sequence_ids (torch.Tensor): 2D Tensor of integers representing a DNA sequence

        Returns:
            dict: Dictionary containing the masked sequences, masks and labels
        """
        labels = sequence_ids.clone()

        # Creating masks
        rand = torch.rand(sequence_ids.shape[0], sequence_ids.shape[1])
        masks = rand < self.mask_ratio

        # One hot encoding
        sequence_ids = F.one_hot(sequence_ids, num_classes=len(self._DNA_BASE_DICT))

        # Masking
        for i in tqdm(range(sequence_ids.shape[0]), desc="Masking"):
            selection = torch.flatten(masks[i].nonzero()).tolist()
            sequence_ids[i, selection] = 0

        return {
            "sequence_ids": sequence_ids,
            "masks": masks,
            "labels": labels
        }


class HG38Dataset(Dataset):
    """Dataset for the hg38 dataset"""

    def __init__(self, data: dict):
        self.sequence_ids = data["sequence_ids"].to(torch.float32)
        self.masks = data["masks"]
        self.labels = data["labels"].to(torch.long)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the item at the given index
        
        Args:
            index (int): Index of the item to be returned

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing the sequence_ids, masks and labels
        """
        return self.sequence_ids[index], self.masks[index], self.labels[index]

    def __len__(self) -> int:
        """
        Returns the length of the dataset
        
        Returns:
            int: Length of the dataset
        """
        return len(self.sequence_ids)


class PretrainedChordMixerDataLoader:
    """DataLoader for the pretrained ChordMixer model"""

    _CHROMOSOMES = [
       "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9",
       "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", 
       "chr18", "chr19", "chr20", "chr21", "chr22", "chrX", "chrY"
    ]

    # _CHROMOSOMES = [
    #     "chr1"
    # ]

    def __init__(self, data_path: str, dataset_filename: str, batch_size: int, mask_ratio: float, sequence_length: int):
        self.data_path = data_path
        self.dataset_filename = dataset_filename
        self.batch_size = batch_size
        self.mask_ratio = mask_ratio
        self.sequence_length = sequence_length

    def _load_sequences(self) -> str:
        """
        Loads and joins the sequences from the fasta file

        Returns:
            str: Concatenated DNA sequences
        """
        data_path = os.path.join(self.data_path, self.dataset_filename)
        sequences_dict = SeqIO.to_dict(SeqIO.parse(data_path, "fasta"))
        sequences = ""
        for chromosome in tqdm(self._CHROMOSOMES, desc="Loading sequences"):
            sequences += str(sequences_dict[chromosome].seq).upper().replace("N", "")
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
    def _split_dataset(masked_sequences: dict) -> Tuple[dict, dict, dict]:
        """
        Splits the sequences into train, validation and test sets

        Args:
            masked_sequences (dict): Dictionary containing the masked sequences, masks and labels

        Returns:
            Tuple[dict, dict, dict]: Tuple containing the train, validation and test sets
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

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Processes the datset and creates dataloaders for the train, validation and test sets

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: Tuple containing the train, validation and test dataloaders
        """
        # sequences = "".join(["ACGT"[random.randint(0, 3)] for _ in range(1000_000)])
        sequences = self._load_sequences()
        masked_sequences = self._process_sequences(sequences)
        train, val, test = self._split_dataset(masked_sequences)

        train_dataloader = DataLoader(HG38Dataset(train), batch_size=self.batch_size, shuffle=True, pin_memory=True)
        val_dataloader = DataLoader(HG38Dataset(val), batch_size=self.batch_size, shuffle=True, pin_memory=True)
        test_dataloader = DataLoader(HG38Dataset(test), batch_size=self.batch_size, shuffle=True, pin_memory=True)

        return train_dataloader, val_dataloader, test_dataloader
