from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List
import pandas as pd
import random
import torch

from .dataloader.dataloader import Dataloader
from .preprocessor.preprocessor import Preprocessor


def complete_batch(dataframe: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    """
    Completes the last batch by copying the first example (batch_size - remainder) times.
    Args:
        dataframe:  dataframe with the sequences.
        batch_size: batch size.

    Returns:
        dataframe with the sequences and the batch_id.
    """
    complete_bins = []
    bins = [bin_df for _, bin_df in dataframe.groupby('bin')]

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


def shuffle_batches(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Shuffles the batches.
    
    Args:
        dataframe: dataframe with the sequences and the batch_id.

    Returns:
        dataframe with the sequences and the batch_id.
    """
    batch_bins = [df_new for _, df_new in dataframe.groupby('batch_id')]
    random.shuffle(batch_bins)
    return pd.concat(batch_bins).reset_index(drop=True)


def concater_collate(batch: List) -> Tuple:
    """
    Collate function for the dataloader. It concatenates the sequences and returns the labels, lengths and bins.
    
    Args:
        batch: list of tuples (sequence, label, length, bin) .

    Returns:
        tuple of concatenated sequences, lengths. bins and labels.
    """
    (sequence, _len, _bin, label) = zip(*batch)
    sequence = torch.cat(sequence, 0)
    label = torch.tensor(label)
    return sequence, list(_len), list(_bin), label


class TaxonomyClassificationDataset(Dataset):
    """Taxonomy classification dataset class"""

    def __init__(self, dataframe, batch_size):
        dataframe = complete_batch(dataframe=dataframe, batch_size=batch_size)
        self.dataframe = shuffle_batches(dataframe=dataframe)[['sequence', 'len', 'bin', 'label']]

    def __getitem__(self, index):
        sequence, _len, _bin, label = self.dataframe.iloc[index, :]
        sequence = torch.from_numpy(sequence)
        label = torch.tensor(label).float()
        return sequence, _len, _bin, label

    def __len__(self):
        return len(self.dataframe)


class HumanVariantEffectPredictionDataset(Dataset):
    """Human variant effect prediction dataset class"""

    def __init__(self, dataframe):
        self.references = dataframe["reference"].values
        self.alternates = dataframe["alternate"].values
        self.tissues = dataframe["tissue"].values
        self.labels = dataframe["label"].values

    def __getitem__(self, index):
        reference = torch.tensor(self.references[index])
        alternate = torch.tensor(self.alternates[index])
        tissue = torch.tensor(self.tissues[index])
        label = torch.tensor(self.labels[index]).float()
        return reference, alternate, tissue, label

    def __len__(self):
        return len(self.references)


class PlantVariantEffectPredictionDataset(Dataset):
    """Plant variant effect prediction dataset class"""

    def __init__(self, dataframe):
        self.sequences = dataframe["sequence"].values
        target_list = dataframe.columns.tolist()[1:]
        self.labels = dataframe[target_list].values

    def __getitem__(self, index):
        sequence = torch.tensor(self.sequences[index])
        label = torch.tensor(self.labels[index]).float()
        return sequence, label

    def __len__(self):
        return len(self.sequences)


class ChordMixerDataLoader(Dataloader, Preprocessor):
    """ChordMixer dataloader class"""

    def create_taxonomy_classification_dataloader(self, dataframe: pd.DataFrame) -> DataLoader:
        dataframe = self.process_taxonomy_classification_dataframe(dataframe=dataframe, model_name="ChordMixer")
        dataset = TaxonomyClassificationDataset(dataframe=dataframe, batch_size=self.batch_size)

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=concater_collate
        )

    def create_human_variant_effect_prediction_dataloader(self, dataframe: pd.DataFrame) -> DataLoader:
        dataframe = self.process_human_variant_effect_prediction_dataframe(dataframe=dataframe, model_name="ChordMixer")
        dataset = HumanVariantEffectPredictionDataset(dataframe=dataframe)

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

    def create_plant_variant_effect_prediction_dataloader(self, dataframe: pd.DataFrame) -> DataLoader:
        dataframe = self.process_plant_variant_effect_prediction_dataframe(dataframe=dataframe, model_name="ChordMixer")
        dataset = PlantVariantEffectPredictionDataset(dataframe=dataframe)

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
