from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import torch
from typing import Tuple

from .dataloader import Dataloader


def complete_batch(dataframe: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    """
    Completes the last batch by copying the first example (batch_size - remainder) times.
    Args:
        dataframe:  dataframe with the sequences
        batch_size: batch size

    Returns:
        dataframe with the sequences and the batch_id
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
    Shuffles the batches
    
    Args:
        dataframe: dataframe with the sequences and the batch_id

    Returns:
        dataframe with the sequences and the batch_id
    """
    batch_bins = [df_new for _, df_new in dataframe.groupby('batch_id')]
    random.shuffle(batch_bins)
    return pd.concat(batch_bins).reset_index(drop=True)


def concater_collate(batch: list) -> tuple:
    """
    Collate function for the dataloader. It concatenates the sequences and returns the labels, lengths and bins.
    
    Args:
        batch: list of tuples (sequence, label, length, bin) 

    Returns:
        tuple of concatenated sequences, labels, lengths and bins
    """
    (xx, yy, lengths, bins) = zip(*batch)
    xx = torch.cat(xx, 0)
    yy = torch.tensor(yy)
    return xx, yy, list(lengths), list(bins)


class TaxonomyClassificationDataset(Dataset):
    """Dataset for the taxonomy classification task."""

    def __init__(self, dataframe, batch_size, var_len=False):
        if var_len:
            dataframe = complete_batch(dataframe=dataframe, batch_size=batch_size)
            self.dataframe = shuffle_batches(dataframe=dataframe)[['sequence', 'label', 'len', 'bin']]
        else:
            self.dataframe = dataframe

    def __getitem__(self, index):
        X, Y, length, bin = self.dataframe.iloc[index, :]
        Y = torch.tensor(Y)
        X = torch.from_numpy(X)
        return X, Y, length, bin

    def __len__(self):
        return len(self.dataframe)


class VariantEffectPredictionDataset(Dataset):
    """Dataset for the variant effect prediction task."""

    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.reference = dataframe["reference"].values
        self.alternate = dataframe["alternate"].values
        self.tissue = dataframe["tissue"].values
        self.label = dataframe["label"].values

    def __getitem__(self, index):
        return self.reference[index], self.alternate[index], self.tissue[index], self.label[index]

    def __len__(self):
        return len(self.dataframe)


class PlantDeepSEADataset(Dataset):
    """Dataset for the PlantDeepSEA task."""

    def __init__(self, dataframe, batch_size, var_len=False):
        if var_len:
            target_list = dataframe.columns.tolist()[:-3]
            dataframe = complete_batch(dataframe=dataframe, batch_size=batch_size)
            columns = ['sequence', 'len', 'bin'] + target_list
            self.dataframe = shuffle_batches(dataframe=dataframe)[columns]
            self.targets = self.dataframe[target_list].values
        else:
            self.dataframe = dataframe

    def __getitem__(self, index):
        X = self.dataframe.iloc[index]['sequence']
        length = self.dataframe.iloc[index]['len']
        bin = self.dataframe.iloc[index]['bin']
        Y = torch.FloatTensor(self.targets[index])
        X = torch.from_numpy(X)
        return X, Y, length, bin

    def __len__(self):
        return len(self.dataframe)


class ChordMixerDataLoader(Dataloader):
    def _create_taxonomic_classification_dataloader(self, dataframe: pd.DataFrame) -> DataLoader:
        """
        Creates a dataloader for the taxonomy classification task.
        
        Args:
            dataframe: dataframe with the sequences

        Returns:
            dataloader for the taxonomy classification task
        """
        dataframe = self.process_taxonomy_classification_dataframe(dataframe, "ChordMixer")
        dataset = TaxonomyClassificationDataset(
            dataframe=dataframe,
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

    def _create_variant_effect_prediction_dataloader(self, dataframe: pd.DataFrame) -> DataLoader:
        """
        Creates a dataloader for the variant effect prediction task.
        
        Args:
            dataframe: dataframe with the sequences

        Returns:
            dataloader for the variant effect prediction task
        """
        dataframe = self.process_variant_effect_prediction_dataframe(dataframe, "ChordMixer")
        dataset = VariantEffectPredictionDataset(dataframe)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )

    def _create_plant_deepsea_dataloader(self, dataframe: pd.DataFrame) -> DataLoader:
        """
        Creates a dataloader for the PlantDeepSEA task.
        
        Args:
            dataframe: dataframe with the sequences

        Returns:
            dataloader for the PlantDeepSEA task
        """
        dataframe = self.process_plantdeepsea_dataframe(dataframe, "ChordMixer")
        dataset = PlantDeepSEADataset(
            dataframe=dataframe,
            batch_size=self.batch_size,
            var_len=False
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=concater_collate,
            drop_last=False
        )

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Creates the dataloaders for the train, validation and test set.
        
        Returns:
            dataloaders for the train, validation and test set
            
        Raises:
            ValueError: if the dataset type is not supported
        """
        train_dataframe = self.read_data(self.train_dataset)
        val_dataframe = self.read_data(self.val_dataset)
        test_dataframe = self.read_data(self.test_dataset)

        if self.dataset_type == "TaxonomyClassification":
            train_dataloader = self._create_taxonomic_classification_dataloader(train_dataframe)
            val_dataloader = self._create_taxonomic_classification_dataloader(val_dataframe)
            test_dataloader = self._create_taxonomic_classification_dataloader(test_dataframe)
        elif self.dataset_type == "VariantEffectPrediction":
            train_dataloader = self._create_variant_effect_prediction_dataloader(train_dataframe)
            val_dataloader = self._create_variant_effect_prediction_dataloader(val_dataframe)
            test_dataloader = self._create_variant_effect_prediction_dataloader(test_dataframe)
        elif self.dataset_type == "PlantDeepSEA":
            train_dataloader = self._create_plant_deepsea_dataloader(train_dataframe)
            val_dataloader = self._create_plant_deepsea_dataloader(val_dataframe)
            test_dataloader = self._create_plant_deepsea_dataloader(test_dataframe)
        else:
            raise ValueError(f"Dataset type {self.dataset_type} not supported.")

        return train_dataloader, val_dataloader, test_dataloader
