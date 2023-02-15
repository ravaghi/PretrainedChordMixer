from torch.utils.data import TensorDataset, DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
import os

from .dataloader import Dataloader

class PlantDeepSEADataset(Dataset):
    def __init__(self, dataframe):
        self.sequences = torch.tensor(np.array(dataframe.sequence.values.tolist(), dtype=np.float32))
        target_list = dataframe.columns.tolist()[:-1]
        self.labels = torch.tensor(dataframe[target_list].values)

    def __getitem__(self, index):
        x = self.sequences[index]
        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.sequences)


class CNNDataLoader(Dataloader):
    def _create_taxonomic_classification_dataloader(self, dataframe: pd.DataFrame) -> DataLoader:
        dataframe = self.process_taxonomy_classification_dataframe(dataframe, "CNN")

        sequences = torch.tensor(np.array(dataframe.sequence.values.tolist(), dtype=np.float32))
        labels = torch.tensor(dataframe.label.values)

        dataset = TensorDataset(sequences, labels)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )

    def _create_variant_effect_prediction_dataloader(self, dataframe: pd.DataFrame) -> DataLoader:
        dataframe = self.process_variant_effect_prediction_dataframe(dataframe, "CNN")

        references = torch.tensor(np.array(dataframe.reference.values.tolist(), dtype=np.float32))
        alternatives = torch.tensor(np.array(dataframe.alternate.values.tolist(), dtype=np.float32))
        tissues = torch.tensor(dataframe.tissue.values)
        labels = torch.tensor(dataframe.label.values)

        dataset = TensorDataset(references, alternatives, tissues, labels)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )

    def _create_plant_deepsea_dataloader(self, dataframe: pd.DataFrame) -> DataLoader:
        dataframe = self.process_plantdeepsea_dataframe(dataframe, "CNN")
        dataset = PlantDeepSEADataset(dataframe)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )
    def create_dataloaders(self):
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
            
