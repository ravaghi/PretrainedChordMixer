from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import os

from .dataloader import Dataloader


class TaxonomyClassificationDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __getitem__(self, index):
        X, Y = self.dataframe.iloc[index, :]
        Y = torch.tensor(Y)
        X = torch.tensor(X)
        return X, Y

    def __len__(self):
        return len(self.dataframe)


class XFormerDataLoader(Dataloader):
    def _create_taxonomic_classification_dataloader(self, dataframe: pd.DataFrame) -> DataLoader:
        dataframe = self.process_taxonomy_classification_dataframe(dataframe, "Xformer")
        dataset = TaxonomyClassificationDataset(dataframe)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )

    def _create_variant_effect_prediction_dataloader(self, dataframe: pd.DataFrame) -> DataLoader:
        pass

    def _create_plant_deepsea_dataloader(self, dataframe: pd.DataFrame) -> DataLoader:
        pass
    
    def create_dataloaders(self):
        train_dataframe = self.read_data(self.train_dataset)[:2000]
        val_dataframe = self.read_data(self.val_dataset)[:2000]
        test_dataframe = self.read_data(self.test_dataset)[:2000]

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
