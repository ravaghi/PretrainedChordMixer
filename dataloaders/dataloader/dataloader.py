from torch.utils.data import DataLoader
from typing import Tuple
from abc import ABC
import pandas as pd
import os


class Dataloader(ABC):
    def __init__(self, batch_size, data_path, dataset_type, dataset_name, train_dataset, val_dataset, test_dataset):
        self.batch_size = batch_size
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.dataset_name = dataset_name
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def read_data(self, filename: str) -> pd.DataFrame:
        """
        Read data from a parquet file

        Args:
            filename: name of the dataset file

        Returns:
            dataframe

        Raises:
            FileNotFoundError: if the file is not found
        """
        path = os.path.join(self.data_path, filename)
        if os.path.exists(path):
            dataframe = pd.read_parquet(path)
        else:
            raise FileNotFoundError(f"File {path} not found.")
        return dataframe

    def create_taxonomy_classification_dataloader(self, dataframe: pd.DataFrame) -> DataLoader:
        """
        Process taxonomy classification dataset and create a dataloader

        Args:
            dataframe: dataframe containing the dataset

        Returns:
            dataloader
        """
        raise NotImplementedError

    def create_human_variant_effect_prediction_dataloader(self, dataframe: pd.DataFrame) -> DataLoader:
        """
        Process human variant effect prediction dataset and create a dataloader

        Args:
            dataframe: dataframe containing the dataset

        Returns:
            dataloader
        """
        raise NotImplementedError

    def create_plant_variant_effect_prediction_dataloader(self, dataframe: pd.DataFrame) -> DataLoader:
        """
        Process plant variant effect prediction dataset and create a dataloader

        Args:
            dataframe: dataframe containing the dataset

        Returns:
            dataloader
        """
        raise NotImplementedError

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create dataloaders for the train, validation and test sets

        Returns:
            train, validation and test dataloaders

        Raises:
            ValueError: if the dataset type is not supported
        """
        train_dataframe = self.read_data(self.train_dataset)
        val_dataframe = self.read_data(self.val_dataset)
        test_dataframe = self.read_data(self.test_dataset)

        if self.dataset_type == "TaxonomyClassification":
            train_dataloader = self.create_taxonomy_classification_dataloader(train_dataframe)
            val_dataloader = self.create_taxonomy_classification_dataloader(val_dataframe)
            test_dataloader = self.create_taxonomy_classification_dataloader(test_dataframe)
        elif self.dataset_type == "HumanVariantEffectPrediction":
            train_dataloader = self.create_human_variant_effect_prediction_dataloader(train_dataframe)
            val_dataloader = self.create_human_variant_effect_prediction_dataloader(val_dataframe)
            test_dataloader = self.create_human_variant_effect_prediction_dataloader(test_dataframe)
        elif self.dataset_type == "PlantVariantEffectPrediction":
            train_dataloader = self.create_plant_variant_effect_prediction_dataloader(train_dataframe)
            val_dataloader = self.create_plant_variant_effect_prediction_dataloader(val_dataframe)
            test_dataloader = self.create_plant_variant_effect_prediction_dataloader(test_dataframe)
        else:
            raise ValueError(f"Dataset type {self.dataset_type} not supported.")

        return train_dataloader, val_dataloader, test_dataloader
