from torch.utils.data import DataLoader
from typing import Tuple
from abc import ABC
import pandas as pd
import os


class Dataloader(ABC):
    """Base class for all dataloaders."""

    def __init__(self,
                 batch_size: int,
                 data_path: str,
                 dataset_type: str,
                 dataset_name: str,
                 train_dataset: str,
                 val_dataset: str,
                 test_dataset: str
                 ):
        self.batch_size = batch_size
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.dataset_name = dataset_name
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def read_data(self, filename: str) -> pd.DataFrame:
        """
        Reads data from a parquet file.

        Args:
            filename: name of the dataset file.

        Returns:
            dataframe containing the dataset.

        Raises:
            FileNotFoundError: if the dataset file is not found.
        """
        path = os.path.join(self.data_path, filename)
        if os.path.exists(path):
            dataframe = pd.read_parquet(path)
        else:
            raise FileNotFoundError(f"File {path} not found.")
        return dataframe

    def create_taxonomy_classification_dataloader(self, dataframe: pd.DataFrame) -> DataLoader:
        """
        Processes taxonomy classification dataset and create a dataloader.

        Args:
            dataframe: dataframe containing the dataset.

        Returns:
            dataloader.
        """
        raise NotImplementedError

    def create_variant_effect_prediction_dataloader(self, dataframe: pd.DataFrame) -> DataLoader:
        """
        Processes variant effect prediction dataset and create a dataloader.

        Args:
            dataframe: dataframe containing the dataset.

        Returns:
            dataloader.
        """
        raise NotImplementedError

    def create_plant_ocr_prediction_dataloader(self, dataframe: pd.DataFrame) -> DataLoader:
        """
        Processes plant ocr prediction dataset and create a dataloader.

        Args:
            dataframe: dataframe containing the dataset.

        Returns:
            dataloader.
        """
        raise NotImplementedError

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Creates dataloaders for the train, validation and test sets.

        Returns:
            train, validation and test dataloaders.

        Raises:
            ValueError: if the dataset type is not supported.
        """
        train_dataframe = self.read_data(self.train_dataset)
        val_dataframe = self.read_data(self.val_dataset)
        test_dataframe = self.read_data(self.test_dataset)

        if self.dataset_type == "TaxonomyClassification":
            train_dataloader = self.create_taxonomy_classification_dataloader(train_dataframe)
            val_dataloader = self.create_taxonomy_classification_dataloader(val_dataframe)
            test_dataloader = self.create_taxonomy_classification_dataloader(test_dataframe)
        elif self.dataset_type == "VariantEffectPrediction":
            train_dataloader = self.create_variant_effect_prediction_dataloader(train_dataframe)
            val_dataloader = self.create_variant_effect_prediction_dataloader(val_dataframe)
            test_dataloader = self.create_variant_effect_prediction_dataloader(test_dataframe)
        elif self.dataset_type == "PlantOcrPrediction":
            train_dataloader = self.create_plant_ocr_prediction_dataloader(train_dataframe)
            val_dataloader = self.create_plant_ocr_prediction_dataloader(val_dataframe)
            test_dataloader = self.create_plant_ocr_prediction_dataloader(test_dataframe)
        else:
            raise ValueError(f"Dataset type {self.dataset_type} not supported.")

        return train_dataloader, val_dataloader, test_dataloader
