from torch.utils.data import DataLoader
import pandas as pd
from typing import Tuple

from .dataloader.dataloader import Dataloader
from .chordmixer import concater_collate, TaxonomyClassificationDataset, VariantEffectPredictionDataset, PlantDeepSEADataset


class PretrainedChordMixerDataLoader(Dataloader):
    def _create_taxonomic_classification_dataloader(self, dataframe: pd.DataFrame) -> DataLoader:
        """
        Creates a dataloader for the taxonomy classification task.
        
        Args:
            dataframe: dataframe with the sequences

        Returns:
            dataloader for the taxonomy classification task
        """
        dataframe = self.process_taxonomy_classification_dataframe(dataframe, "PretrainedChordMixer")
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
        dataframe = self.process_variant_effect_prediction_dataframe(dataframe, "PretrainedChordMixer")
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
        dataframe = self.process_plantdeepsea_dataframe(dataframe, "PretrainedChordMixer")
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
