from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch

from .dataloader.dataloader import Dataloader
from .preprocessor.preprocessor import Preprocessor


class TaxonomyClassificationDataset(Dataset):
    """Taxonomy classification dataset class"""

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __getitem__(self, index):
        sequence, label = self.dataframe.iloc[index, :]
        sequence = torch.tensor(sequence)
        label = torch.tensor(label).float()
        return sequence, label

    def __len__(self):
        return len(self.dataframe)


class HumanVariantEffectPredictionDataset(Dataset):
    """Human variant effect prediction dataset class"""

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __getitem__(self, index):
        reference, alternate, tissue, label = self.dataframe.iloc[index, :]
        reference = torch.tensor(reference)
        alternate = torch.tensor(alternate)
        tissue = torch.tensor(tissue)
        label = torch.tensor(label).float()
        return reference, alternate, tissue, label

    def __len__(self):
        return len(self.dataframe)


class PlantVariantEffectPredictionDataset(Dataset):
    """Plant variant effect prediction dataset class"""

    def __init__(self, dataframe):
        self.sequences = np.array(dataframe.sequence.values.tolist())
        target_list = dataframe.columns.tolist()[1:]
        self.labels = dataframe[target_list].values

    def __getitem__(self, index):
        sequence = torch.tensor(self.sequences[index])
        label = torch.tensor(self.labels[index]).float()
        return sequence, label

    def __len__(self):
        return len(self.sequences)


class XFormerDataLoader(Dataloader, Preprocessor):
    """ XFormer dataloader class """

    def create_taxonomy_classification_dataloader(self, dataframe: pd.DataFrame) -> DataLoader:
        dataframe = self.process_taxonomy_classification_dataframe(
            dataframe=dataframe,
            model_name="Xformer",
            max_sequence_length=25_000
        )
        dataset = TaxonomyClassificationDataset(dataframe=dataframe)

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

    def create_human_variant_effect_prediction_dataloader(self, dataframe: pd.DataFrame) -> DataLoader:
        dataframe = self.process_human_variant_effect_prediction_dataframe(
            dataframe=dataframe,
            model_name="Xformer"
        )
        dataset = HumanVariantEffectPredictionDataset(dataframe=dataframe)

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

    def create_plant_variant_effect_prediction_dataloader(self, dataframe: pd.DataFrame) -> DataLoader:
        dataframe = self.process_plant_variant_effect_prediction_dataframe(
            dataframe=dataframe,
            model_name="Xformer"
        )
        dataset = PlantVariantEffectPredictionDataset(dataframe=dataframe)

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
