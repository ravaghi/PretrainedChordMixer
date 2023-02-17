from torch.utils.data import TensorDataset, DataLoader, Dataset
import pandas as pd
import numpy as np
import torch

from .dataloader.dataloader import Dataloader
from .preprocessor.preprocessor import Preprocessor


class PlantVariantEffectPredictionDataset(Dataset):
    """Plant variant effect prediction dataset class"""

    def __init__(self, dataframe):
        self.sequences = np.array(dataframe.sequence.values.tolist())
        target_list = dataframe.columns.tolist()[:-1]
        self.labels = dataframe[target_list].values

    def __getitem__(self, index):
        sequence = torch.tensor(self.sequences[index])
        label = torch.tensor(self.labels[index])
        return sequence, label

    def __len__(self):
        return len(self.sequences)


class CNNDataLoader(Dataloader, Preprocessor):
    """CNN dataloader class"""

    def create_taxonomy_classification_dataloader(self, dataframe: pd.DataFrame) -> DataLoader:
        dataframe = self.process_taxonomy_classification_dataframe(dataframe=dataframe, model_name="CNN")

        sequences = torch.tensor(np.array(dataframe.sequence.values.tolist(), dtype=np.float32))
        labels = torch.tensor(dataframe.label.values)
        dataset = TensorDataset(sequences, labels)

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

    def create_human_variant_effect_prediction_dataloader(self, dataframe: pd.DataFrame) -> DataLoader:
        dataframe = self.process_human_variant_effect_prediction_dataframe(dataframe=dataframe, model_name="CNN")

        references = torch.tensor(np.array(dataframe.reference.values.tolist(), dtype=np.float32))
        alternatives = torch.tensor(np.array(dataframe.alternate.values.tolist(), dtype=np.float32))
        tissues = torch.tensor(dataframe.tissue.values)
        labels = torch.tensor(dataframe.label.values)
        dataset = TensorDataset(references, alternatives, tissues, labels)

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

    def create_plant_variant_effect_prediction_dataloader(self, dataframe: pd.DataFrame) -> DataLoader:
        dataframe = self.process_plant_variant_effect_prediction_dataframe(dataframe=dataframe, model_name="CNN")
        dataset = PlantVariantEffectPredictionDataset(dataframe=dataframe)

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
