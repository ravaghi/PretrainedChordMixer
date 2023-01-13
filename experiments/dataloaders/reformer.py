from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import os

from .dataloader import Dataloader


class DatasetCreator(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __getitem__(self, index):
        X, Y = self.dataframe.iloc[index, :]
        Y = torch.tensor(Y)
        X = torch.tensor(X)
        return X, Y

    def __len__(self):
        return len(self.dataframe)


class ReformerDataLoader(Dataloader):
    def create_dataloader(self):
        data_path = os.path.join(self.data_path, self.dataset_filename)
        dataframe = pd.read_csv(data_path)

        if self.dataset_type == "TaxonomyClassification":
            dataframe = self.process_taxonomy_classification_dataframe(dataframe, "Reformer")

            dataset = DatasetCreator(dataframe)
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False
            )

        elif self.dataset_type == "VariantEffectPrediction":
            pass

        elif self.dataset_type == "PlantDeepSEA":
            pass

        else:
            raise ValueError(f"Dataset type {self.dataset_type} not supported.")
