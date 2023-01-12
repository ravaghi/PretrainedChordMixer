from torch.utils.data import TensorDataset, DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
import os

from .dataloader import Dataloader


class PlantDeepSeaDatasetCreator(Dataset):
    def __init__(self, df):
        self.df = df
        target_list = df.columns.tolist()[:-3]
        self.targets = self.df[target_list].values

    def __getitem__(self, index):
        X = self.df.iloc[index]['sequence']
        length = self.df.iloc[index]['len']
        bin = self.df.iloc[index]['bin']
        Y = torch.FloatTensor(self.targets[index])
        X = torch.from_numpy(X)
        return X, Y, length, bin

    def __len__(self):
        return len(self.df)


class CNNDataLoader(Dataloader):
    def create_dataloader(self):
        data_path = os.path.join(self.data_path, self.dataset_filename)
        dataframe = pd.read_csv(data_path)

        if self.dataset_type == "TaxonomyClassification":
            dataframe = self.process_taxonomy_classification_dataframe(dataframe, "CNN")

            sequences = np.array(dataframe.sequence.values.tolist(), dtype=np.float32)
            labels = dataframe.label.values.tolist()

            sequences = torch.tensor(sequences)
            labels = torch.tensor(labels)

            dataset = TensorDataset(sequences, labels)

            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False
            )

        elif self.dataset_type == "VariantEffectPrediction":
            dataframe = self.process_variant_effect_prediction_dataframe(dataframe, "CNN")
            
            references = np.array(dataframe.reference.values.tolist(), dtype=np.float32)
            alternates = np.array(dataframe.alternate.values.tolist(), dtype=np.float32)
            tissues = dataframe.tissue.values.tolist()
            labels = dataframe.label.values.tolist()

            references = torch.tensor(references)
            alternates = torch.tensor(alternates)
            tissues = torch.tensor(tissues, dtype=torch.int64)
            labels = torch.tensor(labels)

            dataset = TensorDataset(references, alternates, tissues, labels)

            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False
            )

        elif self.dataset_type == "PlantDeepSEA":
            dataframe = self.process_plantdeepsea_dataframe(dataframe, "CNN")
            dataset = PlantDeepSeaDatasetCreator(df=dataframe)
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False
            )

        else:
            raise ValueError(f"Dataset type {self.dataset_type} not supported.")
            
