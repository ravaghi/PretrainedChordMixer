from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import torch
import os

from .dataloader import Dataloader


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
