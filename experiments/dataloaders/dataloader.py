from abc import ABC
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

DNA_BASE_DICT = {
    'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, 'Y': 5, 'R': 6, 'M': 7,
    'W': 8, 'K': 9, 'S': 10, 'B': 11, 'H': 12, 'D': 13, 'V': 14
}
DNA_BASE_DICT_REVERSED = {
    0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N', 5: 'Y', 6: 'R', 7: 'M',
    8: 'W', 9: 'K', 10: 'S', 11: 'B', 12: 'H', 13: 'D', 14: 'V'
}


class Dataloader(ABC):
    def __init__(self, data_path, dataset, dataset_name, batch_size):
        self.data_path = data_path
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.batch_size = batch_size

    def create_dataloader(self) -> DataLoader:
        raise NotImplementedError

    @staticmethod
    def process_taxonomy_classification_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Process the dataframe for taxonomy classification

        Args:
            dataframe: dataframe to process

        Returns:
            processed dataframe
        """
        dataframe["sequence"] = dataframe["sequence"].apply(lambda x: np.array([DNA_BASE_DICT[base] for base in x]))
        dataframe = dataframe[["sequence", "label", "bin", "len"]]
        return dataframe

    @staticmethod
    def process_plantdeepsea_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Process the dataframe for plant deepsea

        Args:
            dataframe: dataframe to process

        Returns:
            processed dataframe
        """
        dataframe["sequence"] = dataframe["sequence"].apply(lambda x: np.array([DNA_BASE_DICT[base] for base in x]))
        dataframe["len"] = dataframe["sequence"].apply(lambda x: len(x))
        dataframe["bin"] = -1
        return dataframe

    @staticmethod
    def process_variant_effect_prediction_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Process the dataframe for variant effect prediction

        Args:
            dataframe: dataframe to process

        Returns:
            processed dataframe
        """
        dataframe["reference"] = dataframe["reference"].apply(lambda x: np.array([DNA_BASE_DICT[base] for base in x]))
        dataframe["alternate"] = dataframe["alternate"].apply(lambda x: np.array([DNA_BASE_DICT[base] for base in x]))
        dataframe = dataframe[["reference", "alternate", "tissue", "label"]]
        return dataframe
