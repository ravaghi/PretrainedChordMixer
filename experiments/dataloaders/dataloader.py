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
    def __init__(self, data_path, dataset_filename, dataset_type, dataset_name, batch_size):
        self.data_path = data_path
        self.dataset_filename = dataset_filename
        self.dataset_type = dataset_type
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        

    def create_dataloader(self) -> DataLoader:
        raise NotImplementedError

    @staticmethod
    def pad_sequences(dataframe: pd.DataFrame, max_len: int=1000) -> pd.DataFrame:
        """
        Pad sequences to max length

        Args:
            dataframe: dataframe to pad
            max_len: max length to pad or truncate to
        
        Returns:
            padded dataframe
        """
        max_seq_len = dataframe["sequence"].apply(lambda x: len(x)).max()
        if max_seq_len < max_len:
            max_seq_len = max_len
        dataframe["sequence"] = dataframe["sequence"].str.pad(max_seq_len, side="right", fillchar="A")
        dataframe["sequence"] = dataframe["sequence"].apply(lambda x: x[:max_len].upper())
        return dataframe

    @staticmethod
    def convert_base_to_index(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Convert DNA sequence bases to indices

        Args:
            dataframe: dataframe to convert

        Returns:
            converted dataframe
        """
        dataframe["new_sequence"] = dataframe["sequence"].apply(lambda x: [DNA_BASE_DICT[base] for base in x])
        dataframe = dataframe.drop(columns=["sequence"])
        dataframe = dataframe.rename(columns={"new_sequence": "sequence"})
        dataframe = dataframe.sample(frac=1).reset_index(drop=True)
        return dataframe

    def process_taxonomy_classification_dataframe(self, dataframe: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """
        Process the taxonomy classification dataset for a specific model

        Args:
            dataframe: dataframe to process
            model_name: model

        Returns:
            processed dataframe
        """
        if model_name == "ChordMixer":
            dataframe["seq"] = dataframe["sequence"].apply(lambda x: np.array([DNA_BASE_DICT[base] for base in x]))
            dataframe = dataframe.drop(columns=['sequence'])
            dataframe = dataframe.rename(columns={'seq': 'sequence'})
            return dataframe[["sequence", "label", "bin", "len"]]

        elif model_name in ["CNN", "Linformer", "Nystromformer", "Poolformer", "Reformer", "Transformer"]:
            dataframe = dataframe.drop(columns=["len", "bin"])

            if self.dataset_type == "TaxonomyClassification":
                max_len = 25_000

            dataframe = self.pad_sequences(dataframe, max_len)
            dataframe = self.convert_base_to_index(dataframe)
    
            return dataframe[["sequence", "label"]]

        elif model_name == "KeGRU":
            dataframe = dataframe.drop(columns=["len", "bin"])

            if self.dataset_type == "TaxonomyClassification":
                max_len = 10_000

            dataframe = self.pad_sequences(dataframe, max_len)
    
            return dataframe[["sequence", "label"]]
            
        else:
            raise ValueError(f"Model: {model_name} not supported")

    @staticmethod
    def process_variant_effect_prediction_dataframe(dataframe: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """
        Process the variant effect prediction dataset for a specific model

        Args:
            dataframe: dataframe to process
            model_name: model

        Returns:
            processed dataframe
        """
        if model_name in ["ChordMixer", "CNN"]:
            dataframe["reference"] = dataframe["reference"].apply(lambda x: np.array([DNA_BASE_DICT[base] for base in x]))
            dataframe["alternate"] = dataframe["alternate"].apply(lambda x: np.array([DNA_BASE_DICT[base] for base in x]))
            dataframe = dataframe[["reference", "alternate", "tissue", "label"]]

        elif model_name == "KeGRU":
            dataframe = dataframe[["reference", "alternate", "tissue", "label"]]

        else:
            raise ValueError(f"Model: {model_name} not supported")

        return dataframe

    @staticmethod
    def process_plantdeepsea_dataframe(dataframe: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """
        Process the PlantDeepSEA dataset for a specific model

        Args:
            dataframe: dataframe to process
            model_name: model

        Returns:
            processed dataframe
        """
        if model_name in ["ChordMixer", "CNN"]:
            dataframe["seq"] = dataframe["sequence"].apply(lambda x: np.array([DNA_BASE_DICT[base] for base in x]))
            dataframe = dataframe.drop(columns=['sequence'])
            dataframe = dataframe.rename(columns={'seq': 'sequence'})
            dataframe["len"] = dataframe["sequence"].apply(lambda x: len(x))
            dataframe["bin"] = -1
        else:
            raise ValueError(f"Model: {model_name} not supported")
        return dataframe
