from sklearn.preprocessing import LabelBinarizer
from abc import ABC
import pandas as pd
import numpy as np


class Preprocessor(ABC):
    _DNA_BASE_DICT = {
        'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4,
        'Y': 5, 'R': 6, 'M': 7, 'W': 8, 'K': 9,
        'S': 10, 'B': 11, 'H': 12, 'D': 13, 'V': 14
    }

    def tokenize(self, dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Convert sequences to tokenized arrays of integers

        Args:
            dataframe: dataframe
            column: name of the column containing the sequences

        Returns:
            dataframe with tokenized sequences
        """
        dataframe[column] = dataframe[column].apply(
            lambda x: np.array([self._DNA_BASE_DICT[base] for base in x], dtype=np.int32)
        )
        return dataframe

    @staticmethod
    def pad_or_truncate(dataframe: pd.DataFrame, length: int) -> pd.DataFrame:
        """
        Pad or truncate sequences to the specified length

        Args:
            dataframe: dataframe to pad or truncate
            length: Length to pad or truncate to

        Returns:
            dataframe with padded or truncated sequences
        """

        # Pad if max_sequence_len < length, otherwise truncate
        max_sequence_len = dataframe["sequence"].apply(lambda x: len(x)).max()
        if max_sequence_len < length:
            max_sequence_len = length

        # Pad to max_sequence_len
        dataframe["sequence"] = dataframe["sequence"].str.pad(max_sequence_len, side="right", fillchar="N")
        # Truncate to length
        dataframe["sequence"] = dataframe["sequence"].apply(lambda x: x[:length])
        return dataframe

    @staticmethod
    def get_bins(dataframe: pd.DataFrame) -> pd.DataFrame:
        percentiles = [i * 0.1 for i in range(10)] + [.95, .99, .995]
        bins = np.quantile(dataframe['len'], percentiles)
        bin_labels = [i for i in range(len(bins) - 1)]
        dataframe['bin'] = pd.cut(dataframe['len'], bins=bins, labels=bin_labels)
        return dataframe

    @staticmethod
    def one_hot_encode(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        One hot encode sequences

        Args:
            dataframe: dataframe containing the DNA sequences
            columns: name of the column containing the sequences

        Returns:
            dataframe with one hot encoded sequences
        """
        label_binarizer = LabelBinarizer()
        label_binarizer.fit(["A", "C", "G", "T"])
        dataframe[column] = dataframe[column].apply(list)
        dataframe[column] = dataframe[column].apply(label_binarizer.transform)
        return dataframe

    def process_taxonomy_classification_dataframe(
            self,
            dataframe: pd.DataFrame,
            model_name: str,
            max_sequence_length: int = None) -> pd.DataFrame:
        """
        Process the taxonomy classification dataset for a specific model

        Args:
            dataframe: dataframe to process
            model_name: model to process for
            max_sequence_length: max sequence length if padding is required

        Returns:
            processed dataframe

        Raises:
            ValueError: if model is not supported
        """
        if model_name == "ChordMixer":
            dataframe = self.tokenize(dataframe, "sequence")
            dataframe["len"] = dataframe["sequence"].apply(lambda x: len(x))
            dataframe = self.get_bins(dataframe)
            dataframe = dataframe[["sequence", "len", "bin", "label"]]

        elif model_name == "KeGRU":
            dataframe = self.pad_or_truncate(dataframe, max_sequence_length)
            dataframe = dataframe[["sequence", "label"]]

        elif model_name == "CNN":
            dataframe = self.pad_or_truncate(dataframe, max_sequence_length)
            dataframe = self.one_hot_encode(dataframe, "sequence")
            dataframe = dataframe[["sequence", "label"]]

        elif model_name == "Xformer":
            dataframe = self.pad_or_truncate(dataframe, max_sequence_length)
            dataframe = self.tokenize(dataframe, "sequence")
            dataframe = dataframe[["sequence", "label"]]

        elif model_name == "FineTunedChordMixer":
            dataframe["len"] = dataframe["sequence"].apply(lambda x: len(x))
            dataframe = self.get_bins(dataframe)
            dataframe = self.one_hot_encode(dataframe, "sequence")
            dataframe = dataframe[["sequence", "len", "bin", "label"]]
        else:
            raise ValueError(f"Model: {model_name} not supported")

        return dataframe

    def process_human_variant_effect_prediction_dataframe(
            self,
            dataframe: pd.DataFrame,
            model_name: str) -> pd.DataFrame:
        """
        Process the human variant effect prediction dataset for a specific model

        Args:
            dataframe: dataframe to process
            model_name: model

        Returns:
            processed dataframe

        Raises:
            ValueError: if model is not supported
        """
        if model_name in ["ChordMixer", "Xformer"]:
            dataframe = self.tokenize(dataframe, "reference")
            dataframe = self.tokenize(dataframe, "alternate")
            dataframe = dataframe[["reference", "alternate", "tissue", "label"]]

        elif model_name == "KeGRU":
            dataframe = dataframe[["reference", "alternate", "tissue", "label"]]

        elif model_name in ["CNN", "FineTunedChordMixer"]:
            dataframe = self.one_hot_encode(dataframe, "reference")
            dataframe = self.one_hot_encode(dataframe, "alternate")
            dataframe = dataframe[["reference", "alternate", "tissue", "label"]]

        else:
            raise ValueError(f"Model: {model_name} not supported")

        return dataframe

    def process_plant_variant_effect_prediction_dataframe(
            self,
            dataframe: pd.DataFrame,
            model_name: str) -> pd.DataFrame:
        """
        Process the plant variant effect prediction dataset for a specific model

        Args:
            dataframe: dataframe to process
            model_name: name of the model

        Returns:
            processed dataframe

        Raises:
            ValueError: if model is not supported
        """
        if model_name in ["ChordMixer", "Xformer"]:
            dataframe = self.tokenize(dataframe, "sequence")

        elif model_name in ["CNN", "FineTunedChordMixer"]:
            dataframe = self.one_hot_encode(dataframe, "sequence")
        
        else:
            raise ValueError(f"Model: {model_name} not supported")
        
        return dataframe
