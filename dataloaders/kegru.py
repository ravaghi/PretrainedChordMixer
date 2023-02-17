from torch.utils.data import Dataset, DataLoader
from gensim.models.word2vec import Word2Vec, KeyedVectors
from typing import List
import pandas as pd
import gensim
import torch
import os

from .preprocessor.preprocessor import Preprocessor


def convert_data_to_index(kmers: List, embeddings: KeyedVectors) -> List:
    """
    Converts a list of kmer sequences to a list of indices.

    Args:
        kmers: list of kmer sequences
        embeddings: pretrained embeddings

    Returns:
        list of indices
    """
    idx_data = []
    for kmer in kmers:
        if kmer in embeddings:
            idx_data.append(embeddings.key_to_index[kmer])
        else:
            idx_data.append(1)
    return idx_data


def generate_kmers(sequences: List, kmer_length: int, stride: int) -> List:
    """
    Generates kmers from a list of sequences.

    Args:
        sequences: list of sequences
        kmer_length: length of the kmers
        stride: stride of the kmers

    Returns:
        list of kmers
    """
    kmers = []
    for sequence in sequences:
        temp_kmers = []
        for i in range(0, (len(sequence) - kmer_length) + 1, stride):
            temp_kmers.append(sequence[i:i + kmer_length])
        kmers.append(temp_kmers)
    return kmers


class TaxonomyClassificationDataset(Dataset):
    """ Taxonomy Classification Dataset """

    def __init__(self, dataframe, embeddings, kmer_length, stride):
        self.embeddings = embeddings
        self.kmer_length = kmer_length
        self.stride = stride

        sequences = dataframe.sequence.values
        sequences = generate_kmers(sequences=sequences, kmer_length=kmer_length, stride=stride)
        self.sequences = [convert_data_to_index(kmers=kmer, embeddings=embeddings) for kmer in sequences]

        self.labels = dataframe.label.values

    def __getitem__(self, index):
        sequence = torch.tensor(self.sequences[index])
        label = torch.tensor(self.labels[index])
        return sequence, label

    def __len__(self):
        return len(self.labels)


class HumanVariantEffectPredictionDataset(Dataset):
    """ Human Variant Effect Prediction Dataset """

    def __init__(self, dataframe, embeddings, kmer_length, stride):
        self.embeddings = embeddings
        self.kmer_length = kmer_length
        self.stride = stride

        references = dataframe.reference.values
        references = generate_kmers(sequences=references, kmer_length=kmer_length, stride=stride)
        self.references = [convert_data_to_index(kmers=kmer, embeddings=embeddings) for kmer in references]

        alternates = dataframe.alternate.values
        alternates = generate_kmers(sequences=alternates, kmer_length=kmer_length, stride=stride)
        self.alternates = [convert_data_to_index(kmers=kmer, embeddings=embeddings) for kmer in alternates]

        self.tissues = dataframe.tissue.values
        self.labels = dataframe.label.values

    def __getitem__(self, index):
        reference = torch.tensor(self.references[index])
        alternate = torch.tensor(self.alternates[index])
        tissue = torch.tensor(self.tissues[index])
        label = torch.tensor(self.labels[index])
        return reference, alternate, tissue, label

    def __len__(self):
        return len(self.labels)


class PlantVariantEffectPredictionDataset(Dataset):
    """ Plant Variant Effect Prediction Dataset """

    def __init__(self, dataframe, embeddings, kmer_length, stride):
        self.embeddings = embeddings
        self.kmer_length = kmer_length
        self.stride = stride

        sequences = dataframe.sequence.values
        sequences = generate_kmers(sequences=sequences, kmer_length=kmer_length, stride=stride)
        self.sequences = [convert_data_to_index(kmers=kmer, embeddings=embeddings) for kmer in sequences]

        target_list = dataframe.columns.tolist()[1:]
        self.labels = dataframe[target_list].values

    def __getitem__(self, index):
        sequence = torch.tensor(self.sequences[index])
        label = torch.tensor(self.labels[index])
        return sequence, label

    def __len__(self):
        return len(self.labels)


class KeGruDataLoader(Preprocessor):
    """KeGRU data loader class"""

    def __init__(self,
                 batch_size,
                 data_path,
                 dataset_type,
                 dataset_name,
                 train_dataset,
                 val_dataset,
                 test_dataset,
                 kmer_length,
                 stride,
                 embedding_size,
                 kmer_embedding_path,
                 kmer_embedding_name
                 ):
        self.batch_size = batch_size
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.dataset_name = dataset_name
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.kmer_length = kmer_length
        self.stride = stride
        self.embedding_size = embedding_size
        self.kmer_embedding_path = kmer_embedding_path
        self.kmer_embedding_name = kmer_embedding_name

    def read_data(self, filename: str) -> pd.DataFrame:
        """
        Read data from a parquet file

        Args:
            filename: name of the dataset file

        Returns:
            dataframe

        Raises:
            FileNotFoundError: if the file is not found
        """
        path = os.path.join(self.data_path, filename)
        if os.path.exists(path):
            dataframe = pd.read_parquet(path)
        else:
            raise FileNotFoundError(f"File {path} not found.")
        return dataframe

    def create_taxonomy_classification_dataloader(self, dataframe: pd.DataFrame,
                                                  word_vectors: KeyedVectors) -> DataLoader:
        """
        Process taxonomy classification dataset and create a dataloader

        Args:
            dataframe: dataframe containing the dataset

        Returns:
            dataloader
        """
        dataframe = self.process_taxonomy_classification_dataframe(dataframe=dataframe, model_name="KeGRU")
        dataset = TaxonomyClassificationDataset(
            dataframe=dataframe,
            embeddings=word_vectors,
            kmer_length=self.kmer_length,
            stride=self.stride
        )

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

    def create_human_variant_effect_prediction_dataloader(self,
                                                          dataframe: pd.DataFrame,
                                                          word_vectors: KeyedVectors) -> DataLoader:
        """
        Process human variant effect prediction dataset and create a dataloader

        Args:
            dataframe: dataframe containing the dataset

        Returns:
            dataloader
        """
        dataframe = self.process_human_variant_effect_prediction_dataframe(dataframe=dataframe, model_name="KeGRU")
        dataset = HumanVariantEffectPredictionDataset(
            dataframe=dataframe,
            embeddings=word_vectors,
            kmer_length=self.kmer_length,
            stride=self.stride
        )

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

    def create_plant_variant_effect_prediction_dataloader(self,
                                                          dataframe: pd.DataFrame,
                                                          word_vectors: KeyedVectors) -> DataLoader:
        """
        Process plant variant effect prediction dataset and create a dataloader

        Args:
            dataframe: dataframe containing the dataset

        Returns:
            dataloader
        """
        dataset = PlantVariantEffectPredictionDataset(
            dataframe=dataframe,
            embeddings=word_vectors,
            kmer_length=self.kmer_length,
            stride=self.stride
        )

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

    def create_dataloaders(self):
        """
        Create dataloaders for the train, validation and test sets

        Returns:
            train, validation and test dataloaders

        Raises:
            ValueError: if the dataset type is not supported
        """
        train_dataframe = self.read_data(self.train_dataset)
        val_dataframe = self.read_data(self.val_dataset)
        test_dataframe = self.read_data(self.test_dataset)

        # Load pretrained embeddings
        model_path = os.path.join(self.kmer_embedding_path, self.kmer_embedding_name)
        model = gensim.models.Word2Vec.load(model_path)
        word_vectors = model.wv

        if self.dataset_type == "TaxonomyClassification":
            train_dataloader = self.create_taxonomy_classification_dataloader(train_dataframe, word_vectors)
            val_dataloader = self.create_taxonomy_classification_dataloader(val_dataframe, word_vectors)
            test_dataloader = self.create_taxonomy_classification_dataloader(test_dataframe, word_vectors)
        elif self.dataset_type == "HumanVariantEffectPrediction":
            train_dataloader = self.create_human_variant_effect_prediction_dataloader(train_dataframe, word_vectors)
            val_dataloader = self.create_human_variant_effect_prediction_dataloader(val_dataframe, word_vectors)
            test_dataloader = self.create_human_variant_effect_prediction_dataloader(test_dataframe, word_vectors)
        elif self.dataset_type == "PlantVariantEffectPrediction":
            train_dataloader = self.create_plant_variant_effect_prediction_dataloader(train_dataframe, word_vectors)
            val_dataloader = self.create_plant_variant_effect_prediction_dataloader(val_dataframe, word_vectors)
            test_dataloader = self.create_plant_variant_effect_prediction_dataloader(test_dataframe, word_vectors)
        else:
            raise ValueError(f"Dataset type {self.dataset_type} not supported.")

        return train_dataloader, val_dataloader, test_dataloader
