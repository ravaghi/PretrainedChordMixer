from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import gensim
import torch
import os
from gensim.models.word2vec import Word2Vec

from .dataloader import Dataloader


class TaxonomyClassificationDataset(Dataset):
    def __init__(self, data, embeddings, kmer_length, stride):
        self.kmer_length = kmer_length
        self.stride = stride
        self.embeddings = embeddings

        kmers = self.generate_kmers(data)
        kmer_indices = [self.convert_data_to_index(kmer) for kmer in kmers]
        kmer_indices = np.asarray(kmer_indices)
        self.idx_data = torch.LongTensor(kmer_indices)
        self.labels = torch.from_numpy(np.asarray([y for y in data.label.to_list()]))
        self.len = len(self.idx_data)

    def __getitem__(self, index):
        return self.idx_data[index], self.labels[index]

    def __len__(self):
        return self.len

    def convert_data_to_index(self, kmers):
        idx_data = []
        for kmer in kmers:
            if kmer in self.embeddings:
                idx_data.append(self.embeddings.key_to_index[kmer])
            else:
                idx_data.append(1)
        return idx_data

    def generate_kmers(self, data):
        sequences = data.sequence.to_list()
        kmers = []
        for seq in sequences:
            temp_kmers = []
            for i in range(0, (len(seq) - self.kmer_length) + 1, self.stride):
                temp_kmers.append(seq[i:i + self.kmer_length])
            kmers.append(temp_kmers)
        return kmers

class VariantEffectPredictionDataset(Dataset):
    def __init__(self, dataframe, embeddings, kmer_length, stride):
        self.kmer_length = kmer_length
        self.stride = stride
        self.embeddings = embeddings

        references = dataframe.reference.to_list()
        reference_kmers = self.generate_kmers(references)
        reference_kmer_indices = [self.convert_data_to_index(kmer) for kmer in reference_kmers]
        reference_kmer_indices = np.asarray(reference_kmer_indices)
        self.reference_kmer_indices = torch.LongTensor(reference_kmer_indices)

        alternatives = dataframe.alternate.to_list()
        alternative_kmers = self.generate_kmers(alternatives)
        alternative_kmer_indices = [self.convert_data_to_index(kmer) for kmer in alternative_kmers]
        alternative_kmer_indices = np.asarray(alternative_kmer_indices)
        self.alternative_kmer_indices = torch.LongTensor(alternative_kmer_indices)

        self.labels = torch.from_numpy(np.asarray([y for y in dataframe.label.to_list()]))
        self.tissues = torch.from_numpy(np.asarray([y for y in dataframe.tissue.to_list()]))
        self.len = len(self.alternative_kmer_indices)

    def __getitem__(self, index):
        return self.reference_kmer_indices[index], self.alternative_kmer_indices[index], self.tissues[index], self.labels[index]

    def __len__(self):
        return self.len

    def convert_data_to_index(self, kmers):
        idx_data = []
        for kmer in kmers:
            if kmer in self.embeddings:
                idx_data.append(self.embeddings.key_to_index[kmer])
            else:
                idx_data.append(1)
        return idx_data

    def generate_kmers(self, sequences):
        kmers = []
        for seq in sequences:
            temp_kmers = []
            for i in range(0, (len(seq) - self.kmer_length) + 1, self.stride):
                temp_kmers.append(seq[i:i + self.kmer_length])
            kmers.append(temp_kmers)
        return kmers


class PlantDeepSEADataset(Dataset):
    def __init__(self, dataframe, embeddings, kmer_length, stride):
        self.kmer_length = kmer_length
        self.stride = stride
        self.embeddings = embeddings

        kmers = self.generate_kmers(dataframe)
        kmer_indices = [self.convert_data_to_index(kmer) for kmer in kmers]
        kmer_indices = np.asarray(kmer_indices)
        self.idx_data = torch.LongTensor(kmer_indices)
        
        self.len = len(self.idx_data)

        target_list = dataframe.columns.tolist()[1:]
        self.targets = dataframe[target_list].values

    def __getitem__(self, index):
        x = self.idx_data[index]
        y = self.targets[index]
        y = torch.from_numpy(y).float()
        return x, y

    def __len__(self):
        return self.len

    def convert_data_to_index(self, kmers):
        idx_data = []
        for kmer in kmers:
            if kmer in self.embeddings:
                idx_data.append(self.embeddings.key_to_index[kmer])
            else:
                idx_data.append(1)
        return idx_data

    def generate_kmers(self, data):
        sequences = data.sequence.to_list()
        kmers = []
        for seq in sequences:
            temp_kmers = []
            for i in range(0, (len(seq) - self.kmer_length) + 1, self.stride):
                temp_kmers.append(seq[i:i + self.kmer_length])
            kmers.append(temp_kmers)
        return kmers


class KeGruDataLoader(Dataloader):
    def __init__(self, kmer_length, stride, embedding_size, kmer_embedding_path, kmer_embedding_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kmer_length = kmer_length
        self.stride = stride
        self.embedding_size = embedding_size
        self.kmer_embedding_path = kmer_embedding_path
        self.kmer_embedding_name = kmer_embedding_name

    def _create_taxonomic_classification_dataloader(self, dataframe: pd.DataFrame, word_vectors: Word2Vec) -> DataLoader:
        dataframe = self.process_taxonomy_classification_dataframe(dataframe, "KeGRU")
        dataset = TaxonomyClassificationDataset(dataframe, word_vectors, self.kmer_length, self.stride)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )

    def _create_variant_effect_prediction_dataloader(self, dataframe: pd.DataFrame, word_vectors: Word2Vec) -> DataLoader:
        dataframe = self.process_variant_effect_prediction_dataframe(dataframe, "KeGRU")
        dataset = VariantEffectPredictionDataset(dataframe, word_vectors, self.kmer_length, self.stride)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )

    def _create_plant_deepsea_dataloader(self, dataframe: pd.DataFrame, word_vectors: Word2Vec) -> DataLoader:
        dataset = PlantDeepSEADataset(dataframe, word_vectors, self.kmer_length, self.stride)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )

    def create_dataloaders(self):
        train_dataframe = self.read_data(self.train_dataset)
        val_dataframe = self.read_data(self.val_dataset)
        test_dataframe = self.read_data(self.test_dataset)

        # Load pretrained embeddings
        model_path = os.path.join(self.kmer_embedding_path, self.kmer_embedding_name)
        model = gensim.models.Word2Vec.load(model_path)
        word_vectors = model.wv

        if self.dataset_type == "TaxonomyClassification":
            train_dataloader = self._create_taxonomic_classification_dataloader(train_dataframe, word_vectors)
            val_dataloader = self._create_taxonomic_classification_dataloader(val_dataframe, word_vectors)
            test_dataloader = self._create_taxonomic_classification_dataloader(test_dataframe, word_vectors)
        elif self.dataset_type == "VariantEffectPrediction":
            train_dataloader = self._create_variant_effect_prediction_dataloader(train_dataframe, word_vectors)
            val_dataloader = self._create_variant_effect_prediction_dataloader(val_dataframe, word_vectors)
            test_dataloader = self._create_variant_effect_prediction_dataloader(test_dataframe, word_vectors)
        elif self.dataset_type == "PlantDeepSEA":
            train_dataloader = self._create_plant_deepsea_dataloader(train_dataframe, word_vectors)
            val_dataloader = self._create_plant_deepsea_dataloader(val_dataframe, word_vectors)
            test_dataloader = self._create_plant_deepsea_dataloader(test_dataframe, word_vectors)
        else:
            raise ValueError(f"Dataset type {self.dataset_type} not supported.")

        return train_dataloader, val_dataloader, test_dataloader
