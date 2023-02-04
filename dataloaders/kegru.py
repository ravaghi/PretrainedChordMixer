from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import gensim
import torch
import os

from .dataloader import Dataloader


class DatasetCreator(Dataset):
    def __init__(self, data, embeddings, kmer_length, stride):
        self.kmer_length = kmer_length
        self.stride = stride
        self.embeddings = embeddings

        kmers = self.generate_kmers(data)
        kmer_indices = [self.convert_data_to_index(kmer) for kmer in kmers]
        kmer_indices = np.asarray(kmer_indices)
        self.idx_data = torch.LongTensor(kmer_indices)
        self.labels = torch.from_numpy(np.asarray([[y] for y in data.label.to_list()], dtype=np.float32))
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

class VEPDatasetCreator(Dataset):
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

        self.labels = torch.from_numpy(np.asarray([[y] for y in dataframe.label.to_list()], dtype=np.float32))
        self.tissues = torch.from_numpy(np.asarray([[y] for y in dataframe.tissue.to_list()], dtype=np.int64))
        self.len = len(self.labels)

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


class KeGruDataLoader(Dataloader):
    def __init__(self, kmer_length, stride, embedding_size, kmer_embedding_path, kmer_embedding_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kmer_length = kmer_length
        self.stride = stride
        self.embedding_size = embedding_size
        self.kmer_embedding_path = kmer_embedding_path
        self.kmer_embedding_name = kmer_embedding_name

    def create_dataloader(self):
        data_path = os.path.join(self.data_path, self.dataset_filename)
        dataframe = pd.read_csv(data_path)[:1000]

        # Load pretrained embeddings
        model_path = os.path.join(self.kmer_embedding_path, self.kmer_embedding_name)
        model = gensim.models.Word2Vec.load(model_path)
        word_vectors = model.wv

        if self.dataset_type == "TaxonomyClassification":
            dataframe = self.process_taxonomy_classification_dataframe(dataframe, "KeGRU")
            dataset = DatasetCreator(dataframe, word_vectors, self.kmer_length, self.stride)

        elif self.dataset_type == "VariantEffectPrediction":
            dataframe = self.process_variant_effect_prediction_dataframe(dataframe, "KeGRU")
            dataset = VEPDatasetCreator(dataframe, word_vectors, self.kmer_length, self.stride)

        elif self.dataset_type == "PlantDeepSEA":
            pass
            
        else:
            raise ValueError(f"Dataset type {self.dataset_type} not supported.")

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )
