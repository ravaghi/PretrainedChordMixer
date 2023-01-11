from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
import pandas as pd
import numpy as np
import gensim
import torch
import math
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))




def pad_sequences(dataframe, strategy="mean"):
    max_seq_len = dataframe["sequence"].apply(lambda x: len(x)).max()
    mean_seq_len = dataframe["sequence"].apply(lambda x: len(x)).mean()
    mean_seq_len = int(math.ceil(mean_seq_len))
    dataframe["sequence"] = dataframe["sequence"].str.pad(max_seq_len, side="right", fillchar="A")

    if strategy == "mean":
        dataframe["sequence"] = dataframe["sequence"].apply(lambda x: x[:mean_seq_len].upper())
    elif strategy == "max":
        dataframe["sequence"] = dataframe["sequence"].apply(lambda x: x.upper())
    elif strategy == "constant":
        dataframe["sequence"] = dataframe["sequence"].apply(lambda x: x[10_000:20_000].upper())
    return dataframe


def process_taxonomy_classification_dataframe(dataframe):
    dataframe["new_sequences"] = dataframe["sequence"].apply(
        lambda x: "".join([DNA_BASE_DICT_REVERSED[base] for base in x]))
    dataframe = dataframe.drop(columns=["sequence", "len", "bin"])
    dataframe = dataframe.rename(columns={"new_sequences": "sequence"})
    dataframe = pad_sequences(dataframe)
    dataframe = dataframe.reset_index(drop=True)
    return dataframe[["sequence", "label"]]


def process_variant_effect_prediction_dataframe(dataframe):
    sequence_path = os.path.join(BASE_DIR, "data", "variant_effect_prediction", "hg38.fa")
    sequences = SeqIO.to_dict(SeqIO.parse(sequence_path, "fasta"))
    dataframe["sequence"] = dataframe.apply(lambda x: str(sequences[x.chr].seq[x.pos - 20_000:x.pos + 20_000]), axis=1)
    dataframe = pad_sequences(dataframe, "constant")
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    return dataframe[["sequence", "label"]]


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


class KeGruDataLoader:
    def __init__(self, data_path, dataset, dataset_name, batch_size, kmer_length, stride, embedding_size,
                 kmer_embedding_path, kmer_embedding_name):
        self.data_path = data_path
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.kmer_length = kmer_length
        self.stride = stride
        self.embedding_size = embedding_size
        self.kmer_embedding_path = kmer_embedding_path
        self.kmer_embedding_name = kmer_embedding_name

    def create_dataloader(self):
        data_path = os.path.join(self.data_path, self.dataset)

        if "taxonomy" in self.dataset_name.lower():
            dataframe = pd.read_pickle(data_path)
            dataframe = process_taxonomy_classification_dataframe(dataframe)
        elif "variant" in self.dataset_name.lower():
            dataframe = pd.read_csv(data_path)
            dataframe = process_variant_effect_prediction_dataframe(dataframe)
        else:
            raise ValueError(f"Dataset {self.dataset_name} name not recognized")

        # Load pretrained embeddings
        model_path = os.path.join(self.kmer_embedding_path, self.kmer_embedding_name)
        model = gensim.models.Word2Vec.load(model_path)
        word_vectors = model.wv

        dataset = DatasetCreator(dataframe, word_vectors, self.kmer_length, self.stride)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )
