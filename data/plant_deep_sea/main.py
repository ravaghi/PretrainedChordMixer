from sequence_processor import GenomeSequence, GenomicFeatures
from sampler import BedFileSampler
import numpy as np
import csv
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

dataset_metadata = [
    {
        "species": "Arabidopsis Thaliana",
        "reference_genome": "Tair.fa",
        "features": "distinct_features.txt",
        "data": "sorted_Dnase_rm_low_and_ATAC_all_peak_sorted.bed.gz",
        "model": {
            "train": "train_data.bed",
            "val": "validate_data.bed",
            "test": "test_data.bed"
        }
    },
    {
        "species": "Oryza Sativa MH",
        "reference_genome": "MH63.fasta",
        "features": "distinct_features.txt",
        "data": "sorted_mh.bed.gz",
        "model": {
            "train": "train_data.bed",
            "val": "validate_data.bed",
            "test": "test_data.bed"
        }
    },
    {
        "species": "Oryza Sativa ZS",
        "reference_genome": "ZS97.fasta",
        "features": "distinct_features.txt",
        "data": "sorted_zs97_15tissues.bed.gz",
        "model": {
            "train": "train_data.bed",
            "val": "validate_data.bed",
            "test": "test_data.bed"
        }
    },
    {
        "species": "Brachypodium Distachyon",
        "reference_genome": "Bdistachyon.fasta",
        "features": "distinct_features.txt",
        "data": "sorted_bd.bed.gz",
        "model": {
            "train": "train_data.bed",
            "val": "validate_data.bed",
            "test": "test_data.bed"
        }
    },
    {
        "species": "Setaria Italica",
        "reference_genome": "Sitalica.fasta",
        "features": "distinct_features.txt",
        "data": "sorted_si.bed.gz",
        "model": {
            "train": "train_data.bed",
            "val": "validate_data.bed",
            "test": "test_data.bed"
        }
    },
    {
        "species": "Sorghum Bicolor",
        "reference_genome": "Sbicolor.fasta",
        "features": "distinct_features.txt",
        "data": "sorted_sb.bed.gz",
        "model": {
            "train": "train_data.bed",
            "val": "validate_data.bed",
            "test": "test_data.bed"
        }
    },
    {
        "species": "Zea Mays",
        "reference_genome": "Zmays.fasta",
        "features": "distinct_features.txt",
        "data": "sorted_zm.bed.gz",
        "model": {
            "train": "train_data.bed",
            "val": "validate_data.bed",
            "test": "test_data.bed"
        }
    },
]


def create_dataset(fields: list, dataset: list, dataset_type: str, dataset_name: str) -> None:
    """
    Creates and saves a csv file for the given dataset.

    Args:
        fields: List of column headers
        dataset: List of data points
        dataset_type: Type of dataset (train, val, test)
        dataset_name: Name of dataset (e.g. Arabidopsis Thaliana)

    Returns:
        None
    """
    print("-- Creating {}_{}.csv".format(dataset_name, dataset_type))
    rows = []
    for batch in dataset:
        X = batch[0]
        Y = batch[1]

        for x, y in zip(X, Y):
            rows.append([x.upper()] + y.astype(int).tolist())

    if not os.path.exists(f"../{dataset_name}"):
        os.mkdir(f"../{dataset_name}")

    with open(f"../{dataset_name}/{dataset_name}_{dataset_type}.csv", "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)


for metadata in dataset_metadata:
    print("\nProcessing {}".format(metadata["species"]))

    folder_name = metadata["species"].replace(" ", "_").lower()

    reference_genome_path = os.path.join(BASE_DIR, 'processing/reference_genomes', metadata["reference_genome"])
    data_path = os.path.join(BASE_DIR, "processing/data", folder_name, metadata["data"])
    feature_path = os.path.join(BASE_DIR, "processing/data", folder_name, metadata["features"])

    sequences = GenomeSequence(reference_genome_path)
    features = GenomicFeatures(data_path, feature_path)

    train_path = os.path.join(BASE_DIR, "processing/models", folder_name, metadata["model"]["train"])
    val_path = os.path.join(BASE_DIR, "processing/models", folder_name, metadata["model"]["val"])
    test_path = os.path.join(BASE_DIR, "processing/models", folder_name, metadata["model"]["test"])

    train_sampler = BedFileSampler(train_path, sequences, features)
    val_sampler = BedFileSampler(val_path, sequences, features)
    test_sampler = BedFileSampler(test_path, sequences, features)

    batch_size = 250
    train_size = 80_000
    val_size = 10_000
    test_size = 10_000

    print("- Sampling training data")
    train_data = train_sampler.get_data_and_targets(batch_size, train_size, encoding=False)
    print("- Sampling validation data")
    val_data = val_sampler.get_data_and_targets(batch_size, val_size, encoding=False)
    print("- Sampling test data")
    test_data = test_sampler.get_data_and_targets(batch_size, test_size, encoding=False)

    fields = ["sequence"] + np.loadtxt(feature_path, dtype=str).tolist()

    create_dataset(fields, train_data, "train", folder_name)
    create_dataset(fields, val_data, "val", folder_name)
    create_dataset(fields, test_data, "test", folder_name)
