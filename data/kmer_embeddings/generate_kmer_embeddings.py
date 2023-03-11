import multiprocessing
from Bio import SeqIO
from tqdm import tqdm
import gensim
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_sequences_list(fasta_file_path: str, max_sequence_length: int) -> list:
    """
    Collects sequences from a fasta file and returns them as a list of strings.

    Args:
        fasta_file_path: The path to the fasta file.
        max_sequence_length: The maximum length of a sequence.

    Returns:
        A list of strings, each string is a sequence.
    """
    sequences = SeqIO.parse(fasta_file_path, "fasta")

    sequences_list = []
    for sequence in sequences:
        sequence_str = str(sequence.seq).upper()
        if len(sequence_str) < max_sequence_length:
            sequences_list.append(sequence_str)
        else:
            sequences_list.append(sequence_str[:max_sequence_length])
    return sequences_list


def generate_kmers(sequences: list, kmer_length: int, stride: int) -> list:
    """
    Generates kmers from a list of sequences.

    Args:
        sequences: A list of strings, each string is a sequence.
        kmer_length: The length of the kmers to generate.
        stride: The stride to use when generating kmers.

    Returns:
        A list of lists, each list contains the kmers for a sequence.
    """
    kmers = []
    for sequence in tqdm(sequences):
        temp_kmers = []
        for i in range(0, (len(sequence) - kmer_length) + 1, stride):
            current_kmer = sequence[i:i + kmer_length]
            if "N" not in current_kmer:
                temp_kmers.append(current_kmer)
        kmers.append(temp_kmers)
    return kmers


def generate_kmer_embeddings(sequences: list, kmer_length: int, stride: int, embedding_size: int, epochs: int) -> None:
    """
    Generates kmer embeddings using gensim's Word2Vec model and save.

    Args:
        sequences: A list of strings, each string is a sequence.
        kmer_length: The length of the kmers to generate.
        stride: The stride to use when generating kmers.
        embedding_size: The size of the embedding vectors.
        epochs: The number of epochs to train the model for.

    Returns:
        None
    """
    model_name = "K{}_S{}_L{}.model".format(kmer_length, stride, embedding_size)
    kmers = generate_kmers(sequences, kmer_length, stride)
    model = gensim.models.Word2Vec(kmers, vector_size=embedding_size, workers=multiprocessing.cpu_count() - 1)
    print(f"Training {model_name}...")
    model.train(kmers, total_examples=len(kmers), epochs=epochs)
    model.save(model_name)


# Best parameters
# Kmer length: 5
# Stride: 2
# Embedding size: 50
if __name__ == "__main__":
    _FASTA_FILE_PATH = BASE_DIR + "/data/variant_effect_prediction/hg38.fa"
    _MAX_SEQUENCE_LENGTH = 10_000_000

    sequences = get_sequences_list(_FASTA_FILE_PATH, _MAX_SEQUENCE_LENGTH)
    for kmer_length in [5, 4, 6]:
        for stride in [2, 1, 3]:
            for embedding_size in [50, 80, 100]:
                model_name = "K{}_S{}_L{}.model".format(kmer_length, stride, embedding_size)
                if os.path.exists(BASE_DIR + "/data/kmer_embeddings/" + model_name):
                    print("{} already exists".format(model_name))
                else:
                    print("Generating {}".format(model_name))
                    generate_kmer_embeddings(sequences, kmer_length, stride, embedding_size, 100)
