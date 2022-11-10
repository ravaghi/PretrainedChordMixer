import multiprocessing
from Bio import SeqIO
from tqdm import tqdm
import gensim
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_sequences_list():
    sequences_list = []
    sequences = SeqIO.parse(BASE_DIR + "/data/variant_effect_prediction/hg38.fa", "fasta")
    print("Reading sequences...")
    for record in sequences:
        sequence_str = str(record.seq).upper()
        if len(sequence_str) < 1000_000:
            sequences_list.append(sequence_str)
        else:
            sequences_list.append(sequence_str[:1000_000])

    return sequences_list


def generate_kmers(sequences, kmer_length, stride):
    kmers = []
    for sequence in tqdm(sequences):
        temp_kmers = []
        for i in range(0, (len(sequence) - kmer_length) + 1, stride):
            temp_kmers.append(sequence[i:i + kmer_length])
        kmers.append(temp_kmers)
    return kmers


def generate_kmer_embeddings(sequences, kmer_length, stride, embedding_size, epochs):
    model_name = "K{}_S{}_L{}.model".format(kmer_length, stride, embedding_size)
    kmers = generate_kmers(sequences, kmer_length, stride)
    model = gensim.models.Word2Vec(kmers, vector_size=embedding_size, workers=multiprocessing.cpu_count() - 1)
    print(f"Training {model_name}...")
    model.train(kmers, total_examples=len(kmers), epochs=epochs)
    model.save(model_name)


if __name__ == "__main__":
    sequences = get_sequences_list()
    for kmer_length in [5, 4, 6]:
        for stride in [2, 1, 3]:
            for embedding_size in [50, 80, 100]:
                model_name = "K{}_S{}_L{}.model".format(kmer_length, stride, embedding_size)
                if os.path.exists(BASE_DIR + "/data/kmer_embeddings/" + model_name):
                    print("{} already exists".format(model_name))
                else:
                    print("Generating {}".format(model_name))
                    generate_kmer_embeddings(sequences, kmer_length, stride, embedding_size, 50)
