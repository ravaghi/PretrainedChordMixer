from kipoiseq import Interval, Variant
from kipoiseq.extractors import VariantSeqExtractor
import pandas as pd
import pyfaidx
import csv


class FastaStringExtractor:
    def __init__(self, fasta_file: str):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        """
        Extracts the sequence from the interval

        Args:
            interval: Interval object
            **kwargs: additional arguments

        Returns:
            str: sequence
        """
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(
            interval.chrom,
            max(interval.start, 0),
            min(interval.end, chromosome_length),
        )
        sequence = self.fasta.get_seq(
            trimmed_interval.chrom, trimmed_interval.start + 1, trimmed_interval.stop
        ).seq
        sequence = str(sequence).upper()
        pad_upstream = "N" * max(-interval.start, 0)
        pad_downstream = "N" * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()


def variant_sequence_extractor(
        chr: int,
        pos: int,
        ref: int,
        alt: int,
        length: int,
        fasta_extractor: FastaStringExtractor
) -> (str, str):
    """
    Extracts the reference and alternate sequences of a variant

    Args:
        chr: chromosome number
        pos: position of the variant
        ref: reference sequence position
        alt: alternate sequence position
        length: length of the sequence
        fasta_extractor: FastaStringExtractor object

    Returns:
        (str, str): reference and alternate sequences
    """
    variant = Variant(chr, pos, ref, alt)
    interval = Interval(variant.chrom, variant.start, variant.start).resize(length)
    seq_extractor = VariantSeqExtractor(reference_sequence=fasta_extractor)
    center = interval.center() - interval.start
    reference = seq_extractor.extract(interval, [], anchor=center)
    alternate = seq_extractor.extract(interval, [variant], anchor=center)
    return reference, alternate


def get_data(dataframe: pd.DataFrame, length: int, fasta_extractor: FastaStringExtractor) -> (list, list, list, list):
    """
    Extracts the reference and alternate sequences of a variant

    Args:
        dataframe: dataframe containing the variants
        length: length of the sequence
        fasta_extractor: FastaStringExtractor object

    Returns:
        (list, list, list, list): reference and alternate sequences
    """
    ref_all = []
    alt_all = []
    tissue_all = []
    label_all = []
    for _, seq in dataframe.iterrows():
        ref, alt = variant_sequence_extractor(
            seq["chr"], seq["pos"], seq["ref"], seq["alt"], length, fasta_extractor
        )
        tissue = seq["tissue"]
        label = seq["label"]
        ref_all.append(ref)
        alt_all.append(alt)
        tissue_all.append(tissue)
        label_all.append(label)

    return ref_all, alt_all, tissue_all, label_all


fasta_extractor = FastaStringExtractor("hg38.fa")
cols = ["reference", "alternate", "tissue", "label"]

for dataset in ["train", "val", "test"]:
    dataframe = pd.read_csv(f"label811_49_{dataset}.csv")
    data = get_data(dataframe, 1000, fasta_extractor)

    with open(f"variant_effect_prediction_{dataset}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(zip(*data))
