from kipoiseq import Interval, Variant
from kipoiseq.extractors import VariantSeqExtractor
import pandas as pd
import pyfaidx
from typing import Tuple, List


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


def variant_sequence_extractor(chromosome: str,
                               position: int,
                               reference: str,
                               alternate: str,
                               length: int,
                               fasta_extractor: FastaStringExtractor
                               ) -> Tuple[str, str]:
    """
    Extracts the reference and alternate sequences of a variant

    Args:
        chromosome: chromosome name
        position: position of the variant
        reference: reference base
        alternate: alternate base
        length: length of the sequence
        fasta_extractor: FastaStringExtractor object

    Returns:
        (str, str): reference and alternate sequences
    """
    variant = Variant(chromosome, position, reference, alternate)
    interval = Interval(variant.chrom, variant.start, variant.start).resize(length)
    seq_extractor = VariantSeqExtractor(reference_sequence=fasta_extractor)
    center = interval.center() - interval.start
    reference = seq_extractor.extract(interval, [], anchor=center)
    alternate = seq_extractor.extract(interval, [variant], anchor=center)
    return reference, alternate


def sample_data(dataframe: pd.DataFrame,
                length: int,
                fasta_extractor: FastaStringExtractor
                ) -> Tuple[List, List, List, List]:
    """
    Extracts the reference and alternate sequences of a variant

    Args:
        dataframe: dataframe containing the variants
        length: length of the sequence
        fasta_extractor: FastaStringExtractor object

    Returns:
        (List, List, List, List): reference, alternate, tissue and label sequences
    """
    references = []
    alternates = []
    tissues = []
    labels = []
    for _, seq in dataframe.iterrows():
        reference, alternate = variant_sequence_extractor(
            seq["chr"],
            seq["pos"],
            seq["ref"],
            seq["alt"],
            length,
            fasta_extractor
        )
        tissue = seq["tissue"]
        label = seq["label"]
        references.append(reference)
        alternates.append(alternate)
        tissues.append(tissue)
        labels.append(label)

    return references, alternates, tissues, labels


if __name__ == "__main__":
    fasta_extractor = FastaStringExtractor("hg38.fa")
    columns = ["reference", "alternate", "tissue", "label"]

    for dataset in ["train", "val", "test"]:
        print(f"Creating homo_sapien_{dataset}.parquet...")
        
        dataframe = pd.read_csv(f"label811_49_{dataset}.csv")
        rows = sample_data(dataframe, 2000, fasta_extractor)

        data = pd.DataFrame(zip(*rows), columns=columns)
        data.to_parquet(f"homo_sapien/homo_sapien_{dataset}.parquet", index=False)
