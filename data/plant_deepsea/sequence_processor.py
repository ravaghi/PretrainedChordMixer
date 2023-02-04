import tabix
import pyfaidx
from abc import ABCMeta
from _cython_genome import _fast_sequence_to_encoding, _fast_get_feature_data


def get_chrs_len(genome, chrs):
    len_chrs = {}
    for chrom in chrs:
        len_chrs[chrom] = len(genome[chrom])
    return len_chrs


def check_sequence(chrs_len, chrom, start, end):
    return chrom in chrs_len and \
        0 <= start < chrs_len[chrom] and \
        start < end and \
        0 < end <= chrs_len[chrom]


def get_sequence_from_coords(genome, chrom, start, end, strand):
    if strand == '+':
        return genome[chrom][start:end].seq
    elif strand == '-':
        return genome[chrom][start:end].reverse.complement.seq
    else:
        raise ValueError('Strand must be + or -, not {}.'.format(strand))


def sequence_to_encoding(sequence, base_to_index, bases_arr):
    return _fast_sequence_to_encoding(sequence, base_to_index, len(bases_arr))


def get_feature_name(feature_path):
    features = []
    with open(feature_path, 'r') as f:
        for line in f:
            features.append(line.split('\n')[0])
    f.close()
    return features


def get_feature_data(chrom, start, end, feature_index_dict, get_feature_rows):
    rows = get_feature_rows(chrom, start, end)
    return _fast_get_feature_data(start, end, feature_index_dict, rows)


class GenomeSequence(metaclass=ABCMeta):
    def __init__(self, fasta_path):
        self.BASES_ARR = ['A', 'C', 'G', 'T']
        self.BASE_TO_INDEX = {
            'A': 0, 'C': 1, 'G': 2, 'T': 3,
            'a': 0, 'c': 1, 'g': 2, 't': 3,
        }

        self.genome = pyfaidx.Fasta(fasta_path)
        self.chrs_len = get_chrs_len(self.genome, sorted(self.genome.keys()))

    def sequence_from_coords(self, chrom, start, end, strand):
        if check_sequence(self.chrs_len, chrom, start, end):
            return get_sequence_from_coords(self.genome, chrom, start, end, strand)
        else:
            print('{}:{}-{} is out of boundary'.format(chrom, start, end))
            if start < 0:
                return 'n' * -start + get_sequence_from_coords(self.genome, chrom, 0, end, strand)
            else:
                return get_sequence_from_coords(self.genome, chrom, start, self.chrs_len[chrom], strand) + 'n' * (
                        end - self.chrs_len[chrom])

    def sequence_element_count(self, chrom, start, end, strand):
        sequence = self.sequence_from_coords(chrom, start, end, strand)
        elements = sorted(set(sequence))
        counts = {}
        for e in elements:
            counts[e] = sequence.count(e)
        return counts

    def encoding_from_coords(self, chrom, start, end, strand):
        sequence = self.sequence_from_coords(chrom, start, end, strand)
        encoding = sequence_to_encoding(sequence, self.BASE_TO_INDEX, self.BASES_ARR)
        return encoding


class GenomicFeatures(metaclass=ABCMeta):
    def __init__(self, bedgz_path, feature_path):
        self.data = tabix.open(bedgz_path)
        self.feature_index_dict = dict(
            [(feat, index) for index, feat in enumerate(get_feature_name(feature_path))])

    def _query_tabix(self, chrom, start, end):
        return self.data.query(chrom, start, end)

    def feature_data(self, chrom, start, end):
        return get_feature_data(chrom, start, end, self.feature_index_dict, self._query_tabix)
