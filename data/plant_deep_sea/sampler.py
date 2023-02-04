import sys
import numpy as np
from abc import ABCMeta
from _cython_genome import _fast_get_feature_data_thresholds


def _define_feature_thresholds(feature_thresholds, features):
    feature_thresholds_vec = np.zeros(len(features))
    if 0 < feature_thresholds <= 1:
        feature_thresholds_vec += feature_thresholds
    else:
        print('wrong thresholds: {}'.format(feature_thresholds))
        sys.exit()
    return feature_thresholds_vec.astype(np.float32)


class BedFileSampler(metaclass=ABCMeta):
    def __init__(self, filepath, genome_sequence, genome_features):
        super(BedFileSampler, self).__init__()
        self.filepath = filepath
        self._file_handle = open(self.filepath, 'r')
        self.genome_sequence = genome_sequence
        self.genome_features = genome_features

    def sample(self, batch_size, encoding, position, center, thresholds, add):
        sequences = []
        targets = []

        while len(sequences) < batch_size:
            line = self._file_handle.readline()
            if not line:
                self._file_handle.close()
                self._file_handle = open(self.filepath, 'r')
                line = self._file_handle.readline()
            cols = line.split('\n')[0].split('\t')
            chrom = cols[0]
            start = int(cols[1]) - add
            end = int(cols[2]) + add
            strand = cols[3]

            if encoding:
                sequence = self.genome_sequence.encoding_from_coords(chrom, start, end, strand)
            else:
                sequence = self.genome_sequence.sequence_from_coords(chrom, start, end, strand)

            target_position = self.genome_features.feature_data(chrom, start, end)
            new_start = int((end - start - center) / 2)
            if position:
                target = target_position[new_start:new_start + center, :]
            else:
                feature_thresholds_vec = _define_feature_thresholds(thresholds, self.genome_features.feature_index_dict)
                targets_center = target_position[new_start:new_start + center, :]
                target = _fast_get_feature_data_thresholds(targets_center, feature_thresholds_vec, center)

            sequences.append(sequence)
            targets.append(target.astype(float))

        sequences = np.array(sequences)
        targets = np.array(targets)
        return sequences, targets

    def get_data_and_targets(self, batch_size, n_samples, position=False, center=200, thresholds=0.5, drop_last=False,
                             encoding=True, add=0):
        sequences_and_targets = []
        count = batch_size
        while count < n_samples:
            seqs, tgts = self.sample(batch_size, encoding, position, center, thresholds, add)
            sequences_and_targets.append((seqs, tgts))
            count += batch_size
        if drop_last:
            return sequences_and_targets
        else:
            remainder = batch_size - (count - n_samples)
            seqs, tgts = self.sample(remainder, encoding, position, center, thresholds, add)
            sequences_and_targets.append((seqs, tgts))
            return sequences_and_targets
