# cython: language_level=3
import numpy as np

cimport cython
cimport numpy as np

ctypedef np.int_t DTYPE_t
ctypedef np.float32_t FDTYPE_t

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def _fast_sequence_to_encoding(str sequence, dict base_to_index, int bases_size):
    cdef int sequence_len = len(sequence)
    cdef np.ndarray[FDTYPE_t, ndim=2] encoding = np.zeros(
        (sequence_len, bases_size), dtype=np.float32)
    cdef int index
    cdef str base
    cdef FDTYPE_t n_fill = np.divide(1, bases_size, dtype=np.float32)

    for index in range(sequence_len):
        base = sequence[index]
        if base in base_to_index:
            encoding[index, base_to_index[base]] = 1
        else:
            encoding[index, :] = n_fill

    return encoding

def _fast_get_feature_data(int start, int end, dict feature_index_dict, rows):
    cdef int n_features = len(feature_index_dict)
    cdef int query_length = end - start
    cdef int feature_start, feature_end, index_start, index_end, index_feat
    cdef np.ndarray[DTYPE_t, ndim=2] encoding = np.zeros(
        (query_length, n_features), dtype=np.int)
    cdef np.ndarray[DTYPE_t, ndim=1] targets = np.zeros(
        n_features, dtype=np.int)
    cdef list row

    if rows is None:
        return np.zeros((n_features,))

    for row in rows:
        feature_start = int(row[1])
        feature_end = int(row[2])
        index_start = max(0, feature_start - start)
        index_end = min(feature_end - start, query_length)
        index_feat = feature_index_dict[row[3]]
        if index_start == index_end:
            index_end += 1
        encoding[index_start:index_end, index_feat] = 1

    return encoding

def _fast_get_feature_data_thresholds(np.ndarray[DTYPE_t, ndim=2] encoding,
                                      np.ndarray[FDTYPE_t, ndim=1] thresholds,
                                      int query_length):
    thresholds = (thresholds * query_length - 1).clip(min=0)
    targets = (np.sum(encoding, axis=0) > thresholds.astype(int)).astype(int)
    return targets
