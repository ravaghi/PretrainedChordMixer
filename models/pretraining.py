import math
import torch
import numpy as np
from torch import nn

from .chordmixer import ChordMixerBlock


class ChordMixerEncoder(nn.Module):
    """ChordMixerEncoder, to be used as a pretrained model in subsequent downstream tasks."""
    def __init__(self, vocab_size, n_blocks, track_size, hidden_size, prelinear_out_features, mlp_dropout, layer_dropout, variable_length=False):
        super(ChordMixerEncoder, self).__init__()
        self.variable_length = variable_length
        self.n_blocks = n_blocks
        n_tracks = n_blocks

        self.prelinear = nn.Linear(vocab_size, prelinear_out_features)
        self.chordmixer_blocks = nn.ModuleList(
            [
                ChordMixerBlock(prelinear_out_features, n_tracks, track_size, hidden_size, mlp_dropout, layer_dropout)
                for _ in range(n_blocks)
            ]
        )

    def forward(self, data, lengths=None):
        if lengths:
            n_layers = math.ceil(np.log2(lengths[0]))
        else:
            n_layers = self.n_blocks

        data = self.prelinear(data)
        for layer in range(n_layers):
            data = self.chordmixer_blocks[layer](data, lengths)

        if lengths:
            data = [torch.mean(t, dim=0) for t in torch.split(data, lengths)]
            data = torch.stack(data)

        return data


class ChordMixerDecoder(nn.Module):
    def __init__(self, vocab_size, n_blocks, track_size, hidden_size, prelinear_in_features, prelinear_out_features, mlp_dropout, layer_dropout):
        super(ChordMixerDecoder, self).__init__()
        self.n_blocks = n_blocks
        n_tracks = n_blocks

        self.prelinear = nn.Linear(prelinear_in_features, prelinear_out_features)
        self.chordmixer_blocks = nn.ModuleList(
            [
                ChordMixerBlock(prelinear_out_features, n_tracks, track_size, hidden_size, mlp_dropout, layer_dropout)
                for _ in range(n_blocks)
            ]
        )
        self.mlm_classifier = nn.Linear(prelinear_out_features, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, data):
        data = self.prelinear(data)
        for layer in range(self.n_blocks):
            data = self.chordmixer_blocks[layer](data)
        data = self.mlm_classifier(data)
        data = self.softmax(data)
        return data


class PretrainedChordMixer(nn.Module):
    def __init__(self,
                 vocab_size,
                 n_blocks,
                 encoder_track_size,
                 decoder_track_size,
                 encoder_prelinear_out_features,
                 decoder_prelinear_in_features,
                 decoder_prelinear_out_features,
                 hidden_size,
                 mlp_dropout,
                 layer_dropout):
        super(PretrainedChordMixer, self).__init__()
        self.encoder = ChordMixerEncoder(vocab_size,
                                         n_blocks,
                                         encoder_track_size,
                                         hidden_size,
                                         encoder_prelinear_out_features,
                                         mlp_dropout,
                                         layer_dropout)
        self.decoder = ChordMixerDecoder(vocab_size,
                                         n_blocks,
                                         decoder_track_size,
                                         hidden_size,
                                         decoder_prelinear_in_features,
                                         decoder_prelinear_out_features,
                                         mlp_dropout,
                                         layer_dropout)

    def forward(self, sequence_ids):
        encoded = self.encoder(sequence_ids)
        decoded = self.decoder(encoded)
        return decoded
