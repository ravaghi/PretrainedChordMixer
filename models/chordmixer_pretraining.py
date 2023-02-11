import math
import torch
import numpy as np
from torch import nn
from collections import OrderedDict

from .chordmixer import ChordMixerBlock


class ChordMixerEncoder(nn.Module):
    """ChordMixerEncoder, to be used as a pretrained model in subsequent downstream tasks."""
    def __init__(self, vocab_size, n_blocks, track_size, hidden_size, prelinear_out_features, mlp_dropout, layer_dropout, variable_length=False):
        super(ChordMixerEncoder, self).__init__()
        self.variable_length = variable_length
        self.prelinear_out_features = prelinear_out_features
        self.n_blocks = n_blocks
        n_tracks = n_blocks

        self.prelinear = nn.Linear(vocab_size, prelinear_out_features)
        self.chordmixer_blocks = nn.ModuleList(
            [
                ChordMixerBlock(prelinear_out_features, n_tracks, track_size, hidden_size, mlp_dropout, layer_dropout)
                for _ in range(n_blocks)
            ]
        )

    @staticmethod
    def _get_encoder_state_dict(model):
        state_dict = OrderedDict()
        for key, value in model.items():
            if "encoder" in key:
                new_key = key.replace("module.encoder.", "")
                state_dict[new_key] = value
        return state_dict

    @classmethod
    def from_pretrained(cls, model, freeze=True, variable_length=False):
        """Load a pretrained model from a file."""
        print(f"Loading {model}")
        model = torch.load(model)
        encoder_state_dict = cls._get_encoder_state_dict(model)

        encoder = cls(
            vocab_size=5,
            n_blocks=20,
            track_size=16,
            hidden_size=196,
            prelinear_out_features=1000,
            mlp_dropout=0,
            layer_dropout=0,
            variable_length=variable_length
        )

        encoder.load_state_dict(encoder_state_dict)

        if freeze:
            for param in encoder.parameters():
                param.requires_grad = False

        return encoder

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