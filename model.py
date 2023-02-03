import math
import torch
import numpy as np
from torch import nn

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RotateChord(nn.Module):
    def __init__(self, n_tracks, track_size):
        super(RotateChord, self).__init__()
        self.n_tracks = n_tracks
        self.track_size = track_size

    def forward(self, x, lengths=None):
        if lengths:
            ys = torch.split(tensor=x, split_size_or_sections=lengths, dim=0)
            
            zs = []
            for y in ys:
                y = torch.split(tensor=y, split_size_or_sections=self.track_size, dim=-1)
                z = [y[0]]
                for i in range(1, len(y)):
                    offset = -2 ** (i - 1)
                    z.append(torch.roll(y[i], shifts=offset, dims=0))
                z = torch.cat(z, -1)
                zs.append(z)
            z = torch.cat(zs, 0)
            assert z.shape == x.shape, 'shape mismatch'
        else:
            y = torch.split(tensor=x, split_size_or_sections=self.track_size, dim=-1)
            z = [y[0]]
            for i in range(1, len(y)):
                offset = -2 ** (i - 1)
                z.append(torch.roll(y[i], shifts=offset, dims=1))
            z = torch.cat(z, -1)
        return z


class ChordMixerBlock(nn.Module):
    def __init__(self, prelinear_output_size, n_tracks, track_size, hidden_size, mlp_dropout, layer_dropout):
        super(ChordMixerBlock, self).__init__()
        self.mixer = Mlp(
            prelinear_output_size,
            hidden_size,
            prelinear_output_size,
            act_layer=nn.GELU,
            drop=mlp_dropout
        )
        self.dropout = nn.Dropout(layer_dropout)
        self.rotator = RotateChord(n_tracks, track_size)

    def forward(self, data, lengths=None):
        res_con = data
        data = self.mixer(data)
        data = self.dropout(data)
        data = self.rotator(data, lengths)
        data = data + res_con
        return data


class ChordMixerEncoder(nn.Module):
    """ChordMixerEncoder, to be used as a pretrained model in subsequent downstream tasks."""
    def __init__(self, vocab_size, max_seq_len, track_size, hidden_size, prelinear_out_features, mlp_dropout, layer_dropout, variable_length=False):
        super(ChordMixerEncoder, self).__init__()
        self.variable_length = variable_length
        self.max_n_layers = math.ceil(np.log2(max_seq_len))
        n_tracks = math.ceil(np.log2(max_seq_len))

        self.prelinear = nn.Linear(vocab_size, prelinear_out_features)
        self.chordmixer_blocks = nn.ModuleList(
            [
                ChordMixerBlock(prelinear_out_features, n_tracks, track_size, hidden_size, mlp_dropout, layer_dropout)
                for _ in range(self.max_n_layers)
            ]
        )

    def forward(self, data, lengths=None):
        if lengths:
            n_layers = math.ceil(np.log2(lengths[0]))
        else:
            n_layers = self.max_n_layers

        data = self.prelinear(data)
        for layer in range(n_layers):
            data = self.chordmixer_blocks[layer](data, lengths)

        if lengths:
            data = [torch.mean(t, dim=0) for t in torch.split(data, lengths)]
            data = torch.stack(data)

        return data


class ChordMixerDecoder(nn.Module):
    def __init__(self, vocab_size, max_seq_len, track_size, hidden_size, prelinear_in_features, prelinear_out_features, mlp_dropout, layer_dropout):
        super(ChordMixerDecoder, self).__init__()
        self.max_n_layers = math.ceil(np.log2(max_seq_len))
        n_tracks = math.ceil(np.log2(max_seq_len))

        self.prelinear = nn.Linear(prelinear_in_features, prelinear_out_features)
        self.chordmixer_blocks = nn.ModuleList(
            [
                ChordMixerBlock(prelinear_out_features, n_tracks, track_size, hidden_size, mlp_dropout, layer_dropout)
                for _ in range(self.max_n_layers)
            ]
        )
        self.mlm_classifier = nn.Linear(prelinear_out_features, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, data):
        data = self.prelinear(data)
        for layer in range(self.max_n_layers):
            data = self.chordmixer_blocks[layer](data)
        data = self.mlm_classifier(data)
        data = self.softmax(data)
        return data


class PretrainedChordMixer(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_seq_len,
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
                                         max_seq_len,
                                         encoder_track_size,
                                         hidden_size,
                                         encoder_prelinear_out_features,
                                         mlp_dropout,
                                         layer_dropout)
        self.decoder = ChordMixerDecoder(vocab_size,
                                         max_seq_len,
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
