import math
import torch
import numpy as np
from torch import nn


class MLP(nn.Module):
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
    """
    Parameter-free module to perform tracks shift.
    """

    def __init__(self, n_tracks, track_size):
        super(RotateChord, self).__init__()
        self.n_tracks = n_tracks
        self.track_size = track_size

    def forward(self, x, lengths=None):
        if not lengths:

            y = torch.split(
                tensor=x,
                split_size_or_sections=self.track_size,
                dim=-1
            )
            # roll sequences in a batch jointly
            z = [y[0]]
            for i in range(1, len(y)):
                offset = -2 ** (i - 1)
                z.append(torch.roll(y[i], shifts=offset, dims=1))
            z = torch.cat(z, -1)

        else:

            ys = torch.split(
                tensor=x,
                split_size_or_sections=lengths,
                dim=0
            )

            zs = []

            # roll sequences separately
            for y in ys:
                y = torch.split(
                    tensor=y,
                    split_size_or_sections=self.track_size,
                    dim=-1
                )
                z = [y[0]]
                for i in range(1, len(y)):
                    offset = -2 ** (i - 1)
                    z.append(torch.roll(y[i], shifts=offset, dims=0))
                z = torch.cat(z, -1)
                zs.append(z)

            z = torch.cat(zs, 0)
            assert z.shape == x.shape, 'shape mismatch'
        return z


class ChordMixerBlock(nn.Module):
    def __init__(
            self,
            embedding_size,
            n_tracks,
            track_size,
            hidden_size,
            mlp_dropout,
            layer_dropout
    ):
        super(ChordMixerBlock, self).__init__()

        self.mixer = MLP(
            embedding_size,
            hidden_size,
            embedding_size,
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


class ChordMixer(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 variable_length,
                 track_size,
                 hidden_size,
                 mlp_dropout,
                 layer_dropout,
                 n_class
                 ):
        super(ChordMixer, self).__init__()
        self.max_n_layers = math.ceil(np.log2(max_seq_len))
        self.variable_length = variable_length
        self.n_class = n_class
        n_tracks = math.ceil(np.log2(max_seq_len))
        embedding_size = int(n_tracks * track_size)
        # Init embedding layer
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_size
        )
        self.chordmixer_blocks = nn.ModuleList(
            [
                ChordMixerBlock(
                    embedding_size,
                    n_tracks,
                    track_size,
                    hidden_size,
                    mlp_dropout,
                    layer_dropout
                )
                for _ in range(self.max_n_layers)
            ]
        )

        self.final = nn.Linear(
            embedding_size,
            n_class
        )

    def forward(self, input_data):
        if input_data["task"] == "TaxonomyClassification":
            x = input_data["x"]
            lengths = input_data["seq_len"]

            n_layers = math.ceil(np.log2(lengths[0]))

            y_hat = self.embedding(x)
            for layer in range(n_layers):
                y_hat = self.chordmixer_blocks[layer](y_hat, lengths)

            y_hat = [torch.mean(t, dim=0) for t in torch.split(y_hat, lengths)]
            y_hat = torch.stack(y_hat)

            y_hat = self.final(y_hat)
            y_hat = y_hat.view(-1)

            return y_hat

        elif input_data["task"] == "HumanVariantEffectPrediction":
            x1 = input_data["x1"]
            x2 = input_data["x2"]
            tissue = input_data["tissue"]

            n_layers = self.max_n_layers

            y_hat_1 = self.embedding(x1)
            for layer in range(n_layers):
                y_hat_1 = self.chordmixer_blocks[layer](y_hat_1, None)

            y_hat_2 = self.embedding(x2)
            for layer in range(n_layers):
                y_hat_2 = self.chordmixer_blocks[layer](y_hat_2, None)

            y_hat = y_hat_1 - y_hat_2
            y_hat = torch.mean(y_hat, dim=1)
            y_hat = self.final(y_hat)

            tissue = tissue.unsqueeze(0).t()
            y_hat = torch.gather(y_hat, 1, tissue)
            y_hat = y_hat.reshape(-1)

            return y_hat

        elif input_data["task"] == "PlantVariantEffectPrediction":
            x = input_data["x"]

            n_layers = self.max_n_layers

            y_hat = self.embedding(x)
            for layer in range(n_layers):
                y_hat = self.chordmixer_blocks[layer](y_hat, None)

            y_hat = y_hat[:, 400:600, :]

            y_hat = torch.mean(y_hat, dim=1)
            y_hat = self.final(y_hat)

            return y_hat

        else:
            raise ValueError(f"Task: {input_data['task']} is not supported.")
