import math
import torch
import numpy as np
from torch import nn
from collections import OrderedDict
from typing import Dict

from .chordmixer import ChordMixerBlock


class ChordMixerEncoder(nn.Module):
    """ChordMixerEncoder, to be used as a pretrained model in subsequent downstream tasks."""

    def __init__(self,
                 vocab_size: int,
                 track_size: int,
                 hidden_size: int,
                 mlp_dropout: float,
                 layer_dropout: float,
                 max_seq_len: int,
                 variable_length: bool = False
                 ):
        super(ChordMixerEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.track_size = track_size
        self.hidden_size = hidden_size
        self.mlp_dropout = mlp_dropout
        self.layer_dropout = layer_dropout
        self.max_seq_len = max_seq_len
        self.variable_length = variable_length

        self.n_tracks = math.ceil(np.log2(max_seq_len))
        self.n_layers = math.ceil(np.log2(max_seq_len))
        self.prelinear_output_size  = int(self.n_tracks * track_size)

        self.prelinear = nn.Linear(vocab_size, self.prelinear_output_size)
        self.chordmixer_blocks = nn.ModuleList(
            [
                ChordMixerBlock(self.prelinear_output_size, self.n_tracks, track_size, hidden_size, mlp_dropout, layer_dropout)
                for _ in range(self.n_layers)
            ]
        )

    @staticmethod
    def _get_encoder_state_dict(model: Dict) -> Dict:
        """
        Gets the state dict of the encoder from a pretrained model.

        Args:
            model: The state dict of the pretrained model.

        Returns:
            The state dict of the encoder, with every other component removed.
        """
        state_dict = OrderedDict()
        for key, value in model.items():
            if "encoder" in key:
                new_key = key.replace("encoder.", "")
                state_dict[new_key] = value
        return state_dict

    @classmethod
    def from_pretrained(cls, model_path: str, freeze: bool = True, variable_length: bool = False) -> nn.Module:
        """
        Loads a pretrained model and return the encoder.

        Args:
            model_path: The path to the pretrained model.
            freeze: Whether to freeze the parameters of the encoder.
            variable_length: Whether the encoder should be able to handle variable length sequences.

        Returns:
            The encoder module.
        """
        print(f"Loading pretrained model: {model_path}")
        model = torch.load(model_path)
        encoder_state_dict = cls._get_encoder_state_dict(model=model)

        encoder = cls(
            vocab_size=4,
            track_size=16,
            hidden_size=196,
            mlp_dropout=0.0,
            layer_dropout=0.0,
            max_seq_len=1000,
            variable_length=variable_length
        )

        encoder.load_state_dict(encoder_state_dict)

        if freeze:
            for param in encoder.parameters():
                param.requires_grad = False

        return encoder

    def forward(self, data, lengths=None):
        if self.variable_length:
            n_layers = math.ceil(np.log2(lengths[0]))
        else:
            n_layers = self.n_layers

        data = self.prelinear(data)
        for layer in range(n_layers):
            data = self.chordmixer_blocks[layer](data, lengths)            
        return data


class ChordMixerDecoder(nn.Module):
    """ChordMixerDecoder, used only during pretraining"""

    def __init__(self,
                 vocab_size: int,
                 track_size: int,
                 hidden_size: int,
                 prelinear_input_size: int,
                 mlp_dropout: float,
                 layer_dropout: float,
                 max_seq_len: int
                 ):
        super(ChordMixerDecoder, self).__init__()

        self.n_tracks = math.ceil(np.log2(max_seq_len))
        self.n_layers = math.ceil(np.log2(max_seq_len))
        self.prelinear_output_size  = int(self.n_tracks * track_size)

        self.prelinear = nn.Linear(prelinear_input_size, self.prelinear_output_size)
        self.chordmixer_blocks = nn.ModuleList(
            [
                ChordMixerBlock(self.prelinear_output_size, self.n_tracks, track_size, hidden_size, mlp_dropout, layer_dropout)
                for _ in range(self.n_layers)
            ]
        )
        self.mlm_classifier = nn.Linear(self.prelinear_output_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, data):
        data = self.prelinear(data)
        for layer in range(self.n_layers):
            data = self.chordmixer_blocks[layer](data)
        data = self.mlm_classifier(data)
        data = self.softmax(data)
        return data


class PretrainedChordMixer(nn.Module):
    """Complete architecture of the pretrained model."""

    def __init__(self,
                 vocab_size: int,
                 encoder_track_size: int,
                 encoder_hidden_size: int,
                 encoder_mlp_dropout: float,
                 encoder_layer_dropout: float,
                 decoder_track_size: int,
                 decoder_hidden_size: int,
                 decoder_mlp_dropout: float,
                 decoder_layer_dropout: float,
                 max_seq_len: int,
                 variable_length: bool
                 ):
        super(PretrainedChordMixer, self).__init__()
        self.encoder = ChordMixerEncoder(
            vocab_size=vocab_size,
            track_size=encoder_track_size,
            hidden_size=encoder_hidden_size,
            mlp_dropout=encoder_mlp_dropout,
            layer_dropout=encoder_layer_dropout,
            max_seq_len=max_seq_len,
            variable_length=variable_length
        )
        decoder_output_size = int(math.ceil(np.log2(max_seq_len)) * encoder_track_size)
        self.decoder = ChordMixerDecoder(
            vocab_size=vocab_size,
            track_size=decoder_track_size,
            hidden_size=decoder_hidden_size,
            prelinear_input_size=decoder_output_size,
            mlp_dropout=decoder_mlp_dropout,
            layer_dropout=decoder_layer_dropout,
            max_seq_len=max_seq_len
        )

    def forward(self, sequence_ids, lengths):
        if lengths:
            lengths = [length.item() for length in lengths]
            sequence_ids = sequence_ids.squeeze()
        encoded = self.encoder(sequence_ids, lengths)
        decoded = self.decoder(encoded)
        if lengths:
            decoded = decoded.unsqueeze(0)
        return decoded
