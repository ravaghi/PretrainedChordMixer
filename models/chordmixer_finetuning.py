from torch import nn
import torch
import math
import numpy as np

from .chordmixer_pretraining import ChordMixerEncoder
from .chordmixer import ChordMixerBlock


class ChordMixerClassifier(nn.Module):
    """ChordMixerClassifier, used for fine-tuning"""

    def __init__(self,
                 track_size: int,
                 hidden_size: int,
                 prelinear_input_size: int,
                 mlp_dropout: float,
                 layer_dropout: float,
                 max_seq_len: int,
                 variable_length: bool
                 ):
        super(ChordMixerClassifier, self).__init__()
        self.variable_length = variable_length

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

    def forward(self, data):
        data = self.prelinear(data)
        for layer in range(self.n_layers):
            data = self.chordmixer_blocks[layer](data, None)
        return data


class FineTunedChordMixer(nn.Module):
    """ChordMixer fine-tuning model."""

    def __init__(self,
                model_path: str,
                freeze: bool,
                variable_length: bool,
                n_class: int,
                decoder_track_size: int,
                decoder_hidden_size: int,
                decoder_mlp_dropout: float,
                decoder_layer_dropout: float
                ):
        super(FineTunedChordMixer, self).__init__()
        self.encoder = ChordMixerEncoder.from_pretrained(
            model_path=model_path,
            freeze=freeze,
            variable_length=variable_length
        )
        self.decoder = ChordMixerClassifier(
            track_size=decoder_track_size,
            hidden_size=decoder_hidden_size,
            prelinear_input_size=self.encoder.prelinear_output_size,
            mlp_dropout=decoder_mlp_dropout,
            layer_dropout=decoder_layer_dropout,
            max_seq_len=self.encoder.max_seq_len,
            variable_length=variable_length
        )
        #self.classifier = nn.Linear(self.decoder.prelinear_output_size, n_class)
        self.classifier = nn.Sequential(
            nn.Linear(4 * self.decoder.prelinear_output_size, self.decoder.prelinear_output_size // 2),
            nn.ReLU(),
            nn.Linear(self.decoder.prelinear_output_size // 2, n_class)
        )

    def forward(self, input_data):
        if input_data["task"] == "TaxonomyClassification":
            x = input_data["x"].float()
            lengths = input_data["seq_len"]

            y_hat = self.encoder(x, lengths)
            y_hat = self.decoder(y_hat)
    
            y_hat = [torch.mean(t, dim=0) for t in torch.split(y_hat, lengths)]
            y_hat = torch.stack(y_hat)
            
            y_hat = self.classifier(y_hat)
            y_hat = y_hat.view(-1)

            return y_hat

        elif input_data["task"] == "HumanVariantEffectPrediction":
            x1 = input_data["x1"].float()
            x2 = input_data["x2"].float()
            tissue = input_data["tissue"]

            y_hat_1 = self.encoder(x1)
            y_hat_2 = self.encoder(x2)

            y_hat_1 = self.decoder(y_hat_1)
            y_hat_2 = self.decoder(y_hat_2)

            y_hat_1 = torch.mean(y_hat_1, dim=1)
            y_hat_2 = torch.mean(y_hat_2, dim=1)

            y_hat = torch.cat([y_hat_1, y_hat_2, y_hat_1 * y_hat_2, y_hat_1 - y_hat_2], dim=1)
            y_hat = self.classifier(y_hat)

            tissue = tissue.unsqueeze(0).t()
            y_hat = torch.gather(y_hat, 1, tissue)
            y_hat = y_hat.reshape(-1)

            return y_hat

        elif input_data["task"] == "PlantVariantEffectPrediction":
            x = input_data["x"].float()

            y_hat = self.encoder(x)
            y_hat = self.decoder(y_hat)

            y_hat = y_hat[:, 400:600, :]

            y_hat = torch.mean(y_hat, dim=1)
            y_hat = self.classifier(y_hat)

            return y_hat

        else:
            raise ValueError(f"Task: {input_data['task']} is not supported.")
