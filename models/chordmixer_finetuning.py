from torch import nn
import torch

from .chordmixer_pretraining import ChordMixerEncoder, ChordMixerClassifier


class FineTunedChordMixer(nn.Module):
    """ChordMixer fine-tuning model."""

    def __init__(self,
                 model_path: str,
                 hidden_size: int,
                 freeze: bool,
                 variable_length: bool,
                 n_class: int
                 ):
        super(FineTunedChordMixer, self).__init__()
        self.encoder = ChordMixerEncoder.from_pretrained(
            model_path=model_path,
            freeze=freeze,
            variable_length=variable_length
        )
        self.decoder = ChordMixerClassifier(
            n_blocks=10,
            track_size=self.encoder.track_size,
            hidden_size=self.encoder.hidden_size,
            prelinear_out_features=self.encoder.prelinear_out_features,
            mlp_dropout=self.encoder.mlp_dropout,
            layer_dropout=self.encoder.layer_dropout,
            variable_length=False
        )
        self.classifier = nn.Linear(self.encoder.prelinear_out_features, n_class)

    def forward(self, input_data):
        if input_data["task"] == "TaxonomyClassification":
            x = input_data["x"].float()
            lengths = input_data["seq_len"]

            y_hat = self.encoder(x, lengths)
            y_hat = self.decoder(y_hat)
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

            y_hat = y_hat_1 - y_hat_2
            y_hat = torch.mean(y_hat, dim=1)
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
