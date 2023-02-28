from torch import nn
import torch

from .chordmixer_pretraining import ChordMixerEncoder


class FineTunedChordMixer(nn.Module):
    """Fine-tuned ChordMixer"""

    def __init__(self, model, hidden_size, freeze, variable_length, n_class):
        super(FineTunedChordMixer, self).__init__()
        self.encoder = ChordMixerEncoder.from_pretrained(
            model=model,
            freeze=freeze,
            variable_length=variable_length
        )
        self.hidden = nn.Linear(self.encoder.prelinear_out_features, hidden_size)
        self.classifier = nn.Linear(hidden_size, n_class)

    def forward(self, input_data):
        if input_data["task"] == "TaxonomyClassification":
            x = input_data["x"].float()
            lengths = input_data["seq_len"]

            y_hat = self.encoder(x, lengths)
            y_hat = self.hidden(y_hat)
            y_hat = self.classifier(y_hat)
            y_hat = y_hat.view(-1)

            return y_hat

        elif input_data["task"] == "HumanVariantEffectPrediction":
            x1 = input_data["x1"].float()
            x2 = input_data["x2"].float()
            tissue = input_data["tissue"]

            y_hat_1 = self.encoder(x1)
            y_hat_2 = self.encoder(x2)
            
            y_hat = y_hat_1 - y_hat_2
            y_hat = torch.mean(y_hat, dim=1)    
            y_hat = self.hidden(y_hat)       
            y_hat = self.classifier(y_hat)

            tissue = tissue.unsqueeze(0).t()
            y_hat = torch.gather(y_hat, 1, tissue)
            y_hat = y_hat.reshape(-1)

            return y_hat

        elif input_data["task"] == "PlantVariantEffectPrediction":
            x = input_data["x"].float()

            y_hat = self.encoder(x)

            y_hat = y_hat[:, 400:600, :]
            
            y_hat = torch.mean(y_hat, dim=1)
            y_hat = self.hidden(y_hat)
            y_hat = self.classifier(y_hat)

            return y_hat

        else:
            raise ValueError(f"Task: {input_data['task']} is not supported.")
