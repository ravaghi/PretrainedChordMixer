import torch
from .pretraining import ChordMixerEncoder


class FineTunedChordMixer(torch.nn.Module):
    """Fine-tuned ChordMixer"""

    def __init__(self, freeze, variable_length, n_class):
        super(FineTunedChordMixer, self).__init__()
        self.encoder = ChordMixerEncoder.from_pretrained(
            model="/cluster/home/mahdih/PDT/models/PretrainedChordMixer-AUC0.7445990444685686.pt",
            freeze=freeze,
            variable_length=variable_length
        )
        self.classifier = torch.nn.Linear(self.encoder.prelinear_out_features, n_class)

    def forward(self, input_data):
        if input_data["task"] == "TaxonomyClassification":
            x = input_data["x"].float()
            lengths = input_data["seq_len"]

            encoded = self.encoder(x, lengths)
            return self.classifier(encoded)

        elif input_data["task"] == "HumanVariantEffectPrediction":
            pass

        elif input_data["task"] == "PlantVariantEffectPrediction":
            pass

        else:
            raise ValueError(f"Task: {input_data['task']} is not supported.")
