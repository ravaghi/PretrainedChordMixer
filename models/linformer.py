import torch
import torch.nn as nn
from linformer import Linformer as LinformerModel


class Linformer(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_heads, num_layers, dataset_type, n_class, device_id):
        super(Linformer, self).__init__()
        self.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        
        if dataset_type == "TaxonomyClassification":
            self.sequence_length = 25000
        else:
            self.sequence_length = 1000
        
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.positional_encoder = nn.Embedding(self.sequence_length, embedding_size)
        self.linformer = LinformerModel(
            dim=embedding_size,
            seq_len=self.sequence_length,
            depth=num_layers,
            heads=num_heads,
            k=256,
            one_kv_head=True,
            share_kv=True
        )
        self.classifier = nn.Linear(embedding_size * self.sequence_length, n_class)

    def forward(self, input_data):
        if input_data["task"] == "TaxonomyClassification":
            x = input_data["x"]

            y_hat = self.embedding(x)
            positions = torch.arange(0, self.sequence_length) \
                .expand(y_hat.size(0), self.sequence_length) \
                .to(self.device)
            y_hat = self.positional_encoder(positions) + y_hat
            y_hat = self.linformer(y_hat)
            y_hat = y_hat.view(y_hat.size(0), -1)
            y_hat = self.classifier(y_hat)
            y_hat = y_hat.view(-1)

            return y_hat

        elif input_data["task"] == "HumanVariantEffectPrediction":
            x1 = input_data["x1"]
            x2 = input_data["x2"]
            tissue = input_data["tissue"]

            y_hat_1 = self.embedding(x1)
            positions1 = torch.arange(0, self.sequence_length) \
                .expand(y_hat_1.size(0), self.sequence_length) \
                .to(self.device)
            y_hat_1 = self.positional_encoder(positions1) + y_hat_1
            y_hat_1 = self.linformer(y_hat_1)

            y_hat_2 = self.embedding(x2)
            positions2 = torch.arange(0, self.sequence_length) \
                .expand(y_hat_2.size(0), self.sequence_length) \
                .to(self.device)
            y_hat_2 = self.positional_encoder(positions2) + y_hat_2
            y_hat_2 = self.linformer(y_hat_2)

            y_hat = y_hat_1 - y_hat_2
            y_hat = y_hat.view(y_hat.size(0), -1)
            y_hat = self.classifier(y_hat)

            tissue = tissue.unsqueeze(0).t()
            y_hat = torch.gather(y_hat, 1, tissue)
            y_hat = y_hat.reshape(-1)

            return y_hat

        elif input_data["task"] == "PlantVariantEffectPrediction":
            x = input_data["x"]

            y_hat = self.embedding(x)
            positions = torch.arange(0, self.sequence_length) \
                .expand(y_hat.size(0), self.sequence_length) \
                .to(self.device)
            y_hat = self.positional_encoder(positions) + y_hat
            y_hat = self.linformer(y_hat)
            y_hat = y_hat.view(y_hat.size(0), -1)
            y_hat = self.classifier(y_hat)

            return y_hat

        else:
            raise ValueError(f"Task: {input_data['task']} is not supported.")
