import torch
from torch import nn


class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_heads, num_layers, const_vector_length, n_class, device_id):
        super(Transformer, self).__init__()
        const_vector_length = 1000
        self.device = "cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu"
        self.const_vector_length = const_vector_length
        self.encoder = nn.Embedding(vocab_size, embedding_size)
        self.posenc = nn.Embedding(self.const_vector_length, embedding_size)
        encoder_layers = nn.TransformerEncoderLayer(embedding_size, num_heads, embedding_size)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.final = nn.Linear(embedding_size * const_vector_length, n_class)

    def forward(self, input_data):
        if input_data["task"] == "TaxonomyClassification":
            x = input_data["x"]

            y_hat = self.encoder(x)
            positions = torch.arange(0, self.const_vector_length) \
                .expand(y_hat.size(0), self.const_vector_length) \
                .to(self.device)
            y_hat = self.posenc(positions) + y_hat
            y_hat = self.transformer_encoder(y_hat)
            y_hat = self.final(y_hat.view(y_hat.size(0), -1))
            y_hat = y_hat.view(-1)
            
            return y_hat

        elif input_data["task"] == "HumanVariantEffectPrediction":
            x1 = input_data["x1"]
            x2 = input_data["x2"]
            tissue = input_data["tissue"]

            y_hat_1 = self.encoder(x1)
            positions1 = torch.arange(0, self.const_vector_length) \
                .expand(y_hat_1.size(0), self.const_vector_length) \
                .to(self.device)
            y_hat_1 = self.posenc(positions1) + y_hat_1
            y_hat_1 = self.transformer_encoder(y_hat_1)

            y_hat_2 = self.encoder(x2)
            positions2 = torch.arange(0, self.const_vector_length) \
                .expand(y_hat_2.size(0), self.const_vector_length) \
                .to(self.device)
            y_hat_2 = self.posenc(positions2) + y_hat_2
            y_hat_2 = self.transformer_encoder(y_hat_2)

            y_hat = y_hat_1 - y_hat_2
            y_hat = self.final(y_hat.view(y_hat.size(0), -1))

            tissue = tissue.unsqueeze(0).t()
            y_hat = torch.gather(y_hat, 1, tissue)
            y_hat = y_hat.reshape(-1)

            return y_hat

        elif input_data["task"] == "PlantVariantEffectPrediction":
            x = input_data["x"]

            y_hat = self.encoder(x)
            positions = torch.arange(0, self.const_vector_length) \
                .expand(y_hat.size(0), self.const_vector_length) \
                .to(self.device)
            y_hat = self.posenc(positions) + y_hat
            y_hat = self.transformer_encoder(y_hat)
            y_hat = self.final(y_hat.view(y_hat.size(0), -1))
            
            return y_hat

        else:
            raise ValueError(f"Task: {input_data['task']} is not supported.")
