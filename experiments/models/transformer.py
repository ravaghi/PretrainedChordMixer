import torch
from torch import nn


class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_heads, num_layers, const_vector_length, n_class, pooling,
                 device_id):
        super(Transformer, self).__init__()
        self.device = "cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu"
        self.const_vector_length = const_vector_length
        self.encoder = nn.Embedding(vocab_size, embedding_size)
        self.posenc = nn.Embedding(self.const_vector_length, embedding_size)
        encoder_layers = nn.TransformerEncoderLayer(embedding_size, num_heads, embedding_size)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.pooling = pooling
        self.final = nn.Linear(embedding_size, n_class)
        if self.pooling == 'flatten':
            self.final = nn.Linear(embedding_size * const_vector_length, n_class)
        self.linear = nn.Linear(2, embedding_size, bias=True)

    def forward(self, input_data):
        if input_data["task"] == "TaxonomyClassification":
            x = input_data["x"]
            x = self.encoder(x)
            positions = torch.arange(0, self.const_vector_length).expand(x.size(0), self.const_vector_length).to(
                self.device)
            x = self.posenc(positions) + x
            x = self.transformer_encoder(x)
            x = self.final(x.view(x.size(0), -1))
            return x

        elif input_data["task"] == "VariantEffectPrediction":
            pass

        elif input_data["task"] == "PlantDeepSEA":
            pass

        else:
            raise ValueError(f"Task: {input_data['task']} is not supported.")
