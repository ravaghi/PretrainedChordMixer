import torch
import torch.nn as nn
from reformer_pytorch import Reformer as ReformerModel


class Reformer(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_heads, num_layers, const_vector_length, n_class, device_id):
        super(Reformer, self).__init__()
        self.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        self.encoder = nn.Embedding(vocab_size, embedding_size)
        self.posenc = nn.Embedding(const_vector_length, embedding_size)
        self.reformermodel = ReformerModel(
            dim=embedding_size,
            depth=num_layers,
            heads=num_heads,
            lsh_dropout=0.1,
            causal=True
        )
        self.const_vector_length = const_vector_length
        self.final = nn.Linear(embedding_size * const_vector_length, n_class)
        self.linear = nn.Linear(2, embedding_size, bias=True)

    def forward(self, input_data):
        if input_data["task"] == "TaxonomyClassification":
            x = input_data["x"]
            x = self.encoder(x)
            positions = torch.arange(0, self.const_vector_length) \
                .expand(x.size(0), self.const_vector_length) \
                .to(self.device)
            x = self.posenc(positions) + x
            x = self.reformermodel(x)
            x = self.final(x.view(x.size(0), -1))
            return x

        elif input_data["task"] == "HumanVariantEffectPrediction":
            pass

        elif input_data["task"] == "PlantVariantEffectPrediction":
            pass

        else:
            raise ValueError(f"Task: {input_data['task']} is not supported.")
