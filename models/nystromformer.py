import torch
import torch.nn as nn
from nystrom_attention import Nystromformer as NystromformerModel


class Nystromformer(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_heads, num_layers, const_vector_length, n_class, pooling,
                 device_id):
        super(Nystromformer, self).__init__()
        self.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        self.encoder = nn.Embedding(vocab_size, embedding_size)
        self.posenc = nn.Embedding(const_vector_length, embedding_size)
        self.nystromformermodel = NystromformerModel(
            dim=embedding_size,
            dim_head=int(embedding_size / num_heads),
            heads=num_heads,
            depth=num_layers,
            num_landmarks=256,  # number of landmarks
            pinv_iterations=6
        )
        self.const_vector_length = const_vector_length
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
            # x = self.dropout1(x)
            x = self.posenc(positions) + x
            x = self.nystromformermodel(x)
            if self.pooling == 'avg':
                x = torch.mean(x, 1)
            elif self.pooling == 'cls':
                x = x[:, 0, :]
            x = self.final(x.view(x.size(0), -1))
            return x

        elif input_data["task"] == "VariantEffectPrediction":
            pass

        elif input_data["task"] == "PlantDeepSEA":
            pass

        else:
            raise ValueError(f"Task: {input_data['task']} is not supported.")
