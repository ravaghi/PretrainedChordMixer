import torch
import torch.nn as nn
from linformer import Linformer as LinformerModel


class Linformer(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_heads, num_layers, const_vector_length, n_class, device_id):
        super(Linformer, self).__init__()
        const_vector_length = 1000
        self.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        self.encoder = nn.Embedding(vocab_size, embedding_size)
        self.posenc = nn.Embedding(const_vector_length, embedding_size)
        self.linformermodel = LinformerModel(
            dim=embedding_size,
            seq_len=const_vector_length,
            depth=num_layers,
            heads=num_heads,
            k=256,
            one_kv_head=True,
            share_kv=True
        )
        self.const_vector_length = const_vector_length
        self.final = nn.Linear(embedding_size * const_vector_length, n_class)
        self.linear = nn.Linear(2, embedding_size, bias=True)

    def forward(self, input_data):
        if input_data["task"] == "TaxonomyClassification":
            x = input_data["x"]
            x = self.encoder(x)
            positions = torch.arange(0, self.const_vector_length).expand(x.size(0), self.const_vector_length).to(
                self.device)
            x = self.posenc(positions) + x
            x = self.linformermodel(x)
            x = self.final(x.view(x.size(0), -1))
            return x

        elif input_data["task"] == "HumanVariantEffectPrediction":
            x1 = input_data["x1"]
            x2 = input_data["x2"]
            tissue = input_data["tissue"]

            x1 = self.encoder(x1)
            positions1 = torch.arange(0, self.const_vector_length).expand(x1.size(0), self.const_vector_length).to(
                self.device)
            x1 = self.posenc(positions1) + x1
            x1 = self.linformermodel(x1)
            x1 = x1.view(x1.size(0), -1)

            x2 = self.encoder(x2)
            positions2 = torch.arange(0, self.const_vector_length).expand(x2.size(0), self.const_vector_length).to(
                self.device)
            x2 = self.posenc(positions2) + x2
            x2 = self.linformermodel(x2)
            x2 = x2.view(x2.size(0), -1)

            y = x2 - x1
            y = self.final(y)

            tissue = tissue.unsqueeze(0).t()
            y = torch.gather(y, 1, tissue)
            y = y.reshape(-1)

            return y



        elif input_data["task"] == "PlantVariantEffectPrediction":
            pass

        else:
            raise ValueError(f"Task: {input_data['task']} is not supported.")
