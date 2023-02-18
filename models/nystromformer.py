import torch
import torch.nn as nn
from nystrom_attention import Nystromformer as NystromformerModel


class Nystromformer(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_heads, num_layers, const_vector_length, n_class, device_id):
        super(Nystromformer, self).__init__()
        const_vector_length = 1000
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
        self.final = nn.Linear(embedding_size * const_vector_length, n_class)

    def forward(self, input_data):
        if input_data["task"] == "TaxonomyClassification":
            x = input_data["x"]

            y_hat = self.encoder(x)
            positions = torch.arange(0, self.const_vector_length) \
                .expand(y_hat.size(0), self.const_vector_length) \
                .to(self.device)
            y_hat = self.posenc(positions) + y_hat
            y_hat = self.nystromformermodel(y_hat)
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
            y_hat_1 = self.nystromformermodel(y_hat_1)

            y_hat_2 = self.encoder(x2)
            positions2 = torch.arange(0, self.const_vector_length) \
                .expand(y_hat_2.size(0), self.const_vector_length) \
                .to(self.device)
            y_hat_2 = self.posenc(positions2) + y_hat_2
            y_hat_2 = self.nystromformermodel(y_hat_2)

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
            y_hat = self.nystromformermodel(y_hat)
            y_hat = self.final(y_hat.view(y_hat.size(0), -1))

            return y_hat

        else:
            raise ValueError(f"Task: {input_data['task']} is not supported.")
