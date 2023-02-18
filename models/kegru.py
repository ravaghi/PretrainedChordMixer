import torch.nn as nn
import torch
import gensim
import os


class KeGru(nn.Module):
    def __init__(self,
                 num_layers,
                 hidden_size,
                 dropout,
                 kmer_embedding_path,
                 kmer_embedding_name,
                 embedding_size,
                 device_id,
                 num_class
                 ):
        super(KeGru, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

        # Load pretrained embeddings
        model_path = os.path.join(kmer_embedding_path, kmer_embedding_name)
        model = gensim.models.Word2Vec.load(model_path)
        ke_weights = torch.FloatTensor(model.wv.vectors)

        self.embedding = nn.Embedding.from_pretrained(ke_weights, freeze=False)
        self.gru = nn.GRU(embedding_size,
                          hidden_size,
                          num_layers,
                          bidirectional=True,
                          batch_first=True,
                          dropout=dropout)
        self.linear = nn.Linear(hidden_size * 2, num_class)

    def forward(self, input_data):
        if input_data["task"] == "TaxonomyClassification":
            x = input_data["x"]

            h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
            y_hat = self.embedding(x)
            y_hat, _ = self.gru(y_hat, h0)
            y_hat = y_hat[:, -1, :]
            y_hat = self.linear(y_hat)
            y_hat = y_hat.view(-1)
            return y_hat

        elif input_data["task"] == "HumanVariantEffectPrediction":
            x1 = input_data["x1"]
            x2 = input_data["x2"]
            tissue = input_data["tissue"]

            h0_1 = torch.zeros(self.num_layers * 2, x1.size(0), self.hidden_size).to(self.device)
            y_hat_1 = self.embedding(x1)
            y_hat_1, _ = self.gru(y_hat_1, h0_1)
            y_hat_1 = y_hat_1[:, -1, :]

            h0_2 = torch.zeros(self.num_layers * 2, x2.size(0), self.hidden_size).to(self.device)
            y_hat_2 = self.embedding(x2)
            y_hat_2, _ = self.gru(y_hat_2, h0_2)
            y_hat_2 = y_hat_2[:, -1, :]

            y_hat = y_hat_2 - y_hat_1

            y_hat = self.linear(y_hat)

            tissue = tissue.unsqueeze(0).t()
            y_hat = torch.gather(y_hat, 1, tissue)
            y_hat = y_hat.reshape(-1)

            return y_hat

        elif input_data["task"] == "PlantVariantEffectPrediction":
            x = input_data["x"]

            h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
            y_hat = self.embedding(x)
            y_hat, _ = self.gru(y_hat, h0)
            y_hat = y_hat[:, -1, :]
            y_hat = self.linear(y_hat)

            return y_hat

        else:
            raise ValueError(f"Task: {input_data['task']} is not supported.")
