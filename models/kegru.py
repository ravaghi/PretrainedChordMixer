import torch
import torch.nn as nn
import gensim
import os


class KeGru(nn.Module):
    def __init__(self, num_layers, hidden_size, dropout, kmer_embedding_path, kmer_embedding_name, embedding_size, device_id, num_class):
        super(KeGru, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

        # Load pretrained embeddings
        model_path = os.path.join(kmer_embedding_path, kmer_embedding_name)
        model = gensim.models.Word2Vec.load(model_path)
        ke_weights = torch.FloatTensor(model.wv.vectors)

        self.embedding = nn.Embedding.from_pretrained(ke_weights, freeze=False)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size * 2, num_class)

    def forward(self, input_data):
        if input_data["task"] == "TaxonomyClassification":
            x = input_data["x"] 
            h0 =  torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
            y = self.embedding(y) 
            y, _ = self.gru(y, h0)
            y = y[:, -1, :]
            y = self.linear(y)
            return y

        elif input_data["task"] == "VariantEffectPrediction":
            x1 = input_data["x1"]
            x2 = input_data["x2"]
            tissue = input_data["tissue"]

            h0_1 = torch.zeros(self.num_layers * 2, x1.size(0), self.hidden_size).to(self.device)
            y1 = self.embedding(x1)
            y1, _ = self.gru(y1, h0_1)
            y1 = y1[:, -1, :]

            h0_2 = torch.zeros(self.num_layers * 2, x2.size(0), self.hidden_size).to(self.device)
            y2 = self.embedding(x2)
            y2, _ = self.gru(y2, h0_2)
            y2 = y2[:, -1, :]

            y = y2 - y1

            y = self.linear(y)

            tissue = tissue.unsqueeze(0).t()
            y = torch.gather(y, 1, tissue)
            y = y.reshape(-1)

            return y

        elif input_data["task"] == "PlantDeepSEA":
            x = input_data["x"]

            h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
            y = self.embedding(x)
            y, _ = self.gru(y, h0)
            y = y[:, -1, :]
            y = self.linear(y)

            return y

        else:
            raise ValueError(f"Task: {input_data['task']} is not supported.")
