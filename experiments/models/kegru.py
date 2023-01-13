import torch
import torch.nn as nn
import gensim
import os


class KeGru(nn.Module):
    def __init__(self, num_layers, hidden_size, kmer_embedding_path, kmer_embedding_name, embedding_size, dropout,
                 device_id):
        super(KeGru, self).__init__()

        device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

        # Load pretrained embeddings
        model_path = os.path.join(kmer_embedding_path, kmer_embedding_name)
        model = gensim.models.Word2Vec.load(model_path)
        ke_weights = torch.FloatTensor(model.wv.vectors)

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding.from_pretrained(ke_weights, freeze=False)
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, bidirectional=True).to(device)
        self.hidden_weights = torch.normal(torch.zeros(2 * hidden_size, 49)).to(device)
        self.hidden_bias = torch.randn(1).to(device)
        torch.nn.init.xavier_uniform_(self.hidden_weights)
        self.hidden_weights.requires_grad = True
        self.hidden_bias.requires_grad = True
        self.drop1 = nn.Dropout(p=dropout).to(device)
        self.linear = nn.Linear(100, 1, bias=True).to(device)

    def forward(self, input_data):
        if input_data["task"] == "TaxonomyClassification":
            x = input_data["x"]
            x_embed = self.embedding(x).permute(1, 0, 2)
            out, _ = self.rnn(x_embed)
            forward_gru = out[-1, :, :self.hidden_size]
            reverse_gru = out[0, :, self.hidden_size:]
            merged_gru = self.drop1(torch.cat((forward_gru, reverse_gru), 1))
            hid = merged_gru @ self.hidden_weights
            hid.add_(self.hidden_bias)
            prediction = torch.sigmoid(hid)
            return prediction

        elif input_data["task"] == "VariantEffectPrediction":
            x1 = input_data["x1"]
            x2 = input_data["x2"]
            tissue = input_data["tissue"]

            x_embed1 = self.embedding(x1).permute(1, 0, 2)
            out1, _ = self.rnn(x_embed1)
            forward_gru1 = out1[-1, :, :self.hidden_size]
            reverse_gru1 = out1[0, :, self.hidden_size:]
            merged_gru1 = self.drop1(torch.cat((forward_gru1, reverse_gru1), 1))
            hid1 = merged_gru1 @ self.hidden_weights
            hid1 = hid1.add(self.hidden_bias)

            x_embed2 = self.embedding(x2).permute(1, 0, 2)
            out2, _ = self.rnn(x_embed2)
            forward_gru2 = out2[-1, :, :self.hidden_size]
            reverse_gru2 = out2[0, :, self.hidden_size:]
            merged_gru2 = self.drop1(torch.cat((forward_gru2, reverse_gru2), 1))
            hid2 = merged_gru2 @ self.hidden_weights
            hid2 = hid2.add(self.hidden_bias)

            data = hid1 - hid2

            tissue = tissue.t()

            data = torch.gather(data, 1, tissue)  
            data = data.reshape(-1)
            data = self.linear(data)
            data = torch.sigmoid(data)

            return data

        elif input_data["task"] == "PlantDeepSEA":
            pass

        else:
            raise ValueError(f"Task: {input_data['task']} is not supported.")
