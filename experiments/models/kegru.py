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
        self.hidden_weights = torch.normal(torch.zeros(2 * hidden_size, 1)).to(device)
        self.hidden_bias = torch.randn(1).to(device)
        torch.nn.init.xavier_uniform_(self.hidden_weights)
        self.hidden_weights.requires_grad = True
        self.hidden_bias.requires_grad = True
        self.drop1 = nn.Dropout(p=dropout).to(device)
        self.linear = nn.Linear(20, 1, bias=True).to(device)

    def forward(self, x):
        x_embed = self.embedding(x).permute(1, 0, 2)
        out, _ = self.rnn(x_embed)
        forward_gru = out[-1, :, :self.hidden_size]
        reverse_gru = out[0, :, self.hidden_size:]

        merged_gru = self.drop1(torch.cat((forward_gru, reverse_gru), 1))
        hid = merged_gru @ self.hidden_weights
        hid.add_(self.hidden_bias)
        prediction = torch.sigmoid(hid)

        return prediction
