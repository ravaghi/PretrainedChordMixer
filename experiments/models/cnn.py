import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, vocab_size, feature_size):
        super(CNN, self).__init__()
        self.vocab_size = vocab_size
        self.feature_size = feature_size

        self.encoder = nn.Embedding(self.vocab_size, self.feature_size)

        self.conv1 = nn.Conv1d(in_channels=2,
                               out_channels=100,
                               kernel_size=6,
                               stride=3,
                               padding=1)
        self.conv2 = nn.Conv1d(in_channels=100,
                               out_channels=400,
                               kernel_size=6,
                               stride=3,
                               padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(3, 2, 1)
        # Enhancer Prediction: 55600
        # Carassius VS Labeo: 277600
        # Danio VS Cyprinus:
        # Sus VS Bos: 
        self.fc1 = nn.Linear(277600, 100)
        self.dropout = nn.Dropout(p=0.5)
        self.softmx = nn.Softmax()

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        x = x.to(torch.int64)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.dropout(x)
        x = self.fc1(x)

        return x
