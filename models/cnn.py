import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, vocab_size, feature_size, n_class, dataset_type):
        super(CNN, self).__init__()
        self.vocab_size = vocab_size
        self.feature_size = feature_size
        self.n_class = n_class
        self.dataset_type = dataset_type

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
        self.dropout = nn.Dropout(p=0.5)
        if self.dataset_type == "TaxonomyClassification": 
            in_features = 277600
        elif self.dataset_type == "VariantEffectPrediction":
            in_features = 11200
        elif self.dataset_type == "PlantDeepSEA":
            in_features = 11200
        self.fc1 = nn.Linear(in_features, self.n_class)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, input_data):
        if input_data["task"] == "TaxonomyClassification":
            x = input_data['x']
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
        
        elif input_data["task"] == "VariantEffectPrediction":
            x1 = input_data["x1"]
            x2 = input_data["x2"]
            tissue = input_data["tissue"]

            x1 = x1.to(torch.int64)
            x1 = self.encoder(x1)
            x1 = x1.permute(0, 2, 1)
            x1 = self.conv1(x1)
            x1 = self.relu(x1)
            x1 = self.pool(x1)
            x1 = self.conv2(x1)
            x1 = self.relu(x1)
            x1 = self.pool(x1)
            x1 = x1.view(-1, self.num_flat_features(x1))
            #x1 = self.dropout(x1)

            x2 = x2.to(torch.int64)
            x2 = self.encoder(x2)
            x2 = x2.permute(0, 2, 1)
            x2 = self.conv1(x2)
            x2 = self.relu(x2)
            x2 = self.pool(x2)
            x2 = self.conv2(x2)
            x2 = self.relu(x2)
            x2 = self.pool(x2)
            x2 = x2.view(-1, self.num_flat_features(x2))
            #x2 = self.dropout(x2)

            x = x1 - x2
            data = self.fc1(x)

            tissue = tissue.unsqueeze(0).t()
            data = torch.gather(data, 1, tissue)  
            data = data.reshape(-1)

            return data

        elif input_data["task"] == "PlantDeepSEA":
            x = input_data['x']

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
        
        else:
            raise ValueError(f"Task: {input_data['task']} is not supported.")
