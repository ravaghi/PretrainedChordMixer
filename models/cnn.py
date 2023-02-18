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
        elif self.dataset_type == "HumanVariantEffectPrediction":
            in_features = 11200
        elif self.dataset_type == "PlantVariantEffectPrediction":
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
            y_hat = self.encoder(x)
            y_hat = y_hat.permute(0, 2, 1)
            y_hat = self.conv1(y_hat)
            y_hat = self.relu(y_hat)
            y_hat = self.pool(y_hat)
            y_hat = self.conv2(y_hat)
            y_hat = self.relu(y_hat)
            y_hat = self.pool(y_hat)
            y_hat = y_hat.view(-1, self.num_flat_features(y_hat))
            y_hat = self.dropout(y_hat)
            y_hat = self.fc1(y_hat)
            y_hat = y_hat.view(-1)

            return y_hat

        elif input_data["task"] == "HumanVariantEffectPrediction":
            x1 = input_data["x1"]
            x2 = input_data["x2"]
            tissue = input_data["tissue"]

            x1 = x1.to(torch.int64)
            y_hat_1 = self.encoder(x1)
            y_hat_1 = y_hat_1.permute(0, 2, 1)
            y_hat_1 = self.conv1(y_hat_1)
            y_hat_1 = self.relu(y_hat_1)
            y_hat_1 = self.pool(y_hat_1)
            y_hat_1 = self.conv2(y_hat_1)
            y_hat_1 = self.relu(y_hat_1)
            y_hat_1 = self.pool(y_hat_1)
            y_hat_1 = y_hat_1.view(-1, self.num_flat_features(y_hat_1))

            x2 = x2.to(torch.int64)
            y_hat_2 = self.encoder(x2)
            y_hat_2 = y_hat_2.permute(0, 2, 1)
            y_hat_2 = self.conv1(y_hat_2)
            y_hat_2 = self.relu(y_hat_2)
            y_hat_2 = self.pool(y_hat_2)
            y_hat_2 = self.conv2(y_hat_2)
            y_hat_2 = self.relu(y_hat_2)
            y_hat_2 = self.pool(y_hat_2)
            y_hat_2 = y_hat_2.view(-1, self.num_flat_features(y_hat_2))

            y_hat = y_hat_1 - y_hat_2
            y_hat = self.fc1(y_hat)

            tissue = tissue.unsqueeze(0).t()
            y_hat = torch.gather(y_hat, 1, tissue)
            y_hat = y_hat.reshape(-1)

            return y_hat

        elif input_data["task"] == "PlantVariantEffectPrediction":
            x = input_data['x']

            x = x.to(torch.int64)
            y_hat = self.encoder(x)
            y_hat = y_hat.permute(0, 2, 1)
            y_hat = self.conv1(y_hat)
            y_hat = self.relu(y_hat)
            y_hat = self.pool(y_hat)
            y_hat = self.conv2(y_hat)
            y_hat = self.relu(y_hat)
            y_hat = self.pool(y_hat)
            y_hat = y_hat.view(-1, self.num_flat_features(y_hat))
            y_hat = self.dropout(y_hat)
            y_hat = self.fc1(y_hat)

            return y_hat

        else:
            raise ValueError(f"Task: {input_data['task']} is not supported.")
