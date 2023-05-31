import numpy as np
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, n_class, dataset_type):
        super(CNN, self).__init__()

        if dataset_type == "TaxonomyClassification":
            sequence_length = 25_000
        elif dataset_type in ["VariantEffectPrediction", "PlantOcrPrediction"]:
            sequence_length = 1000

        conv_kernel_size = 8
        pool_kernel_size = 4

        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(320, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size,
                stride=pool_kernel_size
            ),
            nn.BatchNorm1d(320),

            nn.Conv1d(320, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(480, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size,
                stride=pool_kernel_size
            ),
            nn.BatchNorm1d(480),
            nn.Dropout(p=0.2),

            nn.Conv1d(480, 960, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(960, 960, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(960),
            nn.Dropout(p=0.2)
        )

        reduce_by = 2 * (conv_kernel_size - 1)
        pool_kernel_size = float(pool_kernel_size)
        self._n_channels = int(
            np.floor(
                (np.floor((sequence_length - reduce_by) / pool_kernel_size) - reduce_by) / pool_kernel_size
            ) - reduce_by
        )

        self.classifier = nn.Sequential(
            nn.Linear(960 * self._n_channels, n_class),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(n_class),
            nn.Linear(n_class, n_class)
        )

    def forward(self, input_data):
        if input_data["task"] == "TaxonomyClassification":
            x = input_data['x']

            x = x.permute(0, 2, 1)
            y_hat = self.conv_net(x)
            y_hat = y_hat.view(y_hat.size(0), 960 * self._n_channels)
            y_hat = self.classifier(y_hat)
            y_hat = y_hat.view(-1)

            return y_hat

        elif input_data["task"] == "VariantEffectPrediction":
            x1 = input_data["x1"]
            x2 = input_data["x2"]
            tissue = input_data["tissue"]

            x1 = x1.permute(0, 2, 1)
            y_hat_1 = self.conv_net(x1)

            x2 = x2.permute(0, 2, 1)
            y_hat_2 = self.conv_net(x2)

            y_hat = y_hat_1 - y_hat_2
            y_hat = y_hat.view(y_hat.size(0), 960 * self._n_channels)
            y_hat = self.classifier(y_hat)

            tissue = tissue.unsqueeze(0).t()
            y_hat = torch.gather(y_hat, 1, tissue)
            y_hat = y_hat.reshape(-1)

            return y_hat

        elif input_data["task"] == "PlantOcrPrediction":
            x = input_data['x']

            x = x.permute(0, 2, 1)
            y_hat = self.conv_net(x)
            y_hat = y_hat.view(y_hat.size(0), 960 * self._n_channels)
            y_hat = self.classifier(y_hat)

            return y_hat