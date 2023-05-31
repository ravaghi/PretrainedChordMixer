import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool1d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        x = self.pool(x) - x
        return x


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv1d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv1d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PoolFormerBlock(nn.Module):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth, 
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale, 
        refer to https://arxiv.org/abs/2103.17239
    """

    def __init__(self, dim, pool_size=3, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PoolFormerModel(nn.Module):
    def __init__(self, embedding_size, num_layers, pool_size=3, mlp_ratio=4., act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop_rate=.0, drop_path_rate=0., use_layer_scale=False, layer_scale_init_value=1e-5):
        super(PoolFormerModel, self).__init__()
        self.layers = num_layers
        self.pool_former_nn = nn.ModuleList(
            PoolFormerBlock(
                embedding_size, pool_size=pool_size, mlp_ratio=mlp_ratio,
                act_layer=act_layer, norm_layer=norm_layer,
                drop=drop_rate, drop_path=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            )
            for i in range(self.layers)
        )

    def forward(self, x):
        for mixer_block in self.pool_former_nn:
            x = mixer_block(x)
        return x


class Poolformer(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_layers, dataset_type, n_class, device_id):
        super(Poolformer, self).__init__()
        self.device = "cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu"

        if dataset_type == "TaxonomyClassification":
            self.sequence_length = 25000
        else:
            self.sequence_length = 1000

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.positional_encoder = nn.Embedding(self.sequence_length, embedding_size)
        self.poolformer = PoolFormerModel(embedding_size, num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(4 * embedding_size, embedding_size // 2),
            nn.ReLU(),
            nn.Linear(embedding_size // 2, n_class)
        )

    def forward(self, input_data):
        if input_data["task"] == "TaxonomyClassification":
            x = input_data["x"]

            y_hat = self.embedding(x).squeeze(-2)
            positions = torch.arange(0, self.sequence_length) \
                .expand(y_hat.size(0), self.sequence_length) \
                .to(self.device)
            y_hat = self.positional_encoder(positions) + y_hat
            y_hat = torch.permute(y_hat, (0, 2, 1))
            y_hat = self.poolformer(y_hat)
            y_hat = torch.permute(y_hat, (0, 2, 1))
            y_hat = y_hat.view(y_hat.size(0), -1)
            y_hat = self.classifier(y_hat)
            y_hat = y_hat.view(-1)

            return y_hat

        elif input_data["task"] == "HumanVariantEffectPrediction":
            x1 = input_data["x1"]
            x2 = input_data["x2"]
            tissue = input_data["tissue"]

            y_hat_1 = self.embedding(x1).squeeze(-2)
            positions_1 = torch.arange(0, self.sequence_length) \
                .expand(y_hat_1.size(0), self.sequence_length) \
                .to(self.device)
            y_hat_1 = self.positional_encoder(positions_1) + y_hat_1
            y_hat_1 = torch.permute(y_hat_1, (0, 2, 1))
            y_hat_1 = self.poolformer(y_hat_1)
            y_hat_1 = torch.permute(y_hat_1, (0, 2, 1))

            y_hat_2 = self.embedding(x2).squeeze(-2)
            positions_2 = torch.arange(0, self.sequence_length) \
                .expand(y_hat_2.size(0), self.sequence_length) \
                .to(self.device)
            y_hat_2 = self.positional_encoder(positions_2) + y_hat_2
            y_hat_2 = torch.permute(y_hat_2, (0, 2, 1))
            y_hat_2 = self.poolformer(y_hat_2)
            y_hat_2 = torch.permute(y_hat_2, (0, 2, 1))

            y_hat_1 = torch.mean(y_hat_1, dim=1)
            y_hat_2 = torch.mean(y_hat_2, dim=1)

            y_hat = torch.cat([y_hat_1, y_hat_2, y_hat_1 * y_hat_2, y_hat_1 - y_hat_2], dim=1)
            y_hat = self.classifier(y_hat)

            tissue = tissue.unsqueeze(0).t()
            y_hat = torch.gather(y_hat, 1, tissue)
            y_hat = y_hat.reshape(-1)

            return y_hat

        elif input_data["task"] == "PlantVariantEffectPrediction":
            x = input_data["x"]

            y_hat = self.embedding(x).squeeze(-2)
            positions = torch.arange(0, self.sequence_length) \
                .expand(y_hat.size(0), self.sequence_length) \
                .to(self.device)
            y_hat = self.positional_encoder(positions) + y_hat
            y_hat = torch.permute(y_hat, (0, 2, 1))
            y_hat = self.poolformer(y_hat)
            y_hat = torch.permute(y_hat, (0, 2, 1))
            y_hat = y_hat.view(y_hat.size(0), -1)
            y_hat = self.classifier(y_hat)

            return y_hat

        else:
            raise ValueError(f"Task: {input_data['task']} is not supported.")
