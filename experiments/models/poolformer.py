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
    def __init__(self, vocab_size, embedding_size, num_layers, const_vector_length, n_class, pooling, device_id):
        super(Poolformer, self).__init__()
        self.device = "cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu"
        self.const_vector_length = int(const_vector_length)
        self.encoder = nn.Embedding(vocab_size, embedding_size)
        self.posenc = nn.Embedding(const_vector_length, embedding_size)
        self.poolformer = PoolFormerModel(embedding_size, num_layers)
        self.pooling = pooling
        self.final = nn.Linear(embedding_size, n_class)
        if self.pooling == 'flatten':
            self.final = nn.Linear(embedding_size * const_vector_length, n_class)
        self.linear = nn.Linear(2, embedding_size, bias=True)

    def forward(self, x):
        x = self.encoder(x).squeeze(-2)
        positions = torch.arange(0, self.const_vector_length).expand(x.size(0), self.const_vector_length).to(
            self.device)
        x = self.posenc(positions) + x
        x = torch.permute(x, (0, 2, 1))
        x = self.poolformer(x)
        x = torch.permute(x, (0, 2, 1))
        if self.pooling == 'avg':
            x = torch.mean(x, 1)
        elif self.pooling == 'cls':
            x = x[:, 0, :]
        x = self.final(x.view(x.size(0), -1))
        return x
