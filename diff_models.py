import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import utils_sde

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        diffusion_step = diffusion_step.to('cuda')
        x = self.embedding[diffusion_step]
        x = x.to('cuda')
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


@utils_sde.register_model(name='csdi')
class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step):

        # print("DIFFCSDI x has input shape ", x.shape)



        B, inputdim, K, L = x.shape
        #inputdim = 2:::: x.shape: (16,2,2,51)=(B,2,K,L)

        x = x.reshape(B, inputdim, K * L)
        # print("DIFFCSDI  x after input projection", x.shape)

        x = self.input_projection(x)
        # print("DIFFCSDI x after input projection", x.shape)

        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)
        # print("DIFFCSDI x reshape after relu", x.shape)


        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        # print("DIFFCSDI x after Residual", x.shape)


        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))

        # print("DIFFCSDI x after ALL NET Residual", x.shape)

        x = x.reshape(B, self.channels, K * L)
        # print("DIFFCSDI x after prior to output reshape ", x.shape)

        x = self.output_projection1(x)  # (B,channel,K*L)
        # print("DIFFCSDI x after output projection ", x.shape)
        x = F.relu(x)
        # print("DIFFCSDI x after Filter output projection ", x.shape)

        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        # print("DIFFCSDI x as final ouutput of model ", x.shape)
        # print("#######################################")
        # print("#######################################")
        # print("#######################################")

        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):

        B, channel, K, L = base_shape

        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)

        return y

    def forward(self, x, cond_info, diffusion_emb):

        # print("residual input x shape", x.shape)
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        # print("diffusion input diffusion_emb shape", diffusion_emb.shape)


        # diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        # print("diffusion input diffusion_emb resized to shape ", diffusion_emb.shape)

        # diffusion_emb = diffusion_emb.expand(-1, -1, 102)  # Expand along the third dimension
        y = x #+ diffusion_emb


        # print("adjusting y input shape from ", y.shape)

        y = self.forward_time(y, base_shape)

        # print(" y after  time transformer layer  ", y.shape)

        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        # print(" y after  feature  transformer layer  ", y.shape)

        y = self.mid_projection(y)  # (B,2*channel,K*L)
        # print(" y after adjusting feature>time transformer layer output  ", y.shape)



        _, cond_dim, _, _ = cond_info.shape  #  (B, cond_dim, K, L) (16, 145, 2, 51)  ; 145 = feature emb+ time emb +1 = 16+128+1
        cond_info = cond_info.reshape(B, cond_dim, K * L)

        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)

        # print(" Y shape before cond_info add  ", y.shape)
        # print(" cond_info shape add  ", cond_info.shape)

        y = y + cond_info

        # print(" Y shape after add  ", y.shape)


        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        # print(" y after filtering shape ", y.shape)

        y = self.output_projection(y)
        # print(" y final output projection ", y.shape)


        residual, skip = torch.chunk(y, 2, dim=1)

        # print(" y shape after residual  projection ", residual.shape)
        #


        x = x.reshape(base_shape)
        # print(" x shape as base shape ", x.shape)

        residual = residual.reshape(base_shape)
        # print(" residual as   base shape ", residual.shape)
        # print("........................")

        skip = skip.reshape(base_shape)

        return (x + residual) / math.sqrt(2.0), skip
