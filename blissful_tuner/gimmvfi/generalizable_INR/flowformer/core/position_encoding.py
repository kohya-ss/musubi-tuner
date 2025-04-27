from loguru import logger
import math
import torch
from torch import nn


class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256)):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        div_term = torch.exp(
            torch.arange(0, d_model // 2, 2).float()
            * (-math.log(10000.0) / d_model // 2)
        )
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, : x.size(2), : x.size(3)]


class LinearPositionEncoding(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256)):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = (
            torch.ones(max_shape).cumsum(0).float().unsqueeze(0) - 1
        ) / max_shape[0]
        x_position = (
            torch.ones(max_shape).cumsum(1).float().unsqueeze(0) - 1
        ) / max_shape[1]
        div_term = torch.arange(0, d_model // 2, 2).float()
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term * math.pi)
        pe[1::4, :, :] = torch.cos(x_position * div_term * math.pi)
        pe[2::4, :, :] = torch.sin(y_position * div_term * math.pi)
        pe[3::4, :, :] = torch.cos(y_position * div_term * math.pi)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        # assert x.shape[2] == 80 and x.shape[3] == 80

        return x + self.pe[:, :, : x.size(2), : x.size(3)]


class LearnedPositionEncoding(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(80, 80)):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
        """
        super().__init__()

        self.pe = nn.Parameter(torch.randn(1, max_shape[0], max_shape[1], d_model))

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        # assert x.shape[2] == 80 and x.shape[3] == 80

        return x + self.pe
