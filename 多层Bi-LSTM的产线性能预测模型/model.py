import torch
from torch import nn


class ManufacturingDBiLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int = 5,
        conv_channels: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                input_size,
                conv_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Conv1d(
                conv_channels,
                conv_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
        )

        self.bilstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        embedding_size = hidden_size * 2
        if embedding_size % attention_heads != 0:
            raise ValueError(
                "2 * hidden_size must be divisible by attention_heads"
            )

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(embedding_size)

        self.attention_score = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        self.regressor = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, sequence, feature]
        conv_output = self.conv(x.transpose(1, 2)).transpose(1, 2)

        sequence_output, _ = self.bilstm(conv_output)

        attended, _ = self.attention(
            sequence_output,
            sequence_output,
            sequence_output,
            need_weights=False,
        )
        attended = self.attention_norm(
            sequence_output + attended
        )

        weights = torch.softmax(
            self.attention_score(attended),
            dim=1,
        )
        context = torch.sum(attended * weights, dim=1)
        return self.regressor(context)
