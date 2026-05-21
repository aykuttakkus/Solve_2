import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ---------------------------------------------------------------------------
# Block 1: Convolutional Denoising Autoencoder (DAE)
# Week 13 — Parts 4 & 8
# ---------------------------------------------------------------------------

class ConvDAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: 112 -> 56 -> 28 -> 14 -> 7
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1,   32,  3, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Conv2d(32,  64,  3, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64,  128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.encoder_fc = nn.Linear(256 * 7 * 7, latent_dim)

        # Decoder: 7 -> 14 -> 28 -> 56 -> 112
        self.decoder_fc = nn.Linear(latent_dim, 256 * 7 * 7)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.ConvTranspose2d(32,  1,   4, stride=2, padding=1), nn.Sigmoid(),
        )

    def encode(self, x):
        # x: (B, 1, 112, 112)
        h = self.encoder_conv(x)          # (B, 256, 7, 7)
        h = h.view(h.size(0), -1)         # (B, 256*7*7)
        return self.encoder_fc(h)         # (B, latent_dim)

    def decode(self, z):
        h = self.decoder_fc(z)            # (B, 256*7*7)
        h = h.view(h.size(0), 256, 7, 7) # (B, 256, 7, 7)
        return self.decoder_conv(h)       # (B, 1, 112, 112)

    def forward(self, x_noisy):
        z = self.encode(x_noisy)
        x_hat = self.decode(z)
        return x_hat, z


# ---------------------------------------------------------------------------
# Block 2: Frame-Level CNN Encoder
# Week 9 — Parts 6 & 7 (BetterCNN pattern)
# ---------------------------------------------------------------------------

class CNNEncoder(nn.Module):
    def __init__(self, cnn_dim=256):
        super().__init__()
        self.cnn_dim = cnn_dim

        # Same stride-2 structure as DAE encoder (enables weight transfer)
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1,   32,  3, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Conv2d(32,  64,  3, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64,  128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        # Global spatial pooling then projection
        self.pool = nn.MaxPool2d(7)
        self.fc   = nn.Linear(256, cnn_dim)

    def forward(self, x):
        # x: (B*T, 1, 112, 112)
        h = self.conv_blocks(x)      # (B*T, 256, 7, 7)
        h = self.pool(h)             # (B*T, 256, 1, 1)
        h = h.view(h.size(0), -1)   # (B*T, 256)
        return self.fc(h)            # (B*T, cnn_dim)


def load_dae_weights(cnn_encoder, dae):
    """Transfer the 4 conv block weights from a trained ConvDAE to CNNEncoder."""
    dae_state  = dae.encoder_conv.state_dict()
    cnn_state  = cnn_encoder.conv_blocks.state_dict()
    cnn_encoder.conv_blocks.load_state_dict(dae_state)
    print("DAE weights transferred to CNNEncoder conv_blocks.")


# ---------------------------------------------------------------------------
# Block 3: Bidirectional LSTM
# Week 10 — Parts 4, 5, 8
# ---------------------------------------------------------------------------

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self._init_forget_bias(1.0)

    def _init_forget_bias(self, value):
        # Jozefowicz et al. (2015): initialise forget gate bias to 1
        # Week 10 Part 4
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                n = param.size(0)
                # forget gate occupies the second quarter of the bias vector
                param.data[n // 4: n // 2].fill_(value)

    def forward(self, x, lengths):
        # x: (B, T, input_size)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        output_packed, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(output_packed, batch_first=True)
        # output: (B, T, 2*hidden_size)
        return output


# ---------------------------------------------------------------------------
# Block 4: GRU Temporal Summarizer
# Week 10 — Part 7
# ---------------------------------------------------------------------------

class GRUSummarizer(nn.Module):
    def __init__(self, input_size=512, hidden_size=256):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self._init_update_bias(1.0)

    def _init_update_bias(self, value):
        # Week 10 Part 4: initialise update gate bias to 1
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                n = param.size(0)
                param.data[n // 3: 2 * n // 3].fill_(value)

    def forward(self, x):
        # x: (B, T, input_size)
        output, h_n = self.gru(x)
        # output : (B, T, hidden_size)  — full sequence for attention
        # h_n    : (1, B, hidden_size)  — final hidden state
        summary = h_n[-1]              # (B, hidden_size)
        return summary, output


# ---------------------------------------------------------------------------
# Block 5: Sparse Autoencoder Bottleneck
# Week 13 — Part 6
# ---------------------------------------------------------------------------

class SparseAEBottleneck(nn.Module):
    def __init__(self, input_size=256, K=128, sparsity_lambda=0.05):
        super().__init__()
        self.sparsity_lambda = sparsity_lambda

        # Sparse encoder: ReLU ensures non-negative codes
        self.encoder = nn.Linear(input_size, K)

        # Auxiliary decoder: reconstructs GRU hidden state
        self.decoder = nn.Linear(K, input_size)

        # Regression head: predicts EF in [0, 100]
        self.regressor = nn.Sequential(
            nn.Linear(K, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, h_gru):
        # h_gru: (B, input_size)
        z       = torch.relu(self.encoder(h_gru))   # (B, K) — sparse codes
        h_recon = self.decoder(z)                    # (B, input_size)
        ef_pred = self.regressor(z) * 100.0          # (B, 1) in [0, 100]
        return ef_pred.squeeze(1), z, h_recon

    def loss(self, ef_pred, ef_true, z, h_gru, h_recon):
        L_ef     = nn.functional.mse_loss(ef_pred, ef_true)
        L_recon  = nn.functional.mse_loss(h_recon, h_gru.detach())
        L_sparse = z.abs().mean()
        return L_ef + 0.1 * L_recon + self.sparsity_lambda * L_sparse
