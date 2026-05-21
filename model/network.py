import torch
import torch.nn as nn
from model.blocks import (
    ConvDAE, CNNEncoder, BiLSTMEncoder,
    GRUSummarizer, SparseAEBottleneck, load_dae_weights
)


class EchoCardioNet(nn.Module):
    def __init__(
        self,
        latent_dim=256,
        cnn_dim=256,
        H_lstm=256,
        H_gru=256,
        K=128,
        sparsity_lambda=0.05,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cnn_dim    = cnn_dim

        self.dae    = ConvDAE(latent_dim)
        self.cnn    = CNNEncoder(cnn_dim)
        self.bilstm = BiLSTMEncoder(
            input_size=latent_dim + cnn_dim,
            hidden_size=H_lstm,
        )
        self.gru    = GRUSummarizer(
            input_size=2 * H_lstm,
            hidden_size=H_gru,
        )
        self.sparse = SparseAEBottleneck(
            input_size=H_gru,
            K=K,
            sparsity_lambda=sparsity_lambda,
        )

    def transfer_dae_weights(self):
        """Copy DAE encoder conv weights into CNNEncoder (call after DAE pre-training)."""
        load_dae_weights(self.cnn, self.dae)

    def forward(self, frames, lengths):
        """
        frames  : (B, T, 1, 112, 112)
        lengths : (B,)  — actual frame count per sample (sorted descending)
        """
        B, T, C, H, W = frames.shape
        frames_flat = frames.view(B * T, C, H, W)   # (B*T, 1, 112, 112)

        # Block 1: DAE encode — unsupervised latent codes
        z_dae = self.dae.encode(frames_flat)          # (B*T, latent_dim)
        z_dae = z_dae.view(B, T, self.latent_dim)     # (B, T, latent_dim)

        # Block 2: CNN encode — supervised spatial features
        z_cnn = self.cnn(frames_flat)                 # (B*T, cnn_dim)
        z_cnn = z_cnn.view(B, T, self.cnn_dim)        # (B, T, cnn_dim)

        # Concat frame-level representations
        z_fused = torch.cat([z_dae, z_cnn], dim=-1)  # (B, T, latent_dim+cnn_dim)

        # Block 3: BiLSTM — long-range temporal context
        z_seq = self.bilstm(z_fused, lengths)         # (B, T, 2*H_lstm)

        # Block 4: GRU — compact summarization
        z_summary, gru_output = self.gru(z_seq)       # (B, H_gru), (B, T, H_gru)

        # Block 5: Sparse AE — interpretable bottleneck + regression
        ef_pred, z_sparse, h_recon = self.sparse(z_summary)  # (B,), (B,K), (B,H_gru)

        return ef_pred, z_sparse, h_recon, z_summary, gru_output

    def compute_loss(self, frames, lengths, ef_true, noise_std=0.3):
        """
        Full multi-task loss:
        L = L_EF + 0.1*L_DAE + L_sparse (from SparseAEBottleneck.loss)
        """
        B, T, C, H, W = frames.shape
        frames_flat = frames.view(B * T, C, H, W)

        # Forward pass
        ef_pred, z_sparse, h_recon, z_summary, _ = self.forward(frames, lengths)

        # Primary regression loss
        L_EF = nn.functional.mse_loss(ef_pred, ef_true)

        # Auxiliary DAE reconstruction loss (keeps encoder anatomy-aware)
        x_noisy = frames_flat + torch.randn_like(frames_flat) * noise_std
        x_hat, _ = self.dae(x_noisy)
        L_DAE = nn.functional.mse_loss(x_hat, frames_flat)

        # Sparse AE loss (reconstruction + sparsity)
        L_sparse = self.sparse.loss(ef_pred, ef_true, z_sparse, z_summary, h_recon)

        return L_EF + 0.1 * L_DAE + L_sparse
