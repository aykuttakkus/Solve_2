import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from model.blocks import (
    ConvDAE, CNNEncoder, BiLSTMEncoder,
    GRUSummarizer, SparseAEBottleneck,
)
from model.network import EchoCardioNet
from model.dataset import EchoDataset, pad_collate


# ---------------------------------------------------------------------------
# Phase 1 Training — DAE Pre-training
# ---------------------------------------------------------------------------

def pretrain_dae(model, train_loader, epochs=10, lr=3e-4, device='cpu', save_path='results/checkpoints/dae_pretrained.pt'):
    model.to(device)
    optimizer = Adam(model.dae.parameters(), lr=lr)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for frames, _, _, lengths in train_loader:
            frames = frames.to(device)
            B, T, C, H, W = frames.shape
            frames_flat = frames.view(B * T, C, H, W)

            x_noisy = frames_flat + torch.randn_like(frames_flat) * 0.3
            x_noisy = torch.clamp(x_noisy, 0.0, 1.0)

            x_hat, _ = model.dae(x_noisy)
            loss = nn.functional.mse_loss(x_hat, frames_flat)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.dae.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(train_loader)
        history.append(avg)
        print(f"[DAE] Epoch {epoch:02d}/{epochs}  loss={avg:.5f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.dae.state_dict(), save_path)
    print(f"DAE checkpoint saved → {save_path}")
    return history


# ---------------------------------------------------------------------------
# Phase 2 Training — End-to-End Fine-tuning
# ---------------------------------------------------------------------------

def train_full(model, train_loader, val_loader, epochs=15, device='cpu',
               save_path='results/checkpoints/model_F_full.pt'):
    model.to(device)

    param_groups = [
        {'params': model.dae.parameters(),    'lr': 3e-5},   # 0.1× — pre-trained
        {'params': model.cnn.parameters(),    'lr': 3e-4},
        {'params': model.bilstm.parameters(), 'lr': 3e-4},
        {'params': model.gru.parameters(),    'lr': 3e-4},
        {'params': model.sparse.parameters(), 'lr': 3e-4},
    ]
    optimizer = Adam(param_groups, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    train_losses, val_maes = [], []
    best_val_mae = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for frames, ef_true, _, lengths in train_loader:
            frames  = frames.to(device)
            ef_true = ef_true.to(device)
            lengths = lengths.to(device)

            loss = model.compute_loss(frames, lengths, ef_true)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_mae  = evaluate(model, val_loader, device)['MAE']

        train_losses.append(avg_loss)
        val_maes.append(val_mae)
        scheduler.step()

        print(f"[Train] Epoch {epoch:02d}/{epochs}  loss={avg_loss:.4f}  val_MAE={val_mae:.3f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

    print(f"Best val MAE: {best_val_mae:.3f}  checkpoint → {save_path}")
    return {'train_loss': train_losses, 'val_mae': val_maes}


# ---------------------------------------------------------------------------
# Ablation Model
# ---------------------------------------------------------------------------

class AblationModel(nn.Module):
    """
    Configurable model for ablation variants A–F.

    variant | dae_init | bidirectional | use_gru | use_sparse_ae
    A       | False    | False         | False   | False
    B       | True     | False         | False   | False
    C       | True     | True          | False   | False
    D       | True     | True          | True    | False
    E/F     | True     | True          | True    | True
    """
    def __init__(self, dae_init=False, bidirectional=True,
                 use_gru=True, use_sparse_ae=True,
                 latent_dim=256, cnn_dim=256, H_lstm=256, H_gru=256, K=128):
        super().__init__()
        self.use_gru       = use_gru
        self.use_sparse_ae = use_sparse_ae
        self.latent_dim    = latent_dim
        self.cnn_dim       = cnn_dim

        self.dae = ConvDAE(latent_dim)
        self.cnn = CNNEncoder(cnn_dim)

        lstm_in  = latent_dim + cnn_dim
        lstm_out = 2 * H_lstm if bidirectional else H_lstm
        self.bilstm = BiLSTMEncoder(
            input_size=lstm_in, hidden_size=H_lstm,
        )
        # Override bidirectional setting
        self.bilstm.lstm = nn.LSTM(
            input_size=lstm_in, hidden_size=H_lstm,
            num_layers=2, bidirectional=bidirectional,
            dropout=0.2, batch_first=True,
        )

        if use_gru:
            self.gru = GRUSummarizer(input_size=lstm_out, hidden_size=H_gru)
            head_in  = H_gru
        else:
            self.gru = None
            head_in  = lstm_out

        if use_sparse_ae:
            self.head = SparseAEBottleneck(input_size=head_in, K=K)
        else:
            self.head = nn.Sequential(
                nn.Linear(head_in, 64), nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1), nn.Sigmoid(),
            )

        if dae_init:
            from model.blocks import load_dae_weights
            load_dae_weights(self.cnn, self.dae)

    def forward(self, frames, lengths):
        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W)

        z_dae = self.dae.encode(flat).view(B, T, self.latent_dim)
        z_cnn = self.cnn(flat).view(B, T, self.cnn_dim)
        z_fused = torch.cat([z_dae, z_cnn], dim=-1)

        z_seq = self.bilstm(z_fused, lengths)

        if self.use_gru:
            z_vec, _ = self.gru(z_seq)
        else:
            z_vec = z_seq.mean(dim=1)  # mean pooling over time

        if self.use_sparse_ae:
            ef_pred, _, _ = self.head(z_vec)
        else:
            ef_pred = self.head(z_vec).squeeze(1) * 100.0

        return ef_pred

    def compute_loss(self, frames, lengths, ef_true, noise_std=0.3):
        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W)

        ef_pred = self.forward(frames, lengths)
        L_EF    = nn.functional.mse_loss(ef_pred, ef_true)

        x_noisy = flat + torch.randn_like(flat) * noise_std
        x_hat, _ = self.dae(x_noisy)
        L_DAE = nn.functional.mse_loss(x_hat, flat)

        return L_EF + 0.1 * L_DAE


ABLATION_CONFIGS = {
    'A': dict(dae_init=False, bidirectional=False, use_gru=False, use_sparse_ae=False),
    'B': dict(dae_init=True,  bidirectional=False, use_gru=False, use_sparse_ae=False),
    'C': dict(dae_init=True,  bidirectional=True,  use_gru=False, use_sparse_ae=False),
    'D': dict(dae_init=True,  bidirectional=True,  use_gru=True,  use_sparse_ae=False),
    'E': dict(dae_init=True,  bidirectional=True,  use_gru=True,  use_sparse_ae=True),
    'F': dict(dae_init=True,  bidirectional=True,  use_gru=True,  use_sparse_ae=True),
}


def run_ablation(train_loader, val_loader, test_loader,
                 epochs=8, device='cpu',
                 save_dir='results/checkpoints'):
    os.makedirs(save_dir, exist_ok=True)
    results = {}

    for variant_id, cfg in ABLATION_CONFIGS.items():
        print(f"\n{'='*50}")
        print(f"Ablation variant {variant_id}: {cfg}")
        print('='*50)

        torch.manual_seed(42)
        model = AblationModel(**cfg).to(device)
        optimizer = Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        for epoch in range(1, epochs + 1):
            model.train()
            total = 0.0
            for frames, ef_true, _, lengths in train_loader:
                frames  = frames.to(device)
                ef_true = ef_true.to(device)
                lengths = lengths.to(device)
                loss = model.compute_loss(frames, lengths, ef_true)
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += loss.item()
            scheduler.step()
            print(f"  Epoch {epoch}/{epochs}  loss={total/len(train_loader):.4f}")

        ckpt = os.path.join(save_dir, f'model_{variant_id}.pt')
        torch.save(model.state_dict(), ckpt)

        metrics = evaluate_ablation(model, test_loader, device)
        results[variant_id] = metrics
        print(f"  Test — MAE={metrics['MAE']:.3f}  RMSE={metrics['RMSE']:.3f}  R2={metrics['R2']:.3f}")

    out_path = 'results/ablation_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation results saved → {out_path}")
    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, loader, device='cpu'):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for frames, ef_true, _, lengths in loader:
            frames  = frames.to(device)
            lengths = lengths.to(device)
            ef_pred, *_ = model(frames, lengths)
            preds.append(ef_pred.cpu())
            trues.append(ef_true)

    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()

    mae  = float(np.mean(np.abs(preds - trues)))
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    ss_res = np.sum((preds - trues) ** 2)
    ss_tot = np.sum((trues - trues.mean()) ** 2)
    r2   = float(1 - ss_res / (ss_tot + 1e-8))

    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'preds': preds, 'trues': trues}


def evaluate_ablation(model, loader, device='cpu'):
    """Same as evaluate() but works with AblationModel (single output)."""
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for frames, ef_true, _, lengths in loader:
            frames  = frames.to(device)
            lengths = lengths.to(device)
            ef_pred = model(frames, lengths)
            preds.append(ef_pred.cpu())
            trues.append(ef_true)

    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()

    mae  = float(np.mean(np.abs(preds - trues)))
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    ss_res = np.sum((preds - trues) ** 2)
    ss_tot = np.sum((trues - trues.mean()) ** 2)
    r2   = float(1 - ss_res / (ss_tot + 1e-8))

    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}


# ---------------------------------------------------------------------------
# Plot Functions
# ---------------------------------------------------------------------------

def plot_training_curves(history, save_path='results/figures/training_curves.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('Training Loss')
    ax1.legend(); ax1.grid(True)

    ax2.plot(history['val_mae'], color='orange', label='Val MAE')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('MAE (%)'); ax2.set_title('Validation MAE')
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved → {save_path}")


def plot_dae_reconstructions(model, loader, device='cpu',
                              save_path='results/figures/dae_reconstructions.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.eval()
    frames, *_ = next(iter(loader))
    frames = frames.to(device)
    B, T, C, H, W = frames.shape
    flat = frames.view(B * T, C, H, W)[:8]

    with torch.no_grad():
        noisy = flat + torch.randn_like(flat) * 0.3
        noisy = torch.clamp(noisy, 0.0, 1.0)
        recon, _ = model.dae(noisy)

    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    titles = ['Original', 'Noisy', 'Reconstructed']
    for col in range(8):
        for row, img in enumerate([flat[col], noisy[col], recon[col]]):
            axes[row, col].imshow(img.squeeze().cpu().numpy(), cmap='gray')
            axes[row, col].axis('off')
        axes[0, col].set_title(f'#{col}', fontsize=8)
    for row, t in enumerate(titles):
        axes[row, 0].set_ylabel(t, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved → {save_path}")


def plot_scatter(trues, preds, save_path='results/figures/scatter_pred_vs_true.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    colors = ['green' if t >= 55 else ('orange' if t >= 40 else 'red') for t in trues]

    plt.figure(figsize=(6, 6))
    plt.scatter(trues, preds, c=colors, alpha=0.5, s=20)
    lim = [min(trues.min(), preds.min()) - 2, max(trues.max(), preds.max()) + 2]
    plt.plot(lim, lim, 'k--', lw=1, label='y = x')
    plt.xlabel('True EF (%)'); plt.ylabel('Predicted EF (%)')
    plt.title('Predicted vs True EF')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved → {save_path}")


def plot_ablation(results, save_path='results/figures/ablation_barplot.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    variants = list(results.keys())
    maes  = [results[v]['MAE']  for v in variants]
    rmses = [results[v]['RMSE'] for v in variants]

    x = np.arange(len(variants))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.2, maes,  0.35, label='MAE',  color='steelblue')
    ax.bar(x + 0.2, rmses, 0.35, label='RMSE', color='salmon')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Model {v}' for v in variants])
    ax.set_ylabel('Error (%)'); ax.set_title('Ablation Study')
    ax.legend(); ax.grid(axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved → {save_path}")


def plot_attention_weights(model, loader, device='cpu',
                            save_path='results/figures/attention_heatmap.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.eval()
    frames, ef_true, _, lengths = next(iter(loader))
    frames  = frames.to(device)
    lengths = lengths.to(device)

    with torch.no_grad():
        _, _, _, z_summary, gru_output = model(frames, lengths)
        # Attention: project each time step hidden state to scalar
        attn = torch.softmax(gru_output.norm(dim=-1), dim=-1)  # (B, T)

    n = min(3, frames.shape[0])
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n))
    if n == 1:
        axes = [axes]

    for i in range(n):
        T = lengths[i].item()
        w = attn[i, :T].cpu().numpy()
        axes[i].bar(range(T), w, color='steelblue')
        axes[i].set_title(f'Sample {i+1}  EF={ef_true[i]:.1f}%')
        axes[i].set_xlabel('Frame'); axes[i].set_ylabel('Attention')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved → {save_path}")


def plot_ef_distribution(df, save_path='results/figures/eda_ef_distribution.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ef = df['EF'].values
    plt.figure(figsize=(8, 5))
    plt.hist(ef[ef >= 55], bins=20, color='green',  alpha=0.7, label='Normal (≥55%)')
    plt.hist(ef[(ef >= 40) & (ef < 55)], bins=20, color='orange', alpha=0.7, label='Mild (40-55%)')
    plt.hist(ef[ef < 40],  bins=20, color='red',    alpha=0.7, label='Dysfunction (<40%)')
    plt.axvline(55, color='green',  linestyle='--', lw=1)
    plt.axvline(40, color='red',    linestyle='--', lw=1)
    plt.xlabel('EF (%)'); plt.ylabel('Count')
    plt.title('EF Distribution — A4C Train Split')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved → {save_path}")
