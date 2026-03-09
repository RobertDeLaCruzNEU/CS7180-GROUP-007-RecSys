from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Autoencoder(nn.Module, BaseEstimator, TransformerMixin):
    """
    Autoencoder model for dimensionality reduction.
    Args:
        encoder: List of layers for the encoder part of the autoencoder
        decoder: List of layers for the decoder part of the autoencoder
        epochs: Number of training epochs
        lr: Learning rate for optimizer
        batch_size: Batch size for training
        min_delta: Minimum change in loss to trigger early stopping
        device: Device to run the model on ("cpu" or "cuda")
        seed: Random seed for reproducibility
    """
    def __init__(
            self,
            encoder: list,
            decoder: list,
            epochs: int = 10,
            lr: float = 0.001,
            batch_size: int = 32,
            min_delta: float = 0.0,
            device: str = "cpu",
            seed: int = 42
        ):
            super().__init__()
            # Crucial: Store these exactly as passed for sklearn compatibility
            self.encoder = encoder
            self.decoder = decoder
            self.seed = seed
            self.epochs = epochs
            self.lr = lr
            self.batch_size = batch_size
            self.min_delta = min_delta
            self.device = device

            # Use different names for the actual PyTorch modules
            self.encoder_net_ = nn.Sequential(*self.encoder)
            self.decoder_net_ = nn.Sequential(*self.decoder)
            self.device_ = torch.device(device)
            
            self.losses = []
            self.to(self.device_)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        Args:
            X: Input tensor of shape (batch_size, input_dim)
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        z = self.encoder_net_(X)
        return self.decoder_net_(z)
    
    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def fit(self, X: np.ndarray, y: np.ndarray=None):
        """
        Fit the autoencoder to the training data.
        Args:
            X: Input features of shape (num_samples, input_dim)
            y: Ignored (not used for autoencoder training)
        Returns: 
            self
        """
        self._set_seed(self.seed)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device_)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        total_batches = self.epochs * len(loader)
        pbar = tqdm(total=total_batches, desc="Training")

        self.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for [batch] in loader:
                optimizer.zero_grad()
                output = self.forward(batch)
                loss = criterion(output, batch)
                loss.backward()
                optimizer.step()
                
                current_loss = loss.item()
                epoch_loss += current_loss
                
                pbar.set_postfix(loss=f"{current_loss:.4f}", epoch=epoch + 1)
                pbar.update(1)

            avg_loss = epoch_loss / len(loader)
            self.losses.append(avg_loss)

            if len(self.losses) > 1:
                change = abs(self.losses[-2] - self.losses[-1])
                if change <= self.min_delta:
                    pbar.write(f"Early stopping at epoch {epoch + 1}")
                    break
        
        pbar.close()
        self.fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform input data to latent space using the encoder.
        Args:
            X: Input features of shape (num_samples, input_dim)
        Returns:
            Latent representations of shape (num_samples, latent_dim)
        """
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device_)
            z = self.encoder_net_(X_tensor)
            return z.cpu().numpy()