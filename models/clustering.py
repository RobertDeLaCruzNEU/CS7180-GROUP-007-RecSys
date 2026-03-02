from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class ClusteringLayer(nn.Module):
    """
    Clustering layer that computes soft assignments of inputs to cluster centers.
    Args:
        n_clusters: Number of clusters
        embedding_dim: Dimension of input features
        alpha: Parameter for controlling the softness of assignments
    """
    def __init__(self, n_clusters: int, embedding_dim: int, alpha: float=1.0):
        super().__init__()
        self.alpha = alpha
        # Learnable cluster centers
        self.centroids = nn.Parameter(torch.Tensor(n_clusters, embedding_dim))
        nn.init.xavier_uniform_(self.centroids)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute soft assignments of inputs to cluster centers.
        Args:
            X: Input tensor of shape (batch_size, embedding_dim)
        Returns:
            Soft assignment tensor of shape (batch_size, n_clusters)"""
        # Calculate Student's t-distribution kernels
        norm = torch.sum((X.unsqueeze(1) - self.centroids) ** 2, dim=2)
        q = 1.0 / (1.0 + norm / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        # Normalize to get probabilities
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q

class DeepEmbeddingClustering(nn.Module, BaseEstimator, TransformerMixin):
    """
    Deep Embedded Clustering (DEC) model for unsupervised clustering.
    Args:
        n_clusters: Number of clusters
        alpha: Parameter for controlling the softness of assignments
        epochs: Number of training epochs
        lr: Learning rate for optimizer
        batch_size: Batch size for training
        device: Device to run the model on ("cpu" or "cuda")
        seed: Random seed for reproducibility
    """
    def __init__(self, n_clusters: int=10, alpha: float=1.0, epochs: int=50, lr: float=0.001, batch_size: int=32, device=None, seed: int=42):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        
        self.encoder = None
        self.clustering_layer = None
        self.device = torch.device(device)
        self.to(self.device)

    def _set_seed(self, seed: int):
        """
        Set random seed for reproducibility.
        Args:
            seed: Random seed value
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def fit(self, X: np.ndarray, y: np.ndarray=None):
        """
        Fit the DEC model to the data.
        Args:
            X: Input features of shape (num_samples, embedding_dim)
            y: Ignored (not used for DEC training)
        Returns:
            self
        """
        self._set_seed(self.seed)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        embedding_dim = X_tensor.shape[1]

        self.clustering_layer = ClusteringLayer(self.n_clusters, embedding_dim, self.alpha)
        self.to(self.device)

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
        initial_z = X_tensor.cpu().numpy()
        kmeans.fit(initial_z)
        self.clustering_layer.centroids.data = torch.tensor(kmeans.cluster_centers_).to(self.device)

        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.KLDivLoss(reduction='batchmean')
        loader = DataLoader(TensorDataset(X_tensor), batch_size=self.batch_size, shuffle=False)
        
        pbar = tqdm(total=self.epochs, desc="Training")
        self.train()
        for epoch in range(self.epochs):
            with torch.no_grad():
                q = self.clustering_layer(X_tensor)
                p = self.target_distribution(q)

            epoch_loss = 0.0
            for i, [batch] in enumerate(loader):
                batch_p = p[i * self.batch_size : (i+1) * self.batch_size]
                optimizer.zero_grad()
                q_batch = self.clustering_layer(batch)
                loss = criterion(q_batch.log(), batch_p)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            pbar.set_postfix(loss=f"{epoch_loss:.4f}", epoch=epoch + 1)
            pbar.update(1)
        
        pbar.close()
        self.fitted_ = True
        return self

    def target_distribution(self, q: torch.Tensor) -> torch.Tensor:
        """
        Compute target distribution for clustering.
        Args:
            q: Soft assignment tensor of shape (num_samples, n_clusters)
        Returns:
            Target distribution tensor of shape (num_samples, n_clusters)
        """
        weight = q**2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform input data to soft cluster assignments.
        Args:
            X: Input features of shape (num_samples, embedding_dim)
        Returns:
            Soft assignment tensor of shape (num_samples, n_clusters)
        """
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            q = self.clustering_layer(X_tensor)
            return q.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster assignments for input data.
        Args:
            X: Input features of shape (num_samples, embedding_dim)
        Returns:
            Cluster labels of shape (num_samples,)
        """
        q = self.transform(X)
        return np.argmax(q, axis=1)