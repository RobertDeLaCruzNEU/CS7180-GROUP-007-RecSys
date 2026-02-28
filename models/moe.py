from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Expert(nn.Module):
    """Simple feedforward expert network."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the expert network.
        Args:
            X: Input tensor of shape (batch_size, input_dim)
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.net(X)

class MoELayer(nn.Module):
    """
    Mixture of Experts layer that routes inputs to top-k experts based on gating.
    Args:
        input_dim: Dimension of input features
        output_dim: Dimension of output features
        num_experts: Number of experts
        k: Number of top experts to route to
    """
    def __init__(self, input_dim: int, output_dim: int, num_experts: int, k=2):
        super().__init__()
        self.k = k
        self.experts = nn.ModuleList([
            Expert(input_dim, input_dim * 2, output_dim) 
            for _ in range(num_experts)
        ])
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MoE layer.
        Args:
            X: Input tensor of shape (batch_size, input_dim)
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        gate_logits = self.gate(X)
        weights, indices = torch.topk(gate_logits, self.k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        batch_size = X.size(0)
        output_dim = self.experts[0].net[-1].out_features
        combined_output = torch.zeros(batch_size, output_dim, device=X.device)

        for i in range(self.k):
            exp_indices = indices[:, i]
            exp_weights = weights[:, i].unsqueeze(1)
            
            for idx in range(len(self.experts)):
                mask = (exp_indices == idx)
                if mask.any():
                    expert_out = self.experts[idx](X[mask])
                    combined_output[mask] += exp_weights[mask] * expert_out

        return combined_output, gate_logits

class MixtureOfExperts(nn.Module, BaseEstimator, TransformerMixin):
    """
    Mixture of Experts (MoE) model for multi-objective optimization.
    Args:
        input_dim: Dimension of input features
        output_dim: Dimension of output features (number of objectives)
        num_experts: Number of experts in the MoE layer
        k: Number of top experts to route to
        epochs: Number of training epochs
        lr: Learning rate for optimizer
        batch_size: Batch size for training
        balancing_coef: Coefficient for balancing task loss and load loss
        device: Device to run the model on ("cpu" or "cuda")
        seed: Random seed for reproducibility
    """
    def __init__(
            self,
            input_dim: int,
            output_dim: int = 3,
            num_experts: int = 8,
            k: int = 2,
            epochs: int = 10,
            lr: float = 0.001,
            batch_size: int = 32,
            balancing_coef: float = 0.01,
            device: str = "cpu",
            seed: int = 42
        ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.k = k
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.balancing_coef = balancing_coef
        self.device = torch.device(device)
        self.seed = seed
        
        # Default business weights (equal priority)
        self.pareto_weights = np.ones(output_dim)

        self.moe_layer = MoELayer(input_dim, output_dim, num_experts, k)
        self.to(self.device)

    def set_pareto_weights(self, weights: list):
        """Sets the weights used for Pareto optimization in transform()."""
        if len(weights) != self.output_dim:
            raise ValueError(f"Weights must match output_dim ({self.output_dim})")
        self.pareto_weights = np.array(weights)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MoE model.
        Args:
            X: Input tensor of shape (batch_size, input_dim)
        Returns:
            Tuple of (predictions, gate_logits)
        """
        return self.moe_layer(X)

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the MoE model to the training data.
        Args:
            X: Input features of shape (num_samples, input_dim)
            y: Target values of shape (num_samples, output_dim)
        Returns:
            self
        """
        self._set_seed(self.seed)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        
        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor), 
            batch_size=self.batch_size, 
            shuffle=True
        )
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        total_steps = self.epochs * len(loader)
        pbar = tqdm(total=total_steps, desc="Training MoE")

        self.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                preds, gate_logits = self.forward(batch_x)
                
                task_loss = criterion(preds, batch_y)

                probs = F.softmax(gate_logits, dim=-1)
                importance = torch.mean(probs, dim=0)
                load_loss = torch.std(importance) / (torch.mean(importance) + 1e-6)

                total_loss = task_loss + (self.balancing_coef * load_loss)
                total_loss.backward()
                optimizer.step()

                pbar.set_postfix({"loss": f"{total_loss.item():.4f}", "epoch": epoch + 1})
                pbar.update(1)
        
        pbar.close()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates a single Pareto-optimized score for each sample.
        Useful for ranking items in a scikit-learn pipeline.
        """
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            preds, _ = self.forward(X_tensor)
            raw_preds = preds.cpu().numpy()

            scores = np.dot(raw_preds, self.pareto_weights)
            return scores.reshape(-1, 1)