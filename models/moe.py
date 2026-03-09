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
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
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
        self.num_experts = num_experts
        self.output_dim = output_dim
        self.experts = nn.ModuleList([
            Expert(input_dim, input_dim * 2, output_dim) 
            for _ in range(num_experts)
        ])
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, X: torch.Tensor):
        gate_logits = self.gate(X)
        weights, indices = torch.topk(gate_logits, self.k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        batch_size = X.size(0)
        # Use self.output_dim instead of inspecting the expert layers
        combined_output = torch.zeros(batch_size, self.output_dim, device=X.device)

        for i in range(self.k):
            for expert_idx in range(self.num_experts):
                mask = (indices[:, i] == expert_idx)
                if mask.any():
                    expert_out = self.experts[expert_idx](X[mask])
                    combined_output[mask] += weights[mask, i].unsqueeze(1) * expert_out

        return combined_output, gate_logits

class MixtureOfExperts(nn.Module, BaseEstimator, TransformerMixin):
    def __init__(
            self, 
            input_dim:int,
            output_dim:int,
            num_experts:int=8, 
            k:int=2, 
            epochs:int=10, 
            lr:float=0.001, 
            batch_size:int=32, 
            min_delta: int = 1e-4, 
            patience: int = 3,
            balancing_coef:float=0.01,
            device:str="cpu", 
            seed:int=42
        ):
        super().__init__() 
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pareto_weights = np.ones(self.output_dim)

        # 2. Store parameters (Sklearn requires these names to match __init__ args)
        self.num_experts = num_experts
        self.k = k
        self.epochs = epochs
        self.lr = lr
        self.min_delta = min_delta
        self.patience = patience
        self.batch_size = batch_size
        self.balancing_coef = balancing_coef
        self.device = device 
        self.seed = seed
        
        # 3. Create the layers (Must be assigned to self to register parameters)
        self._device = torch.device(device)
        # 4. Move model to device
        self.to(self._device)
        self.moe_layer_ = MoELayer(input_dim, output_dim, num_experts, k)
        self.fitted_ = False

    def set_pareto_weights(self, weights: list):
        """Sets business priority for transform() scoring."""
        if len(weights) != self.output_dim:
            raise ValueError(f"Expected {self.output_dim} weights.")
        self.pareto_weights = np.array(weights)

    def forward(self, X: torch.Tensor):
        return self.moe_layer_(X)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._set_seed(self.seed)
        self.to(self._device)
        
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=self.seed)

        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train, dtype=torch.float32).to(self._device), 
                torch.tensor(y_train, dtype=torch.float32).to(self._device)
            ), 
            batch_size=self.batch_size, shuffle=True
        )
        
        val_x = torch.tensor(X_val, dtype=torch.float32).to(self._device)
        val_y = torch.tensor(y_val, dtype=torch.float32).to(self._device)

        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        pos_weight = torch.tensor([(1 - y.sum()) / y.sum()]).to(self._device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # criterion = nn.BCELoss()

        best_loss = float('inf')
        counter = 0

        # Change 1: Total is epochs * batches per epoch
        total_steps = self.epochs * len(train_loader)
        pbar = tqdm(total=total_steps, desc="Training MoE")

        for epoch in range(self.epochs):
            self.train()
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                preds, gate_logits = self.moe_layer_(batch_x)
                
                task_loss = criterion(preds, batch_y)
                probs = F.softmax(gate_logits, dim=-1)
                importance = probs.sum(0)
                load_loss = self.num_experts * torch.sum(importance**2) / (importance.sum()**2)
                
                total_loss = task_loss + (self.balancing_coef * load_loss)
                total_loss.backward()
                optimizer.step()
                
                # Change 2: Update bar on every batch
                pbar.update(1)
                pbar.set_postfix({"loss": f"{total_loss.item():.4f}", "epoch": epoch + 1})

            # Evaluation Step
            self.eval()
            with torch.no_grad():
                val_preds, _ = self.moe_layer_(val_x)
                current_val_loss = criterion(val_preds, val_y).item()

            # Optional: Overwrite postfix at end of epoch to show validation status
            pbar.set_postfix({"val_loss": f"{current_val_loss:.4f}", "best": f"{best_loss:.4f}"})

            # Early Stopping Check
            if current_val_loss < best_loss - self.min_delta:
                best_loss = current_val_loss
                counter = 0
                best_model_state = self.state_dict()
            else:
                counter += 1
                if counter >= self.patience:
                    pbar.write(f"Early stopping at epoch {epoch + 1}")
                    # Change 3: Ensure bar reaches 100% or stops cleanly on early exit
                    pbar.update(total_steps - pbar.n) 
                    break

        if 'best_model_state' in locals():
            self.load_state_dict(best_model_state)
            
        pbar.close()
        self.fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns raw multi-objective predictions.
        Shape: (n_samples, output_dim)
        """
        if not self.fitted_:
            raise RuntimeError("Model must be fit before predicting.")
        
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self._device)
            preds, _ = self.forward(X_tensor)
            return preds.cpu().numpy()

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Returns a single scalar score based on pareto_weights.
        Useful for ranking items in a pipeline.
        """
        raw_preds = self.predict(X)
        # Dot product for weighted score
        scores = np.dot(raw_preds, self.pareto_weights)
        return scores.reshape(-1, 1)

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)