from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Expert(nn.Module):
    """Feedforward expert network operating on a shared hidden representation."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)


class MoELayer(nn.Module):
    """
    Mixture of Experts layer with noisy top-k gating to prevent expert collapse.

    Implements the noisy gating mechanism from Shazeer et al. (2017):
    - Adds learned, input-dependent noise to gate logits during training
    - This prevents early expert collapse by encouraging exploration
    - Returns full gate logits (pre-topk) for use in the auxiliary load loss

    Args:
        hidden_dim:   Dimension of the shared input/output representation
        num_experts:  Total number of expert networks
        k:            Number of experts to activate per token
    """
    def __init__(self, hidden_dim: int, num_experts: int, k: int = 2):
        super().__init__()
        self.k = k
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim

        self.experts = nn.ModuleList([
            Expert(hidden_dim) for _ in range(num_experts)
        ])
        # Main gating network
        self.gate = nn.Linear(hidden_dim, num_experts)
        # Noise gating network — learns how much noise to inject per input
        self.noise_gate = nn.Linear(hidden_dim, num_experts)

    def forward(self, X: torch.Tensor):
        gate_logits = self.gate(X)

        # Inject learned noise during training to encourage exploration
        if self.training:
            noise_std = F.softplus(self.noise_gate(X))
            noise = torch.randn_like(gate_logits) * noise_std
            gate_logits = gate_logits + noise

        # Select top-k experts and compute normalised routing weights
        topk_logits, indices = torch.topk(gate_logits, self.k, dim=-1)
        weights = F.softmax(topk_logits, dim=-1)

        batch_size = X.size(0)
        combined_output = torch.zeros(batch_size, self.hidden_dim, device=X.device)

        for i in range(self.k):
            for expert_idx in range(self.num_experts):
                mask = (indices[:, i] == expert_idx)
                if mask.any():
                    expert_out = self.experts[expert_idx](X[mask])
                    combined_output[mask] += weights[mask, i].unsqueeze(1) * expert_out

        # Return full (pre-topk) logits so auxiliary loss sees all experts
        return combined_output, gate_logits


class MixtureOfExperts(nn.Module, BaseEstimator, TransformerMixin):
    """
    Mixture of Experts classifier/ranker compatible with sklearn Pipelines.

    Architecture:
        input -> LayerNorm -> input_proj -> MoELayer -> residual -> output_proj -> logits

    The MoE layer operates on a shared hidden_dim representation, meaning:
    - All experts share the same input/output dimensionality (hidden_dim)
    - input_proj learns a task-specific embedding before routing
    - output_proj maps the expert outputs to target space
    - A residual connection around the MoE layer stabilises training

    Gating uses noisy top-k routing (Shazeer 2017) to prevent expert collapse,
    combined with a variance-based load balancing auxiliary loss.

    Args:
        input_dim:       Number of input features
        output_dim:      Number of output targets (supports multi-label)
        hidden_dim:      Internal representation dimension (default: 128)
        num_experts:     Total number of expert networks (default: 8)
        k:               Number of active experts per forward pass (default: 2)
        epochs:          Maximum training epochs (default: 10)
        lr:              Adam learning rate (default: 1e-3)
        batch_size:      Training batch size (default: 32)
        min_delta:       Minimum val loss improvement for early stopping (default: 1e-4)
        patience:        Early stopping patience in epochs (default: 3)
        balancing_coef:  Weight of the load balancing auxiliary loss (default: 0.01)
        pareto_weights:  Per-output weights for transform() scoring (default: uniform)
        device:          Torch device string, e.g. 'cpu' or 'cuda' (default: 'cpu')
        seed:            Random seed for reproducibility (default: 42)
    """
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int = 128,
            num_experts: int = 8,
            k: int = 2,
            epochs: int = 10,
            lr: float = 1e-3,
            batch_size: int = 32,
            min_delta: float = 1e-4,
            patience: int = 3,
            balancing_coef: float = 0.01,
            pareto_weights: np.ndarray = None,
            device: str = "cpu",
            seed: int = 42
        ):
        super().__init__()

        # --- Sklearn-visible hyperparameters (names must match __init__ args) ---
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.k = k
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.min_delta = min_delta
        self.patience = patience
        self.balancing_coef = balancing_coef
        self.pareto_weights = pareto_weights if pareto_weights is not None else np.ones(output_dim)
        self.device = device
        self.seed = seed

        # --- Architecture ---
        # _device is derived from self.device via property; not stored as attribute
        # to avoid breaking sklearn clone()
        self.norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.moe_layer_ = MoELayer(hidden_dim, num_experts, k)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        self.to(self._device)

    # --- Properties ---

    @property
    def _device(self) -> torch.device:
        """Derived from self.device so sklearn clone() doesn't need to track it."""
        return torch.device(self.device)

    # --- Public API ---

    def set_pareto_weights(self, weights: list):
        """Sets per-output business priority weights used in transform()."""
        if len(weights) != self.output_dim:
            raise ValueError(f"Expected {self.output_dim} weights, got {len(weights)}.")
        self.pareto_weights = np.array(weights, dtype=np.float32)

    def forward(self, X: torch.Tensor):
        """
        Forward pass through the full architecture.

        Args:
            X: (batch_size, input_dim)
        Returns:
            logits:      (batch_size, output_dim)
            gate_logits: (batch_size, num_experts) — used for load balancing loss
        """
        # Shared representation
        h = self.input_proj(self.norm(X))       # (B, hidden_dim)
        moe_out, gate_logits = self.moe_layer_(h)
        # Residual connection stabilises gradient flow through sparse routing
        h = h + moe_out                          # (B, hidden_dim)
        logits = self.output_proj(h)             # (B, output_dim)
        return logits, gate_logits

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model using BCE loss with per-target class weighting.

        Supports both single-label (y shape: (n,)) and multi-label
        (y shape: (n, output_dim)) targets.
        """
        self._set_seed(self.seed)
        self.to(self._device)

        # Ensure y is always 2D: (n_samples, output_dim)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=self.seed
        )

        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

        val_x = torch.tensor(X_val, dtype=torch.float32).to(self._device)
        val_y = torch.tensor(y_val, dtype=torch.float32).to(self._device)

        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        # Per-target pos_weight: shape (output_dim,) handled correctly by BCE
        pos_count = y_train.sum(axis=0).clip(min=1)           # avoid div-by-zero
        neg_count = (len(y_train) - y_train.sum(axis=0)).clip(min=1)
        pos_weight = torch.tensor(
            neg_count / pos_count, dtype=torch.float32, device=self._device
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_loss = float("inf")
        best_model_state = None
        counter = 0

        total_steps = self.epochs * len(train_loader)
        pbar = tqdm(total=total_steps, desc="Training MoE")

        for epoch in range(self.epochs):
            self.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device)

                optimizer.zero_grad()
                preds, gate_logits = self.forward(batch_x)

                task_loss = criterion(preds, batch_y)

                # Load balancing: penalise variance of mean routing probability
                # across experts. Zero when all experts are used equally.
                # Uses full softmax (not topk) so all experts receive gradient signal.
                probs = F.softmax(gate_logits, dim=-1)   # (B, num_experts)
                importance = probs.mean(0)               # (num_experts,)
                load_loss = torch.var(importance) * self.num_experts

                total_loss = task_loss + self.balancing_coef * load_loss
                total_loss.backward()
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix({"loss": f"{total_loss.item():.4f}", "epoch": epoch + 1})

            # Validation
            self.eval()
            with torch.no_grad():
                val_preds, _ = self.forward(val_x)
                current_val_loss = criterion(val_preds, val_y).item()

            pbar.set_postfix({"val_loss": f"{current_val_loss:.4f}", "best": f"{best_loss:.4f}"})

            if current_val_loss < best_loss - self.min_delta:
                best_loss = current_val_loss
                counter = 0
                best_model_state = self.state_dict()
            else:
                counter += 1
                if counter >= self.patience:
                    pbar.write(f"Early stopping at epoch {epoch + 1}")
                    pbar.update(total_steps - pbar.n)
                    break

        if best_model_state is not None:
            self.load_state_dict(best_model_state)

        pbar.close()
        # Sklearn check_is_fitted looks for attributes ending in '_'
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns raw logits for all output targets.
        Shape: (n_samples, output_dim)
        """
        check_is_fitted(self, "is_fitted_")
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self._device)
            preds, _ = self.forward(X_tensor)
            return preds.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns sigmoid-activated probabilities.
        Shape: (n_samples, output_dim)
        """
        return torch.sigmoid(torch.tensor(self.predict(X))).numpy()

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Returns a single weighted score per sample using pareto_weights.
        Applies sigmoid first so all targets are on the same [0, 1] scale.
        Shape: (n_samples, 1)
        """
        probs = self.predict_proba(X)
        scores = np.dot(probs, self.pareto_weights)
        return scores.reshape(-1, 1)

    # --- Internals ---

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)