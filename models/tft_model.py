"""
Temporal Fusion Transformer Model (ATTENTION model)
Variable selection + multi-head attention for interpretable time series prediction.
Replaces GRU — shows WHICH past bars and features influenced the prediction.
Input: (batch, 60, ~180). Output: P(price_up) in [0,1] + attention weights.
"""
import logging
import os

import numpy as np
import joblib

import config

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not installed — tft_model disabled")


# All nn.Module subclasses must be guarded — they reference nn at class definition time
if HAS_TORCH:
    class GatedResidualNetwork(nn.Module):
        """GRN block used throughout TFT."""

        def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.gate = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(output_dim)
            self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

        def forward(self, x):
            residual = self.skip(x)
            h = torch.relu(self.fc1(x))
            h = self.dropout(h)
            out = self.fc2(h)
            gate = torch.sigmoid(self.gate(h))
            return self.layer_norm(gate * out + residual)


    class VariableSelectionNetwork(nn.Module):
        """Selects relevant features at each timestep."""

        def __init__(self, n_features, hidden_dim, dropout=0.1):
            super().__init__()
            self.grn = GatedResidualNetwork(n_features, hidden_dim, n_features, dropout)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x):
            # x: (batch, seq_len, n_features)
            weights = self.softmax(self.grn(x))
            return x * weights, weights


    class SimplifiedTFT(nn.Module):
        """Simplified Temporal Fusion Transformer for classification."""

        def __init__(self, n_features, hidden_dim=64, n_heads=4, n_layers=2, dropout=0.2):
            super().__init__()
            self.variable_selection = VariableSelectionNetwork(n_features, hidden_dim, dropout)
            self.encoder = nn.LSTM(n_features, hidden_dim, n_layers, batch_first=True, dropout=dropout)
            self.attention = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
            self.grn_out = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
            self.fc_out = nn.Linear(hidden_dim, 1)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            # Variable selection
            selected, var_weights = self.variable_selection(x)

            # LSTM encoder
            lstm_out, _ = self.encoder(selected)

            # Multi-head attention
            attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

            # Take last timestep
            last = attn_out[:, -1, :]
            out = self.grn_out(last)
            out = self.dropout(out)
            logit = self.fc_out(out)
            return torch.sigmoid(logit), attn_weights, var_weights


class TemporalFusionTransformerModel:
    """TFT wrapper for training and prediction."""

    def __init__(self):
        self.model = None
        self.is_trained = False
        self.lookback = config.ML_LOOKBACK_TIMESTEPS
        self.device = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"

    def build(self, n_features: int = 180):
        if not HAS_TORCH:
            return
        # Smaller model on CPU (NVS315/CPU mode) — faster training, similar accuracy
        hidden = 32 if self.device == "cpu" else 64
        heads = 2 if self.device == "cpu" else 4
        self.model = SimplifiedTFT(
            n_features=n_features,
            hidden_dim=hidden,
            n_heads=heads,
            n_layers=2,
            dropout=0.2,
        ).to(self.device)

    # Max sequences for CPU training — keeps each epoch fast enough to be responsive
    _CPU_MAX_SEQUENCES = 1500

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=25, batch_size=64, lr=0.001, on_epoch=None):
        """Train TFT model.

        Args:
            X_train:  3D array (samples, lookback, features)
            y_train:  binary labels
            on_epoch: optional callback(epoch, total_epochs, train_loss, val_loss)
                      called after every epoch — used for progress reporting
        """
        if not HAS_TORCH:
            logger.error("PyTorch not available")
            return {"error": "torch_not_installed"}

        if self.model is None:
            self.build(n_features=X_train.shape[2])

        # ── CPU guard: subsample to keep training fast ─────────────
        if self.device == "cpu" and len(X_train) > self._CPU_MAX_SEQUENCES:
            idx = np.random.choice(len(X_train), self._CPU_MAX_SEQUENCES, replace=False)
            idx.sort()
            X_train = X_train[idx]
            y_train = y_train[idx]
            logger.info(f"TFT CPU mode: subsampled to {self._CPU_MAX_SEQUENCES} sequences")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        n_pos = max(np.sum(y_train == 1), 1)
        n_neg = max(np.sum(y_train == 0), 1)
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(self.device)

        best_val_loss = float("inf")
        patience_counter = 0
        patience_limit = 5  # early-stop patience

        for epoch in range(epochs):
            self.model.train()
            indices = np.random.permutation(len(X_train))
            total_loss = 0
            n_batches = 0

            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start:start + batch_size]
                xb = X_t[batch_idx]
                yb = y_t[batch_idx]
                optimizer.zero_grad()
                pred, _, _ = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            val_loss = None

            if X_val is not None and len(X_val) > 0:
                val_loss = self._eval_loss(X_val, y_val, criterion)
                scheduler.step(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience_limit:
                        if on_epoch:
                            on_epoch(epoch + 1, epochs, avg_loss, val_loss, stopped_early=True)
                        break
            else:
                scheduler.step(avg_loss)

            # ── per-epoch progress callback ─────────────────────────
            if on_epoch:
                on_epoch(epoch + 1, epochs, avg_loss, val_loss, stopped_early=False)

        self.is_trained = True
        metrics = self._compute_metrics(X_val, y_val) if X_val is not None and len(X_val) > 0 else {}
        logger.info(f"TFT trained — metrics: {metrics}")
        return metrics

    def predict(self, X) -> float:
        if not self.is_trained or self.model is None:
            return 0.5
        if X.ndim == 2:
            X = X.reshape(1, X.shape[0], X.shape[1])
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            pred, _, _ = self.model(X_t)
        return float(pred[0][0].cpu())

    def predict_with_attention(self, X):
        """Predict with attention weights for interpretability."""
        if not self.is_trained or self.model is None:
            return 0.5, None, None
        if X.ndim == 2:
            X = X.reshape(1, X.shape[0], X.shape[1])
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            pred, attn_weights, var_weights = self.model(X_t)
        return (
            float(pred[0][0].cpu()),
            attn_weights[0].cpu().numpy() if attn_weights is not None else None,
            var_weights[0].cpu().numpy() if var_weights is not None else None,
        )

    def predict_batch(self, X) -> np.ndarray:
        if not self.is_trained or self.model is None:
            return np.full(len(X), 0.5)
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            pred, _, _ = self.model(X_t)
        return pred.cpu().numpy().flatten()

    def save(self, symbol: str):
        if not self.is_trained or self.model is None:
            return
        ticker = symbol.replace("NSE:", "").replace("-EQ", "")
        path = os.path.join(config.MODELS_DIR, ticker)
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "tft_model.pt"))

    def load(self, symbol: str, n_features: int = 180) -> bool:
        if not HAS_TORCH:
            return False
        ticker = symbol.replace("NSE:", "").replace("-EQ", "")
        path = os.path.join(config.MODELS_DIR, ticker, "tft_model.pt")
        if not os.path.exists(path):
            return False
        self.build(n_features)
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.model.eval()
        self.is_trained = True
        return True

    def _eval_loss(self, X_val, y_val, criterion):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(self.device)
            pred, _, _ = self.model(X_t)
            loss = criterion(pred, y_t)
        return loss.item()

    def _compute_metrics(self, X_val, y_val):
        if X_val is None or y_val is None:
            return {}
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        proba = self.predict_batch(X_val)
        preds = (proba > 0.5).astype(int)
        return {
            "accuracy": float(accuracy_score(y_val, preds)),
            "f1": float(f1_score(y_val, preds, zero_division=0)),
            "auc": float(roc_auc_score(y_val, proba)) if len(set(y_val)) > 1 else 0.0,
        }
