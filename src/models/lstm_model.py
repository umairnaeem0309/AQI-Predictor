"""
LSTM Model for AQI Multi-Horizon Forecasting.

Implements a sequence-to-vector LSTM that takes 24 timesteps of features
and predicts AQI at 24h, 48h, and 72h horizons simultaneously.

Architecture:
- Input: (batch_size, sequence_length, n_features)
- LSTM layers with dropout for regularization
- Dense output layer for multi-horizon prediction

Usage:
    from src.models.lstm_model import LSTMModel
    model = LSTMModel(sequence_length=24, n_features=79, n_targets=3)
    model.fit(X_train_seq, y_train, epochs=50, batch_size=32)
    predictions = model.predict(X_test_seq)
"""

import logging
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _check_tensorflow():
    """Check if TensorFlow is available."""
    try:
        import tensorflow as tf
        # Suppress TF warnings
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.get_logger().setLevel('ERROR')
        return tf
    except ImportError:
        raise ImportError(
            "TensorFlow is required for LSTM model. "
            "Install with: pip install tensorflow-cpu"
        )


class LSTMModel:
    """LSTM model for multi-horizon AQI forecasting.

    Takes sequential feature data (24 timesteps) and predicts
    AQI at 24h, 48h, and 72h horizons.

    Attributes:
        sequence_length: Number of timesteps in input sequence.
        n_features: Number of input features per timestep.
        n_targets: Number of output targets (default: 3).
        model: Compiled Keras model.
        scaler: Feature scaler (fitted on training data).
    """

    def __init__(
        self,
        sequence_length: int = 24,
        n_features: int = 79,
        n_targets: int = 3,
        lstm_units: Optional[List[int]] = None,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        random_seed: int = 42,
    ):
        """Initialize LSTM model.

        Args:
            sequence_length: Number of historical timesteps (default 24h).
            n_features: Number of input features.
            n_targets: Number of prediction targets (24h, 48h, 72h).
            lstm_units: List of units per LSTM layer. Default [64, 32].
            dropout_rate: Dropout rate for regularization.
            learning_rate: Adam optimizer learning rate.
            random_seed: Random seed for reproducibility.
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_targets = n_targets
        self.lstm_units = lstm_units or [64, 32]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.model = None
        self._is_fitted = False

    def _build_model(self):
        """Build the Keras model architecture."""
        tf = _check_tensorflow()

        tf.random.set_seed(self.random_seed)
        np.random.seed(self.random_seed)

        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam

        model = Sequential()

        # First LSTM layer
        model.add(LSTM(
            units=self.lstm_units[0],
            return_sequences=len(self.lstm_units) > 1,
            input_shape=(self.sequence_length, self.n_features),
        ))
        model.add(Dropout(self.dropout_rate))

        # Additional LSTM layers
        for i, units in enumerate(self.lstm_units[1:], start=1):
            return_seq = i < len(self.lstm_units) - 1
            model.add(LSTM(units=units, return_sequences=return_seq))
            model.add(Dropout(self.dropout_rate))

        # Output layer
        model.add(Dense(self.n_targets))

        # Compile
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae'],
        )

        self.model = model
        logger.info(
            "LSTM model built: %d params, layers=%s",
            model.count_params(), self.lstm_units,
        )
        return model

    def create_sequences(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Create sliding window sequences from flat data.

        Converts (n_rows, n_features) → (n_sequences, seq_len, n_features)
        by grouping by location and creating windows.

        Args:
            X: Feature array of shape (n_rows, n_features).
            y: Target array of shape (n_rows, n_targets). Optional.

        Returns:
            Tuple of (X_sequences, y_sequences).
            X_sequences shape: (n_sequences, sequence_length, n_features)
            y_sequences shape: (n_sequences, n_targets)
        """
        n_rows = len(X)
        if n_rows < self.sequence_length:
            logger.warning(
                "Not enough rows (%d) for sequence length (%d)",
                n_rows, self.sequence_length,
            )
            return np.array([]), np.array([]) if y is not None else None

        n_sequences = n_rows - self.sequence_length + 1
        X_seq = np.zeros((n_sequences, self.sequence_length, self.n_features))

        for i in range(n_sequences):
            X_seq[i] = X[i:i + self.sequence_length]

        if y is not None:
            # Target is at the end of the sequence
            y_seq = y[self.sequence_length - 1:]
            return X_seq, y_seq

        return X_seq, None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 0,
    ) -> Dict[str, Any]:
        """Train the LSTM model.

        Args:
            X_train: Training features, shape (n_rows, n_features).
            y_train: Training targets, shape (n_rows, n_targets).
            X_val: Validation features. Optional.
            y_val: Validation targets. Optional.
            epochs: Number of training epochs.
            batch_size: Batch size.
            verbose: Keras verbose level (0=silent, 1=progress, 2=one line).

        Returns:
            Training history dictionary.
        """
        tf = _check_tensorflow()

        if self.model is None:
            self._build_model()

        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train)
        if len(X_train_seq) == 0:
            raise ValueError("Not enough data to create training sequences")

        logger.info(
            "LSTM training: sequences=%d, shape=%s",
            len(X_train_seq), X_train_seq.shape,
        )

        # Validation data
        callbacks = []
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self.create_sequences(X_val, y_val)
            if len(X_val_seq) > 0:
                validation_data = (X_val_seq, y_val_seq)
            else:
                validation_data = None
        else:
            validation_data = None

        # Early stopping
        from tensorflow.keras.callbacks import EarlyStopping
        callbacks.append(EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=10,
            restore_best_weights=True,
            verbose=0,
        ))

        # Train
        history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
        )

        self._is_fitted = True
        logger.info("LSTM training complete: %d epochs", len(history.history['loss']))

        return {
            'loss': history.history['loss'],
            'val_loss': history.history.get('val_loss', []),
            'mae': history.history['mae'],
            'val_mae': history.history.get('val_mae', []),
            'epochs_trained': len(history.history['loss']),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict AQI at 24h, 48h, 72h horizons.

        Args:
            X: Feature array of shape (n_rows, n_features).

        Returns:
            Predictions array of shape (n_sequences, 3).
        """
        if not self._is_fitted:
            raise RuntimeError("Model not trained. Call fit() first.")

        X_seq, _ = self.create_sequences(X)
        if len(X_seq) == 0:
            raise ValueError("Not enough data to create prediction sequences")

        predictions = self.model.predict(X_seq, verbose=0)
        return predictions

    def save(self, path: Path) -> None:
        """Save model to disk.

        Args:
            path: Directory to save model files.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save Keras model
        self.model.save(str(path / "lstm_model.keras"))

        # Save config
        config = {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'n_targets': self.n_targets,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'random_seed': self.random_seed,
            'is_fitted': self._is_fitted,
        }
        with open(path / "config.pkl", "wb") as f:
            pickle.dump(config, f)

        logger.info("LSTM model saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> 'LSTMModel':
        """Load model from disk.

        Args:
            path: Directory containing saved model files.

        Returns:
            Loaded LSTMModel instance.
        """
        tf = _check_tensorflow()
        path = Path(path)

        with open(path / "config.pkl", "rb") as f:
            config = pickle.load(f)

        instance = cls(
            sequence_length=config['sequence_length'],
            n_features=config['n_features'],
            n_targets=config['n_targets'],
            lstm_units=config['lstm_units'],
            dropout_rate=config['dropout_rate'],
            learning_rate=config['learning_rate'],
            random_seed=config['random_seed'],
        )

        instance.model = tf.keras.models.load_model(
            str(path / "lstm_model.keras")
        )
        instance._is_fitted = config['is_fitted']

        logger.info("LSTM model loaded from %s", path)
        return instance

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters for logging."""
        return {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'n_targets': self.n_targets,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'random_seed': self.random_seed,
            'total_params': self.model.count_params() if self.model else 0,
        }
