"""
Logistic Regression model wrapper for factuality classification.

This module provides a wrapper class around sklearn's LogisticRegression
with methods for training, prediction, and model persistence.
"""

import numpy as np
import pickle
from typing import Optional, Tuple
from sklearn.linear_model import LogisticRegression
from scipy import sparse
import warnings


class FactoidLogRegClassifier:
    """
    Logistic Regression classifier for Reddit post factuality prediction.

    This class wraps sklearn's LogisticRegression with utilities for
    training on sparse TF-IDF features and metadata.
    """

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = 'l2',
        solver: str = 'saga',
        max_iter: int = 1000,
        class_weight: Optional[str] = 'balanced',
        random_state: int = 42,
        verbose: int = 0
    ):
        """
        Initialize the Logistic Regression classifier.

        Args:
            C: Inverse of regularization strength (smaller = stronger regularization)
            penalty: Regularization penalty ('l1', 'l2', or 'elasticnet')
            solver: Optimization algorithm. 'saga' supports sparse matrices and all penalties
            max_iter: Maximum number of iterations for optimization
            class_weight: 'balanced' to automatically adjust weights, or None
            random_state: Random seed for reproducibility
            verbose: Verbosity level for training output
        """
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        self.verbose = verbose

        # Initialize the model
        self.model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
            verbose=verbose
        )

        self.is_fitted = False
        self.classes_ = None
        self.n_features_ = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> 'FactoidLogRegClassifier':
        """
        Fit the logistic regression model on training data.

        Args:
            X_train: Training features (sparse or dense)
            y_train: Training labels
            X_val: Validation features (optional, for monitoring)
            y_val: Validation labels (optional, for monitoring)

        Returns:
            Self (for method chaining)
        """
        print(f"\nTraining Logistic Regression Classifier...")
        print(f"  C: {self.C}")
        print(f"  Penalty: {self.penalty}")
        print(f"  Solver: {self.solver}")
        print(f"  Class weight: {self.class_weight}")
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Features: {X_train.shape[1]}")

        # Fit the model
        self.model.fit(X_train, y_train)

        self.is_fitted = True
        self.classes_ = self.model.classes_
        self.n_features_ = X_train.shape[1]

        # Report training performance
        train_score = self.model.score(X_train, y_train)
        print(f"\nTraining accuracy: {train_score:.4f}")

        # Report validation performance if provided
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            print(f"Validation accuracy: {val_score:.4f}")

        print(f"\nModel fitted successfully!")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples.

        Args:
            X: Feature matrix (sparse or dense)

        Returns:
            Array of predicted class labels

        Raises:
            RuntimeError: If model has not been fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples.

        Args:
            X: Feature matrix (sparse or dense)

        Returns:
            Array of class probabilities with shape (n_samples, n_classes)

        Raises:
            RuntimeError: If model has not been fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score on given data.

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            Accuracy score
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")

        return self.model.score(X, y)

    def get_coefficients(self) -> np.ndarray:
        """
        Get model coefficients (feature weights).

        Returns:
            Array of coefficients with shape (n_classes, n_features) or (n_features,)
            For binary classification, returns (n_features,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before accessing coefficients")

        coef = self.model.coef_

        # For binary classification, return flat array
        if coef.shape[0] == 1:
            return coef[0]

        return coef

    def get_intercept(self) -> np.ndarray:
        """
        Get model intercept.

        Returns:
            Array of intercepts
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before accessing intercept")

        return self.model.intercept_

    def get_top_features(
        self,
        feature_names: Optional[list] = None,
        top_n: int = 20,
        class_idx: int = 0
    ) -> Tuple[list, list]:
        """
        Get top positive and negative features by coefficient magnitude.

        Args:
            feature_names: List of feature names (optional)
            top_n: Number of top features to return
            class_idx: Which class to analyze (for multiclass)

        Returns:
            Tuple of (positive_features, negative_features)
            Each is a list of (feature_name/idx, coefficient) tuples
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before accessing features")

        # Get coefficients
        coef = self.model.coef_
        if coef.shape[0] > 1:
            coef = coef[class_idx]
        else:
            coef = coef[0]

        # Get top positive features
        top_positive_idx = np.argsort(coef)[-top_n:][::-1]
        top_positive = []
        for idx in top_positive_idx:
            feat_name = feature_names[idx] if feature_names else f"feature_{idx}"
            top_positive.append((feat_name, float(coef[idx])))

        # Get top negative features
        top_negative_idx = np.argsort(coef)[:top_n]
        top_negative = []
        for idx in top_negative_idx:
            feat_name = feature_names[idx] if feature_names else f"feature_{idx}"
            top_negative.append((feat_name, float(coef[idx])))

        return top_positive, top_negative

    def print_top_features(
        self,
        feature_names: Optional[list] = None,
        top_n: int = 20
    ):
        """
        Print top positive and negative features.

        Args:
            feature_names: List of feature names
            top_n: Number of features to display
        """
        top_pos, top_neg = self.get_top_features(feature_names, top_n)

        print(f"\nTop {top_n} Positive Features (predict class 1):")
        print("-" * 60)
        for i, (feat, coef) in enumerate(top_pos, 1):
            print(f"{i:2d}. {feat:40s} {coef:8.4f}")

        print(f"\nTop {top_n} Negative Features (predict class 0):")
        print("-" * 60)
        for i, (feat, coef) in enumerate(top_neg, 1):
            print(f"{i:2d}. {feat:40s} {coef:8.4f}")

    def save(self, path: str):
        """
        Save the fitted model to disk.

        Args:
            path: Output file path (should end in .pkl)
        """
        if not self.is_fitted:
            warnings.warn("Saving unfitted model")

        with open(path, 'wb') as f:
            pickle.dump(self, f)

        print(f"Model saved to: {path}")

    @classmethod
    def load(cls, path: str) -> 'FactoidLogRegClassifier':
        """
        Load a fitted model from disk.

        Args:
            path: Path to saved model file

        Returns:
            Loaded FactoidLogRegClassifier instance
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)

        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}")

        print(f"Model loaded from: {path}")
        if model.is_fitted:
            print(f"  Classes: {model.classes_}")
            print(f"  Features: {model.n_features_}")

        return model

    def __repr__(self) -> str:
        """String representation of the model."""
        status = "fitted" if self.is_fitted else "not fitted"
        return (f"FactoidLogRegClassifier(C={self.C}, penalty='{self.penalty}', "
                f"solver='{self.solver}', {status})")


def create_baseline_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    C: float = 1.0,
    **kwargs
) -> FactoidLogRegClassifier:
    """
    Convenience function to create and train a baseline logistic regression model.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        C: Regularization parameter
        **kwargs: Additional arguments for FactoidLogRegClassifier

    Returns:
        Fitted FactoidLogRegClassifier instance
    """
    model = FactoidLogRegClassifier(C=C, **kwargs)
    model.fit(X_train, y_train, X_val, y_val)

    return model
