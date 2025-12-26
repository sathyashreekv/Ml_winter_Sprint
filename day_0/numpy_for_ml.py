"""
Essential NumPy for Machine Learning - Senior ML Engineer Guide
Core operations every ML practitioner must master
"""

import numpy as np

# ============================================================================
# 1. ARRAY CREATION & INITIALIZATION (Foundation)
# ============================================================================

# Data loading simulation
X = np.random.randn(1000, 10)  # Features: 1000 samples, 10 features
y = np.random.randint(0, 2, 1000)  # Binary labels

# Essential initializations
zeros = np.zeros((100, 50))  # Weight initialization
ones = np.ones(100)  # Bias initialization
identity = np.eye(10)  # Identity matrix for regularization
random_weights = np.random.normal(0, 0.01, (784, 128))  # Xavier/He initialization

print(f"Data shape: {X.shape}, Labels: {y.shape}")

# ============================================================================
# 2. MATRIX OPERATIONS (Core ML Math)
# ============================================================================

# Forward pass simulation
W1 = np.random.randn(10, 64) * 0.01  # Layer 1 weights
b1 = np.zeros(64)  # Layer 1 bias
z1 = X @ W1 + b1  # Linear transformation
a1 = np.maximum(0, z1)  # ReLU activation

# Batch operations
batch_size = 32
X_batch = X[:batch_size]
predictions = X_batch @ W1

print(f"Forward pass output shape: {a1.shape}")

# ============================================================================
# 3. STATISTICAL OPERATIONS (Feature Engineering)
# ============================================================================

# Normalization (Critical for ML)
X_mean = np.mean(X, axis=0)  # Feature means
X_std = np.std(X, axis=0)    # Feature standard deviations
X_normalized = (X - X_mean) / (X_std + 1e-8)  # Z-score normalization

# Min-Max scaling
X_min = np.min(X, axis=0)
X_max = np.max(X, axis=0)
X_minmax = (X - X_min) / (X_max - X_min + 1e-8)

# Correlation analysis
correlation_matrix = np.corrcoef(X.T)  # Feature correlations

print(f"Original mean: {X_mean[:3]}")
print(f"Normalized mean: {np.mean(X_normalized, axis=0)[:3]}")

# ============================================================================
# 4. INDEXING & SLICING (Data Manipulation)
# ============================================================================

# Train/validation split
train_size = int(0.8 * len(X))
indices = np.random.permutation(len(X))
train_idx, val_idx = indices[:train_size], indices[train_size:]

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

# Boolean indexing for outlier removal
outlier_mask = np.abs(X) < 3 * np.std(X, axis=0)  # Remove 3-sigma outliers
X_clean = X[np.all(outlier_mask, axis=1)]

# Advanced indexing
top_features = np.argsort(np.var(X, axis=0))[-5:]  # Top 5 most variant features
X_selected = X[:, top_features]

print(f"Train: {X_train.shape}, Val: {X_val.shape}")
print(f"Selected features: {X_selected.shape}")

# ============================================================================
# 5. BROADCASTING (Efficient Computation)
# ============================================================================

# Batch normalization simulation
batch_mean = np.mean(X_batch, axis=0, keepdims=True)  # (1, 10)
batch_var = np.var(X_batch, axis=0, keepdims=True)    # (1, 10)
X_batch_norm = (X_batch - batch_mean) / np.sqrt(batch_var + 1e-8)

# Distance calculations (for KNN, clustering)
centroids = np.random.randn(5, 10)  # 5 cluster centers
distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)  # Broadcasting

print(f"Batch norm shape: {X_batch_norm.shape}")
print(f"Distances shape: {distances.shape}")

# ============================================================================
# 6. LINEAR ALGEBRA (ML Algorithms)
# ============================================================================

# Eigendecomposition (PCA)
cov_matrix = np.cov(X.T)
eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
sorted_idx = np.argsort(eigenvals)[::-1]
principal_components = eigenvecs[:, sorted_idx[:3]]  # Top 3 PCs

# SVD (Dimensionality reduction, matrix factorization)
U, s, Vt = np.linalg.svd(X, full_matrices=False)
X_reduced = U[:, :5] @ np.diag(s[:5])  # Keep top 5 components

# Pseudo-inverse (Linear regression)
X_with_bias = np.column_stack([np.ones(len(X)), X])  # Add bias column
weights = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y

print(f"PCA reduced shape: {X_reduced.shape}")
print(f"Regression weights shape: {weights.shape}")

# ============================================================================
# 7. ACTIVATION FUNCTIONS (Neural Networks)
# ============================================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Test activations
logits = np.random.randn(32, 10)  # Batch of 32, 10 classes
probabilities = softmax(logits)

print(f"Softmax output shape: {probabilities.shape}")
print(f"Probability sums: {np.sum(probabilities, axis=1)[:3]}")  # Should be ~1.0

# ============================================================================
# 8. LOSS FUNCTIONS (Model Training)
# ============================================================================

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def cross_entropy_loss(y_true, y_pred):
    # y_true: one-hot encoded, y_pred: probabilities
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-15), axis=1))

def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))

# Example usage
y_pred_prob = sigmoid(np.random.randn(100))
bce_loss = binary_cross_entropy(y[:100], y_pred_prob)

print(f"Binary cross-entropy loss: {bce_loss:.4f}")

# ============================================================================
# 9. GRADIENT COMPUTATION (Backpropagation)
# ============================================================================

def compute_gradients(X, y, weights):
    """Logistic regression gradients"""
    m = len(X)
    z = X @ weights
    predictions = sigmoid(z)
    
    # Gradients
    dw = (1/m) * X.T @ (predictions - y)
    db = (1/m) * np.sum(predictions - y)
    
    return dw, db

# Gradient descent step
learning_rate = 0.01
w = np.random.randn(X.shape[1])
dw, db = compute_gradients(X, y, w)
w_updated = w - learning_rate * dw

print(f"Gradient shape: {dw.shape}")

# ============================================================================
# 10. PERFORMANCE METRICS (Model Evaluation)
# ============================================================================

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision_recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp + 1e-15)
    recall = tp / (tp + fn + 1e-15)
    f1 = 2 * precision * recall / (precision + recall + 1e-15)
    
    return precision, recall, f1

# Example evaluation
y_pred_binary = (y_pred_prob > 0.5).astype(int)
acc = accuracy(y[:100], y_pred_binary)
prec, rec, f1 = precision_recall(y[:100], y_pred_binary)

print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")

# ============================================================================
# CRITICAL NUMPY PATTERNS FOR ML
# ============================================================================

print("\n" + "="*50)
print("CRITICAL NUMPY PATTERNS FOR ML:")
print("="*50)

patterns = {
    "Data Loading": "X = np.loadtxt('data.csv', delimiter=',')",
    "Train/Test Split": "indices = np.random.permutation(len(X))",
    "Normalization": "X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)",
    "One-Hot Encoding": "np.eye(num_classes)[labels]",
    "Batch Processing": "for i in range(0, len(X), batch_size):",
    "Matrix Multiplication": "output = input @ weights + bias",
    "Activation": "activated = np.maximum(0, linear_output)  # ReLU",
    "Loss Computation": "loss = np.mean((y_true - y_pred) ** 2)",
    "Gradient Descent": "weights -= learning_rate * gradients",
    "Regularization": "loss += lambda_reg * np.sum(weights ** 2)"
}

for operation, code in patterns.items():
    print(f"{operation:.<20} {code}")

print("\nMaster these patterns and you'll handle 90% of ML NumPy operations!")