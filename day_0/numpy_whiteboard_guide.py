"""
NUMPY FOR ML - WHITEBOARD EXPLAINABLE CONCEPTS
Deep dive into each concept with visual understanding
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*60)
print("1. ARRAYS & MEMORY LAYOUT - Foundation of Speed")
print("="*60)

# WHY NUMPY IS FAST: Contiguous memory layout
python_list = [1, 2, 3, 4, 5]  # Each element is a Python object (overhead)
numpy_array = np.array([1, 2, 3, 4, 5])  # Contiguous memory block

print(f"Python list size: {python_list.__sizeof__()} bytes")
print(f"NumPy array size: {numpy_array.nbytes} bytes")
print(f"Memory efficiency: {python_list.__sizeof__() / numpy_array.nbytes:.1f}x overhead in Python")

# WHITEBOARD CONCEPT: Memory Layout
"""
Python List:     [ptr] -> [obj1] [ptr] -> [obj2] [ptr] -> [obj3]  (scattered)
NumPy Array:     [1][2][3][4][5]                                  (contiguous)

Why this matters:
- CPU cache efficiency
- Vectorized operations
- SIMD (Single Instruction, Multiple Data)
"""

print("\n" + "="*60)
print("2. BROADCASTING - The Magic Behind Efficient Computation")
print("="*60)

# BROADCASTING RULES (Critical to understand)
a = np.array([[1, 2, 3],      # Shape: (2, 3)
              [4, 5, 6]])
b = np.array([10, 20, 30])    # Shape: (3,)

result = a + b  # Broadcasting happens automatically

print("Broadcasting Example:")
print(f"Array a (2,3):\n{a}")
print(f"Array b (3,):   {b}")
print(f"Result (2,3):\n{result}")

# WHITEBOARD EXPLANATION:
"""
Broadcasting Rules:
1. Start from rightmost dimension
2. Dimensions are compatible if:
   - They are equal, OR
   - One of them is 1, OR  
   - One is missing (treated as 1)

Example: (2,3) + (3,) = (2,3)
Step 1: Align right    (2,3) + (1,3)
Step 2: Broadcast      (2,3) + (2,3)  ✓

Visual:
[1 2 3]   +   [10 20 30]   =   [11 22 33]
[4 5 6]       [10 20 30]       [14 25 36]
              ↑ broadcasted
"""

# Common broadcasting patterns in ML
X = np.random.randn(1000, 10)  # 1000 samples, 10 features
mean = np.mean(X, axis=0)      # Shape: (10,)
X_centered = X - mean          # Broadcasting: (1000,10) - (10,) = (1000,10)

print(f"\nML Example - Centering data:")
print(f"Data shape: {X.shape}")
print(f"Mean shape: {mean.shape}")
print(f"Centered shape: {X_centered.shape}")

print("\n" + "="*60)
print("3. VECTORIZATION - Eliminating Loops")
print("="*60)

# SLOW: Python loops
def slow_dot_product(a, b):
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

# FAST: Vectorized operations
def fast_dot_product(a, b):
    return np.dot(a, b)

# Timing comparison
a = np.random.randn(10000)
b = np.random.randn(10000)

import time
start = time.time()
for _ in range(100):
    slow_result = slow_dot_product(a, b)
slow_time = time.time() - start

start = time.time()
for _ in range(100):
    fast_result = np.dot(a, b)
fast_time = time.time() - start

print(f"Slow (Python loop): {slow_time:.4f} seconds")
print(f"Fast (Vectorized):  {fast_time:.4f} seconds")
print(f"Speedup: {slow_time/fast_time:.1f}x faster")

# WHITEBOARD CONCEPT: Why vectorization works
"""
Python Loop (slow):
for i in range(n):
    result[i] = a[i] * b[i]  # Python interpreter overhead each iteration

Vectorized (fast):
result = a * b  # Single C function call, SIMD instructions

CPU Level:
Loop:     [load a[0]] [load b[0]] [multiply] [store] [repeat...]
Vector:   [load a[0:4]] [load b[0:4]] [multiply 4 at once] [store 4]
"""

print("\n" + "="*60)
print("4. MATRIX OPERATIONS - Heart of Machine Learning")
print("="*60)

# Forward pass in neural network
def neural_network_forward(X, W1, b1, W2, b2):
    """
    X: input (batch_size, input_dim)
    W1: first layer weights (input_dim, hidden_dim)
    b1: first layer bias (hidden_dim,)
    W2: second layer weights (hidden_dim, output_dim)
    b2: second layer bias (output_dim,)
    """
    # Layer 1
    z1 = X @ W1 + b1           # Linear transformation
    a1 = np.maximum(0, z1)     # ReLU activation
    
    # Layer 2  
    z2 = a1 @ W2 + b2         # Linear transformation
    a2 = 1 / (1 + np.exp(-z2)) # Sigmoid activation
    
    return a2

# Example network
batch_size, input_dim, hidden_dim, output_dim = 32, 784, 128, 10
X = np.random.randn(batch_size, input_dim)
W1 = np.random.randn(input_dim, hidden_dim) * 0.01
b1 = np.zeros(hidden_dim)
W2 = np.random.randn(hidden_dim, output_dim) * 0.01
b2 = np.zeros(output_dim)

output = neural_network_forward(X, W1, b1, W2, b2)
print(f"Neural network output shape: {output.shape}")

# WHITEBOARD EXPLANATION:
"""
Matrix Multiplication Intuition:

X @ W = Output
(32, 784) @ (784, 128) = (32, 128)

Each row of X (one sample) gets transformed by W:
sample_1 • W = new_representation_1
sample_2 • W = new_representation_2
...

Geometric interpretation:
- W defines a linear transformation
- Each column of W is a "feature detector"
- X @ W projects input onto new feature space
"""

print("\n" + "="*60)
print("5. AXIS OPERATIONS - Understanding Dimensions")
print("="*60)

# 3D tensor example (common in deep learning)
tensor = np.random.randn(2, 3, 4)  # (batch, height, width) or (batch, seq_len, features)

print(f"Original tensor shape: {tensor.shape}")
print(f"Sum along axis=0 (batch): {np.sum(tensor, axis=0).shape}")
print(f"Sum along axis=1 (height): {np.sum(tensor, axis=1).shape}")  
print(f"Sum along axis=2 (width): {np.sum(tensor, axis=2).shape}")

# WHITEBOARD VISUALIZATION:
"""
3D Tensor: (2, 3, 4)

axis=0 ↓
┌─────────┐ ┌─────────┐
│ 3 x 4   │ │ 3 x 4   │  → Sum along axis=0 → (3, 4)
│ matrix  │ │ matrix  │
└─────────┘ └─────────┘

axis=1 →
┌─ 4 ─┐
│ row │ ┐
│ row │ │ 3  → Sum along axis=1 → (2, 4)
│ row │ ┘
└─────┘

axis=2 →
[col col col col] → Sum along axis=2 → (2, 3)
"""

# Practical ML example: Batch normalization
batch_data = np.random.randn(64, 10)  # 64 samples, 10 features

# Compute statistics along batch dimension (axis=0)
batch_mean = np.mean(batch_data, axis=0, keepdims=True)  # (1, 10)
batch_var = np.var(batch_data, axis=0, keepdims=True)    # (1, 10)

# Normalize
normalized = (batch_data - batch_mean) / np.sqrt(batch_var + 1e-8)

print(f"\nBatch Normalization:")
print(f"Input shape: {batch_data.shape}")
print(f"Mean shape: {batch_mean.shape}")
print(f"Normalized shape: {normalized.shape}")
print(f"New mean (should be ~0): {np.mean(normalized, axis=0)[:3]}")
print(f"New std (should be ~1): {np.std(normalized, axis=0)[:3]}")

print("\n" + "="*60)
print("6. LINEAR ALGEBRA - The Mathematical Foundation")
print("="*60)

# Principal Component Analysis (PCA) from scratch
def pca_from_scratch(X, n_components):
    """
    X: data matrix (n_samples, n_features)
    n_components: number of principal components
    """
    # Step 1: Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Step 2: Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Step 3: Eigendecomposition
    eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
    
    # Step 4: Sort by eigenvalues (descending)
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    # Step 5: Select top components
    components = eigenvecs[:, :n_components]
    
    # Step 6: Transform data
    X_pca = X_centered @ components
    
    return X_pca, components, eigenvals

# Example: Reduce 10D data to 3D
X = np.random.randn(1000, 10)
X_pca, components, eigenvals = pca_from_scratch(X, 3)

print(f"Original data: {X.shape}")
print(f"PCA reduced: {X_pca.shape}")
print(f"Explained variance ratio: {eigenvals[:3] / np.sum(eigenvals)}")

# WHITEBOARD EXPLANATION:
"""
PCA Intuition:

1. Data Cloud in High-D Space:
   • • •
  • • • •  ← Find directions of maximum variance
   • • •

2. Principal Components:
   PC1 ────→ (direction of max variance)
   PC2 ↗     (orthogonal, next max variance)

3. Projection:
   Original: (x₁, x₂, ..., x₁₀)
   PCA:      (pc₁, pc₂, pc₃)  ← Lower dimensional representation

Mathematical Steps:
1. Center data: X' = X - μ
2. Covariance: C = X'ᵀX' / (n-1)
3. Eigendecomposition: C = VΛVᵀ
4. Transform: Y = X'V
"""

print("\n" + "="*60)
print("7. GRADIENT COMPUTATION - Backpropagation Foundation")
print("="*60)

def logistic_regression_gradients(X, y, w, b):
    """
    Compute gradients for logistic regression
    X: features (m, n)
    y: labels (m,)
    w: weights (n,)
    b: bias (scalar)
    """
    m = X.shape[0]
    
    # Forward pass
    z = X @ w + b                    # Linear combination
    a = 1 / (1 + np.exp(-z))        # Sigmoid activation
    
    # Cost function
    cost = -np.mean(y * np.log(a + 1e-15) + (1 - y) * np.log(1 - a + 1e-15))
    
    # Gradients
    dw = (1/m) * X.T @ (a - y)      # Weight gradients
    db = (1/m) * np.sum(a - y)      # Bias gradient
    
    return cost, dw, db

# Example
X = np.random.randn(1000, 5)
y = np.random.randint(0, 2, 1000)
w = np.random.randn(5) * 0.01
b = 0.0

cost, dw, db = logistic_regression_gradients(X, y, w, b)
print(f"Cost: {cost:.4f}")
print(f"Weight gradients shape: {dw.shape}")
print(f"Bias gradient: {db:.4f}")

# WHITEBOARD EXPLANATION:
"""
Gradient Computation Chain Rule:

Cost = -[y·log(σ(z)) + (1-y)·log(1-σ(z))]
       ↑
       σ(z) = 1/(1+e⁻ᶻ)
       ↑
       z = Xw + b

Chain rule:
∂Cost/∂w = ∂Cost/∂σ · ∂σ/∂z · ∂z/∂w

Working backwards:
∂z/∂w = X
∂σ/∂z = σ(1-σ)  
∂Cost/∂σ = -(y/σ - (1-y)/(1-σ))

Combined: ∂Cost/∂w = Xᵀ(σ - y) / m

Geometric intuition:
- Gradient points in direction of steepest increase
- We move opposite to gradient (gradient descent)
- Step size controlled by learning rate
"""

print("\n" + "="*60)
print("8. NUMERICAL STABILITY - Production-Ready Code")
print("="*60)

def unstable_softmax(x):
    """Numerically unstable version"""
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def stable_softmax(x):
    """Numerically stable version"""
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)  # Subtract max for stability
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Test with large numbers
large_logits = np.array([[1000, 1001, 999]])

print("Numerical Stability Test:")
try:
    unstable_result = unstable_softmax(large_logits)
    print(f"Unstable softmax: {unstable_result}")
except:
    print("Unstable softmax: OVERFLOW ERROR!")

stable_result = stable_softmax(large_logits)
print(f"Stable softmax: {stable_result}")

# WHITEBOARD EXPLANATION:
"""
Numerical Stability Issues:

Problem: e^1000 = ∞ (overflow)

Solution: Softmax invariance property
softmax(x) = softmax(x - c) for any constant c

Choose c = max(x):
e^(x-max) / Σe^(x-max)

Example:
x = [1000, 1001, 999]
x - max = [1000-1001, 1001-1001, 999-1001] = [-1, 0, -2]
e^[-1, 0, -2] = [0.368, 1.0, 0.135] ← No overflow!

Other stability tricks:
- Add small ε to prevent log(0)
- Clip gradients to prevent explosion
- Use double precision for accumulation
"""

print("\n" + "="*60)
print("SUMMARY: Key NumPy Patterns for ML Interviews")
print("="*60)

key_patterns = {
    "Data Preprocessing": {
        "Normalization": "(X - X.mean(axis=0)) / X.std(axis=0)",
        "Train/Test Split": "idx = np.random.permutation(len(X))",
        "One-hot Encoding": "np.eye(n_classes)[labels]"
    },
    "Neural Networks": {
        "Forward Pass": "output = input @ weights + bias",
        "Activation": "np.maximum(0, x)  # ReLU",
        "Softmax": "exp_x / exp_x.sum(axis=-1, keepdims=True)"
    },
    "Linear Algebra": {
        "PCA": "U, s, Vt = np.linalg.svd(X)",
        "Regression": "weights = np.linalg.pinv(X) @ y",
        "Distance": "np.linalg.norm(a - b, axis=1)"
    },
    "Optimization": {
        "Gradient Descent": "weights -= lr * gradients",
        "Momentum": "velocity = beta * velocity + gradients",
        "Regularization": "loss += lambda_reg * np.sum(weights**2)"
    }
}

for category, patterns in key_patterns.items():
    print(f"\n{category}:")
    for name, code in patterns.items():
        print(f"  {name}: {code}")

print(f"\n{'='*60}")
print("🎯 WHITEBOARD READY: You can now explain each concept visually!")
print("🚀 INTERVIEW READY: You understand the 'why' behind each operation!")
print("💡 PRODUCTION READY: You know the numerical stability considerations!")
print("="*60)