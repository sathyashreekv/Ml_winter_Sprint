# NumPy Whiteboard Explanations - Visual Guide

## 1. Broadcasting Visualization

```
BROADCASTING RULES:
Rule: Align dimensions from right, make compatible

Example 1: (3,4) + (4,) = (3,4)
┌─────────┐     ┌───────┐     ┌─────────┐
│ 1 2 3 4 │  +  │ 10 20 │  =  │11 22 33 │
│ 5 6 7 8 │     │ 30 40 │     │35 46 57 │  
│ 9 0 1 2 │     └───────┘     │39 40 41 │
└─────────┘    (broadcasted)   └─────────┘

Example 2: (2,3,4) + (3,1) = (2,3,4)
3D tensor + 2D matrix → 3D result
```

## 2. Matrix Multiplication Intuition

```
NEURAL NETWORK FORWARD PASS:

Input X     Weights W1      Output Z1
┌─────┐     ┌─────────┐     ┌─────┐
│ x11 │  @  │ w11 w12 │  =  │ z11 │
│ x12 │     │ w21 w22 │     │ z12 │
└─────┘     │ w31 w32 │     └─────┘
(1,3)       └─────────┘     (1,2)
            (3,2)

Each output neuron = weighted sum of all inputs
z11 = x11*w11 + x12*w21 + x13*w31
z12 = x11*w12 + x12*w22 + x13*w32
```

## 3. Axis Operations Visual

```
3D TENSOR OPERATIONS:

Original: (2, 3, 4)
┌─────────────┐ ┌─────────────┐
│ ┌─ 4 cols ─┐ │ │ ┌─ 4 cols ─┐ │
│ │ row1     │ │ │ │ row1     │ │
│ │ row2     │3│ │ │ row2     │3│
│ │ row3     │ │ │ │ row3     │ │
│ └──────────┘ │ │ └──────────┘ │
└─────────────┘ └─────────────┘
    batch 0         batch 1

axis=0 (batch): Sum across batches → (3,4)
axis=1 (rows):  Sum across rows → (2,4)  
axis=2 (cols):  Sum across cols → (2,3)
```

## 4. PCA Geometric Intuition

```
PRINCIPAL COMPONENT ANALYSIS:

Original Data Cloud:        After PCA:
     •                      
   • • •                    PC2 ↑
 • • • • •                     │ •
   • • •          →             │• •
     •                          └──→ PC1
                               (max variance)

Steps:
1. Center data: X' = X - mean
2. Find directions of max variance
3. Project onto these directions
4. Keep top k components
```

## 5. Gradient Descent Visualization

```
LOSS LANDSCAPE:

     Loss
      ↑
      │    ╭─╮
      │   ╱   ╲
      │  ╱     ╲
      │ ╱   •   ╲  ← Current position
      │╱    ↓    ╲
      └──────────→ Weight
           ∇L     (gradient direction)

Update rule: w_new = w_old - α * ∇L
- α (learning rate) controls step size
- ∇L points uphill, so we go opposite direction
```

## 6. Softmax Temperature Effect

```
SOFTMAX WITH DIFFERENT TEMPERATURES:

Logits: [1, 2, 3]

T=1 (normal):     [0.09, 0.24, 0.67]  ← Confident
T=0.5 (sharp):    [0.02, 0.12, 0.86]  ← Very confident  
T=2 (smooth):     [0.16, 0.26, 0.58]  ← Less confident

Visual:
T=0.5: |    |▓▓▓▓▓▓▓▓|  (peaked)
T=1.0: |  ▓▓|▓▓▓▓▓▓  |  (normal)
T=2.0: | ▓▓▓|▓▓▓▓    |  (smooth)
```

## 7. Batch Normalization Flow

```
BATCH NORMALIZATION:

Input Batch (N, D):
┌─────────────┐
│ sample 1    │ ← μ₁, σ₁²
│ sample 2    │ ← μ₂, σ₂²  
│ sample 3    │ ← μ₃, σ₃²
│ ...         │
└─────────────┘
      ↓
Compute batch stats:
μ_batch = mean(samples, axis=0)  # (D,)
σ²_batch = var(samples, axis=0)   # (D,)
      ↓
Normalize:
x_norm = (x - μ_batch) / √(σ²_batch + ε)
      ↓
Scale & Shift:
y = γ * x_norm + β  # Learnable parameters
```

## 8. Convolution Operation

```
CONVOLUTION (for CNN understanding):

Input (5x5):          Kernel (3x3):      Output (3x3):
┌─────────┐          ┌─────┐            ┌─────┐
│1 2 3 4 5│          │1 0 1│            │? ? ?│
│6 7 8 9 0│    *     │0 1 0│     =      │? ? ?│
│1 2 3 4 5│          │1 0 1│            │? ? ?│
│6 7 8 9 0│          └─────┘            └─────┘
│1 2 3 4 5│
└─────────┘

Each output = sum of element-wise multiplication
Output[0,0] = 1*1 + 2*0 + 3*1 + 6*0 + 7*1 + 8*0 + 1*1 + 2*0 + 3*1
```

## Key Whiteboard Drawing Tips:

1. **Always show dimensions** - write shapes next to matrices
2. **Use arrows** to show data flow direction  
3. **Draw coordinate systems** for geometric concepts
4. **Show before/after** for transformations
5. **Use different colors** for different concepts
6. **Include numerical examples** - makes it concrete
7. **Draw the "why"** - show intuition, not just mechanics

## Interview Questions You Can Now Answer:

1. "Explain broadcasting and why it's important"
2. "How does matrix multiplication work in neural networks?"
3. "What's the geometric intuition behind PCA?"
4. "Why do we need numerical stability in softmax?"
5. "How does gradient descent find the minimum?"
6. "Explain batch normalization step by step"
7. "What happens when we change softmax temperature?"

## Practice Exercises:

1. Draw a 3-layer neural network with dimensions
2. Show how broadcasting works with (64,10) + (10,)
3. Illustrate PCA on 2D data projected to 1D
4. Draw the gradient descent path on a loss surface
5. Show batch normalization on a mini-batch