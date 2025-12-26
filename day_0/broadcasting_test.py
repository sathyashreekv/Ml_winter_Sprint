"""
Testing Broadcasting: (64,10) + (10,)
Let's verify if this actually works and understand why
"""

import numpy as np

# Test the broadcasting example
print("="*50)
print("TESTING BROADCASTING: (64,10) + (10,)")
print("="*50)

# Create test arrays
A = np.random.randn(64, 10)  # Shape: (64, 10)
B = np.random.randn(10)      # Shape: (10,)

print(f"Array A shape: {A.shape}")
print(f"Array B shape: {B.shape}")

# Test if broadcasting works
try:
    result = A + B
    print(f"✅ Broadcasting WORKS!")
    print(f"Result shape: {result.shape}")
    print(f"Expected: (64, 10)")
    
    # Show how it works
    print(f"\nHow it works:")
    print(f"A: (64, 10)")
    print(f"B: (10,)    ← Gets broadcasted to (1, 10)")
    print(f"Result: (64, 10)")
    
except Exception as e:
    print(f"❌ Broadcasting FAILED: {e}")

print("\n" + "="*50)
print("BROADCASTING RULES VERIFICATION")
print("="*50)

def check_broadcasting(shape1, shape2):
    """Check if two shapes can be broadcasted"""
    # Pad shorter shape with 1s on the left
    ndim1, ndim2 = len(shape1), len(shape2)
    max_ndim = max(ndim1, ndim2)
    
    shape1_padded = [1] * (max_ndim - ndim1) + list(shape1)
    shape2_padded = [1] * (max_ndim - ndim2) + list(shape2)
    
    result_shape = []
    
    for i in range(max_ndim):
        dim1, dim2 = shape1_padded[i], shape2_padded[i]
        
        if dim1 == dim2:
            result_shape.append(dim1)
        elif dim1 == 1:
            result_shape.append(dim2)
        elif dim2 == 1:
            result_shape.append(dim1)
        else:
            return None, f"Incompatible dimensions: {dim1} vs {dim2}"
    
    return tuple(result_shape), "Compatible"

# Test various broadcasting scenarios
test_cases = [
    ((64, 10), (10,)),      # Our case
    ((64, 10), (64, 1)),    # Column vector
    ((64, 10), (1, 10)),    # Row vector  
    ((64, 10), (64, 10)),   # Same shape
    ((64, 10), (5,)),       # Incompatible
    ((3, 4, 5), (5,)),      # 3D + 1D
    ((3, 4, 5), (4, 1)),    # 3D + 2D
]

for shape1, shape2 in test_cases:
    result_shape, status = check_broadcasting(shape1, shape2)
    if result_shape:
        print(f"✅ {shape1} + {shape2} = {result_shape}")
    else:
        print(f"❌ {shape1} + {shape2} → {status}")

print("\n" + "="*50)
print("VISUAL BROADCASTING EXPLANATION")
print("="*50)

print("""
BROADCASTING (64,10) + (10,):

Step 1: Align dimensions from the right
A: (64, 10)
B:     (10)  ← Missing dimension treated as 1

Step 2: Pad with 1s on the left
A: (64, 10)
B: ( 1, 10)

Step 3: Check compatibility
- Dimension 0: 64 vs 1 → 1 broadcasts to 64 ✅
- Dimension 1: 10 vs 10 → Same size ✅

Step 4: Result shape = (64, 10)

Visual representation:
┌──────────────────┐     ┌──────────┐
│ row 1: [a b c d] │  +  │ [x y z w] │ (broadcasted to all rows)
│ row 2: [e f g h] │     └──────────┘
│ row 3: [i j k l] │          ↓
│ ...              │     ┌──────────────────┐
│ row64: [m n o p] │  =  │ [a+x b+y c+z d+w]│
└──────────────────┘     │ [e+x f+y g+z h+w]│
                         │ [i+x j+y k+z l+w]│
                         │ ...              │
                         │ [m+x n+y o+z p+w]│
                         └──────────────────┘

The (10,) array gets "copied" to match each of the 64 rows.
""")

print("\n" + "="*50)
print("COMMON ML BROADCASTING PATTERNS")
print("="*50)

# Practical ML examples
print("1. Batch Normalization:")
batch_data = np.random.randn(32, 128)  # 32 samples, 128 features
batch_mean = np.mean(batch_data, axis=0)  # Shape: (128,)
centered = batch_data - batch_mean  # Broadcasting: (32,128) - (128,)
print(f"   Data: {batch_data.shape} - Mean: {batch_mean.shape} = {centered.shape}")

print("\n2. Adding Bias in Neural Networks:")
linear_output = np.random.randn(64, 10)  # 64 samples, 10 neurons
bias = np.random.randn(10)  # 10 bias values
with_bias = linear_output + bias  # Broadcasting: (64,10) + (10,)
print(f"   Linear: {linear_output.shape} + Bias: {bias.shape} = {with_bias.shape}")

print("\n3. Feature Scaling:")
features = np.random.randn(1000, 20)  # 1000 samples, 20 features
feature_std = np.std(features, axis=0)  # Shape: (20,)
scaled = features / feature_std  # Broadcasting: (1000,20) / (20,)
print(f"   Features: {features.shape} / Std: {feature_std.shape} = {scaled.shape}")

print("\n4. Distance Computation:")
points = np.random.randn(100, 3)  # 100 points in 3D
center = np.array([0, 0, 0])  # Shape: (3,)
distances = np.linalg.norm(points - center, axis=1)  # Broadcasting: (100,3) - (3,)
print(f"   Points: {points.shape} - Center: {center.shape} = distances: {distances.shape}")

print(f"\n{'='*50}")
print("✅ CONCLUSION: (64,10) + (10,) DEFINITELY WORKS!")
print("This is one of the most common broadcasting patterns in ML!")
print("="*50)