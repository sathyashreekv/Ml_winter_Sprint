"""
BROADCASTING EXPLAINED: (100,3) - (3,)
Step-by-step breakdown with visual examples
"""

import numpy as np

print("="*60)
print("BROADCASTING: (100,3) - (3,) DETAILED EXPLANATION")
print("="*60)

# Create example arrays
points = np.array([
    [1, 2, 3],    # Point 1
    [4, 5, 6],    # Point 2  
    [7, 8, 9],    # Point 3
    [2, 1, 4],    # Point 4
    [5, 3, 2]     # Point 5
])  # Shape: (5, 3) - using 5 points for clarity

center = np.array([2, 2, 2])  # Shape: (3,)

print(f"Points shape: {points.shape}")
print(f"Center shape: {center.shape}")
print(f"\nPoints array:")
print(points)
print(f"\nCenter array: {center}")

print("\n" + "="*60)
print("STEP-BY-STEP BROADCASTING PROCESS")
print("="*60)

print("Step 1: Align dimensions from the right")
print("Points: (5, 3)")
print("Center:    (3)  ← Missing left dimension")

print("\nStep 2: Pad shorter array with 1s on the left")
print("Points: (5, 3)")
print("Center: (1, 3)  ← Padded with 1")

print("\nStep 3: Check dimension compatibility")
print("- Dimension 0: 5 vs 1 → 1 can broadcast to 5 ✅")
print("- Dimension 1: 3 vs 3 → Same size ✅")

print("\nStep 4: Broadcasting happens automatically")
result = points - center
print(f"Result shape: {result.shape}")

print("\n" + "="*60)
print("VISUAL REPRESENTATION")
print("="*60)

print("BEFORE BROADCASTING:")
print("Points (5,3):           Center (3,):")
print("┌─────────┐            ┌─────────┐")
print("│ 1  2  3 │            │ 2  2  2 │")
print("│ 4  5  6 │            └─────────┘")
print("│ 7  8  9 │")
print("│ 2  1  4 │")
print("│ 5  3  2 │")
print("└─────────┘")

print("\nAFTER BROADCASTING:")
print("Points (5,3):           Center (broadcasted to 5,3):")
print("┌─────────┐            ┌─────────┐")
print("│ 1  2  3 │     -      │ 2  2  2 │")
print("│ 4  5  6 │            │ 2  2  2 │")
print("│ 7  8  9 │            │ 2  2  2 │")
print("│ 2  1  4 │            │ 2  2  2 │")
print("│ 5  3  2 │            │ 2  2  2 │")
print("└─────────┘            └─────────┘")

print("\nRESULT (5,3):")
print("┌─────────┐")
for i, row in enumerate(result):
    print(f"│{row[0]:2.0f} {row[1]:2.0f} {row[2]:2.0f} │  ← Point {i+1} - Center")
print("└─────────┘")

print("\n" + "="*60)
print("ELEMENT-WISE CALCULATION")
print("="*60)

print("What happens under the hood:")
for i, point in enumerate(points):
    calculation = f"Point {i+1}: [{point[0]}, {point[1]}, {point[2]}] - [2, 2, 2] = [{point[0]-2}, {point[1]-2}, {point[2]-2}]"
    print(calculation)

print("\n" + "="*60)
print("REAL-WORLD ML EXAMPLE: DISTANCE CALCULATION")
print("="*60)

# Realistic example with 100 points
np.random.seed(42)  # For reproducible results
points_3d = np.random.randn(100, 3) * 10  # 100 random 3D points
origin = np.array([0, 0, 0])  # Origin point

print(f"100 3D points shape: {points_3d.shape}")
print(f"Origin shape: {origin.shape}")

# Calculate distances from origin
vectors_from_origin = points_3d - origin  # Broadcasting: (100,3) - (3,)
distances = np.linalg.norm(vectors_from_origin, axis=1)

print(f"\nVectors from origin shape: {vectors_from_origin.shape}")
print(f"Distances shape: {distances.shape}")

print(f"\nFirst 5 points:")
for i in range(5):
    point = points_3d[i]
    vector = vectors_from_origin[i]
    dist = distances[i]
    print(f"Point {i+1}: [{point[0]:6.2f}, {point[1]:6.2f}, {point[2]:6.2f}] → Distance: {dist:6.2f}")

print("\n" + "="*60)
print("COMMON BROADCASTING PATTERNS IN ML")
print("="*60)

patterns = [
    ("Centering data", "(N, D) - (D,)", "X - X.mean(axis=0)"),
    ("Scaling features", "(N, D) / (D,)", "X / X.std(axis=0)"),
    ("Adding bias", "(N, D) + (D,)", "linear_output + bias"),
    ("Distance to centroid", "(N, D) - (D,)", "points - centroid"),
    ("Batch normalization", "(N, D) - (D,)", "(X - batch_mean) / batch_std"),
]

for description, shapes, example in patterns:
    print(f"{description:.<20} {shapes:.<15} {example}")

print("\n" + "="*60)
print("WHY BROADCASTING IS POWERFUL")
print("="*60)

print("Without broadcasting (inefficient):")
print("""
# Manual way - memory intensive
center_repeated = np.tile(center, (100, 1))  # Shape: (100, 3)
result = points - center_repeated
""")

print("\nWith broadcasting (efficient):")
print("""
# NumPy way - memory efficient  
result = points - center  # Broadcasting handles it automatically
""")

print("\nBenefits:")
print("✅ Memory efficient - no need to create large repeated arrays")
print("✅ Faster computation - optimized C code")
print("✅ Cleaner code - more readable and intuitive")
print("✅ Automatic - no manual dimension handling")

print("\n" + "="*60)
print("WHITEBOARD DRAWING GUIDE")
print("="*60)

print("""
How to draw (100,3) - (3,) on whiteboard:

1. Draw the larger array as a matrix:
   ┌─────────┐
   │ • • • │ ← 100 rows
   │ • • • │   3 columns  
   │ • • • │
   │  ...  │
   └─────────┘

2. Draw the smaller array as a single row:
   [• • •] ← 3 elements

3. Show the broadcasting with an arrow:
   [• • •] ──┐
   [• • •] ──┤ Broadcast to
   [• • •] ──┤ all 100 rows
   [ ... ] ──┘

4. Show the result:
   ┌─────────┐
   │ ○ ○ ○ │ ← Each row operated
   │ ○ ○ ○ │   with broadcast array
   │ ○ ○ ○ │
   │  ...  │
   └─────────┘
""")

print("="*60)
print("🎯 KEY TAKEAWAY:")
print("Broadcasting (100,3) - (3,) subtracts the 3-element array")
print("from EACH of the 100 rows automatically!")
print("="*60)