vector=[1,2] #vector [i ,j]
matrix=[
    [1,2],
    [5,6]
]
import numpy as np

def manual_check(v1,v2):
    try:
        k1= v2[0]/v1[0] if v1[0]!=0 else 0
        
        k2= v2[1]/v1[1] if v2[1]!=0 else 0
    except ZeroDivisionError:
        return False
    
    return True if k1==k2 else False

v1=[1,0]
v2=[0,1]
print(f" Linear dependent ?{manual_check(v1,v2)}")

theta=np.radians(90)
rotation_matrix=np.array([
    [np.cos(theta),-np.sin(theta)],
    [np.sin(theta),np.cos(theta)]
])

v=np.array([1,0])
v_rotated=rotation_matrix @ v
print(f"original :{v} , Rotated by  90 degrees:{v_rotated.round()}")


matrix=np.array([
    [3,0],
    [0,2]
])
area_change=np.linalg.det(matrix)
print(f"Area change after transformation :{area_change}")

#Null space ->if our data items lie on a same plane or line in higher dimensional space it become linearly dependent leads to null space (redundant data get trapped and never contribute to the final result)
#numpy dont have np.null _space so we are using SVD ->ml algorithm

import numpy as np
A=np.array([
    [1,2,3],
    [3,6,9]
])
U,Sigma,VT=np.linalg.svd(A)
print(f"Singular value decomposition")
print(f"U shape: {U.shape}")
print(f"Sigma shape: {Sigma.shape}")
print(f"VT shape: {VT.shape}")
print(f"Sigma values: {Sigma}")

# Correct reconstruction - pad Sigma to match dimensions
Sigma_full = np.zeros((U.shape[1], VT.shape[0]))
np.fill_diagonal(Sigma_full, Sigma)
A_reconstructed = U @ Sigma_full @ VT
print(f"Original A:\n{A}")
print(f"Reconstructed A:\n{A_reconstructed}")

#Numpy 
image=np.random.rand(28,28)
flat_image=image.reshape(1,-1)
print(f"Original :{image.shape},Flattened:{flat_image.shape}")

W=np.random.rand(100,5)
print(f"Original W:{W.shape} transposed:{W.T.shape}")

prices=np.array([20000,25000,30000,35000,40000])
expensive_mask=prices>20000
print(f"Prices:{prices}")
print(f"Expensive mask:{expensive_mask}")

# This matrix is singular (determinant = 0)
# Row 3 = Row 1 + Row 2, making it linearly dependent
singular_matrix=np.array([
    [1,2,3],
    [4,5,6],
    [5,7,9]  # This row = [1,2,3] + [4,5,6]
])
print(f"Singular matrix:\n{singular_matrix}")
print(f"Determinant: {np.linalg.det(singular_matrix)}")

# Use a non-singular (invertible) matrix instead
invertible_matrix=np.array([
    [1,2,3],
    [0,1,4],
    [5,6,0]
])
print(f"\nInvertible matrix:\n{invertible_matrix}")
I=np.eye(3)
print(f"Identity matrix:\n{I}")
result_matrix=invertible_matrix@I
print(f"Matrix @ Identity:\n{result_matrix}")

# Check if matrix is invertible (determinant != 0)
det = np.linalg.det(invertible_matrix)
print(f"Determinant: {det}")

if abs(det) > 1e-10:  # Check if not singular
    inverse_matrix=np.linalg.inv(invertible_matrix)
    final=invertible_matrix @ inverse_matrix
    print(f"Matrix @ Inverse:\n{final.round(10)}")
else:
    print("Matrix is singular (not invertible)")

# Show why original matrix was singular
print(f"\nWhy [1,2,3; 4,5,6; 7,8,9] is singular:")
print(f"Row 3 - Row 2 - Row 1 = {[7,8,9]} - {[4,5,6]} - {[1,2,3]} = {[7-4-1, 8-5-2, 9-6-3]}")

A=np.arange(24).reshape(2,3,4)
A=np.sum(A,axis=1)
print(f" A.shape: {A.shape} and {A}")


sample=np.random.rand(10,5)
print(f"sample mattrix :{sample} with 10 samples with 5 features")
mean_sample=np.mean(sample,axis=0)
print(f" mean of sample matrix :{mean_sample}")
new_sample=sample-mean_sample
print(f" Centred matrix :{new_sample}")
uh,ps,vh=np.linalg.svd(new_sample)
print("SVD VALUES")
print(f"Sigma value:{ps}")
print(f"Unitary value:{uh}")
print(f"VT value:{vh}")


