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

