import MiniNumPy as mnp
import numpy as np


MA = mnp.array([[3, -2],[-81, 2]])
ma = np.array([[1, -2],[-1, 2]])

print(np.linalg.eig(ma))
e,v = mnp.linalg.eig(MA)
print(e)

print(MA.determinant())
# print(v)
A = mnp.zeros((3,4,5,1))
B = mnp.zeros((3,1,1,4))
print(mnp.coord_to_index((2,0,3),(3,2,4)))
print(mnp.matmul(A,B).shape)