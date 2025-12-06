import MiniNumPy as mnp
import numpy as np
import random

def random_float_matrix(n, low=-10.0, high=10.0):
    data = [[random.uniform(low, high) for _ in range(n)] for _ in range(n)]
    return mnp.array(data)

T = random_float_matrix(4)
print(T)
P, L, U = T.LU_Decomposition()

print(P)
print(L)
print(U)
print(L@U - P@T)

Q, R = mnp.qr_decomposition(T)
print(Q)
print(R)
print(Q@R - T)

# MA = mnp.array([[3, -2],[-81, 2]])
# ma = np.array([[1, -2],[-1, 2]])

# print(np.linalg.eig(ma))
# e,v = mnp.linalg.eig(MA)
# print(e)

# print(MA.determinant())
# # print(v)
# A = mnp.zeros((3,4,5,1))
# B = mnp.zeros((3,1,1,4))
# print(mnp.coord_to_index((2,0,3),(3,2,4)))
# print(mnp.matmul(A,B).shape)

# A = mnp.array([[1,2,3],[4,5,6]])
# B = mnp.array([1,2,3])

# print(mnp.dot(B,A))
# print(B.shape)
# print(A.shape)

A = mnp.array([[1, 2], [3, 4]])
print(mnp.inv(A))