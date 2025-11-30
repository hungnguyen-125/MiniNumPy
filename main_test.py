import MiniNumPy as mnp
import numpy as np

B = mnp.array([[4, 1],
           [1, 1],
           [1, 0]])     # 1D vector
# X = mnp.linalg.solve(A, B)
# print(A@X-B)
print("---")
E = mnp.array([[2, 7, 1], 
               [3, 8, 0],
               [1, 5, -2]])
F = mnp.array([1, 2, 3])
G = mnp.array([[[0,1,2],[3,4,5]]])
print(G.shape)
print(G.T)

HK = mnp.eye(4,2,-2)
TD = mnp.zeros((4,3,1,2))

DH = mnp.arange(6).reshape((3,2,1))
a = mnp.coord_to_index((2,1,0), (3,2,1))
print(mnp.solve(E,F))
print(mnp.inv(E)@F)

MA = mnp.array([[0, 2, 1], [1, 3, 4],[2, 1, 0]])

print(mnp.det(MA))

A = mnp.array([[2,2,1],[3,4,1]])

print(mnp.norm(A,3))

C = np.array([2,-7,1])

P,L,U = MA.LU_Decomposition()

Q,R = mnp.qr_decomposition(MA)

print(Q.T @ Q)
print(R)

