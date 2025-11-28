import MiniNumPy as mnp
import numpy as np

A = mnp.array([[3, 1, -1],
           [2, 4, 1],
           [-1, 2, 5]])

B = mnp.array([[4, 1],
           [1, 1],
           [1, 0]])     # 1D vector
X = mnp.linalg.solve(A, B)
print(A@X-B)
print("---")
E = mnp.array([[2, 7, 1], 
               [3, 8, 0],
               [1, 5, -2]])
F = mnp.array([[5, 2, 9],[1, 7, 3]])
G = mnp.array([[[0,1,2],[3,4,5]]])
print(G.shape)
print(A.T)
print(G.T)

HK = mnp.zeros((4,1,2,1))
TD = mnp.zeros((4,3,1,2))

print(mnp.matmul(HK,TD).size)
print((1,2,3)+(0,))