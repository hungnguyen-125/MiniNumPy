import sys
from pathlib import Path

# Add the src folder to the Python import path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import MiniNumPy as mnp
import numpy as np

# 2D array
a = mnp.array([1,1,1])
print(a.shape)  # shape (3,2)
print('-------')

# 4D array
c = mnp.array([[1,2],
              [3,4],
              [5,6]])
print(mnp.linalg.dot(a,c))  # shape (2,2,2,1)

print('-------')
d = np.array([[2, 7, 1], 
               [3, -2, 0],
               [1, 5, 3]])
print(np.linalg.inv(d))




A = np.array([[2,7,1],
              [3,-2,0],
              [1,5,3]])

b = np.array([1,2,3])

print(np.dot(A,b))
print('-------')

E = mnp.array([[2, 7, 1], 
               [3, -2, 0],
               [1, 5, 3]])
f = mnp.array([1, 2, 3])
print(mnp.dot(E,f))
