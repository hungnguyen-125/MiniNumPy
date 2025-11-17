import sys
from pathlib import Path

# Add the src folder to the Python import path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from MiniNumPy import MiniNumPy as mnp
import numpy as np

# 2D array
a = mnp.array([[[[1, 2, 3], 
               [4, 5, 6]]]])
print(a/2)  # shape (3,2)
print('-------')

# 3D array
b = mnp.array([[[1, 2], [3, 4]], 
               [[5, 6], [7, 8]]])
print(b)  # shape (2,2,2)
print('-------')

# 4D array
c = mnp.array([[[[1], [2]], [[3], [4]]], 
               [[[5], [6]], [[7], [8]]]])
print(c)  # shape (2,2,2,1)

print('-------')
d = np.array([[[[1], [2]], [[3], [4]]], 
               [[[5], [6]], [[7], [8]]]])

e = d **3
print(b.arg_max())