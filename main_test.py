import sys
from pathlib import Path

# Add the src folder to the Python import path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from MiniNumPy import MiniNumPy as mnp
import numpy as np

a = mnp.Array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Array data:", a.data)
print("Array shape:", a.shape)
print("Array ndim:", a.ndim)
print("Array size:", a.size)


b = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
c = np.zeros((2, 3, 4, 5))
print(a.flatten())
print(c)
# print("NumPy Array shape:", b.shape)
# print("NumPy Array ndim:", b.ndim)
# print("NumPy Array size:", b.size)
print(b)