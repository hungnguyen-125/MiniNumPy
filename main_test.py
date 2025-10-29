import sys
from pathlib import Path

# Add the src folder to the Python import path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from MiniNumPy import MiniNumPy as mnp
import numpy as np

# a = mnp.Array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
# print("Array data:", a.data)
# print("Array shape:", a.shape)
# print("Array ndim:", a.ndim)
# print("Array size:", a.size)


# b = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
# print(a)
# print(b)

a = mnp.array([[1, 2, 3], [4, 5, 6]])
print(a)
b = mnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(b.reshape((4,2)))
print(b)
