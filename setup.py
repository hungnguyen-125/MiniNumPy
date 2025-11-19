import numpy as np

a = np.array([1,1,1])     # shape (3,)
B = np.array([[1,2],
              [3,4],
              [5,6]])     # shape (3,2)

print(a @ B)