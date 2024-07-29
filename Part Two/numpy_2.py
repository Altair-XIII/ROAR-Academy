import numpy as np

a = np.array([2])
b = np.array([[6, -9, 1], [4, 24, 8]])
print(a*b)
print()

a = np.array([[1, 0], [0, 1]])
b = np.array([[6, -9, 1], [4, 24, 8]])
print(a@b)
print()

a = np.array([[4, 3], [3, 2]])
b = np.array([[-2, 3], [3, -4]])
print(a@b)