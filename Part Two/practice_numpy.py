import numpy as np

a = np.array([1, 2, 3])
print(a[0], a[1], a[2])
print(a.shape)

m = np.array([[1, 2, 3], [4, 5, 6]])
print(m[0])
print(m[0][0], m[0,0]) # second way is more efficient
print(m.shape)

# SPECIAL ARRAYS

# ARRAYS OF ALL ONES
a = np.ones(3); print(a)
print(a.dtype)
m = np.ones((1,3)); print(m)
m = np.ones((3,1)); print(m)

# ARRAYS OF ALL zeros
m = np.zeros((2,2)); print(m)

# ARRAYS OF SAME VALUE
m = np.full((2,2), 5); print(m)

#IDENTITY MATRIX
m = np.eye(3); print(m)

# Allocate an array w/o initialization
m = np.empty(3) # takes any random value

# Infinity
np.inf

# Not a number
np.nan

# Pi
np.pi
np.sin(np.pi)

# There are three ways to call dot-product
u = np.array([1., 2., 3.])
e0 = np.array([1.0, 0, 0])

u.dot(e0)

np.dot(u, e0)

u@e0

# Linear Algebra Routines
A = np.array([[1, 1], [4, 2]])
b = np.array([[35], [110]])
A_inverse = np.linalg.inv(A)
x = A_inverse@b
print(x)
print(np.linalg.solve(A,b))

