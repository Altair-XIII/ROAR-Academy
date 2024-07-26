import numpy as np
array = np.array([[0, 1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25], [30, 31, 32, 33, 34, 35], [40, 41, 42, 43, 44, 45], [50, 51, 52, 53, 54, 55]])

print(array[0][1])
print(array[1][1])

# b = []
# for i in range(6):
#      b.append(array[i][1].item())
b = array[:, 1]
print(b)
print()

c = array[1, 2:4].tolist()
print(c)
print()

d = array[2:4, 4:].tolist()
print(d)
print()

# OR do this: first slice rows, then columns
e = array[2:4, 4:6].tolist()
print(e)
print()

f = array[2::2, ::2].tolist()
print(f)
