import numpy as np


"""
Once an array is already created you can do np.reshape(array, rows, columns)
"""
        
# Method 1
# def set_array(L, rows, cols):
#     list = []
#     for i in range(rows):
#         list.append(L[:cols])
#         for i in range(cols):
#             L.pop(0)
#     array = np.array(list)
#     return array

# Method 2
def set_array(L, rows, cols):
    array = np.array(L).reshape(rows, cols) # reshape takes in dimensions
    return array
    
print(set_array([1, 2, 3, 4, 5, 6], 2, 3))