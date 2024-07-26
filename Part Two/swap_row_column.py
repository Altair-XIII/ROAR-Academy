import numpy as np

M1 = np.array([[1, 2], [3, 4]])

def swap_rows(M, a, b):
    # new_M = M1.copy()
    # new_M[a] = M[b]
    # new_M[b] = M[a]
    # return new_M
    if a > M.shape[0] or b > M.shape[0]:
        raise ValueError("Not enough rows to swap.")
    
    M[[a, b]] = M[[b, a]]
    return M

def swap_columns(M, a, b):
    # new_M = M1.copy()
    # new_M[:, a] = M[:, b]
    # new_M[:, b] = M[:, a]
    # return new_M
    if a > M.shape[1] or b > M.shape[1]:
        raise ValueError("Not enough columns to swap.")
    M[:, [a, b]] = M[:, [b, a]]
    return M


print(swap_rows(M1, 0, 1))
print(swap_rows(M1, 0, 2))
print(swap_columns(M1, 0, 1))