# ============================================================================
# File: py_tools.jl
# Part of: OhMyU1
#
# Description:
#   DEPRECATED FILE!
#   Python functions to build the matrix A and vectors b and c for the
#   constrained problem in the form Ax = b, cx -> min, with x \in {0, 1}^{n}. 
#   It generates the problem of the Euler cycle search.
# Author: Sergei
# Created: Jan 2026
# ============================================================================


py"""
import numpy as np

def get_tsp_eq(n):
    def pair_to_ind(i, j, n):
        return i * n + j

    A = np.zeros((2 * n, n ** 2), dtype=np.int8)
    for j in range(n):
        for i in range(n):
            if j != i:
                A[j, pair_to_ind(i, j, n)] = 1
    for i in range(n):
        for j in range(n):
            if j != i:
                A[n + i, pair_to_ind(i, j, n)] = 1
    b = np.ones(2 * n, dtype=np.int8)
    return A, b

def get_A_b_c():
    A, b = get_tsp_eq(n=10)
    non_zero_columns = np.any(A != 0, axis=0)
    A = A[:, non_zero_columns]

    # Define TSP objective:
    c_matrix = np.load("YOUR_PATH_HERE.npy")
    c = c_matrix.reshape(10 ** 2)
    c = c[non_zero_columns]
    return A, b, c
"""