"""Helpful statistical functions to assist in math and data wrangling.
"""

import numpy as np
import scipy

# TODO: test this function!!
def delete_rows_csr(mat, idx):
    
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError(f"Works only for CSR format, not {type(mat)}")
    
    idx = list(idx)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[idx] = False
    return mat[mask]

# TODO: test this function!!
def delete_row_csr(mat, i):
    
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError(f"Works only for CSR format, not {type(mat)}")
    
    if not isinstance(i, int):
        raise ValueError(f"Works only for a single integer index, not {type(i)}")
    
    n = mat.indptr[i+1] - mat.indptr[i]
    if n > 0:
        mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i+1]:]
        mat.data = mat.data[:-n]
        mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i+1]:]
        mat.indices = mat.indices[:-n]
    mat.indptr[i:-1] = mat.indptr[i+1:]
    mat.indptr[i:] -= n
    mat.indptr = mat.indptr[:-1]
    mat._shape = (mat._shape[0]-1, mat._shape[1])

# TODO: test this function!!
def delete_row_lil(mat, i):
    
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError(f"Works only for LIL format, not {type(mat)}")
    
    if not isinstance(i, int):
        raise ValueError(f"Works only for a single integer index, not {type(i)}")
    
    mat.rows = np.delete(mat.rows, i)
    mat.data = np.delete(mat.data, i)
    mat._shape = (mat._shape[0] - 1, mat._shape[1])

def remove_ids_from_array(a, idx):
    
    if isinstance(a, np.ndarray):
        return np.delete(a, idx, axis=0)
    elif isinstance(a, scipy.sparse.csr_matrix):
        return delete_rows_csr(a, idx)
    elif isinstance(a, scipy.sparse.lil_matrix):
        return delete_row_lil(a, idx)
    else:
        raise ValueError(f"Unknown matrix passed to remove_ids_from_array: {type(a)}")