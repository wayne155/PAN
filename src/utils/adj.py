import torch
import numpy as np
from scipy.sparse import coo_matrix

def adj_to_edge_index_weight(adj_matrix):
    adj_coo = coo_matrix(adj_matrix)
    
    src = adj_coo.row
    dst = adj_coo.col
    
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    weight = torch.tensor(adj_coo.data, dtype=torch.float)
    
    return edge_index, weight



def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave
