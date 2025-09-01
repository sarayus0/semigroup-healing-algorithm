import numpy as np
from itertools import product
import random
from .mace4 import generate_semigroup_with_mace4_retry

def one_hot_table(table, n):
    cube = np.zeros((n, n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            cube[i, j, table[i, j]] = 1.0
    return cube

def table_to_cube(table):
    n = table.shape[0]
    cube = np.zeros((n, n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            cube[i, j, table[i, j]] = 1.0
    return cube

def mace4_randomflip_dataset(n, num_tables, corrupt_percent=0.2):
    clean_vectors = []
    corrupted_vectors = []

    for _ in range(num_tables):
        clean_table = generate_semigroup_with_mace4_retry(n)
        clean_cube = one_hot_table(clean_table, n)
        corrupted_cube = clean_cube.copy()

        total = n * n
        num_corrupt = max(1, int(total * corrupt_percent))
        idxs = list(product(range(n), repeat=2))
        random.shuffle(idxs)
        idxs = idxs[:num_corrupt]

        # flip each corrupted cell to a random *different* integer
        for (i, j) in idxs:
            true_k = clean_table[i, j]
            choices = [k for k in range(n) if k != true_k]
            new_k = random.choice(choices)
            corrupted_cube[i, j, :] = 0.0
            corrupted_cube[i, j, new_k] = 1.0

        corrupted_vectors.append(corrupted_cube.reshape(-1, 1))
        clean_vectors.append(clean_cube.reshape(-1, 1))

    return np.array(corrupted_vectors), np.array(clean_vectors)

