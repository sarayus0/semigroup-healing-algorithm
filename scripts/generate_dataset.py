import random
import numpy as np
from semigroup_healing.data import mace4_randomflip_dataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

n = 3
num_tables = 100
corrupt_percent = 0.15

print(f"Generating {num_tables} Mace4 tables (n={n}, random flips)...")
corrupted_vectors, clean_vectors = mace4_randomflip_dataset(
    n=n,
    num_tables=num_tables,
    corrupt_percent=corrupt_percent
)
np.savez_compressed(
    f"mace4_dataset_n{n}_flip.npz",
    corrupted=corrupted_vectors,
    clean=clean_vectors
)
print(f"Saved dataset to 'mace4_dataset_n{n}_flip.npz'")

data = np.load(f"mace4_dataset_n{n}_flip.npz")

clean_tables   = np.argmax(data["clean"].reshape(-1, n, n, n), axis=3).astype(np.int64)
corrupt_tables = np.argmax(data["corrupted"].reshape(-1, n, n, n), axis=3).astype(np.int64)

clean_vec   = data["clean"].reshape(-1, n*n*n).astype(np.float32)
corrupt_vec = data["corrupted"].reshape(-1, n*n*n).astype(np.float32)

#test, get first table
B = corrupt_tables[0]
T = clean_tables[0]
