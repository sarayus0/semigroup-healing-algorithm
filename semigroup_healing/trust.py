import numpy as np

def compute_trust_map(T: np.ndarray) -> np.ndarray:
    """trust = 1 - fails/participations across associativity triples; robust to valid indices only"""
    n = T.shape[0]
    fails = np.zeros((n, n), dtype=np.int64)
    part  = np.zeros((n, n), dtype=np.int64)

    for i in range(n):
        for j in range(n):
            ij = T[i, j]
            if not (0 <= ij < n):  # skip invalid
                continue
            for k in range(n):
                jk = T[j, k]
                if not (0 <= jk < n):
                    continue

                # participants (count once per side we actually evaluate)
                part[i, j] += 1
                part[j, k] += 1
                part[ij, k] += 1
                part[i, jk] += 1

                a = T[ij, k]
                b = T[i, jk]
                if not (0 <= a < n and 0 <= b < n):
                    # count as failure against any valid participants
                    fails[i, j] += 1
                    fails[j, k] += 1
                    if 0 <= ij < n: fails[ij, k] += 1
                    if 0 <= jk < n: fails[i, jk] += 1
                    continue

                if a != b:
                    fails[i, j] += 1
                    fails[j, k] += 1
                    fails[ij, k] += 1
                    fails[i, jk] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        trust = 1.0 - (fails / np.maximum(part, 1))
    return trust.astype(np.float32)

