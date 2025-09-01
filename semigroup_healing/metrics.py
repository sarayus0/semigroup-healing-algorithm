import numpy as np

def is_associative(T: np.ndarray) -> bool:
    n = T.shape[0]
    for i in range(n):
        for j in range(n):
            ij = T[i, j]
            if ij < 0 or ij >= n: return False
            for k in range(n):
                jk = T[j, k]
                if jk < 0 or jk >= n: return False
                a = T[ij, k]
                b = T[i, jk]
                if a < 0 or a >= n or b < 0 or b >= n: return False
                if a != b:
                    return False
    return True

def percent_fully_associative(tables):
    total = len(tables)
    count = 0
    for T in tables:
        if is_associative(T):
            count += 1
    pct = 100 * count / total
    print(f"✅ {count}/{total} tables are fully associative ({pct:.2f}%)")
    return pct

def count_fully_associative(tables: np.ndarray):
    """
    Count how many tables in a batch are fully associative.
    tables: (N,n,n) array
    """
    count = 0
    for T in tables:
        if is_associative(T):
            count += 1
    print(f"{count}/{len(tables)} tables are fully associative "
          f"({100*count/len(tables):.1f}%)")
    return count

