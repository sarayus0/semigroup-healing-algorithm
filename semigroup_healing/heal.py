from collections import defaultdict, Counter
import numpy as np
from functools import lru_cache
from .trust import compute_trust_map
from .metrics import is_associative
from scripts.train_and_eval import healed_test

def deterministic_repair(table):
    n = table.shape[0]
    repaired = table.copy()
    changed = True

    def factor_pairs(i):
        pairs = []
        for i1 in range(n):
            for i2 in range(n):
                if repaired[i1, i2] == i:
                    pairs.append((i1, i2))
        return pairs

    while changed:
        changed = False
        for i in range(n):
            for j in range(n):
                candidate_counts = {}
                pairs = factor_pairs(i)
                for (i1, i2) in pairs:
                    k = repaired[i2, j]
                    l = repaired[i1, k]
                    candidate_counts[l] = candidate_counts.get(l, 0) + 1

                if candidate_counts:
                    max_count = max(candidate_counts.values())
                    top_candidates = [k for k, v in candidate_counts.items() if v == max_count]
                    # prefer current value if among top candidates
                    if repaired[i, j] in top_candidates:
                        m = repaired[i, j]
                    else:
                        m = top_candidates[0]

                    if repaired[i, j] != m:
                        repaired[i, j] = m
                        changed = True
    return repaired


def make_rf_proba_cache(clf, trust_map):
    """
    Returns proba(i,j,val) = P(correct | set T[i,j]=val), with caching.
    """
    n = trust_map.shape[0]

    @lru_cache(maxsize=200_000)
    def proba(i, j, val):
        i = int(i); j = int(j); val = int(val)
        # features same template you trained on
        row_mean = float(np.mean(trust_map[i, :]))
        col_mean = float(np.mean(trust_map[:, j]))
        i0, i1 = max(0, i-1), min(n, i+2)
        j0, j1 = max(0, j-1), min(n, j+2)
        local_std = float(np.std(trust_map[i0:i1, j0:j1]))
        feats = np.array([[i/n, j/n, val/n,
                           float(trust_map[i, j]), row_mean, col_mean, local_std]],
                         dtype=np.float32)
        return float(clf.predict_proba(feats)[0, 1])  # class 1 = "Correct"
    return proba

def sample_groups(groups, k=None, seed=0):
    """Optionally downsample groups to at most k."""
    if (k is None) or (k <= 0) or (len(groups) <= k):
        return groups
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(groups), size=k, replace=False)
    return [groups[i] for i in idx]

class _FloatCounter(dict):
    def __missing__(self, key): return 0.0

def heal_small_groups_with_ml_fast(
    T_cur, T_ref, groups, clf, trust_map_global,
    min_conf_for_merge=0.55,
    size_weight=True,
    group_cap=None,
    skip_if_locally_assoc=True
):
 
    proba = make_rf_proba_cache(clf, trust_map_global)
    votes = defaultdict(_FloatCounter)

    # optional downsample (huge speedup when there are tons of groups)
    groups = sample_groups(groups, k=group_cap, seed=0)

    for G in groups:
        T_loc = reindex_project_to_group(T_cur, T_ref, G)
        if T_loc is None:
            continue

        # optional skip: don't spend time if already associative
        if skip_if_locally_assoc and is_associative(T_loc):
            continue

        # deterministic local heal (fast); you already have enforce_associativity
        tm_loc = trust_map_global[np.ix_(G, G)]
        T_healed_loc = enforce_associativity(T_loc, trust_map=tm_loc, max_passes=2)

        # vote back using RF probability as weight
        base_size = (1.0 / len(G)) if size_weight else 1.0
        for u, a in enumerate(G):
            for v, b in enumerate(G):
                pred_local  = int(T_healed_loc[u, v])
                pred_global = G[pred_local]
                # weight = RF P(correct) * (optional) size weight * (optional) trust
                w = proba(a, b, pred_global) * base_size * float(trust_map_global[a, b])
                votes[(a, b)][pred_global] += w

    return votes

def rf_proba_correct_for_value(clf, i, j, val_candidate, trust_map):
    """
    Score: P(correct | set T[i,j] = val_candidate) using your RF feature template.
    """
    n = trust_map.shape[0]
    row_mean = float(np.mean(trust_map[i, :]))
    col_mean = float(np.mean(trust_map[:, j]))
    i0, i1 = max(0, i-1), min(n, i+2)
    j0, j1 = max(0, j-1), min(n, j+2)
    local_std = float(np.std(trust_map[i0:i1, j0:j1]))
    feats = np.array([[i/n, j/n, val_candidate/n,
                       float(trust_map[i, j]), row_mean, col_mean, local_std]],
                     dtype=np.float32)
    return float(clf.predict_proba(feats)[0, 1])  # class 1 = "Correct"

def enforce_associativity_ml_local(T_loc, G, clf, trust_map_global, max_passes=4):
    m = T_loc.shape[0]
    T = T_loc.copy()

    for _ in range(max_passes):
        changed = False
        for i in range(m):
            for j in range(m):
                ij = int(T[i, j])
                for k in range(m):
                    jk = int(T[j, k])
                    a = int(T[ij, k])    # (i*j)*k   (local)
                    b = int(T[i, jk])    # i*(j*k)   (local)
                    if a == b:
                        continue

                    # Map local -> global coordinates/values
                    gi, gj, gk  = G[i], G[j], G[k]
                    gij, gjk    = G[ij], G[jk]
                    a_g, b_g    = G[a], G[b]

                    # Option L: write left cell to match right
                    pL_new = rf_proba_correct_for_value(clf, gij, gk, b_g, trust_map_global)
                    pL_old = rf_proba_correct_for_value(clf, gij, gk, a_g, trust_map_global)
                    dL = pL_new - pL_old

                    # Option R: write right cell to match left
                    pR_new = rf_proba_correct_for_value(clf, gi, gjk, a_g, trust_map_global)
                    pR_old = rf_proba_correct_for_value(clf, gi, gjk, b_g, trust_map_global)
                    dR = pR_new - pR_old

                    # Apply whichever increases RF confidence more (if any)
                    if dL > dR and dL > 0:
                        T[ij, k] = b
                        changed = True
                    elif dR >= dL and dR > 0:
                        T[i, jk] = a
                        changed = True
        if not changed:
            break
    return T

class _FloatCounter(dict):
    def __missing__(self, key): return 0.0

def heal_small_groups_with_ml(T_cur, T_ref, groups, clf, trust_map_global, max_passes=4):
    """
    Build local tables per group, ML-heal locally, then vote back to global,
    weighting votes by RF P(correct).
    """
    votes = defaultdict(_FloatCounter)

    for G in groups:
        T_loc = reindex_project_to_group(T_cur, T_ref, G)
        if T_loc is None:
            continue

        T_healed_loc = enforce_associativity_ml_local(
            T_loc, G, clf, trust_map_global, max_passes=max_passes
        )

        # Vote back to global with RF probability as weight
        for u, a in enumerate(G):
            for v, b in enumerate(G):
                pred_local  = int(T_healed_loc[u, v])
                pred_global = G[pred_local]
                w = rf_proba_correct_for_value(clf, a, b, pred_global, trust_map_global)
                votes[(a, b)][pred_global] += w

    return votes

def restore_missing_from_corrupt(T_in, C_in):
    T = T_in.copy()
    mask = (T == -1)
    T[mask] = C_in[mask]
    return T

def fill_remaining_neg1_by_row_mode(T):
    n = T.shape[0]
    out = T.copy()
    for i in range(n):
        row = out[i]
        vals = row[(row >= 0) & (row < n)]
        fallback = 0 if vals.size == 0 else Counter(vals.tolist()).most_common(1)[0][0]
        row[row < 0] = fallback
        out[i] = row
    return np.clip(out, 0, n-1)

def make_reference_table(T0, C0):
    assert T0.shape == C0.shape and T0.ndim == 2 and T0.shape[0] == T0.shape[1]
    T_ref = restore_missing_from_corrupt(T0, C0)
    bad = (T_ref < 0) | (T_ref >= T_ref.shape[0])
    if bad.any():
        T_ref = fill_remaining_neg1_by_row_mode(T_ref)
    return T_ref

def bounded_closure_from_seed(seed, T_ref, max_size):
    G = set(seed)
    changed = True
    while changed:
        changed = False
        new_elems = []
        for x in G:
            for y in G:
                prod = int(T_ref[x, y])
                if prod not in G:
                    new_elems.append(prod)
        if new_elems:
            for z in new_elems:
                G.add(z)
                if len(G) > max_size:
                    return sorted(G), False
            changed = True
    return sorted(G), True

def triple_seed(T_ref, i, j, k):
    ij  = int(T_ref[i, j])
    jk  = int(T_ref[j, k])
    ijk = int(T_ref[ij, k])
    ikj = int(T_ref[i, jk])
    return {i, j, k, ij, jk, ijk, ikj}

def reindex_project_to_group(T_cur, T_ref, G_sorted):
    idx = {g:i for i,g in enumerate(G_sorted)}
    m = len(G_sorted)
    T_loc = np.empty((m, m), dtype=np.int64)
    for u, a in enumerate(G_sorted):
        for v, b in enumerate(G_sorted):
            v_ref = int(T_ref[a, b])
            if v_ref not in idx:
                return None
            v_cur = int(T_cur[a, b])
            T_loc[u, v] = idx[v_cur] if v_cur in idx else idx[v_ref]
    return T_loc

def unmap_local_value_to_global(val_local, G_sorted):
    return G_sorted[int(val_local)]

def generate_closed_groups(T_ref, max_size=3):
    n = T_ref.shape[0]
    seen = set()
    groups = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                seed = triple_seed(T_ref, i, j, k)
                G, ok = bounded_closure_from_seed(seed, T_ref, max_size=max_size)
                if not ok:
                    continue
                key = frozenset(G)
                if key in seen:
                    continue
                seen.add(key)
                groups.append(G)
    return groups

def heal_small_groups_with_existing(T_cur, T_ref, groups, enforce_fn, trust_map_global=None):
    votes = defaultdict(Counter)
    for G in groups:
        T_loc = reindex_project_to_group(T_cur, T_ref, G)
        if T_loc is None:
            continue
        tm_loc = None if trust_map_global is None else trust_map_global[np.ix_(G, G)]
        T_healed_loc = enforce_fn(T_loc, trust_map=tm_loc, max_passes=4)
        for u, a in enumerate(G):
            for v, b in enumerate(G):
                pred_local = int(T_healed_loc[u, v])
                pred_global = unmap_local_value_to_global(pred_local, G)
                votes[(a, b)][pred_global] += 1
    return votes

def merge_votes_simple(T_cur, votes, min_conf):
    out = T_cur.copy()
    for (i, j), ctr in votes.items():
        total = sum(ctr.values())
        if total == 0:
            continue
        val_star, cnt = ctr.most_common(1)[0]
        conf = cnt / total
        if conf >= min_conf:
            out[i, j] = val_star
    return out

def merge_votes_weighted(T_cur, votes, trust, alpha_size=0.6, alpha_trust=0.4, min_conf=0.55):
    out = T_cur.copy()
    # votes[(i,j)] is Counter{val: count}; weight by group size and trust[i,j]
    for (i,j), ctr in votes.items():
        total = 0.0
        best_val, best_w = None, 0.0
        # approximate size-weight: infer from total ballots
        for val, count in ctr.items():
            # smaller groups tend to cast fewer conflicting ballots → emulate with alpha_size/count
            w = alpha_size / max(count,1) + alpha_trust * float(trust[i,j])
            if w > best_w:
                best_w, best_val = w, val
            total += w
        conf = best_w / max(total, 1e-9)
        if conf >= min_conf:
            out[i,j] = best_val
    return out

# ==== end helpers ====

def restore_missing_from_corrupt(healed: np.ndarray, corrupt: np.ndarray) -> np.ndarray:
    T = healed.copy()
    mask = (T == -1)
    T[mask] = corrupt[mask]
    return T

def fill_remaining_neg1_by_row_mode(T: np.ndarray) -> np.ndarray:
    """replace any leftover -1 by the row mode (or 0 if the row is empty/invalid)"""
    n = T.shape[0]
    out = T.copy()
    for i in range(n):
        row = out[i]
        vals = row[(row >= 0) & (row < n)]
        fallback = 0 if vals.size == 0 else Counter(vals.tolist()).most_common(1)[0][0]
        row[row < 0] = fallback
        out[i] = row
    out = np.clip(out, 0, n-1)
    return out

def make_reference_table(T0: np.ndarray, C0: np.ndarray) -> np.ndarray:
    """ensure we have a fully specified reference table with indices in [0..n-1]"""
    assert T0.shape == C0.shape and T0.ndim == 2 and T0.shape[0] == T0.shape[1], "tables must be square and same shape"
    n = T0.shape[0]
    T_ref = restore_missing_from_corrupt(T0, C0)
    # if corrupt also had -1 or out-of-range, clean them deterministically
    bad = (T_ref < 0) | (T_ref >= n)
    if bad.any():
        T_ref = fill_remaining_neg1_by_row_mode(T_ref)
    return T_ref

def triple_closure(T_ref: np.ndarray, i: int, j: int, k: int) -> list:
    """
    G = { i, j, k, i*j, j*k, (i*j)*k, i*(j*k) } computed using T_ref.
    always uses T_ref so elements are valid and closure fits small (≤7).
    """
    ij  = T_ref[i, j]
    jk  = T_ref[j, k]
    ijk = T_ref[ij, k]
    ijk2 = T_ref[i, jk]
    G = {i, j, k, ij, jk, ijk, ijk2}
    return sorted(G)

def reindex_project_to_group(T_cur: np.ndarray, T_ref: np.ndarray, G_sorted: list) -> np.ndarray:
    """
    build a local m×m table using current T_cur values but projecting any out-of-group / invalid
    cell to the reference product (which is guaranteed in-group).
    returns a valid local table over alphabet {0..m-1}.
    """
    idx = {g:i for i,g in enumerate(G_sorted)}
    m = len(G_sorted)
    T_loc = np.empty((m, m), dtype=np.int64)
    for a_t, a in enumerate(G_sorted):
        for b_t, b in enumerate(G_sorted):
            v_ref = int(T_ref[a, b])
            if v_ref not in idx:
                # not closed -> skip this group
                return None
            v = int(T_cur[a, b])
            T_loc[a_t, b_t] = idx[v] if v in idx else idx[v_ref]
    return T_loc


def unmap_local_value_to_global(val_local: int, G_sorted: list) -> int:
    return G_sorted[int(val_local)]

def enforce_associativity(T_in: np.ndarray, trust_map: np.ndarray | None = None, max_passes: int = 4) -> np.ndarray:
    """
    in-place style healing to satisfy (i*j)*k == i*(j*k).
    assumes T_in is a VALID index table over [0..n-1] (no -1 in local use).
    when trust_map provided, overwrites the lower-trust side.
    """
    T = T_in.copy()
    n = T.shape[0]
    tm = trust_map.astype(np.float32) if trust_map is not None else None

    for _ in range(max_passes):
        changed = False
        for i in range(n):
            for j in range(n):
                ij = T[i, j]
                for k in range(n):
                    jk = T[j, k]
                    a = T[ij, k]
                    b = T[i, jk]
                    if a == b:
                        continue
                    if tm is None:
                        # default: force RHS to match LHS
                        if T[i, jk] != a:
                            T[i, jk] = a
                            changed = True
                        continue
                    # trust-guided
                    trust_lhs = float(tm[ij, k])
                    trust_rhs = float(tm[i, jk])
                    if trust_lhs < trust_rhs:
                        if T[ij, k] != b:
                            T[ij, k] = b
                            changed = True
                    else:
                        if T[i, jk] != a:
                            T[i, jk] = a
                            changed = True
        if not changed:
            break
    return T

def merge_by_votes(T_cur: np.ndarray,
                   votes: dict,
                   apply_conf: float = 0.55,
                   lock_conf: float = 0.70,
                   locked: np.ndarray | None = None):
    """
    apply consensus votes into T_cur. returns (T_out, new_locked, n_changed)
    """
    n = T_cur.shape[0]
    T_out = T_cur.copy()
    if locked is None:
        locked = np.zeros((n, n), dtype=bool)
    new_locked = locked.copy()
    changes = 0

    for (i, j), counter in votes.items():
        total_w = sum(counter.values())
        if total_w <= 0:
            continue
        v_star, w_star = max(counter.items(), key=lambda kv: kv[1])
        conf = w_star / total_w

        if not new_locked[i, j] and conf >= apply_conf:
            if T_out[i, j] != v_star:
                T_out[i, j] = v_star
                changes += 1
            if conf >= lock_conf:
                new_locked[i, j] = True

    return T_out, new_locked, changes

def global_polish(T: np.ndarray, tm: np.ndarray, locked: np.ndarray, passes: int = 2) -> np.ndarray:
    """
    light global deterministic sweeps that respect locks
    """
    n = T.shape[0]
    out = T.copy()
    for _ in range(passes):
        changed = False
        for i in range(n):
            for j in range(n):
                ij = out[i, j]
                for k in range(n):
                    jk = out[j, k]
                    a = out[ij, k]
                    b = out[i, jk]
                    if a == b:
                        continue
                    # prefer writing into lower-trust side, but respect locks
                    trust_l = float(tm[ij, k])
                    trust_r = float(tm[i, jk])
                    if trust_l < trust_r:
                        if not locked[ij, k] and out[ij, k] != b:
                            out[ij, k] = b
                            changed = True
                        elif not locked[i, jk] and out[i, jk] != a:
                            out[i, jk] = a
                            changed = True
                    else:
                        if not locked[i, jk] and out[i, jk] != a:
                            out[i, jk] = a
                            changed = True
                        elif not locked[ij, k] and out[ij, k] != b:
                            out[ij, k] = b
                            changed = True
        if not changed:
            break
    return out

def bounded_closure_from_seed(seed, T_ref, max_size):
    """
    Expand `seed` under * using T_ref until closed or size exceeds max_size.
    Returns (G_sorted, closed_ok). If closed_ok=False, you should skip this group.
    """
    G = set(seed)
    changed = True
    while changed:
        changed = False
        # collect new products
        new_elems = []
        for x in G:
            for y in G:
                prod = int(T_ref[x, y])
                if prod not in G:
                    new_elems.append(prod)
        if new_elems:
            for z in new_elems:
                G.add(z)
                if len(G) > max_size:
                    return sorted(G), False  # exceeded cap
            changed = True
    return sorted(G), True

def triple_seed(T_ref, i, j, k):
    ij  = int(T_ref[i, j])
    jk  = int(T_ref[j, k])
    ijk = int(T_ref[ij, k])
    ikj = int(T_ref[i, jk])
    return {i, j, k, ij, jk, ijk, ikj}

def heal_via_triple_closures(
    T0: np.ndarray,
    C0: np.ndarray,
    outer_iters: int = 2,
    apply_conf: float = 0.55,
    lock_conf: float = 0.70,
    weight_small: float = 0.60,   # base weight for small groups (all triple-closures are ≤7)
    weight_trust: float = 0.40,   # trust contribution
    polish_passes: int = 2,
    skip_if_locally_assoc: bool = True
):
    """
    robust, exhaustive pass over all (i,j,k) triple-closures; votes + merge + global polish.
    returns (T_final, stats)
    """

    n = T0.shape[0]
    # 1) build reference + trust (fully specified)
    T_ref = make_reference_table(T0, C0)
    tm = compute_trust_map(T_ref)

    # 2) working table (clip invalids early)
    T_cur = T0.copy()
    T_cur = np.where((T_cur >= 0) & (T_cur < n), T_cur, T_ref)  # fallback invalids to ref
    locked = np.zeros((n, n), dtype=bool)

    stats = []

    for it in range(1, outer_iters+1):
        votes = defaultdict(_FloatCounter)  # custom counter for floats
        n_groups = 0
        n_skipped_assoc = 0

        # iterate all triples
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    # 1. build seed + closure
                    seed = triple_seed(T_ref, i, j, k)
                    G, ok = bounded_closure_from_seed(seed, T_ref, max_size=9)  # cap at 9
                    if not ok:
                        continue  # skip if closure explodes beyond cap

                    # 2. build local table
                    T_loc = reindex_project_to_group(T_cur, T_ref, G)
                    if T_loc is None:
                        continue  # not actually closed, skip

                    tm_loc = tm[np.ix_(G, G)]

                    # 3. optional skip if already associative
                    if skip_if_locally_assoc and is_associative(T_loc):
                        n_skipped_assoc += 1
                        continue

                    # 4. heal locally
                    T_heal = enforce_associativity(T_loc, trust_map=tm_loc, max_passes=4)

                    # 5. accumulate votes
                    base = weight_small / len(G)
                    for u, a in enumerate(G):
                        for v, b in enumerate(G):
                            pred_local = int(T_heal[u, v])
                            pred_global = G[pred_local]
                            w = base + weight_trust * float(tm[a, b])
                            votes[(a, b)][pred_global] += w
                    n_groups += 1


        # merge
        T_merged, locked, n_changes = merge_by_votes(T_cur, votes, apply_conf, lock_conf, locked)

        # global polish
        T_polished = global_polish(T_merged, tm, locked, passes=polish_passes)

        # stats + early stop
        assoc_now = is_associative(T_polished)
        stats.append({
            "iter": it,
            "groups_processed": n_groups,
            "locally_assoc_skipped": n_skipped_assoc,
            "merge_changes": int(n_changes),
            "associative": bool(assoc_now),
        })

        T_cur = T_polished
        if assoc_now:
            break

    return T_cur, stats

# SUB-SEMIGROUPS!!!

# ---------- Associativity + trust ----------
def compute_trust_map(T: np.ndarray) -> np.ndarray:
    """Trust[i,j] = 1 - (fails/participations) based on associativity checks."""
    n = T.shape[0]
    fails = np.zeros((n, n), dtype=np.int64)
    part  = np.zeros((n, n), dtype=np.int64)

    for i in range(n):
        for j in range(n):
            tij = T[i, j]
            if tij < 0 or tij >= n:  # guard invalid
                continue
            for k in range(n):
                jk = T[j, k]
                if jk < 0 or jk >= n:  # guard invalid
                    continue

                a_row, a_col = tij, k
                b_row, b_col = i, jk

                part[i, j] += 1
                part[j, k] += 1
                part[a_row, a_col] += 1
                part[b_row, b_col] += 1

                if T[tij, k] != T[i, jk]:
                    fails[i, j] += 1
                    fails[j, k] += 1
                    fails[a_row, a_col] += 1
                    fails[b_row, b_col] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        trust = 1.0 - (fails / np.maximum(part, 1))
    return trust.astype(np.float32)

def enforce_associativity(T: np.ndarray, trust_map: np.ndarray | None = None,
                          max_passes: int = 8) -> np.ndarray:
    """Deterministic repair: for each violation, overwrite lower-trust side."""
    T_fixed = np.array(T, dtype=np.int64, copy=True)
    n = T_fixed.shape[0]
    tm = None if trust_map is None else trust_map.astype(np.float32)

    for _ in range(max_passes):
        changed = False
        for i in range(n):
            for j in range(n):
                lhs_row = int(T_fixed[i, j])
                if not (0 <= lhs_row < n):  # skip invalid index
                    continue
                for k in range(n):
                    rhs_col = int(T_fixed[j, k])
                    if not (0 <= rhs_col < n):
                        continue

                    # results
                    if not (0 <= T_fixed[lhs_row, k] < n):  # invalid -> favor valid side if possible
                        if 0 <= T_fixed[i, rhs_col] < n:
                            T_fixed[lhs_row, k] = int(T_fixed[i, rhs_col])
                            changed = True
                        continue
                    if not (0 <= T_fixed[i, rhs_col] < n):
                        if 0 <= T_fixed[lhs_row, k] < n:
                            T_fixed[i, rhs_col] = int(T_fixed[lhs_row, k])
                            changed = True
                        continue

                    a = int(T_fixed[lhs_row, k])
                    b = int(T_fixed[i, rhs_col])
                    if a != b:
                        if tm is not None:
                            tl = float(tm[lhs_row, k])
                            tr = float(tm[i, rhs_col])
                            if tl < tr:
                                T_fixed[lhs_row, k] = b
                            else:
                                T_fixed[i, rhs_col] = a
                        else:
                            T_fixed[i, rhs_col] = a
                        changed = True
        if not changed:
            break
    return np.clip(T_fixed, 0, n-1)

# ---------- Closure over triples ----------
def triple_closure_set(T: np.ndarray, i: int, j: int, k: int) -> set[int]:
    """Return closure set G = {i, j, k, i*j, j*k, (i*j)*k, i*(j*k)} if valid; else set()."""
    n = T.shape[0]
    try:
        ij = T[i, j]
        jk = T[j, k]
        a  = T[ij, k]      # (i*j)*k
        b  = T[i, jk]      # i*(j*k)
    except Exception:
        return set()

    S = {i, j, k, ij, jk, a, b}
    if any((x < 0 or x >= n) for x in S):
        return set()
    return S

def is_closed_on_set(T: np.ndarray, idxs: list[int]) -> bool:
    """Check that for all p,q in idxs, T[p,q] ∈ idxs."""
    idx_set = set(idxs)
    for p in idxs:
        for q in idxs:
            v = T[p, q]
            if v not in idx_set:
                return False
    return True

def reindex_subtable(T: np.ndarray, idxs: list[int]) -> np.ndarray:
    """Return local table using 0..m-1 labels, or raise if not closed."""
    if not is_closed_on_set(T, idxs):
        raise ValueError("Set is not closed under the operation.")
    index_map = {g: i for i, g in enumerate(idxs)}
    m = len(idxs)
    local = np.empty((m, m), dtype=np.int64)
    for a, ga in enumerate(idxs):
        for b, gb in enumerate(idxs):
            gv = T[ga, gb]
            local[a, b] = index_map[gv]
    return local, index_map

# ---------- Proposals and merging ----------
def unmap_local_to_global(local_T: np.ndarray, idxs: list[int]) -> dict[tuple[int,int], int]:
    """Return proposed global entries {(gi,gj): gval} from a healed local table."""
    m = len(idxs)
    out = {}
    for a in range(m):
        for b in range(m):
            gi, gj = idxs[a], idxs[b]
            gval = idxs[int(local_T[a, b])]
            out[(gi, gj)] = gval
    return out

def merge_proposals(n: int,
                    base_T: np.ndarray,
                    proposals: list[dict[tuple[int,int], int]],
                    trust: np.ndarray,
                    weight_fn=lambda m: 1.0/(m*m)) -> np.ndarray:
    """
    Merge cell proposals with trust-weighted voting.
    - Each proposal dict comes from one subgroup (size m).
    - weight_fn(m): weight per vote from a subgroup of size m (defaults: favor smaller).
    - trust weights multiply the subgroup weight by cell trust of the proposed (i,j) cell.
    """
    # Collect votes
    buckets = defaultdict(list)  # (i,j) -> list of (val, weight)
    # We need m (subgroup size) for weight; attach it by storing alongside dict
    for prop in proposals:
        # We stored only dicts; attach subgroup size inferred from unique idxs in keys
        idxs = set()
        for (i, j) in prop.keys():
            idxs.add(i); idxs.add(j)
        m = len(idxs)
        w0 = weight_fn(m)
        for (i, j), v in prop.items():
            w = w0 * float(trust[i, j])  # trust of the source cell being set
            buckets[(i, j)].append((v, w))

    T_new = base_T.copy()
    for (i, j), votes in buckets.items():
        if not votes:
            continue
        # Aggregate by value
        wsum = defaultdict(float)
        for val, w in votes:
            wsum[val] += w
        # pick argmax, tie-breaker: keep original
        best_val, best_w = None, -1.0
        for val, totw in wsum.items():
            if totw > best_w:
                best_val, best_w = val, totw
        if best_val is not None:
            T_new[i, j] = best_val
    return np.clip(T_new, 0, n-1)

def generate_closed_groups_from_triples(T: np.ndarray,
                                        min_size=2,
                                        max_size=5) -> dict[int, list[list[int]]]:
    """
    Build a dictionary: size -> list of closed index sets of that size,
    by scanning all triples and taking their associativity-closure set.
    Deduplicate by frozenset.
    """
    n = T.shape[0]
    buckets = {s: set() for s in range(min_size, max_size+1)}
    for i in range(n):
        for j in range(n):
            for k in range(n):
                S = triple_closure_set(T, i, j, k)
                m = len(S)
                if min_size <= m <= max_size:
                    # ensure closed (some triples produce non-closed sets if entries invalid)
                    idxs = sorted(S)
                    if is_closed_on_set(T, idxs):
                        buckets[m].add(frozenset(idxs))
    # Convert to lists of lists
    return {s: [sorted(list(fs)) for fs in bucket] for s, bucket in buckets.items()}

def heal_layer(T_cur: np.ndarray, groups: list[list[int]]) -> np.ndarray:
    """
    Heal one layer (all groups of a given size).
    For each group:
      - reindex to local
      - compute local trust
      - enforce associativity locally
      - produce proposals mapped to global
    Merge proposals at the end of the layer.
    """
    n = T_cur.shape[0]
    proposals = []
    # Precompute a trust map on the current global table to help merging
    global_trust = compute_trust_map(T_cur)

    for idxs in groups:
        try:
            local, index_map = reindex_subtable(T_cur, idxs)
        except ValueError:
            continue  # skip non-closed (should be rare here)

        # local trust (you can also pass global_trust[idxs][:,idxs] if you prefer)
        local_trust = compute_trust_map(local)

        # heal locally (deterministic, trust-guided)
        local_healed = enforce_associativity(local, trust_map=local_trust, max_passes=8)

        # if already associative, you may skip proposing (optional)
        if np.array_equal(local_healed, local):
            # still propose — helps reinforce consistency during merge
            pass

        # map back to global proposals
        prop = unmap_local_to_global(local_healed, idxs)
        proposals.append(prop)

    # merge all subgroup proposals for this layer
    T_next = merge_proposals(n=n, base_T=T_cur, proposals=proposals,
                             trust=global_trust,
                             weight_fn=lambda m: 1.0/(m*m))  # favor smaller
    return T_next

def layered_subgroup_heal(T: np.ndarray,
                          sizes=(2,3,4,5),
                          rounds_per_size=1,
                          final_global_pass=True) -> np.ndarray:
    """
    Layered healing from small to larger closed groups, then optional final global pass.
    """
    T_cur = np.array(T, dtype=np.int64, copy=True)
    for s in sizes:
        groups_dict = generate_closed_groups_from_triples(T_cur, min_size=s, max_size=s)
        groups = groups_dict.get(s, [])
        if not groups:
            continue
        for _ in range(rounds_per_size):
            T_next = heal_layer(T_cur, groups)
            T_cur = T_next

    if final_global_pass:
        # Final deterministic pass on full table with a fresh trust map
        tm = compute_trust_map(T_cur)
        T_cur = enforce_associativity(T_cur, trust_map=tm, max_passes=8)

    return T_cur