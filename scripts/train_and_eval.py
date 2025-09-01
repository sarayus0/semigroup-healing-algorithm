import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from semigroup_healing.trust import compute_trust_map
from semigroup_healing.heal import (
    make_reference_table,
    generate_closed_groups,
    heal_small_groups_with_ml_fast,
    merge_votes_weighted,
    enforce_associativity,
    heal_via_triple_closures,
    restore_missing_from_corrupt,
    layered_subgroup_heal
)
from semigroup_healing.data import load_npz_dataset
from .generate_dataset import table_to_cube, clean_tables, corrupt_tables
from semigroup_healing.metrics import is_associative

min_conf = 0.5

# ---------- Split by TABLE (not by cell) ----------
def split_by_table(clean_tables, corrupt_tables, trust_maps, test_size=0.2, seed=42, strict=False):
    """
    Split by TABLE (not cell). Handles mismatched dataset lengths safely.
    If strict=True, raise if lengths differ. Otherwise truncate to the common min length.
    """
    # lengths
    n_clean  = len(clean_tables)
    n_corrupt = len(corrupt_tables)
    n_trust  = len(trust_maps)
    N = min(n_clean, n_corrupt, n_trust)

    if not (n_clean == n_corrupt == n_trust):
        msg = (f"[split_by_table] Lengths differ: clean={n_clean}, corrupt={n_corrupt}, trust={n_trust}. "
               f"Using first N={N} tables from each.")
        if strict:
            raise ValueError(msg)
        else:
            print(msg)

    # helper that indexes np arrays or stacks lists
    def take(arr, ix):
        if isinstance(arr, np.ndarray):
            return arr[ix]
        else:  # list/sequence of arrays
            return np.stack([arr[i] for i in ix])

    # align to common length N up front
    # (convert to numpy if needed to slice; otherwise slice the list then stack in take)
    def head(arr, n):
        return arr[:n] if not isinstance(arr, np.ndarray) else arr[:n]

    clean_aligned   = head(clean_tables, N)
    corrupt_aligned = head(corrupt_tables, N)
    trust_aligned   = head(trust_maps, N)

    # now split on the aligned length
    idx = np.arange(N)
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed, shuffle=True)

    clean_train,  clean_test   = take(clean_aligned,  train_idx), take(clean_aligned,  test_idx)
    corrupt_train, corrupt_test = take(corrupt_aligned, train_idx), take(corrupt_aligned, test_idx)
    trust_train,  trust_test   = take(trust_aligned,  train_idx), take(trust_aligned,  test_idx)

    return (clean_train, corrupt_train, trust_train, train_idx), (clean_test, corrupt_test, trust_test, test_idx)


# ---------- Feature extraction on a SUBSET ----------
def extract_features_and_labels_from_subset(clean_sub, corrupt_sub, trust_sub):
    num_tables, n, _ = corrupt_sub.shape
    X, y = [], []
    for t in range(num_tables):
        clean = clean_sub[t]; corrupt = corrupt_sub[t]; trust = trust_sub[t]
        for i in range(n):
            for j in range(n):
                feats = [
                    i / n,
                    j / n,
                    corrupt[i, j] / n,
                    trust[i, j],
                    np.mean(trust[i, :]),
                    np.mean(trust[:, j]),
                    np.std(trust[max(0,i-1):i+2, max(0,j-1):j+2]),
                ]
                X.append(feats)
                y.append(int(corrupt[i, j] == clean[i, j]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

# ---------- Train classifier on TRAIN tables  ----------
def train_classifier_on_train_tables(clean_train, corrupt_train, trust_train, seed=42):
    Xtr, ytr = extract_features_and_labels_from_subset(clean_train, corrupt_train, trust_train)
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    clf.fit(Xtr, ytr)
    # Optional: quick train diagnostics
    yhat = clf.predict(Xtr)
    print(" Train (cell-level) diagnostics:")
    print(classification_report(ytr, yhat, target_names=["Corrupted","Correct"]))
    print(confusion_matrix(ytr, yhat))
    return clf

# ---------- Masking ----------
def mask_table_with_classifier(corrupt_table, trust_map, clf, mask_thresh: float = 0.40):
    """
    Mask cells predicted corrupted by RandomForest.
    Uses calibrated probability of 'Correct' (class=1) and masks if P(correct) < mask_thresh.
    Lower mask_thresh => more aggressive masking; higher => more conservative.
    """
    n = corrupt_table.shape[0]
    masked_table = corrupt_table.copy()
    for i in range(n):
        for j in range(n):
            feats = np.array([[
                i / n,
                j / n,
                corrupt_table[i, j] / n,
                trust_map[i, j],
                np.mean(trust_map[i, :]),
                np.mean(trust_map[:, j]),
                np.std(trust_map[max(0,i-1):i+2, max(0,j-1):j+2]),
            ]], dtype=np.float32)

            # P(correct) = proba[:, 1]
            proba = clf.predict_proba(feats)[0, 1]
            if proba < mask_thresh:
                masked_table[i, j] = -1
    for row in masked_table:
        print(" ".join(f"{val:2d}" for val in row))
    return masked_table

# ---------- Associativity healer for -1s ----------
def heal_with_associativity(masked_table):
    n = masked_table.shape[0]
    healed = masked_table.copy()
    changed = True

    def factor_pairs(i):
        pairs = []
        for i1 in range(n):
            for i2 in range(n):
                if healed[i1, i2] == i:  # i = i1 * i2
                    pairs.append((i1, i2))
        return pairs

    while changed:
        changed = False
        for i in range(n):
            for j in range(n):
                if healed[i, j] != -1:
                    continue
                candidates = {}
                pairs = factor_pairs(i)
                for i1, i2 in pairs:
                    if healed[i2, j] == -1:
                        continue
                    mid = healed[i2, j]
                    if healed[i1, mid] == -1:
                        continue
                    val = healed[i1, mid]  # candidate via associativity
                    candidates[val] = candidates.get(val, 0) + 1
                if candidates:
                    best_val = max(candidates.items(), key=lambda x: x[1])[0]
                    healed[i, j] = best_val
                    changed = True
    return healed

# ---------- Evaluation ----------
def per_cell_accuracy(healed_table, clean_table):
    return (healed_table == clean_table).mean()

def associativity_fraction(T):
    n = T.shape[0]
    tot = n*n*n
    sat = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if T[T[i, j], k] == T[i, T[j, k]]:
                    sat += 1
    return sat / tot

def evaluate_healed_table(healed_table, clean_table):
    acc = per_cell_accuracy(healed_table, clean_table)
    assoc = associativity_fraction(healed_table)
    print(f" Per-cell accuracy:       {acc*100:.2f}%")
    print(f" Associativity satisfied: {assoc*100:.2f}%")
    return acc, assoc

def baseline_test_stats(corrupt_test, clean_test):
    accs0, assocs0 = [], []
    for t in range(len(corrupt_test)):
        acc0 = (corrupt_test[t] == clean_test[t]).mean()
        assoc0 = associativity_fraction(corrupt_test[t])
        accs0.append(acc0); assocs0.append(assoc0)
    print("\n Baseline (uncorrected) test stats")
    print(f"• Mean per-cell accuracy:       {100*np.mean(accs0):.2f}%")
    print(f"• Mean associativity satisfied: {100*np.mean(assocs0):.2f}%")

def evaluate_on_test_tables(corrupt_test, clean_test, trust_test, clf):
    healed_tables = []
    accs, assocs = [], []
    for t in range(len(corrupt_test)):
        print(f"\n Test table {t+1}/{len(corrupt_test)}")
        masked = mask_table_with_classifier(corrupt_test[t], trust_test[t], clf)
        healed0 = heal_with_associativity(masked)

        # --- subgroup pass (FAST ML-weighted) ---
        T_ref  = make_reference_table(healed0, corrupt_test[t])
        tm_ref = compute_trust_map(T_ref)

        # you can cap groups per size to keep things snappy (tune these)
        CAP2, CAP3, CAP4 = 400, 300, 200   # set None to use all groups

        # 2×2
        groups_2 = generate_closed_groups(T_ref, max_size=2)
        votes2   = heal_small_groups_with_ml_fast(
            healed0, T_ref, groups_2, clf, tm_ref,
            min_conf_for_merge=0.55, group_cap=CAP2, skip_if_locally_assoc=True
        )
        T1 = merge_votes_weighted(healed0, votes2, trust=tm_ref, min_conf=0.55)

        # 3×3
        groups_3 = generate_closed_groups(T_ref, max_size=3)
        votes3   = heal_small_groups_with_ml_fast(
            T1, T_ref, groups_3, clf, tm_ref,
            min_conf_for_merge=0.55, group_cap=CAP3, skip_if_locally_assoc=True
        )
        T2 = merge_votes_weighted(T1, votes3, trust=tm_ref, min_conf=0.55)

        # 4×4 (optional)
        groups_4 = generate_closed_groups(T_ref, max_size=4)
        votes4   = heal_small_groups_with_ml_fast(
            T2, T_ref, groups_4, clf, tm_ref,
            min_conf_for_merge=0.60, group_cap=CAP4, skip_if_locally_assoc=True
        )
        healed = merge_votes_weighted(T2, votes4, trust=tm_ref, min_conf=0.60)

        # --- light global polish (kept, still cheap) ---
        healed = enforce_associativity(healed, trust_map=tm_ref, max_passes=2)
        tm_live = compute_trust_map(healed)
        healed  = enforce_associativity(healed, trust_map=tm_live, max_passes=1)

        acc, assoc = evaluate_healed_table(healed, clean_test[t])


        healed_tables.append(healed)
        accs.append(acc); assocs.append(assoc)
    print(f"\n TEST summary over {len(corrupt_test)} tables")
    print(f" • Mean per-cell accuracy:       {100*np.mean(accs):.2f}%")
    print(f" • Mean associativity satisfied: {100*np.mean(assocs):.2f}%")
    return np.stack(healed_tables), np.array(accs), np.array(assocs)


def save_healed_tables_npz(healed_tables, filename, n):
    healed_cubes = np.array([table_to_cube(T) for T in healed_tables])
    healed_vectors = healed_cubes.reshape(len(healed_cubes), -1, 1)
    np.savez_compressed(filename, healed=healed_vectors)
    print(f" Saved healed tables to '{filename}'")

trust_maps = np.stack([compute_trust_map(T) for T in corrupt_tables])
print(len(trust_maps))  # should match len(clean_tables)

#  MAIN: split → train on TRAIN → baseline on TEST → heal TEST → save 
(train_pack, test_pack) = split_by_table(clean_tables, corrupt_tables, trust_maps, test_size=0.2, seed=42)
clean_train, corrupt_train, trust_train, train_idx = train_pack
clean_test,  corrupt_test,  trust_test,  test_idx  = test_pack
print(f"Train tables: {len(train_idx)} | Test tables: {len(test_idx)}")

clf = train_classifier_on_train_tables(clean_train, corrupt_train, trust_train, seed=42)

baseline_test_stats(corrupt_test, clean_test)

healed_test, accs_test, assocs_test = evaluate_on_test_tables(corrupt_test, clean_test, trust_test, clf)

n = clean_tables[0].shape[0]
save_healed_tables_npz(healed_test, f"healed_TEST_n{n}_assoc.npz", n)

# Triple Closure Healing

healed_after_closures = []
final_healed = []
stats_all = []

for i in range(healed_test.shape[0]):
    T0 = healed_test[i]
    C0 = corrupt_test[i]

    # --- run triple-closure healer on the ML table ---
    T_local, stats = heal_via_triple_closures(
        T0=T0,
        C0=C0,
        outer_iters=2,        # 2–3 is usually enough
        apply_conf=0.55,
        lock_conf=0.70
    )
    healed_after_closures.append(T_local)
    stats_all.append(stats)

    # --- one light global deterministic pass (polish) ---
    # use trust from a fully specified reference to guide
    T_ref = restore_missing_from_corrupt(T_local, C0)
    tm = compute_trust_map(T_ref)
    T_final = enforce_associativity(T_local, trust_map=tm, max_passes=2)

    final_healed.append(T_final)

healed_after_closures = np.array(healed_after_closures)
final_healed = np.array(final_healed)

# --- evaluation ---
num_assoc = sum(is_associative(T) for T in final_healed)
print(f"{num_assoc}/{len(final_healed)} fully associative ({100*num_assoc/len(final_healed):.1f}%)")

# optional: compare to clean
acc0 = (healed_test == clean_test).mean()
acc1 = (healed_after_closures == clean_test).mean()
acc2 = (final_healed == clean_test).mean()
print(f"per-cell accuracy: ML={acc0:.3f}  +closures={acc1:.3f}  +global={acc2:.3f}")

healed_batch = []
assoc_flags = []
for T in healed_test:  # shape (B,10,10)
    H = layered_subgroup_heal(T, sizes=(2,3,4,5), rounds_per_size=2, final_global_pass=True)
    healed_batch.append(H)
    assoc_flags.append(is_associative(H))
healed_batch = np.array(healed_batch)
print(f"{sum(assoc_flags)}/{len(assoc_flags)} fully associative")