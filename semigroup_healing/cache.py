import os, json, hashlib, datetime, pathlib
import numpy as np
from scripts.train_and_eval import healed_batch, clean_test, corrupt_test, clean_tables
from scripts.generate_dataset import corrupt_percent, SEED, n
from semigroup_healing.metrics import is_fully_associative, associativity_fraction
import os, json, glob, re, datetime
import matplotlib.pyplot as plt

# ---------- utilities ----------
def _ensure_dir(p: str):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def _jsonable(obj):
    """Make dicts/lists/ndarrays/np.number JSON-safe."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": True, "shape": obj.shape, "dtype": str(obj.dtype)}
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj

def _param_sig(params: dict) -> str:
    s = json.dumps(params or {}, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(s.encode()).hexdigest()[:8]

def fingerprint_tables(*arrays) -> str:
    """Stable short fingerprint for (possibly large) arrays."""
    h = hashlib.blake2b(digest_size=12)
    for A in arrays:
        A = np.asarray(A)
        h.update(str(A.shape).encode()); h.update(str(A.dtype).encode())
        # chunk to avoid huge memory spikes
        view = memoryview(A.tobytes())
        step = 1 << 20
        for i in range(0, len(view), step):
            h.update(view[i:i+step])
    return h.hexdigest()

def derive_dataset_key(n: int = None,
                       num_tables: int = None,
                       tag: str = None,
                       seed: int = None,
                       corrupt_pct: float = None,
                       fallback_hash: str = None) -> str:
    parts = []
    if tag: parts.append(tag)
    if n is not None: parts.append(f"n{n}")
    if corrupt_pct is not None: parts.append(f"p{int(round(100*corrupt_pct))}pct")
    if num_tables is not None: parts.append(f"N{num_tables}")
    if seed is not None: parts.append(f"seed{seed}")
    if fallback_hash: parts.append(f"h{fallback_hash[:8]}")
    return "_".join(parts) if parts else f"run_{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"

# ---------- results cache ----------
class ResultsCache:
    def __init__(self, root="cache"):
        self.root = root
        self.dir_results = os.path.join(root, "results")
        _ensure_dir(self.dir_results)

    def save(self,
             dataset_key: str,
             method_name: str,
             params: dict,
             metrics: dict,
             arrays_to_npz: dict | None = None):
        """
        Save a JSON summary + optional NPZ payload of arrays (e.g., healed tables).
        Returns paths.
        """
        param_sig = _param_sig(params)
        base = f"{dataset_key}__{method_name}__{param_sig}"
        json_path = os.path.join(self.dir_results, base + ".json")
        npz_path  = os.path.join(self.dir_results, base + ".npz") if arrays_to_npz else None

        payload = {
            "dataset_key": dataset_key,
            "method_name": method_name,
            "params": params or {},
            "metrics": metrics or {},
            "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "npz_path": npz_path,
            "param_sig": param_sig,
            "schema": 1
        }
        with open(json_path, "w") as f:
            json.dump(_jsonable(payload), f, indent=2)

        if arrays_to_npz:
            # arrays_to_npz is a dict like {"healed": healed_test, "accs": accs_test, ...}
            np.savez_compressed(npz_path, **arrays_to_npz)

        return {"json": json_path, "npz": npz_path}

    def load(self, dataset_key: str, method_name: str, params: dict):
        """Load summary (and npz if present). Returns (summary_dict, npz_dict_or_None)."""
        param_sig = _param_sig(params)
        base = f"{dataset_key}__{method_name}__{param_sig}"
        json_path = os.path.join(self.dir_results, base + ".json")
        npz_path  = os.path.join(self.dir_results, base + ".npz")
        if not os.path.exists(json_path):
            return None, None
        with open(json_path, "r") as f:
            summary = json.load(f)
        npz_payload = None
        if os.path.exists(npz_path):
            npz_payload = dict(np.load(npz_path, allow_pickle=True))
        return summary, npz_payload

# --- after your pipeline finished ---
CACHE = ResultsCache()


# Build a dataset key you like (pick one path)
# A) If you know generation settings:
dataset_key = derive_dataset_key(n=clean_tables[0].shape[0],
                                 num_tables=len(corrupt_test),
                                 tag="mace4_v1",           # <- or any label you use
                                 seed=SEED,                  # <- if you used one
                                 corrupt_pct=corrupt_percent)

# B) Or derive from content (fingerprint):
# dataset_key = "custom_" + fingerprint_tables(corrupt_test, clean_test)

# Describe the method + params you used for this run
method_name = "heal_with_ml"         # or "second_heal", "methodB", etc.
params = {
    "mask_thresh": 0.40,
    "min_conf_small": 0.50,
    "min_conf_medium": 0.55,
    "min_conf_large": 0.60,
    "polish_passes": 2
}

# Compute a few summary metrics (use whatever you already have)
pct_fully_assoc = 100.0 * np.mean([is_fully_associative(T) for T in healed_batch])
mean_assoc_frac = float(np.mean([associativity_fraction(T) for T in healed_batch]))

num_test = healed_batch.shape[0]
accs = np.array([(healed_batch[i] == clean_test[i]).mean() for i in range(num_test)],dtype=np.float32)
mean_cell_acc = float(accs.mean())

metrics = {
    "pct_fully_associative": float(pct_fully_assoc),
    "mean_associativity_fraction": mean_assoc_frac,
    "mean_cell_accuracy": float(mean_cell_acc),
    "num_test_tables": int(num_test)
}

# Save arrays you want to reuse/plot later (optional but handy)
arrays = {
    "healed": healed_batch,      # (T, n, n)
    "accs": accs,          # (T,)
    "assocs": pct_fully_assoc       # (T,)
}

# Mount Drive (Colab)
CACHE = ResultsCache(root="cache")

# ... then call CACHE.save(...) like before

paths = CACHE.save(dataset_key=dataset_key,
                   method_name=method_name,
                   params=params,
                   metrics=metrics,
                   arrays_to_npz=arrays)

print("Saved summary:", paths["json"])
print("Saved arrays:", paths["npz"])

import os, glob, json
import numpy as np

# --- must match what you used when saving ---
method_name = "heal_with_ml"
params = {
    "mask_thresh": 0.40,
    "min_conf_small": 0.50,
    "min_conf_medium": 0.55,
    "min_conf_large": 0.60,
    "polish_passes": 2
}
# if you saved with these:
dataset_key = derive_dataset_key(
    n=n,                      # <— you said n=3
    num_tables=len(corrupt_test),  # same as when you saved
    tag="mace4_v1",
    seed=SEED,
    corrupt_pct=0.15
)

# First try exact load (matches param_sig)
summary, payload = CACHE.load(dataset_key, method_name, params)

# Fallback: if params changed, grab the latest file for that dataset/method
if summary is None:
    pat = f"cache/results/{dataset_key}__{method_name}__*.json"
    cand = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
    if not cand:
        raise FileNotFoundError(f"No cached results found for '{dataset_key}' + '{method_name}'.")
    json_path = cand[0]
    base = json_path[:-5]  # strip .json
    npz_path = base + ".npz" if os.path.exists(base + ".npz") else None
    with open(json_path, "r") as f:
        summary = json.load(f)
    payload = dict(np.load(npz_path, allow_pickle=True)) if npz_path else None

# -------- print what’s saved about associativity --------
print("=== CACHE SUMMARY ===")
print("dataset_key:", summary["dataset_key"])
print("method_name:", summary["method_name"])
print("param_sig  :", summary["param_sig"])
print("created_utc:", summary["created_utc"])
print("\n-- Metrics --")
m = summary.get("metrics", {})
print("pct_fully_associative      :", m.get("pct_fully_associative"))
print("mean_associativity_fraction:", m.get("mean_associativity_fraction"))
print("mean_cell_accuracy         :", m.get("mean_cell_accuracy"))
print("num_test_tables            :", m.get("num_test_tables"))

# If NPZ exists, try to show per-table associativity fractions
if payload is not None:
    key_candidates = ["assocs", "assoc_fracs"]
    found = None
    for k in key_candidates:
        if k in payload:
            found = k
            break
    if found:
        arr = np.array(payload[found])
        print(f"\n-- Arrays --\n'{found}' dtype={arr.dtype}, shape={arr.shape}")
        if arr.ndim == 0:
            # You saved a single scalar under 'assocs'
            print("Stored as a scalar value:", float(arr))
            print("Tip: if you want per-table fractions, save an array of length num_test.")
        else:
            print("first 10 per-table assoc fractions:", np.round(arr[:1]()))

# ------------ config ------------
METHOD_FILTER   = "heal_with_ml"              # e.g. "new_heal_pipeline" or None to accept any
CORRUPT_PCT_FIX = 15                # pick one corruption % to plot (e.g., 15)
N_RANGE         = range(3, 11)       # 3..8 inclusive
RESULTS_DIR = CACHE.dir_results if 'CACHE' in globals() else "cache/results"
# --------------------------------

def _parse_dataset_key(key: str):
    """Extract n, corrupt_pct, N, seed, tag from your dataset_key."""
    out = {"tag": None, "n": None, "corrupt_pct": None, "num_tables": None, "seed": None}
    toks = key.split("_")
    if toks:
        out["tag"] = toks[0]
    for t in toks:
        m = re.match(r"n(\d+)$", t)
        if m: out["n"] = int(m.group(1)); continue
        m = re.match(r"p(\d+)pct$", t)
        if m: out["corrupt_pct"] = int(m.group(1)); continue
        m = re.match(r"N(\d+)$", t)
        if m: out["num_tables"] = int(m.group(1)); continue
        m = re.match(r"seed(\d+)$", t)
        if m: out["seed"] = int(m.group(1)); continue
    return out

def _parse_iso(ts: str):
    try:
        # stored as UTC ISO like "...Z"
        return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None

# Load all cached summaries
rows = []
for j in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
    with open(j, "r") as f:
        meta = json.load(f)
    keybits = _parse_dataset_key(meta.get("dataset_key",""))
    m = {
        "json_path": j,
        "method_name": meta.get("method_name"),
        "created": _parse_iso(meta.get("created_utc","")) or datetime.datetime.utcfromtimestamp(os.path.getmtime(j)),
        "n": keybits["n"],
        "corrupt_pct": keybits["corrupt_pct"],
        "metrics": meta.get("metrics", {}),
    }
    rows.append(m)

# Filter by desired corrupt % and n range and (optionally) method
rows = [r for r in rows
        if r["n"] in N_RANGE
        and r["corrupt_pct"] == CORRUPT_PCT_FIX
        and (METHOD_FILTER is None or r["method_name"] == METHOD_FILTER)]

if not rows:
    print("No cached runs found for your filters. Check RESULTS_DIR / filters.")
else:
    # Keep only the latest run per (n)
    latest_by_n = {}
    for r in rows:
        n = r["n"]
        if (n not in latest_by_n) or (r["created"] > latest_by_n[n]["created"]):
            latest_by_n[n] = r

    # Build x/y for the plot
    xs = sorted(latest_by_n.keys())
    ys = [latest_by_n[n]["metrics"].get("pct_fully_associative", None) for n in xs]

    # Pretty print what we'll plot
    print(f"Using corruption % = {CORRUPT_PCT_FIX} and "
          f"{'any method' if METHOD_FILTER is None else METHOD_FILTER}\n")
    print("n | % fully assoc | mean assoc frac | mean cell acc | when (UTC)               | method")
    print("--+---------------+------------------+---------------+---------------------------+---------------------")
    for n in xs:
        m = latest_by_n[n]["metrics"]
        print(f"{n:<2}| {m.get('pct_fully_associative', float('nan')):>13.2f} | "
              f"{m.get('mean_associativity_fraction', float('nan')):>16.4f} | "
              f"{m.get('mean_cell_accuracy', float('nan')):>13.4f} | "
              f"{latest_by_n[n]['created'].strftime('%Y-%m-%d %H:%M:%S'):>25} | "
              f"{latest_by_n[n]['method_name']}")

    # Plot
    plt.figure(figsize=(6,4))
    plt.plot(xs, ys, marker='o')
    plt.xticks(list(N_RANGE))
    plt.ylim(0, 100)
    plt.xlabel("Cardinality n")
    plt.ylabel("% fully associative (latest per n)")
    title_bits = []
    if METHOD_FILTER: title_bits.append(METHOD_FILTER)
    title_bits.append(f"{CORRUPT_PCT_FIX}% corruption")
    plt.title(" / ".join(title_bits))
    plt.grid(True, alpha=0.3)
    plt.show()
