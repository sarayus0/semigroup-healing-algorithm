"""

semigroup_healing
=================
A toolkit for generating, corrupting, and healing semigroup Cayley tables.
"""

# dataset + Mace4 generation
from .data import mace4_randomflip_dataset, one_hot_table
from .mace4 import generate_semigroup_with_mace4_retry

# trust + associativity checks
from .trust import (
    is_associative,
    compute_trust_map,
    restore_missing_from_corrupt,
    fill_remaining_neg1_by_row_mode,
    make_reference_table,
)

# healing algorithms
from .heal import (
    enforce_associativity,
    heal_via_triple_closures,
    generate_closed_groups,
    heal_small_groups_with_ml_fast,
    merge_votes_weighted,
)

# results cache
from .cache import ResultsCache, derive_dataset_key

__all__ = [
    # data
    "mace4_randomflip_dataset",
    "one_hot_table",
    "generate_semigroup_with_mace4_retry",
    # trust
    "is_associative",
    "compute_trust_map",
    "restore_missing_from_corrupt",
    "fill_remaining_neg1_by_row_mode",
    "make_reference_table",
    # heal
    "enforce_associativity",
    "heal_via_triple_closures",
    "generate_closed_groups",
    "heal_small_groups_with_ml_fast",
    "merge_votes_weighted",
    # cache
    "ResultsCache",
    "derive_dataset_key",
]
