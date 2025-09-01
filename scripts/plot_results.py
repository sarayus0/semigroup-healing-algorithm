import numpy as np
from semigroup_healing.metrics import is_fully_associative, associativity_fraction
from scripts.train_and_eval import clean_test, corrupt_test, healed_test
from semigroup_healing.metrics import is_associative
from semigroup_healing.metrics import percent_fully_associative

def print_all_tables(clean_tables, corrupt_tables, healed_tables, limit=5):
    num_tables = len(clean_tables)
    n = clean_tables[0].shape[0]

    for t in range(min(num_tables, limit)):
        print(f"\n🧪 Table {t+1}/{num_tables} — size {n}x{n}")
        print("=" * (n * 10))

        print("CLEAN TABLE".center(n * 10))
        for row in clean_tables[t]:
            print(" ".join(f"{val:2d}" for val in row))
        print()

        print("CORRUPTED TABLE".center(n * 10))
        for row in corrupt_tables[t]:
            print(" ".join(f"{val:2d}" for val in row))
        print()

        print("HEALED TABLE".center(n * 10))
        for row in healed_tables[t]:
            print(" ".join(f"{val:2d}" for val in row))
        print()

        print("Associative?")
        print(is_fully_associative(healed_tables[t]))

        print("Percent Associativity")
        print(associativity_fraction(healed_tables[t]))

        print("=" * (n * 10))

print("First Healing Results:\n")
print("Each block shows: CLEAN --> CORRUPTED --> HEALED, with associativity check.\n")
print_all_tables(clean_test, corrupt_test, healed_test, limit=10)  # adjust limit as needed
percent_fully_associative(healed_test)