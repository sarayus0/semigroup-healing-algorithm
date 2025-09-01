import numpy as np
import random, re, os
import subprocess, pathlib
from typing import Optional

def ensure_mace4_built(prefix="/content"):
    url = "http://www.cs.unm.edu/~mccune/mace4/download/LADR-2009-11A.tar"
    prefix_path = pathlib.Path(prefix).expanduser().resolve()   
    tar_path = pathlib.Path(prefix) / "LADR-2009-11A.tar"
    ladr_dir = pathlib.Path(prefix) / "LADR-2009-11A"
    mace4_bin = ladr_dir / "bin" / "mace4"

    if mace4_bin.exists():
        return str(mace4_bin)

    prefix_path.mkdir(parents=True, exist_ok=True)  # 🔹 added: make sure folder exists

    # download tarball
    subprocess.run(["wget", url, "-O", str(tar_path)], check=True)

    # extract
    subprocess.run(["tar", "-xvf", str(tar_path), "-C", prefix], check=True)

    # build
    subprocess.run(["make"], cwd=str(ladr_dir), check=True)
    subprocess.run(["make", "all"], cwd=str(ladr_dir), check=True)

    return str(mace4_bin)

MACE4_PATH = ensure_mace4_built()

def random_constraints(n, num_constraints=4):
    constraints = []
    used = set()
    for _ in range(num_constraints):
        i, j = random.randint(0, n-1), random.randint(0, n-1)
        k = random.randint(0, n-1)
        if (i, j) not in used:
            constraints.append(f"{i} * {j} = {k}.")
            used.add((i, j))
    return "\n".join(constraints)

def generate_semigroup_with_mace4(n: int) -> np.ndarray:
    mace_input = "formulas(assumptions).\n"
    mace_input += "(x * y) * z = x * (y * z).\n"
    mace_input += random_constraints(n, num_constraints=4) + "\n"
    mace_input += "end_of_list.\n"

    with open("mace_input.in", "w") as f:
        f.write(mace_input)

    subprocess.run(
        [MACE4_PATH, "-n", str(n), "-t", "10", "-b", "500"],
        stdin=open("mace_input.in", "r"),
        stdout=open("mace_output.out", "w"),
        stderr=open("mace_error.err", "w"),
        check=False
    )

    output_str = open("mace_output.out").read()
    match = re.search(r"function\(\*\(_,_\), \[(.*?)\]\)", output_str, re.S)
    if not match:
        raise ValueError(f"Mace4 could not find a model for n={n}.\nOutput:\n{output_str}")

    numbers = [int(x.strip()) for x in match.group(1).split(",")]
    table = np.array(numbers).reshape(n, n)

    # cleanup
    for ext in ["in", "out", "err"]:
        if os.path.exists(f"mace_input.{ext}"):
            os.remove(f"mace_input.{ext}")
        if os.path.exists(f"mace_output.{ext}"):
            os.remove(f"mace_output.{ext}")
        if os.path.exists(f"mace_error.{ext}"):
            os.remove(f"mace_error.{ext}")

    return table

def generate_semigroup_with_mace4_retry(n, max_tries=10):
    for _ in range(max_tries):
        try:
            return generate_semigroup_with_mace4(n)
        except ValueError:
            continue
    raise ValueError(f"Mace4 failed after {max_tries} tries.")
