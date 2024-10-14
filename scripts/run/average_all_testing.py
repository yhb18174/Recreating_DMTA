# %%
import sys
from pathlib import Path
import pandas as pd
PROJ_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, PROJ_DIR / "scripts/run/")
from average_all import AverageAll

sys.path.insert(0, PROJ_DIR / "scripts/misc/")
from misc_functions import (
    get_sel_mols_between_iters,
    molid_to_smiles,
    molid_ls_to_smiles,
)

avg = AverageAll()

# %%
avg._average_experiment(exp_suffix='10_mp', n_iters=150)
# %%
