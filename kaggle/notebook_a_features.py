# %% [markdown]
# Notebook A - extract classical DSP/ML features on CPU.
# Run this after cloning the repository in Kaggle.

# %%
import json
import os
import sys
from pathlib import Path

import numpy as np
np.random.seed(42)

ROOT = Path(os.environ.get("POKEDEX_ROOT", Path.cwd()))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import load_dataset
from src.features import FEATURE_DIM, FEATURE_GROUPS
from src.preprocess import IMAGE_SIZE

DATA_DIR = os.environ.get("POKEDEX_DATA", str(ROOT / "data"))
OUT_PATH = os.environ.get("POKEDEX_FEATURES", str(ROOT / "features.npz"))

# %%
X, y, labels, groups = load_dataset(DATA_DIR, return_groups=True)
assert X.ndim == 2 and X.shape[1] == FEATURE_DIM == 3157
assert len(X) == len(y) == len(groups)
print(f"X={X.shape}, classes={len(labels)}, groups={len(np.unique(groups))}")

# %%
np.savez_compressed(
    OUT_PATH,
    X=X.astype(np.float32),
    y=y.astype(np.int64),
    groups=groups.astype(str),
    names=np.asarray([labels[i] for i in range(len(labels))], dtype=str),
    group_columns=np.asarray(json.dumps(FEATURE_GROUPS), dtype=str),
    image_size=np.asarray(IMAGE_SIZE, dtype=np.int64),
)
print(f"Saved {OUT_PATH}")
