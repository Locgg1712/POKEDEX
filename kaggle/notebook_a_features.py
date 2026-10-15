# %% [markdown]
# Notebook A - Trích xuất toàn bộ đặc trưng DSP & Machine Learning trên CPU.
# Chạy notebook này sau khi đưa mã nguồn lên Kaggle.

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
from src.evolution import get_pokemon_family

DATA_DIR = os.environ.get("POKEDEX_DATA", str(ROOT / "data"))
OUT_PATH = os.environ.get("POKEDEX_FEATURES", str(ROOT / "features.npz"))

print(f"IMAGE_SIZE: {IMAGE_SIZE}")
print(f"FEATURE_DIM: {FEATURE_DIM}")
print("Feature Groups:")
for k, v in FEATURE_GROUPS.items():
    print(f"  {k:15s}: [{v[0]:5d}, {v[1]:5d}) dim={v[1]-v[0]}")

# %%
X, y, labels, groups = load_dataset(DATA_DIR, return_groups=True)
assert X.ndim == 2 and X.shape[1] == FEATURE_DIM, f"X.shape[1]={X.shape[1]} != FEATURE_DIM={FEATURE_DIM}"
assert len(X) == len(y) == len(groups)
print(f"X={X.shape}, classes={len(labels)}, unique groups={len(np.unique(groups))}")

# %%
# Tạo family labels cho bài toán Hierarchical / Evolution Classification
families = np.asarray([get_pokemon_family(labels[idx]) for idx in y], dtype=str)
unique_families = np.unique(families)
print(f"Số lượng họ tiến hóa (Families): {len(unique_families)}")

# %%
np.savez_compressed(
    OUT_PATH,
    X=X.astype(np.float32),
    y=y.astype(np.int64),
    groups=groups.astype(str),
    families=families,
    names=np.asarray([labels[i] for i in range(len(labels))], dtype=str),
    group_columns=np.asarray(json.dumps(FEATURE_GROUPS), dtype=str),
    image_size=np.asarray(IMAGE_SIZE, dtype=np.int64),
)
print(f"Đã lưu thành công tập đặc trưng tại: {OUT_PATH}")
