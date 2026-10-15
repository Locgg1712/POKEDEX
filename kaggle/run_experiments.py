# kaggle/run_experiments.py
"""
Script tự động hóa quy trình chạy thực nghiệm và ghi log khoa học:
1. Tải đặc trưng đã trích xuất từ features.npz
2. Chạy StratifiedGroupKFold 5-fold (zero leakage)
3. Chạy từng cấu hình bóc tách (Ablation Study)
4. Đánh giá Top-1, Top-3, Top-5, Macro-F1
5. Xuất kết quả ra results/experiments.csv và results/experiments.json
"""

import os
import sys
import json
import csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import FEATURE_GROUPS, FEATURE_DIM


def main():
    features_path = ROOT / "features.npz"
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not features_path.exists():
        print(f"Lỗi: Không tìm thấy file đặc trưng '{features_path}'.")
        print("Vui lòng chạy notebook_a_features.py trước để tạo features.npz.")
        return

    print("=" * 70)
    print("🚀 BẮT ĐẦU CHẠY BỘ THỬ NGHIỆM ĐÁNH GIÁ MÔ HÌNH POKÉDEX DSP")
    print("=" * 70)

    # Chạy trực tiếp module notebook_b_train
    import kaggle.notebook_b_train

    print("\n✅ Hoàn thành tất cả thử nghiệm!")
    print(f"Kết quả chi tiết được lưu tại: {results_dir}")


if __name__ == "__main__":
    main()
