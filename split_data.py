import os
import shutil
import random

source_dir = "data_processed"
output_dir = "data_split"

train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")

split_ratio = 0.8

# 🔥 XÓA folder cũ để tránh lỗi
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

os.makedirs(train_dir)
os.makedirs(val_dir)

print("🚀 Start splitting dataset...\n")

for cls in os.listdir(source_dir):
    cls_path = os.path.join(source_dir, cls)

    if not os.path.isdir(cls_path):
        continue

    images = [
        f for f in os.listdir(cls_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    if len(images) == 0:
        print(f"⚠️ Bỏ qua {cls} (không có ảnh)")
        continue

    random.shuffle(images)

    split_idx = int(len(images) * split_ratio)

    # 🔥 đảm bảo không bị 0 ảnh
    if split_idx == 0:
        split_idx = 1

    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    # tạo folder
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

    # copy train
    for img in train_imgs:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(train_dir, cls, img)
        )

    # copy val
    for img in val_imgs:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(val_dir, cls, img)
        )

    print(f"✅ {cls}: {len(train_imgs)} train | {len(val_imgs)} val")

print("\n🎉 DONE! Dataset ready.")