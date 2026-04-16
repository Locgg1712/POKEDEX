# src/explain.py

import os
import joblib
import numpy as np
from src.features import extract_feature_components
from src.preprocess import extract_pokemon

POKEMON_TRAITS = {
    "bulbasaur": "màu xanh lá chủ đạo, hình dạng hơi tù với củ tỏi trên lưng",
    "charmander": "màu cam nổi bật, dáng đứng có đuôi dài",
    "eevee": "màu nâu nhạt, dáng nhỏ nhắn với phần lông cổ xù",
    "jigglypuff": "màu hồng nhạt, hình dáng tròn trịa như quả bóng",
    "magikarp": "màu đỏ cam, hình dáng thon dài của loài cá",
    "meowth": "màu kem sáng, dáng đứng giống mèo với đồng tiền trên trán",
    "pikachu": "màu vàng tươi, dáng nhỏ nhắn với đôi tai nhọn",
    "psyduck": "màu vàng nhạt, dáng đứng bệ vệ với chiếc mỏ vịt",
    "snorlax": "màu xanh đen và bụng trắng kem, thân hình to béo đồ sộ",
    "squirtle": "màu xanh dương, hình dáng tròn trịa mang mai rùa"
}

def compute_centroids(data_dir="data", model_dir="Model"):
    print("Computing feature centroids...")
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    centroids = {}
    
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        hogs, hists, fds = [], [], []
        
        for file in os.listdir(class_path):
            path = os.path.join(class_path, file)
            img = extract_pokemon(path)
            if img is not None:
                hog_feat, hist, fd_feat = extract_feature_components(img)
                hogs.append(hog_feat)
                hists.append(hist)
                fds.append(fd_feat)
                
        if hogs:
            centroids[class_name] = {
                "hog": np.array(hogs).mean(axis=0),
                "hist": np.array(hists).mean(axis=0),
                "fd": np.array(fds).mean(axis=0)
            }
        print(f"  {class_name}: processed {len(hogs)} original images.")

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(centroids, os.path.join(model_dir, "centroids.pkl"))
    print("Saved centroids to Model/centroids.pkl")

def cosine_similarity(v1, v2):
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    return np.dot(v1, v2) / norm

def analyze_features(img_components, predicted_class, centroids):
    hog_feat, hist, fd_feat = img_components
    
    if predicted_class not in centroids:
        return 0.0, 0.0, 0.0
        
    c = centroids[predicted_class]
    
    sim_texture = cosine_similarity(hog_feat, c["hog"])
    sim_color = cosine_similarity(hist, c["hist"])
    sim_shape = cosine_similarity(fd_feat, c["fd"])
    
    return max(0.0, min(1.0, float(sim_color))), max(0.0, min(1.0, float(sim_shape))), max(0.0, min(1.0, float(sim_texture)))

def generate_explanation(predicted_class, sim_color, sim_shape, sim_texture):
    traits = POKEMON_TRAITS.get(predicted_class, "đặc trưng chưa xác định")
    parts = traits.split(", ")
    color_desc = parts[0] if len(parts) > 0 else "màu sắc đặc trưng"
    shape_desc = parts[1] if len(parts) > 1 else "hình dáng đặc trưng"
    
    c = int(sim_color * 100)
    s = int(sim_shape * 100)
    t = int(sim_texture * 100)
    
    explanation = (
        f"LÝ DO NHẬN DIỆN ({predicted_class.capitalize()}):\n\n"
        f"• Màu sắc chủ đạo (giống {c}%):\n  Phù hợp với {color_desc}.\n\n"
        f"• Hình dạng tổng thể (giống {s}%):\n  Đặc trưng bởi {shape_desc}.\n\n"
        f"• Chi tiết cấu trúc cạnh (giống {t}%):\n  Các đường viền, góc cạnh (tai, mắt...) khớp với nguyên mẫu.\n"
    )
    return explanation

if __name__ == "__main__":
    compute_centroids()
