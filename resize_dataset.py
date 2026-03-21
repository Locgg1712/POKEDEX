import os
import cv2

input_path = "data"
output_path = "data_processed"

os.makedirs(output_path, exist_ok=True)

for label in os.listdir(input_path):
    label_path = os.path.join(input_path, label)
    save_label_path = os.path.join(output_path, label)
    
    os.makedirs(save_label_path, exist_ok=True)
    
    for file in os.listdir(label_path):
        img_path = os.path.join(label_path, file)
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Resize
        img = cv2.resize(img, (128, 128))
        
        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Edge
        edges = cv2.Canny(gray, 100, 200)
        
        save_path = os.path.join(save_label_path, file)
        cv2.imwrite(save_path, edges)

print("✅ Done processing dataset")