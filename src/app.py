# src/app.py

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2

from src.preprocess import extract_pokemon_debug
from src.predict import predict


# ===== convert cv2 -> tkinter =====
def cv2_to_tk(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((200, 200))
    return ImageTk.PhotoImage(img)


# ===== load image + xử lý =====
def load_image():
    path = filedialog.askopenfilename()

    if not path:
        return

    steps = extract_pokemon_debug(path)

    # ===== ORIGINAL =====
    img1 = cv2_to_tk(steps["original"])
    panel_original.config(image=img1)
    panel_original.image = img1  #  giữ reference

    # ===== BLUR =====
    img2 = cv2_to_tk(steps["blur"])
    panel_blur.config(image=img2)
    panel_blur.image = img2

    # ===== EDGE =====
    img3 = cv2_to_tk(steps["edges"])
    panel_edge.config(image=img3)
    panel_edge.image = img3

    # ===== FINAL =====
    img4 = cv2_to_tk(steps["combine"])
    panel_final.config(image=img4)
    panel_final.image = img4

    # ===== PREDICT =====
    name, conf = predict(path)

    result_label.config(text=f"{name} ({conf:.2f})")


# ===== GUI =====
root = tk.Tk()
root.title("Pokédex DSP Visual")

# nút chọn ảnh
btn = tk.Button(root, text="Chọn ảnh", command=load_image)
btn.pack(pady=10)

# frame chứa ảnh
frame = tk.Frame(root)
frame.pack()

# ===== LABEL TEXT =====
tk.Label(frame, text="Original").grid(row=0, column=0)
tk.Label(frame, text="Blur").grid(row=0, column=1)
tk.Label(frame, text="Edge").grid(row=2, column=0)
tk.Label(frame, text="Final").grid(row=2, column=1)

# ===== IMAGE PANELS =====
panel_original = tk.Label(frame)
panel_original.grid(row=1, column=0, padx=10, pady=10)

panel_blur = tk.Label(frame)
panel_blur.grid(row=1, column=1, padx=10, pady=10)

panel_edge = tk.Label(frame)
panel_edge.grid(row=3, column=0, padx=10, pady=10)

panel_final = tk.Label(frame)
panel_final.grid(row=3, column=1, padx=10, pady=10)

# ===== RESULT =====
result_label = tk.Label(root, text="Result", font=("Arial", 16))
result_label.pack(pady=10)

root.mainloop()