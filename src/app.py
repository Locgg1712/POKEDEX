import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from src.predict import predict

def load_image():
    path = filedialog.askopenfilename()

    img = Image.open(path)
    img = img.resize((250,250))
    img_tk = ImageTk.PhotoImage(img)

    panel.config(image=img_tk)
    panel.image = img_tk

    results = predict(path)

    text = ""
    for name, p in results:
        text += f"{name}: {p:.2f}\n"

    result_label.config(text=text)

root = tk.Tk()
root.title("Pokédex DSP")

panel = tk.Label(root)
panel.pack()

btn = tk.Button(root, text="Chọn ảnh", command=load_image)
btn.pack()

result_label = tk.Label(root, font=("Arial", 14))
result_label.pack()

root.mainloop()