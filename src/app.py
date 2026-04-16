# src/app.py

import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import cv2

from src.preprocess import extract_pokemon_debug
from src.predict import predict

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

class PokedexApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Pokédex DSP Visual")
        self.geometry("1100x800")
        self.minsize(900, 700)
        
        # Configure layout (1 row, 2 columns)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=2, minsize=350) # Left panel
        self.grid_columnconfigure(1, weight=3, minsize=450) # Right main content

        # Colors
        self.bg_color = "#1E1E24"
        self.panel_bg = "#2B2B36"
        self.pokeball_red = "#e74c3c"
        self.pokeball_white = "#ffffff"
        self.pokeball_divider = "#000000"
        self.accent_red = "#FF3E3E"
        self.text_color = "#FFFFFF"

        self.configure(fg_color=self.bg_color)
        
        self.placeholder_image = self.create_placeholderImage()

        self.pokeball_state = "closed" # closed, opening, open, closing
        self.top_rely = 0.25
        self.bot_rely = 0.75

        self.setup_ui()

    def create_placeholderImage(self):
        img = Image.new("RGB", (220, 220), self.panel_bg)
        return ctk.CTkImage(img, size=(220, 220))

    def setup_ui(self):
        # ================= LEFT PANEL =================
        self.left_panel = ctk.CTkFrame(self, fg_color=self.panel_bg, corner_radius=0)
        self.left_panel.grid(row=0, column=0, sticky="nsew")
        self.left_panel.grid_rowconfigure(2, weight=1)

        # Header
        self.title_label = ctk.CTkLabel(self.left_panel, text="🔴 Pokédex DSP", 
                                        font=ctk.CTkFont(family="Arial", size=26, weight="bold"),
                                        text_color=self.accent_red)
        self.title_label.pack(anchor="w", padx=20, pady=(30, 5))
        
        self.subtitle_label = ctk.CTkLabel(self.left_panel, text="Mô phỏng Xử lý tín hiệu số",
                                           font=ctk.CTkFont(family="Arial", size=14), text_color="gray")
        self.subtitle_label.pack(anchor="w", padx=20, pady=(0, 20))

        # Action Layer
        self.action_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.action_frame.pack(fill="x", padx=20, pady=10)

        self.upload_btn = ctk.CTkButton(self.action_frame, text="Truy xuất dữ liệu (Chọn Ảnh)", 
                                        fg_color=self.accent_red, hover_color="#D93232",
                                        font=ctk.CTkFont(family="Arial", size=14, weight="bold"),
                                        height=40, corner_radius=8,
                                        command=self.load_image)
        self.upload_btn.pack(fill="x")

        # Result Details
        self.result_container = ctk.CTkFrame(self.left_panel, fg_color=self.bg_color, corner_radius=15)
        self.result_container.pack(fill="both", expand=True, padx=20, pady=(20, 30))
        self.result_container.pack_propagate(False)

        self.result_title = ctk.CTkLabel(self.result_container, text="KẾT QUẢ PHÂN TÍCH", font=ctk.CTkFont(size=14, weight="bold"), text_color="gray")
        self.result_title.pack(pady=(20, 10))

        self.pokemon_name_label = ctk.CTkLabel(self.result_container, text="Chưa xác định", font=ctk.CTkFont(size=32, weight="bold"))
        self.pokemon_name_label.pack(pady=5)

        self.confidence_bar = ctk.CTkProgressBar(self.result_container, width=240, height=12, progress_color="gray", fg_color="#1E1E24")
        self.confidence_bar.set(0)
        self.confidence_bar.pack(pady=(15, 5))

        self.confidence_label = ctk.CTkLabel(self.result_container, text="Độ tin cậy: 0%", font=ctk.CTkFont(size=13))
        self.confidence_label.pack(pady=5)

        self.explanation_box = ctk.CTkTextbox(self.result_container, width=260, height=150, fg_color=self.bg_color, font=ctk.CTkFont(family="Arial", size=15))
        self.explanation_box.pack(pady=(15, 10), padx=10, fill="both", expand=True)
        self.explanation_box.insert("1.0", "Hãy click vào Pokéball hoặc nhấn Chọn Ảnh để phân tích đặc trưng tín hiệu và dự đoán.")
        self.explanation_box.configure(state="disabled")

        # ================= RIGHT MAIN (POKEBALL) =================
        self.right_main = ctk.CTkFrame(self, fg_color=self.bg_color, corner_radius=0)
        self.right_main.grid(row=0, column=1, sticky="nsew")

        # Container fixing aspect ratio/size for Pokeball
        self.pokeball_container = ctk.CTkFrame(self.right_main, width=580, height=580, fg_color="transparent")
        self.pokeball_container.place(relx=0.5, rely=0.5, anchor="center")

        # Inner Content (Pipeline Images) - this stays in background and is revealed
        self.inner_frame = ctk.CTkFrame(self.pokeball_container, width=560, height=560, fg_color=self.panel_bg, corner_radius=15)
        self.inner_frame.place(relx=0.5, rely=0.5, anchor="center")
        self.inner_frame.grid_columnconfigure((0, 1), weight=1)
        self.inner_frame.grid_rowconfigure((0, 1), weight=1)
        self.inner_frame.pack_propagate(False)

        self.panel_original = self.create_image_panel(self.inner_frame, "Original", 0, 0)
        self.panel_blur = self.create_image_panel(self.inner_frame, "Bilateral Filter", 0, 1)
        self.panel_edge = self.create_image_panel(self.inner_frame, "Canny Edges", 1, 0)
        self.panel_final = self.create_image_panel(self.inner_frame, "Final Output", 1, 1)

        # OUTER POKEBALL SHELL
        self.top_half = ctk.CTkFrame(self.pokeball_container, width=580, height=290, fg_color=self.pokeball_red, corner_radius=30)
        self.top_half.place(relx=0.5, rely=self.top_rely, anchor="center")

        self.bot_half = ctk.CTkFrame(self.pokeball_container, width=580, height=290, fg_color=self.pokeball_white, corner_radius=30)
        self.bot_half.place(relx=0.5, rely=self.bot_rely, anchor="center")

        # Middle black lines attached to top and bot halves so they move along!
        self.top_line = ctk.CTkFrame(self.top_half, width=580, height=12, fg_color=self.pokeball_divider)
        self.top_line.place(relx=0.5, rely=1.0, anchor="s")

        self.bot_line = ctk.CTkFrame(self.bot_half, width=580, height=12, fg_color=self.pokeball_divider)
        self.bot_line.place(relx=0.5, rely=0.0, anchor="n")

        # Center Button 
        self.center_ring = ctk.CTkFrame(self.pokeball_container, width=130, height=130, corner_radius=65, fg_color=self.pokeball_divider)
        self.center_ring.place(relx=0.5, rely=0.5, anchor="center")

        self.center_btn_white = ctk.CTkFrame(self.center_ring, width=96, height=96, corner_radius=48, fg_color="#F0F0F0")
        self.center_btn_white.place(relx=0.5, rely=0.5, anchor="center")

        self.center_btn = ctk.CTkButton(self.center_btn_white, width=70, height=70, corner_radius=35, 
                                        fg_color="#FFFFFF", hover_color="#DDDDDD", text="",
                                        border_width=2, border_color="#CCCCCC",
                                        command=self.load_image)
        self.center_btn.place(relx=0.5, rely=0.5, anchor="center")

        # Info text when idle
        self.idle_label = ctk.CTkLabel(self.bot_half, text="Click để tải tín hiệu", font=ctk.CTkFont("Arial", 16, "bold"), text_color="#A0A0A0")
        self.idle_label.place(relx=0.5, rely=0.6, anchor="center")

    def create_image_panel(self, parent, title, row, col):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
        
        lbl_title = ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=12, weight="bold"), text_color="gray")
        lbl_title.pack(pady=(0, 2))

        lbl_img = ctk.CTkLabel(frame, text="", image=self.placeholder_image)
        lbl_img.pack(expand=True)
        return lbl_img

    def cv2_to_ctk(self, img, size=(220, 220)):
        if img is None:
            return self.placeholder_image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        return ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=size)

    def set_pokeball_state(self, state):
        if state == "closed":
            self.top_half.place(rely=0.25)
            self.bot_half.place(rely=0.75)
            self.center_ring.place(relx=0.5, rely=0.5, anchor="center")
            self.idle_label.place(relx=0.5, rely=0.6, anchor="center")
            self.pokeball_state = "closed"
            self.top_rely = 0.25
            self.bot_rely = 0.75

    def open_animation(self, step=0):
        if step == 0:
            self.pokeball_state = "opening"
            self.center_ring.place_forget() # hide center button
            self.idle_label.place_forget()

        if step <= 25:
            # Animate from 0.25 to -0.23 (moves much further out)
            self.top_rely = 0.25 - (0.48 * (step/25))
            self.bot_rely = 0.75 + (0.48 * (step/25))
            self.top_half.place(rely=self.top_rely)
            self.bot_half.place(rely=self.bot_rely)
            self.after(12, lambda: self.open_animation(step+1))
        else:
            self.pokeball_state = "open"

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh Pokémon",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not path:
            return

        # Snap closed to hide image swapping
        self.set_pokeball_state("closed")
        self.update()

        steps = extract_pokemon_debug(path)
        if not steps:
            self.pokemon_name_label.configure(text="Lỗi tải ảnh", text_color=self.accent_red)
            self.confidence_bar.set(0)
            self.confidence_label.configure(text="Độ tin cậy: 0%")
            return

        # Prepare images behind closed doors
        img_org = self.cv2_to_ctk(steps["original"])
        self.panel_original.configure(image=img_org)

        img_blur = self.cv2_to_ctk(steps["blur"])
        self.panel_blur.configure(image=img_blur)

        img_edge = self.cv2_to_ctk(steps["edges"])
        self.panel_edge.configure(image=img_edge)

        img_final = self.cv2_to_ctk(steps["combine"])
        self.panel_final.configure(image=img_final)

        # Pipeline Processing logic
        try:
            name, conf, expl = predict(path)
            
            self.pokemon_name_label.configure(text=name.capitalize(), text_color=self.text_color)
            self.confidence_bar.set(conf)
            self.confidence_label.configure(text=f"Độ tin cậy: {conf*100:.2f}%")
            
            if conf > 0.8: color = "#4CAF50"
            elif conf > 0.5: color = "#FFC107"
            else: color = self.accent_red
                
            self.confidence_bar.configure(progress_color=color)

            self.explanation_box.configure(state="normal")
            self.explanation_box.delete("1.0", "end")
            self.explanation_box.insert("1.0", expl)
            self.explanation_box.configure(state="disabled")

        except Exception as e:
            self.pokemon_name_label.configure(text="Lỗi Model", text_color=self.accent_red)
            self.confidence_bar.set(0)
            self.confidence_label.configure(text=f"Chi tiết: {str(e)}")
            self.explanation_box.configure(state="normal")
            self.explanation_box.delete("1.0", "end")
            self.explanation_box.configure(state="disabled")

        # Open Pokéball!
        self.after(200, self.open_animation)

if __name__ == "__main__":
    app = PokedexApp()
    app.mainloop()