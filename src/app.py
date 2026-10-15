# src/app.py

import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import cv2

from src.preprocess import extract_pokemon_debug
from src.predict import predict
from src.pokedex_data import get_pokedex_info
from src.evolution import get_family_info

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# Bảng màu hệ nguyên tố Pokémon
TYPE_COLORS = {
    "Grass": "#78C850", "Poison": "#A040A0", "Fire": "#F08030",
    "Flying": "#A890F0", "Water": "#6890F0", "Bug": "#A8B820",
    "Normal": "#A8A878", "Electric": "#F8D030", "Ground": "#E0C068",
    "Fairy": "#EE99AC", "Fighting": "#C03028", "Psychic": "#F85888",
    "Rock": "#B8A038", "Steel": "#B8B8D0", "Ice": "#98D8D8",
    "Ghost": "#705598", "Dragon": "#7038F8"
}


class PokedexApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Pokédex DSP Visual — Nhận diện 150 Pokémon")
        self.geometry("1200x860")
        self.minsize(950, 750)

        # Configure layout (1 row, 2 columns)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=2, minsize=420)  # Left panel
        self.grid_columnconfigure(1, weight=3, minsize=520)  # Right main content

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

        self.pokeball_state = "closed"
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

        # Header
        self.title_label = ctk.CTkLabel(
            self.left_panel, text="🔴 Pokédex DSP",
            font=ctk.CTkFont(family="Arial", size=26, weight="bold"),
            text_color=self.accent_red
        )
        self.title_label.pack(anchor="w", padx=20, pady=(20, 3))

        self.subtitle_label = ctk.CTkLabel(
            self.left_panel, text="Xử lý tín hiệu số & Nhận diện Pokémon",
            font=ctk.CTkFont(family="Arial", size=13), text_color="gray"
        )
        self.subtitle_label.pack(anchor="w", padx=20, pady=(0, 15))

        # Action Button
        self.action_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.action_frame.pack(fill="x", padx=20, pady=(0, 10))

        self.upload_btn = ctk.CTkButton(
            self.action_frame, text="Truy xuất dữ liệu (Chọn Ảnh)",
            fg_color=self.accent_red, hover_color="#D93232",
            font=ctk.CTkFont(family="Arial", size=14, weight="bold"),
            height=40, corner_radius=8,
            command=self.load_image
        )
        self.upload_btn.pack(fill="x")

        # Scrollable container for results
        self.result_scroll = ctk.CTkScrollableFrame(
            self.left_panel, fg_color=self.bg_color, corner_radius=15
        )
        self.result_scroll.pack(fill="both", expand=True, padx=20, pady=(5, 20))

        # Result Details Header
        self.result_title = ctk.CTkLabel(
            self.result_scroll, text="KẾT QUẢ PHÂN TÍCH",
            font=ctk.CTkFont(size=13, weight="bold"), text_color="gray"
        )
        self.result_title.pack(pady=(10, 5))

        # Main Name & ID
        self.pokemon_name_label = ctk.CTkLabel(
            self.result_scroll, text="Chưa xác định",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        self.pokemon_name_label.pack(pady=2)

        # Type badges frame
        self.types_frame = ctk.CTkFrame(self.result_scroll, fg_color="transparent")
        self.types_frame.pack(pady=3)

        # Confidence Bar
        self.confidence_bar = ctk.CTkProgressBar(
            self.result_scroll, width=280, height=12,
            progress_color="gray", fg_color="#15151A"
        )
        self.confidence_bar.set(0)
        self.confidence_bar.pack(pady=(10, 3))

        self.confidence_label = ctk.CTkLabel(
            self.result_scroll, text="Độ tin cậy: 0%",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.confidence_label.pack(pady=2)

        # Low Confidence Warning Badge
        self.warning_label = ctk.CTkLabel(
            self.result_scroll,
            text="",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#FFA726"
        )
        self.warning_label.pack(pady=2)

        # Top 3 Candidates Section
        self.top3_title = ctk.CTkLabel(
            self.result_scroll, text="TOP 3 DỰ ĐOÁN (CÂN NHẮC HỌ TIẾN HÓA)",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#A0A0B0"
        )
        self.top3_title.pack(anchor="w", padx=15, pady=(12, 4))

        self.top3_container = ctk.CTkFrame(self.result_scroll, fg_color="#25252F", corner_radius=10)
        self.top3_container.pack(fill="x", padx=10, pady=2)
        self.top3_labels = []
        self.top3_bars = []
        for i in range(3):
            row = ctk.CTkFrame(self.top3_container, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=3)
            lbl = ctk.CTkLabel(row, text=f"#{i+1} --", font=ctk.CTkFont(size=12), width=120, anchor="w")
            lbl.pack(side="left")
            pbar = ctk.CTkProgressBar(row, width=120, height=8, progress_color="#6890F0", fg_color="#15151A")
            pbar.set(0)
            pbar.pack(side="left", padx=5)
            plbl = ctk.CTkLabel(row, text="0%", font=ctk.CTkFont(size=11), width=45, anchor="e")
            plbl.pack(side="right")
            self.top3_labels.append((lbl, plbl))
            self.top3_bars.append(pbar)

        # Pokédex Info Card
        self.info_title = ctk.CTkLabel(
            self.result_scroll, text="THÔNG TIN POKÉDEX",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#A0A0B0"
        )
        self.info_title.pack(anchor="w", padx=15, pady=(12, 4))

        self.info_box = ctk.CTkTextbox(
            self.result_scroll, width=320, height=85,
            fg_color="#25252F", font=ctk.CTkFont(family="Arial", size=13),
            corner_radius=10
        )
        self.info_box.pack(fill="x", padx=10, pady=2)
        self.info_box.insert("1.0", "Thông tin loài Pokémon sẽ hiển thị tại đây khi tải ảnh.")
        self.info_box.configure(state="disabled")

        # DSP Explanation Box
        self.dsp_title = ctk.CTkLabel(
            self.result_scroll, text="GIẢI THÍCH ĐẶC TRƯNG TÍN HIỆU (DSP)",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#A0A0B0"
        )
        self.dsp_title.pack(anchor="w", padx=15, pady=(12, 4))

        self.explanation_box = ctk.CTkTextbox(
            self.result_scroll, width=320, height=130,
            fg_color="#25252F", font=ctk.CTkFont(family="Arial", size=13),
            corner_radius=10
        )
        self.explanation_box.pack(fill="x", padx=10, pady=2)
        self.explanation_box.insert(
            "1.0",
            "Nhấn Pokéball hoặc nút Chọn Ảnh để trích xuất đặc trưng HOG, Fourier, HSV, GLCM, Gabor và dự đoán."
        )
        self.explanation_box.configure(state="disabled")

        # ================= RIGHT MAIN (POKEBALL) =================
        self.right_main = ctk.CTkFrame(self, fg_color=self.bg_color, corner_radius=0)
        self.right_main.grid(row=0, column=1, sticky="nsew")

        self.pokeball_container = ctk.CTkFrame(self.right_main, width=580, height=580, fg_color="transparent")
        self.pokeball_container.place(relx=0.5, rely=0.5, anchor="center")

        # Inner Content (Pipeline Images)
        self.inner_frame = ctk.CTkFrame(self.pokeball_container, width=560, height=560, fg_color=self.panel_bg, corner_radius=15)
        self.inner_frame.place(relx=0.5, rely=0.5, anchor="center")
        self.inner_frame.grid_columnconfigure((0, 1), weight=1)
        self.inner_frame.grid_rowconfigure((0, 1), weight=1)
        self.inner_frame.pack_propagate(False)

        self.panel_original = self.create_image_panel(self.inner_frame, "1. Ảnh Gốc (Original)", 0, 0)
        self.panel_blur = self.create_image_panel(self.inner_frame, "2. Lọc Khử Nhiễu (Bilateral)", 0, 1)
        self.panel_edge = self.create_image_panel(self.inner_frame, "3. Biên Cạnh (Auto Canny)", 1, 0)
        self.panel_final = self.create_image_panel(self.inner_frame, "4. Đầu Ra DSP (Preprocessed)", 1, 1)

        # Outer Pokeball Shell
        self.top_half = ctk.CTkFrame(self.pokeball_container, width=580, height=290, fg_color=self.pokeball_red, corner_radius=30)
        self.top_half.place(relx=0.5, rely=self.top_rely, anchor="center")

        self.bot_half = ctk.CTkFrame(self.pokeball_container, width=580, height=290, fg_color=self.pokeball_white, corner_radius=30)
        self.bot_half.place(relx=0.5, rely=self.bot_rely, anchor="center")

        self.top_line = ctk.CTkFrame(self.top_half, width=580, height=12, fg_color=self.pokeball_divider)
        self.top_line.place(relx=0.5, rely=1.0, anchor="s")

        self.bot_line = ctk.CTkFrame(self.bot_half, width=580, height=12, fg_color=self.pokeball_divider)
        self.bot_line.place(relx=0.5, rely=0.0, anchor="n")

        # Center Button
        self.center_ring = ctk.CTkFrame(self.pokeball_container, width=130, height=130, corner_radius=65, fg_color=self.pokeball_divider)
        self.center_ring.place(relx=0.5, rely=0.5, anchor="center")

        self.center_btn_white = ctk.CTkFrame(self.center_ring, width=96, height=96, corner_radius=48, fg_color="#F0F0F0")
        self.center_btn_white.place(relx=0.5, rely=0.5, anchor="center")

        self.center_btn = ctk.CTkButton(
            self.center_btn_white, width=70, height=70, corner_radius=35,
            fg_color="#FFFFFF", hover_color="#DDDDDD", text="",
            border_width=2, border_color="#CCCCCC",
            command=self.load_image
        )
        self.center_btn.place(relx=0.5, rely=0.5, anchor="center")

        self.idle_label = ctk.CTkLabel(
            self.bot_half, text="Click để tải tín hiệu",
            font=ctk.CTkFont("Arial", 16, "bold"), text_color="#A0A0A0"
        )
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
            self.center_ring.place_forget()
            self.idle_label.place_forget()

        if step <= 25:
            self.top_rely = 0.25 - (0.48 * (step / 25))
            self.bot_rely = 0.75 + (0.48 * (step / 25))
            self.top_half.place(rely=self.top_rely)
            self.bot_half.place(rely=self.bot_rely)
            self.after(12, lambda: self.open_animation(step + 1))
        else:
            self.pokeball_state = "open"

    def update_type_badges(self, types):
        for widget in self.types_frame.winfo_children():
            widget.destroy()

        for t in types:
            bg_col = TYPE_COLORS.get(t, "#606060")
            badge = ctk.CTkLabel(
                self.types_frame, text=f" {t} ",
                fg_color=bg_col, text_color="#FFFFFF",
                corner_radius=6, font=ctk.CTkFont(size=11, weight="bold")
            )
            badge.pack(side="left", padx=3)

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh Pokémon",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not path:
            return

        self.set_pokeball_state("closed")
        self.update()

        steps = extract_pokemon_debug(path)
        if not steps:
            self.pokemon_name_label.configure(text="Lỗi đọc ảnh", text_color=self.accent_red)
            self.confidence_bar.set(0)
            self.confidence_label.configure(text="Độ tin cậy: 0%")
            return

        self.panel_original.configure(image=self.cv2_to_ctk(steps["original"]))
        self.panel_blur.configure(image=self.cv2_to_ctk(steps["blur"]))
        self.panel_edge.configure(image=self.cv2_to_ctk(steps["edges"]))
        self.panel_final.configure(image=self.cv2_to_ctk(steps["combine"]))

        try:
            name, conf, expl, top3 = predict(path)
            poke_info = get_pokedex_info(name)
            fam_info = get_family_info(name)

            # Display Pokémon Name
            id_prefix = f"#{poke_info['id']:03d} " if poke_info.get("id") else ""
            self.pokemon_name_label.configure(text=f"{id_prefix}{name}", text_color=self.text_color)

            # Type Badges
            self.update_type_badges(poke_info.get("types", ["Normal"]))

            # Confidence Bar
            self.confidence_bar.set(conf)
            self.confidence_label.configure(text=f"Độ tin cậy: {conf * 100:.1f}%")

            if conf >= 0.70:
                bar_color = "#4CAF50"
                self.warning_label.configure(text="")
            elif conf >= 0.40:
                bar_color = "#FFC107"
                self.warning_label.configure(text="")
            else:
                bar_color = self.accent_red
                self.warning_label.configure(text="⚠️ Độ tin cậy thấp: Hãy thử chụp lại góc thẳng hơn")

            self.confidence_bar.configure(progress_color=bar_color)

            # Top 3 Candidates
            for i, (cand_name, cand_prob) in enumerate(top3[:3]):
                lbl_name, lbl_pct = self.top3_labels[i]
                lbl_name.configure(text=f"#{i+1} {cand_name}")
                lbl_pct.configure(text=f"{cand_prob*100:.1f}%")
                self.top3_bars[i].set(cand_prob)
                if i == 0:
                    self.top3_bars[i].configure(progress_color=bar_color)
                else:
                    self.top3_bars[i].configure(progress_color="#6890F0")

            # Pokédex Info Box
            fam_name = fam_info['family_name'] if fam_info else 'Độc lập'
            stage_str = f"Giai đoạn {fam_info['stage']}/{fam_info['total_stages']}" if fam_info else "Không tiến hóa"
            info_text = (
                f"Phân loại: {poke_info.get('category', 'Pokémon')} | Dòng họ: {fam_name} ({stage_str})\n"
                f"Chiều cao: {poke_info.get('height', '---')} | Cân nặng: {poke_info.get('weight', '---')}\n\n"
                f"Mô tả: {poke_info.get('description', '')}"
            )
            self.info_box.configure(state="normal")
            self.info_box.delete("1.0", "end")
            self.info_box.insert("1.0", info_text)
            self.info_box.configure(state="disabled")

            # DSP Explanation
            self.explanation_box.configure(state="normal")
            self.explanation_box.delete("1.0", "end")
            self.explanation_box.insert("1.0", expl)
            self.explanation_box.configure(state="disabled")

        except Exception as e:
            self.pokemon_name_label.configure(text="Lỗi Model", text_color=self.accent_red)
            self.confidence_bar.set(0)
            self.confidence_label.configure(text=f"Chi tiết: {str(e)}")

        self.after(200, self.open_animation)


if __name__ == "__main__":
    app = PokedexApp()
    app.mainloop()