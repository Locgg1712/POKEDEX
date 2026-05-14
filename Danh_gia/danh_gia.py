import cv2
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ────────────────────────────────────────────────────────────────
# Đường dẫn thư mục
# ────────────────────────────────────────────────────────────────
DIR_GOC  = "D:/Danh_gia/goc"   # ảnh gốc (và ảnh đã thêm nhiễu)
DIR_DSP  = "D:/Danh_gia/dsp"   # ảnh đã khử nhiễu bằng DSP (Bilateral)
DIR_UNET = "D:/Danh_gia/unet"  # ảnh đã khử nhiễu bằng U-Net

NOISE_TYPES = ["gaussian", "saltpepper", "speckle"]
IMG_EXTS    = [".png", ".jpg", ".jpeg", ".bmp"]

# ────────────────────────────────────────────────────────────────
# Hàm hỗ trợ: tìm file trong thư mục không phân biệt extension
# ────────────────────────────────────────────────────────────────
def find_file(directory: str, stem: str) -> str | None:
    """Trả về đường dẫn đầy đủ của file có tên (không extension) là stem,
    tìm trong directory với các extension phổ biến. Trả None nếu không thấy."""
    for ext in IMG_EXTS:
        path = os.path.join(directory, stem + ext)
        if os.path.isfile(path):
            return path
    return None


def read_gray_3ch(path: str):
    """Đọc ảnh; nếu grayscale thì chuyển sang 3 kênh để tính SSIM đồng nhất."""
    img = cv2.imread(path)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def resize_to_match(ref, target):
    """Resize target về cùng kích thước với ref nếu khác nhau."""
    if ref.shape != target.shape:
        h, w = ref.shape[:2]
        target = cv2.resize(target, (w, h), interpolation=cv2.INTER_AREA)
    return target


def calc_metrics(ref, test):
    """Trả về (PSNR, SSIM) giữa ảnh tham chiếu và ảnh kiểm tra."""
    test = resize_to_match(ref, test)
    p = psnr(ref, test, data_range=255)
    s = ssim(ref, test, data_range=255, channel_axis=2)
    return p, s


# ────────────────────────────────────────────────────────────────
# Thu thập kết quả
# ────────────────────────────────────────────────────────────────
# results[noise_type] = list of dicts {file, psnr_dsp, ssim_dsp, psnr_unet, ssim_unet}
results = {n: [] for n in NOISE_TYPES}
skipped = []

# Lấy danh sách stem của ảnh nhiễu từ thư mục gốc
noisy_stems = []
for fname in os.listdir(DIR_GOC):
    stem, ext = os.path.splitext(fname)
    if ext.lower() not in IMG_EXTS:
        continue
    for noise in NOISE_TYPES:
        if stem.endswith(f"_{noise}"):
            noisy_stems.append((stem, noise))
            break

noisy_stems.sort(key=lambda x: (x[1], x[0]))  # sắp xếp theo loại nhiễu rồi tên

for stem, noise in noisy_stems:
    # Đường dẫn ảnh nhiễu (gốc chưa khử)
    path_goc = find_file(DIR_GOC, stem)
    # Đường dẫn ảnh sau khử nhiễu DSP và U-Net
    path_dsp  = find_file(DIR_DSP,  stem)
    path_unet = find_file(DIR_UNET, stem)

    if path_goc is None:
        skipped.append(f"{stem} (không tìm thấy trong goc/)")
        continue
    if path_dsp is None:
        skipped.append(f"{stem} (không tìm thấy trong dsp/)")
        continue
    if path_unet is None:
        skipped.append(f"{stem} (không tìm thấy trong unet/)")
        continue

    img_goc  = read_gray_3ch(path_goc)
    img_dsp  = read_gray_3ch(path_dsp)
    img_unet = read_gray_3ch(path_unet)

    if img_goc is None or img_dsp is None or img_unet is None:
        skipped.append(f"{stem} (lỗi đọc ảnh)")
        continue

    p_dsp,  s_dsp  = calc_metrics(img_goc, img_dsp)
    p_unet, s_unet = calc_metrics(img_goc, img_unet)

    results[noise].append({
        "file":      stem,
        "psnr_dsp":  p_dsp,
        "ssim_dsp":  s_dsp,
        "psnr_unet": p_unet,
        "ssim_unet": s_unet,
    })

# ────────────────────────────────────────────────────────────────
# In kết quả
# ────────────────────────────────────────────────────────────────
COL = 40   # chieu rong cot ten file
HDR = f"{'File':<{COL}} {'PSNR_DSP':>10} {'SSIM_DSP':>10} {'PSNR_UNet':>10} {'SSIM_UNet':>10}"
SEP = "-" * len(HDR)

all_psnr_dsp, all_ssim_dsp   = [], []
all_psnr_unet, all_ssim_unet = [], []

for noise in NOISE_TYPES:
    rows = results[noise]
    if not rows:
        continue

    label_map = {"gaussian": "Gaussian", "saltpepper": "Salt & Pepper", "speckle": "Speckle"}
    print(f"\n{'='*len(HDR)}")
    print(f"  Loai nhieu: {label_map[noise]}  ({len(rows)} anh)")
    print(f"{'='*len(HDR)}")
    print(HDR)
    print(SEP)

    g_psnr_dsp, g_ssim_dsp   = [], []
    g_psnr_unet, g_ssim_unet = [], []

    for r in rows:
        name = r["file"]
        if len(name) > COL - 1:
            name = "..." + name[-(COL - 3):]
        print(f"{name:<{COL}} {r['psnr_dsp']:>10.2f} {r['ssim_dsp']:>10.4f}"
              f" {r['psnr_unet']:>10.2f} {r['ssim_unet']:>10.4f}")
        g_psnr_dsp.append(r["psnr_dsp"]);  g_ssim_dsp.append(r["ssim_dsp"])
        g_psnr_unet.append(r["psnr_unet"]); g_ssim_unet.append(r["ssim_unet"])

    print(SEP)
    print(f"{'Trung bình':<{COL}}"
          f" {np.mean(g_psnr_dsp):>10.2f} {np.mean(g_ssim_dsp):>10.4f}"
          f" {np.mean(g_psnr_unet):>10.2f} {np.mean(g_ssim_unet):>10.4f}")

    all_psnr_dsp  += g_psnr_dsp;  all_ssim_dsp  += g_ssim_dsp
    all_psnr_unet += g_psnr_unet; all_ssim_unet += g_ssim_unet

# ── Tong hop toan bo ──────────────────────────────────────────
if all_psnr_dsp:
    print(f"\n{'='*len(HDR)}")
    print(f"  TONG HOP  ({len(all_psnr_dsp)} anh)")
    print(f"{'='*len(HDR)}")
    print(HDR)
    print(SEP)
    print(f"{'[DSP  - Bilateral]':<{COL}}"
          f" {np.mean(all_psnr_dsp):>10.2f} {np.mean(all_ssim_dsp):>10.4f}"
          f" {'---':>10} {'---':>10}")
    print(f"{'[UNet - Deep Learning]':<{COL}}"
          f" {'---':>10} {'---':>10}"
          f" {np.mean(all_psnr_unet):>10.2f} {np.mean(all_ssim_unet):>10.4f}")
    print(SEP)
    winner_psnr = "UNet" if np.mean(all_psnr_unet) > np.mean(all_psnr_dsp) else "DSP"
    winner_ssim = "UNet" if np.mean(all_ssim_unet) > np.mean(all_ssim_dsp) else "DSP"
    print(f"\n  >> PSNR tot hon: {winner_psnr}")
    print(f"  >> SSIM tot hon: {winner_ssim}")

if skipped:
    print(f"\n[!] Bo qua {len(skipped)} file:")
    for s in skipped:
        print(f"   - {s}")