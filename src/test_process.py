from src.preprocess import extract_pokemon
import cv2

img = extract_pokemon("image.png")

# hiển thị ảnh
cv2.imshow("DSP Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()