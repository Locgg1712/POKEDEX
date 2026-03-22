from src.predict import predict
from src.dataset import load_dataset

# lấy classes đúng từ dataset
_, _, classes = load_dataset("data_processed")

result = predict("image.png", "model.pth", classes)
print(result)