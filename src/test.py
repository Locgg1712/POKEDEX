from src.predict import predict
from src.dataset import load_dataset

# lấy classes đúng từ dataset
_, _, classes = load_dataset("data")

result = predict("test.jpg", "model.pth", classes)
print(result)