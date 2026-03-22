import torch
from PIL import Image
from torchvision import transforms
from src.model import PokemonCNN
from src.dsp import preprocess_image  # 👈 import DSP

def predict(image_path, model_path, classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ⚙️ Transform giống y chang dataset
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Lambda(preprocess_image),  # 👈 DSP
        transforms.ToTensor()
    ])

    # load ảnh
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    # load model
    model = PokemonCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # predict
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

    return classes[predicted.item()]