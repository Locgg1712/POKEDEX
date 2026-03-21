import torch
from PIL import Image
from torchvision import transforms
from model import PokemonCNN

def predict(image_path, model_path, classes):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    img = Image.open(image_path)
    img = transform(img).unsqueeze(0)

    model = PokemonCNN(len(classes))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

    return classes[predicted.item()]