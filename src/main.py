from src.dataset import load_dataset
from src.model import PokemonCNN
from src.train import train_model

data_dir = "data"

train_loader, val_loader, classes = load_dataset(data_dir)

model = PokemonCNN(num_classes=len(classes))

train_model(model, train_loader, val_loader)