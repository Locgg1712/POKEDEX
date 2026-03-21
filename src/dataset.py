import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_dataset(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=transform
    )

    val_data = datasets.ImageFolder(
        root=os.path.join(data_dir, "val"),
        transform=transform
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    return train_loader, val_loader, train_data.classes