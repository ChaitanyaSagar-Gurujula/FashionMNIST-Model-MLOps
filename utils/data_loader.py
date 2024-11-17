import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from config.config import CONFIG

def get_fashion_mnist_loader():
    """Create and return Fashion MNIST data loader"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.FashionMNIST(
        root=CONFIG['data_dir'], 
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers']
    )
    
    return train_loader 