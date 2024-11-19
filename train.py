import torch
import torch.nn as nn
import torch.optim as optim
from models.light_cnn import LightFashionCNN
from utils.data_loader import get_fashion_mnist_loader
from config.config import CONFIG

def train_model():
    # Setup device
    device = torch.device(CONFIG['device'] if torch.cuda.is_available() else "cpu")
    
    # Initialize model, criterion, and optimizer
    model = LightFashionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Get data loader
    train_loader = get_fashion_mnist_loader()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=1,
        steps_per_epoch=len(train_loader),
        div_factor=10
    )
    # Training loop
    model.train()
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Calculate accuracy
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % CONFIG['print_frequency'] == 0:
            print(f'Batch: {batch_idx}, Loss: {loss.item():.4f}, '
                  f'Accuracy: {100.*correct/total:.2f}%')

    print(f'Final Training Accuracy: {100.*correct/total:.2f}%')

if __name__ == "__main__":
    train_model() 