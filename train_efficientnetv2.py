# train_efficientnetv2.py

import torch
import torch.nn as nn
import torch.optim as optim
from utils.preprocess import load_dataset
from models.efficientnetv2 import build_model
import os

def train(model, train_loader, val_loader, num_epochs=2, lr=0.001, save_path="weights/efficientnetv2.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    try:
        print("Training script has started...")
        train_loader, val_loader, class_names = load_dataset()
        print(f"Classes: {class_names}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")

        model = build_model(num_classes=len(class_names))
        train(model, train_loader, val_loader)
    except Exception as e:
        print("❌ Error during training:", e)