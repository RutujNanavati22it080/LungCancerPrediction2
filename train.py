import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from models.efficientnetv2 import build_model
from utils.preprocess import load_dataset

# Load and prepare data
X_train, X_test, y_train, y_test = load_dataset()
train_dataset = TensorDataset(torch.tensor(X_train).permute(0, 3, 1, 2), torch.tensor(y_train))
test_dataset = TensorDataset(torch.tensor(X_test).permute(0, 3, 1, 2), torch.tensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = build_model().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training
for epoch in range(5):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "model.pth")
