import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset
from utils.preprocess import load_dataset
from models.efficientnetv2 import build_model

X_train, X_test, y_train, y_test = load_dataset()
test_dataset = TensorDataset(torch.tensor(X_test).permute(0, 3, 1, 2), torch.tensor(y_test))
test_loader = DataLoader(test_dataset, batch_size=32)

model = build_model()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for X, y in test_loader:
        outputs = model(X)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(y.numpy())

print(classification_report(all_labels, all_preds))
