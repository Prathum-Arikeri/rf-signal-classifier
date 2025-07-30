import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils.model_utils import SimpleCNN
from sklearn.metrics import accuracy_score, classification_report


def evaluate():
    X = np.load('data/processed/X.npy')
    y = np.load('data/processed/y.npy')

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    model = SimpleCNN(num_classes=len(np.unique(y)))
    model.load_state_dict(torch.load('models/rf_cnn.pth'))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))


if __name__ == "__main__":
    evaluate()
