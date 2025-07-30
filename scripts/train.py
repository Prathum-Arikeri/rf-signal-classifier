import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils.model_utils import SimpleCNN  # Example model, you create this file!


def train():
    # Load preprocessed data
    X = np.load('data/processed/X.npy')
    y = np.load('data/processed/y.npy')

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = SimpleCNN(num_classes=len(np.unique(y)))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(loader):.4f}")

    # Save model weights
    torch.save(model.state_dict(), 'models/rf_cnn.pth')
    print("Training complete. Model saved to models/rf_cnn.pth")


if __name__ == "__main__":
    train()
