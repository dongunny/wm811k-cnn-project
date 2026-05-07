import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
import importlib

dataset_module = importlib.import_module("01_dataset")
get_dataloaders = dataset_module.get_dataloaders

model_module = importlib.import_module("02_cnn_model")
SimpleCNN = model_module.SimpleCNN

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PATIENCE = 7
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "best_cnn_model.pth")

def train_model():
    print("=" * 60)
    print("  WM-811K CNN Training")
    print("=" * 60)
    train_loader, val_loader, test_loader, classes = get_dataloaders(batch_size=BATCH_SIZE)
    num_classes = len(classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    model = SimpleCNN(num_classes=num_classes).to(device)
    y_train = train_loader.dataset.y
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float("inf")
    patience_counter = 0
    print(f"\n[Training] Total {EPOCHS} Epochs")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = train_correct = train_total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}")
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> Improved! Saved: {MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n[Early Stopping] No improvement for {PATIENCE} epochs.")
                break
    print(f"\nTraining complete. Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train_model()
