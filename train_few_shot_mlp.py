"""
USE FOR TRAINING MLP FOR FEW-SHOT SCENERIO

"""

import pandas as pd
import numpy as np
import joblib
import os
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ================================
# Configuration
# ================================
LABEL_COLUMN = 'file'
TEST_SIZE = 0.2
BATCH_SIZE = 32
EPOCHS = 100
LR = 0.001
SEED = 42

RESULT_DIR = 'mlp_results/'
os.makedirs(RESULT_DIR, exist_ok=True)

LOSS_PLOT_PATH = os.path.join(RESULT_DIR, 'loss_curve.png')
CONF_MATRIX_PATH = os.path.join(RESULT_DIR, 'confusion_matrix.png')
METRICS_LOG_PATH = os.path.join(RESULT_DIR, 'metrics_log.csv')
SCALER_PATH = os.path.join(RESULT_DIR, 'scaler.pkl')

# ================================
# Load data
# ================================
df_fingerprints = pd.read_csv('fingerprint_all_emb.csv')
df_fingerprints = df_fingerprints[df_fingerprints['file'].str.contains('-nc-all')]
df = df_fingerprints

X = df.drop(columns=[LABEL_COLUMN]).values.astype(np.float32)
y = df[LABEL_COLUMN].values

# Encode class labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_encoded, test_size=TEST_SIZE, random_state=SEED, stratify=y_encoded
)

# Convert to tensors
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ================================
# Define MLP model
# ================================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.out(x)

# ================================
# Training setup
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(input_dim=X.shape[1], output_dim=len(label_encoder.classes_)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train_losses, val_losses = [], []

# Initialize metrics CSV
with open(METRICS_LOG_PATH, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])

# Track best metrics
best_metrics = {
    'min_train_loss': {'value': float('inf'), 'epoch': None},
    'min_val_loss': {'value': float('inf'), 'epoch': None},
    'max_train_acc': {'value': 0.0, 'epoch': None},
    'max_val_acc': {'value': 0.0, 'epoch': None}
}

# Save checkpoints temporarily for all epochs (to pick best later)
checkpoint_dir = os.path.join(RESULT_DIR, 'epoch_checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

for epoch in range(EPOCHS):
    # Training
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)
    avg_train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            val_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            val_correct += (predicted == y_batch).sum().item()
            val_total += y_batch.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    avg_val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total

    # Log metrics
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    with open(METRICS_LOG_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, avg_train_loss, avg_val_loss, train_acc, val_acc])

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Update best metrics
    if avg_train_loss < best_metrics['min_train_loss']['value']:
        best_metrics['min_train_loss'] = {'value': avg_train_loss, 'epoch': epoch+1}
    if avg_val_loss < best_metrics['min_val_loss']['value']:
        best_metrics['min_val_loss'] = {'value': avg_val_loss, 'epoch': epoch+1}
    if train_acc > best_metrics['max_train_acc']['value']:
        best_metrics['max_train_acc'] = {'value': train_acc, 'epoch': epoch+1}
    if val_acc > best_metrics['max_val_acc']['value']:
        best_metrics['max_val_acc'] = {'value': val_acc, 'epoch': epoch+1}

    # Save model for this epoch in checkpoint directory
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth'))

# ================================
# Save only best models after training
# ================================
for key, info in best_metrics.items():
    best_epoch = info['epoch']
    if best_epoch:
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'epoch_{best_epoch}.pth')))
        model_name = f'best_epoch_{best_epoch}_{key}.pth'
        torch.save(model.state_dict(), os.path.join(RESULT_DIR, model_name))

# Save final scaler
joblib.dump(scaler, SCALER_PATH)

# ================================
# Plot loss curves
# ================================
def plot_loss_curves(train_losses, val_losses, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

plot_loss_curves(train_losses, val_losses, save_path=LOSS_PLOT_PATH)

# ================================
# Confusion matrix
# ================================
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(CONF_MATRIX_PATH)
plt.close()
