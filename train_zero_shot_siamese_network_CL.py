"""
USE FOR TRAINING SIAMESE NETWORK WITH CONTRASTIVE LOSS FOR ZERO-SHOT SCENERIO

"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random

# -------------------------------
# 1. Load and filter fingerprint embeddings
# -------------------------------
df = pd.read_csv("STOPA_ASVspoof2019_embeddings.csv")

df["attack_class"] = df["filename"].str.split("_").str[0]

# Convert embeddings to numpy array
embeddings = df.drop(columns=["filename", "attack_class"]).values.astype(np.float32)
labels = df["attack_class"].values

# -------------------------------
# 2. Dataset for positive/negative pairs
# -------------------------------
class FingerprintPairDataset(Dataset):
    def __init__(self, embeddings, labels, n_pairs=10000):
        self.embeddings = embeddings
        self.labels = labels
        self.n_pairs = n_pairs
        self.unique_classes = list(set(labels))
        self.index_by_class = {c: np.where(labels == c)[0] for c in self.unique_classes}
        self.pairs = self._make_pairs()

    def _make_pairs(self):
        pairs = []
        for _ in range(self.n_pairs):
            if random.random() < 0.5:
                # positive pair
                c = random.choice(self.unique_classes)
                idx1, idx2 = np.random.choice(self.index_by_class[c], 2, replace=False)
                pairs.append((idx1, idx2, 1))
            else:
                # negative pair
                c1, c2 = random.sample(self.unique_classes, 2)
                idx1 = random.choice(self.index_by_class[c1])
                idx2 = random.choice(self.index_by_class[c2])
                pairs.append((idx1, idx2, 0))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i1, i2, label = self.pairs[idx]
        x1 = torch.tensor(self.embeddings[i1])
        x2 = torch.tensor(self.embeddings[i2])
        y = torch.tensor(label, dtype=torch.float32)
        return x1, x2, y


# -------------------------------
# 3. Siamese Encoder + Network
# -------------------------------
class SiameseEncoder(nn.Module):
    def __init__(self, input_dim=160, hidden_dim=64, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, p=2, dim=-1)  # L2 normalize embeddings


class SiameseNetwork(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        return z1, z2


# -------------------------------
# 4. Contrastive Loss
# -------------------------------
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, z1, z2, y):
        # Euclidean distance between embeddings
        dist = F.pairwise_distance(z1, z2)
        # Contrastive loss
        loss_pos = y * dist.pow(2)                     # positive pairs: minimize distance
        loss_neg = (1 - y) * F.relu(self.margin - dist).pow(2)  # negative pairs: enforce margin
        return (loss_pos + loss_neg).mean()


# -------------------------------
# 5. Training Loop
# -------------------------------
def train_model(embeddings, labels, epochs=10, batch_size=64, lr=1e-3, n_pairs=20000, margin=1.0):
    dataset = FingerprintPairDataset(embeddings, labels, n_pairs=n_pairs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    encoder = SiameseEncoder()
    model = SiameseNetwork(encoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = ContrastiveLoss(margin=margin)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x1, x2, y in dataloader:
            optimizer.zero_grad()
            z1, z2 = model(x1, x2)
            loss = criterion(z1, z2, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x1.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return model


# -------------------------------
# 6. Example Usage
# -------------------------------
if __name__ == "__main__":
    model = train_model(
        embeddings, labels,
        epochs=100,
        batch_size=128,
        n_pairs=50000,
        margin=1.0
    )

    # Save model
    torch.save(
        model.state_dict(),
        "models/zero_shot_siamese_network_CL.pt"
    )
