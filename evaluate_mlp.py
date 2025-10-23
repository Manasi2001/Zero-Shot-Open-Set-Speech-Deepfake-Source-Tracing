"""
USE FOR EVALUATING MLP

"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# ==============================
# Paths
# ==============================
model_path = 'mlp_results/best_epoch_97_min_val_loss.pth'
scaler_path = 'mlp_results/scaler.pkl'
fingerprint_file = 'fingerprint_all_emb.csv'
trial_embeddings_file = 'trial_embeddings.npy'
trial_list_file = 'protocols_trials_extended/AA01-co-100_trials.txt'
protocols_folder = 'protocols_trials_extended'
output_file = 'evaluation_files_with_scores/evaluation_mlp.csv'

# ==============================
# Load trial embeddings and metadata
# ==============================
fp_df = pd.read_csv(fingerprint_file)
trial_data = np.load(trial_embeddings_file)
trial_df = pd.DataFrame(trial_data)

trial_list = pd.read_csv(trial_list_file, sep=" ")
trial_df['FileName'] = trial_list['FileName']

# Reorder
last_col = trial_df.columns[-1]
trial_df = trial_df[[last_col] + list(trial_df.columns[:-1])]

# ==============================
# MLP Definition
# ==============================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

# ==============================
# Load Model & Scaler
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = trial_df.shape[1] - 1  # excluding FileName
output_classes = sorted([cls for cls in fp_df['file'].unique() if '-nc-all' in cls])
output_dim = len(output_classes)

model = MLP(input_dim=input_dim, output_dim=5).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

scaler = joblib.load(scaler_path)

# Label mapping
label_mapping = {i: label for i, label in enumerate(output_classes)}

# ==============================
# Compute MLP Scores for Trials
# ==============================
trial_features = trial_df.iloc[:, 1:].values.astype(np.float32)
trial_scaled = scaler.transform(trial_features)

input_tensor = torch.tensor(trial_scaled, dtype=torch.float32).to(device)
with torch.no_grad():
    logits = model(input_tensor)
    probs = F.softmax(logits, dim=1).cpu().numpy()

# ==============================
# Create Score DataFrame
# ==============================
print(output_classes)
score_df = pd.DataFrame(probs, columns=output_classes)
score_df.insert(0, 'FileName', trial_df['FileName'])

# ==============================
# Write updated protocol files
# ==============================
for col in score_df.columns[1:]:
    proto_file = os.path.join(protocols_folder, col + '_trials.txt')
    if os.path.exists(proto_file):
        with open(proto_file, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        with open(proto_file, 'w') as f:
            if lines:
                f.write(lines[0] + ' MLPScore\n')
                for line, score in zip(lines[1:], score_df[col]):
                    f.write(f"{line} {score}\n")
    else:
        print(f'Protocol file not found: {proto_file}')

print("MLP scores written to protocol files.")

# ==============================
# Concatenate protocol files
# ==============================
all_lines = []

for file_name in os.listdir(protocols_folder):
    if file_name.endswith('.txt') and 'nc-all' in file_name and any(k in file_name for k in ['AA01', 'AA03', 'AA05', 'AA07', 'AA10']):  
        path = os.path.join(protocols_folder, file_name)
        with open(path, 'r') as f:
            lines = [line.strip().replace(" ", ",") for line in f.readlines()]
            all_lines.extend(lines)

with open(output_file, 'w') as f:
    for line in all_lines:
        f.write(line + '\n')

print(f"All MLP-scored protocol files saved to {output_file}")
