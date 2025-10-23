"""
USE FOR EVALUATING SIAMESE NETWORKS

"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from train_zero_shot_siamese_network_CL import SiameseEncoder, SiameseNetwork

# -------------------------------
# 1. Load trained encoder
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Instantiate encoder first
encoder = SiameseEncoder()

# 2. Instantiate wrapper network with encoder
net = SiameseNetwork(encoder)

# 3. Load checkpoint
net.load_state_dict(torch.load("models/zero_shot_siamese_network_CL.pt", map_location=device))  # choose the desired model
net.to(device).eval()

# 4. Use encoder for evaluation
encoder = net.encoder

# -------------------------------
# 2. Load fingerprint embeddings
# -------------------------------
fingerprint_df = pd.read_csv("fingerprint_all_emb.csv")
fingerprint_df["attack_class"] = fingerprint_df["file"].str.split("-").str[0] 

fingerprints = {
    row["attack_class"]: torch.tensor(row.drop(["file", "attack_class"]).values.astype(np.float32)).to(device)
    for _, row in fingerprint_df.iterrows()
}

# -------------------------------
# 3. Load trial embeddings from .npy
# -------------------------------
trial_embeddings = np.load("trial_embeddings.npy")  

# -------------------------------
# 4. Process each trials.txt
# -------------------------------
trials_dir = "protocols_trials_extended"
out_dir = "protocols_trials_extended_for_siamese_with_scores"  # this will store files with cosine score appended in them
os.makedirs(out_dir, exist_ok=True)

for fname in os.listdir(trials_dir):
    if not fname.endswith("_trials.txt"):
        continue
    
    trial_path = os.path.join(trials_dir, fname)
    trial_df = pd.read_csv(trial_path, sep=" ")

    # Make DataFrame for embeddings aligned to this file
    trial_emb_df = pd.DataFrame(trial_embeddings[:len(trial_df)])  # assumes order matches
    trial_emb_df["FileName"] = trial_df["FileName"].values

    cos_scores = []
    with torch.no_grad():
        for idx, row in trial_df.iterrows():
            atk = row["AbstractModel"].split("-")[0]
            if atk not in fingerprints:
                cos_scores.append(np.nan)
                continue

            f_emb = encoder(fingerprints[atk].unsqueeze(0))

            t_emb = torch.tensor(
                trial_emb_df.iloc[idx, :-1].values.astype(np.float32)
            ).to(device)
            t_emb = encoder(t_emb.unsqueeze(0))

            sim = F.cosine_similarity(f_emb, t_emb).item()
            cos_scores.append(sim)

    # Add CosScore column
    trial_df["CosScore"] = cos_scores

    out_path = os.path.join(out_dir, fname)
    trial_df.to_csv(out_path, sep=" ", index=False)
    print(f"Processed {fname} → {out_path}")

# ------------------------------------------------------------
# 5. Combine generated trial files into one evaluation CSV
# ------------------------------------------------------------

# Folder where the generated files with cosine scores are saved
protocols_folder = 'protocols_trials_extended_for_siamese_with_scores'
os.makedirs('evaluation_files_with_scores', exist_ok=True)

# Output file for combined results
output_file = 'evaluation_files_with_scores/evaluation_zero_shot_siamese_network_CL.csv'

all_lines = []

# Collect and concatenate selected files
for file_name in os.listdir(protocols_folder):
    file_path = os.path.join(protocols_folder, file_name)

    # Check if the file is a text file and matches specified patterns
    if file_name.endswith('.txt') and os.path.isfile(file_path) and any(pattern in file_name for pattern in ['AA01', 'AA03', 'AA05', 'AA07', 'AA10']):
        with open(file_path, 'r') as f:
            lines = [line.strip().replace(' ', ',') for line in f.readlines()]
            all_lines.extend(lines)

# Write combined file
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'w') as f:
    for line in all_lines:
        f.write(line + '\n')

print(f'All protocol files have been concatenated into {output_file}')