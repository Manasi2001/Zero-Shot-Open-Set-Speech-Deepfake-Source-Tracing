"""
USE FOR EXTRACTING EMBEDDINGS FOR STOPA + ASVspoof2019 COMBINED TRAINING DATASET USING SSL_AASIST_AAM_RegMixup.pth 

"""

import json
import os
import warnings
import random
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
import pandas as pd

# Updated model import to match training script
try:
    from SSL_AASIST_AAM_RegMixup import Model
except ImportError:
    print("Error: Could not import 'Model' from SSL_AASIST_AAM_RegMixup.py")
    sys.exit(1)

from data_utils import Dataset_Custom, trim_audio_silence

warnings.filterwarnings("ignore", category=FutureWarning)


def load_protocol(file_path: Path, total_samples=300):
    """Parse protocol file, group by attack type, and select equal samples per class."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            filename, attack_type = parts[1], parts[3]

            if '.wav' not in filename:
                filename = filename + '.flac'

            data.append((filename, attack_type))

    grouped = {}
    for fname, atk in data:
        grouped.setdefault(atk, []).append(fname)

    num_classes = len(grouped)
    per_class = total_samples // num_classes

    final_data = []
    for atk, files in grouped.items():
        if len(files) > per_class:
            sampled = random.sample(files, per_class)
        else:
            sampled = files  
        final_data.extend((f, atk) for f in sampled)

    return final_data


def collate_fn(batch):
    max_len = 64600
    sample_rate = 16000

    batch_x, utt_ids = [], []

    # Updated to unpack 3 values based on your Dataset_Custom behavior
    # Assuming: waveform, label, utt_id (or similar 3-item structure)
    for items in batch:
        waveform = items[0]
        # In extraction, we usually want the filename/ID which is often the last item
        utt_id = items[-1] 

        if isinstance(waveform, torch.Tensor):
            waveform_np = waveform.squeeze().cpu().numpy()
        else:
            waveform_np = waveform

        trimmed_np = trim_audio_silence(waveform_np, sample_rate=sample_rate)
        if len(trimmed_np) == 0:
            trimmed_np = waveform_np

        # Match the logic in training: concatenate if short, crop if long
        if len(trimmed_np) < max_len:
            while len(trimmed_np) < max_len:
                trimmed_np = np.concatenate((trimmed_np, trimmed_np))
            trimmed_np = trimmed_np[:max_len]
        else:
            trimmed_np = trimmed_np[:max_len]

        trimmed_tensor = torch.tensor(trimmed_np).float().unsqueeze(0)
        batch_x.append(trimmed_tensor)
        utt_ids.append(utt_id)

    batch_x = torch.stack(batch_x)
    return batch_x, utt_ids

def get_custom_loader(data_list, wav_dir: Path, config: dict):
    # Pass the filename list and the attack type list separately 
    # to match the Dataset_Custom(__init__) signature
    files = [d[0] for d in data_list]
    labels = [d[1] for d in data_list]
    
    dataset = Dataset_Custom(files, labels, base_dir=wav_dir)
    
    # Set num_workers=1 as suggested by your system warning to avoid freezing
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=collate_fn,
        num_workers=1 
    )
    return loader


def get_model(device: torch.device, model_path: Path):
    model = Model([], device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.apply_mixup = False # Explicitly disable mixup logic
    model.eval()
    return model


def generate_embeddings(data_loader: DataLoader, model, device: torch.device, data_list):
    results = []
    with torch.no_grad():
        idx = 0
        for batch_x, utt_ids in data_loader:
            batch_x = batch_x.to(device)
            
            # Forward pass returns (embedding, lambda, shuffled_target)
            # We set y=None as done in produce_evaluation_file in training script
            emb, _, _ = model(batch_x.squeeze(1), y=None)
            
            emb = emb.cpu().numpy()
            for fname, e in zip(utt_ids, emb):
                atk = data_list[idx][1]  # get attack type from the original list
                results.append(((fname, atk), e))
                idx += 1
    return results


def save_embeddings_to_csv(results, output_csv: Path):
    rows = []
    for (fname, atk), emb in results:
        # Saving as atk_filename to keep unique identifiers with class info
        row = [f"{atk}_{fname}"] + emb.tolist()
        rows.append(row)
        
    num_cols = len(rows[0]) - 1
    columns = ["filename"] + [str(i) for i in range(num_cols)]

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Embeddings saved to {output_csv}")


if __name__ == "__main__":
    # --- Configuration Paths ---
    config_path = 'config/STOPA_ASVspoof2019.conf'
    with open(config_path, "r") as f_json:
        config = json.loads(f_json.read())

    # Update this path to the specific weights folder from your mixup run
    model_path = Path("models/SSL_AASIST_AAM_RegMixup.pth")
    
    database_path = Path(config["database_path"])
    protocol_file = Path("STOPA+ASVspoof2019/protocols/stopa_asvspoof2019_train.txt")  
    
    emb_config = config["embd_config"]
    output_csv = Path("STOPA_ASVspoof2019_embeddings_from_SSL_AASIST_AAM_RegMixup.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading protocol from {protocol_file}...")
    filtered_data = load_protocol(protocol_file, total_samples=300)
    print(f"Selected {len(filtered_data)} samples across {len(set([d[1] for d in filtered_data]))} attack types.")

    print("Initializing loader and model...")
    loader = get_custom_loader(filtered_data, database_path, config)
    model = get_model(device, model_path)

    print("Extracting embeddings...")
    results = generate_embeddings(loader, model, device, filtered_data)

    save_embeddings_to_csv(results, output_csv)
    print("Embedding extraction complete!")