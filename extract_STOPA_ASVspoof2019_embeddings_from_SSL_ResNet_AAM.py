"""
USE FOR EXTRACTING EMBEDDINGS FOR STOPA + ASVspoof2019 COMBINED TRAINING DATASET USING SSL_ResNet34_AAM.pth 

"""

import argparse
import json
import os
import warnings
import random
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from data_utils import Dataset_Custom, trim_audio_silence

try:
    from ResNet34 import Resnet34Model
except ImportError:
    print("Error: Could not import Resnet34Model. Ensure 'ResNet34.py' is in the directory.")
    sys.exit(1)

warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------------------------------------------------
# 1. Model Wrapper (Must match training exactly)
# -------------------------------------------------------------------
class ResNetWithFrontend(nn.Module):
    def __init__(self, device, model_config):
        super(ResNetWithFrontend, self).__init__()
        self.device = device
        
        self.frontend = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=128,
            f_min=0.0,
            f_max=8000.0,
            window_fn=torch.hamming_window
        ).to(device)

        self.instancenorm = nn.InstanceNorm1d(128)
        self.backbone = Resnet34Model(device=device, input_channels=1, num_classes=2)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1) 
            
        with torch.no_grad():
            x = self.frontend(x) + 1e-6
            x = x.log()
            x = self.instancenorm(x)
            x = x.unsqueeze(1)

        logits, features = self.backbone(x)
        return logits, features

# -------------------------------------------------------------------
# 2. Data Processing & Indexing
# -------------------------------------------------------------------
def collate_fn(batch):
    max_len = 64600  
    sample_rate = 16000
    batch_x = []
    
    list_2 = []
    list_3 = []

    for waveform, item2, item3 in batch:
        if isinstance(waveform, torch.Tensor):
            waveform_np = waveform.squeeze().cpu().numpy()
        else:
            waveform_np = waveform

        trimmed_np = trim_audio_silence(waveform_np, sample_rate=sample_rate)
        if len(trimmed_np) == 0: trimmed_np = waveform_np
        
        if len(trimmed_np) < max_len:
            while len(trimmed_np) < max_len:
                trimmed_np = np.concatenate((trimmed_np, trimmed_np))
            trimmed_np = trimmed_np[:max_len] 
        else:
            trimmed_np = trimmed_np[:max_len] 

        trimmed_tensor = torch.tensor(trimmed_np).float().unsqueeze(0) 
        batch_x.append(trimmed_tensor)
        list_2.append(item2)
        list_3.append(item3)

    batch_x = torch.stack(batch_x)
    return batch_x, list_2, list_3

def index_audio_files(wav_dir: Path):
    """Scans the directory to create a map of filename -> full path."""
    print(f"Indexing files in {wav_dir}...")
    file_map = {}
    for file_path in wav_dir.rglob("*"):
        if file_path.suffix.lower() in ['.flac', '.wav']:
            file_map[file_path.stem] = file_path.resolve()
    print(f"Index complete. Found {len(file_map)} audio files.")
    return file_map

def load_protocol(file_path: Path, file_map: dict, total_samples=300):
    """Parse protocol file, group by attack type, map to absolute paths, and select equal samples."""
    data = []
    missing_count = 0
    
    with open(file_path, "r") as f:
        lines = f.readlines()
        
    start_idx = 0
    if len(lines) > 0 and 'SpkID' in lines[0]:
        start_idx = 1

    for line in lines[start_idx:]:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        
        filename, attack_type = parts[1], parts[3]
        key = Path(filename).stem
        
        if key in file_map:
            data.append((str(file_map[key]), attack_type))
        else:
            missing_count += 1

    if missing_count > 0:
        print(f"Warning: {missing_count} files from the protocol were not found in the audio directory.")

    # Group by attack type
    grouped = {}
    for fname, atk in data:
        grouped.setdefault(atk, []).append(fname)

    # Equal distribution
    num_classes = len(grouped)
    if num_classes == 0:
        print("Error: No classes found to sample from.")
        return []
        
    per_class = total_samples // num_classes

    final_data = []
    for atk, files in grouped.items():
        if len(files) > per_class:
            sampled = random.sample(files, per_class)
        else:
            sampled = files  # if fewer than per_class available
        final_data.extend((f, atk) for f in sampled)

    return final_data

def get_custom_loader(data_list, config: dict):
    # Extract paths and labels from the data_list tuples
    paths = [d[0] for d in data_list]
    labels = [d[1] for d in data_list]
    
    # base_dir is empty because data_list now contains absolute paths from the index
    dataset = Dataset_Custom(paths, labels, base_dir=Path(''))
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return loader

# -------------------------------------------------------------------
# 3. Model Init & Inference
# -------------------------------------------------------------------
def get_model(device: torch.device, model_config: dict, model_path: Path):
    model = ResNetWithFrontend(device, model_config)
    
    state_dict = torch.load(model_path, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def generate_embeddings(data_loader: DataLoader, model, device: torch.device, data_list):
    results = []
    model.eval()
    with torch.no_grad():
        idx = 0
        for batch_x, utt_ids, _ in data_loader:
            batch_x = batch_x.to(device)
            
            # ResNet returns (logits, features)
            _, emb = model(batch_x)
            emb = emb.cpu().numpy()
            
            for fname, e in zip(utt_ids, emb):
                atk = data_list[idx][1]  # get attack type
                results.append(((fname, atk), e))
                idx += 1
    return results

def save_embeddings_to_csv(results, output_csv: Path):
    if len(results) == 0:
        print("No embeddings to save.")
        return
        
    rows = []
    for (fname, atk), emb in results:
        # Extract just the filename from the absolute path for cleaner CSV reading
        clean_fname = Path(fname).name 
        row = [f"{atk}_{clean_fname}"] + emb.tolist()
        rows.append(row)
        
    num_cols = len(rows[0]) - 1
    columns = ["filename"] + [str(i) for i in range(num_cols)]

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Embeddings saved to {output_csv}")

# -------------------------------------------------------------------
# 4. Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    # --- Configuration ---
    config_path = 'config/Resnet34.conf'
    
    with open(config_path, "r") as f_json:
        config = json.loads(f_json.read())

    # Update these paths as needed for your ResNet setup
    database_path = Path('STOPA+ASVspoof2019/all_files/')
    protocol_file = Path("STOPA+ASVspoof2019/protocols/stopa_asvspoof2019_train.txt")  
    model_path = Path("models/SSL_ResNet34_AAM.pth")
    
    wav_path = database_path
    
    # Target CSV output
    output_csv = Path("STOPA_ASVspoof2019_embeddings_from_SSL_ResNet_AAM.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Index Files
    file_map = index_audio_files(wav_path)

    # 2. Parse Protocol & Sample
    print(f"Loading protocol from {protocol_file.name}...")
    filtered_data = load_protocol(protocol_file, file_map, total_samples=300)
    
    if len(filtered_data) == 0:
        print("No valid data found. Exiting.")
        sys.exit(1)
        
    print(f"Selected {len(filtered_data)} samples total across {len(set([d[1] for d in filtered_data]))} attack types.")

    # 3. Create Loader
    loader = get_custom_loader(filtered_data, config)
    
    # 4. Initialize Model
    model = get_model(device, config.get("model_config", {}), model_path)
    print("Model loaded successfully.")

    # 5. Extract & Save
    print("Extracting embeddings...")
    results = generate_embeddings(loader, model, device, filtered_data)

    save_embeddings_to_csv(results, output_csv)
    print("Embedding extraction complete!")
