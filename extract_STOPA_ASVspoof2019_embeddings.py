"""
USE FOR EXTRACTING EMBEDDINGS FOR STOPA + ASVspoof2019 COMBINED TRAINING DATASET USING SSL_AASIST_AAM.pth 

"""

import json
import os
import warnings
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
import pandas as pd
from SSL_AASIST import Model

from data_utils import Dataset_Custom, trim_audio_silence

warnings.filterwarnings("ignore", category=FutureWarning)

def collate_fn(batch):
    max_len = 64600
    sample_rate = 16000

    batch_x, utt_ids = [], []

    for waveform, utt_id in batch:
        if isinstance(waveform, torch.Tensor):
            waveform_np = waveform.squeeze().cpu().numpy()
        else:
            waveform_np = waveform

        trimmed_np = trim_audio_silence(waveform_np, sample_rate=sample_rate)
        if len(trimmed_np) == 0:
            trimmed_np = waveform_np

        trimmed_tensor = torch.tensor(trimmed_np).float().unsqueeze(0)

        if trimmed_tensor.shape[1] < max_len:
            padded_tensor = F.pad(trimmed_tensor, (0, max_len - trimmed_tensor.shape[1]))
        else:
            padded_tensor = trimmed_tensor[:, :max_len]

        batch_x.append(padded_tensor)
        utt_ids.append(utt_id)

    batch_x = torch.stack(batch_x)
    return batch_x, utt_ids


def load_protocol(file_path: Path, total_samples=300):
    """Parse protocol file, group by attack type, and select equal samples per class."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            filename, attack_type = parts[1], parts[3]

            # Append .flac for non-AA* types
            if not attack_type.startswith("AA"):
                if not filename.endswith(".flac"):
                    filename = filename + ".flac"

            data.append((filename, attack_type))

    # Group by attack type
    grouped = {}
    for fname, atk in data:
        grouped.setdefault(atk, []).append(fname)

    # Equal distribution
    num_classes = len(grouped)
    per_class = total_samples // num_classes

    final_data = []
    for atk, files in grouped.items():
        if len(files) > per_class:
            sampled = random.sample(files, per_class)
        else:
            sampled = files  # if fewer than per_class available
        final_data.extend((f, atk) for f in sampled)

    return final_data


def get_custom_loader(data_list, wav_dir: Path, config: dict):
    dataset = Dataset_Custom([d[0] for d in data_list], base_dir=wav_dir)
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return loader


def get_model(device: torch.device, model_path: Path):
    model = Model([], device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def generate_embeddings(data_loader: DataLoader, model, device: torch.device, data_list):
    results = []
    with torch.no_grad():
        idx = 0
        for batch_x, utt_ids in data_loader:
            batch_x = batch_x.to(device)
            emb, _ = model(batch_x.squeeze(1))
            emb = emb.cpu().numpy()
            for fname, e in zip(utt_ids, emb):
                atk = data_list[idx][1]  # get attack type
                results.append(((fname, atk), e))
                idx += 1
    return results


def save_embeddings_to_csv(results, output_csv: Path):
    rows = []
    for (fname, atk), emb in results:
        row = [f"{atk}_{fname}"] + emb.tolist()
        rows.append(row)
        
    num_cols = len(rows[0]) - 1
    columns = ["filename"] + [str(i) for i in range(num_cols)]

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Embeddings saved to {output_csv}")


if __name__ == "__main__":
    config_path = 'config/AASIST_STOPA_ASVspoof2019.conf'
    with open(config_path, "r") as f_json:
        config = json.loads(f_json.read())

    emb_config = config["embd_config"]
    database_path = Path(config["database_path"])
    protocol_file = Path("STOPA+ASVspoof2019/protocols/stopa_asvspoof2019_train.txt")  
    model_path = Path("models/SSL_AASIST_AAM.pth")
    wav_path = database_path
    output_csv = Path("STOPA_ASVspoof2019_embeddings.csv")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Loading protocol...")
    filtered_data = load_protocol(protocol_file, total_samples=300)
    print(f"Selected {len(filtered_data)} samples total across {len(set([d[1] for d in filtered_data]))} attack types.")

    loader = get_custom_loader(filtered_data, wav_path, config)
    model = get_model(device, model_path)

    results = generate_embeddings(loader, model, device, filtered_data)

    save_embeddings_to_csv(results, output_csv)

    print("Embedding extraction complete!")
