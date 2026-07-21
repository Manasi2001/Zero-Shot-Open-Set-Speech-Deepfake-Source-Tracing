"""
Main script for training SSL-ResNet34 with Additive Angular Margin (AAM) loss
on the combined STOPA + ASVspoof2019 dataset.

"""

import argparse
import json
import os
import sys
import warnings
import numpy as np
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_utils import Dataset_Custom, trim_audio_silence
from evaluation import compute_EER_custom
from utils import seed_worker, set_seed

# Import your specific ResNet model
try:
    from ResNet34 import Resnet34Model
except ImportError:
    print("Error: Could not import Resnet34Model. Ensure 'ResNet34.py' is in the directory.")
    sys.exit(1)

warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------------------------------------------------
# 1. Feature Extraction (Frontend) + Model Wrapper
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
# 2. Loss Function
# -------------------------------------------------------------------
class MAMLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(MAMLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, labels):
        input_norm = F.normalize(input, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(input_norm, weight_norm)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cosine)
        theta_m = theta + self.m
        cosine_m = torch.cos(theta_m)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = self.s * (one_hot * cosine_m + (1.0 - one_hot) * cosine)
        loss = F.cross_entropy(output, labels)
        return loss
    
    def infer(self, input):
        embedding_norm = F.normalize(input, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        cosine_theta = torch.matmul(embedding_norm, weight_norm.t())
        cosine_theta = cosine_theta.clamp(-1, 1)
        logits = cosine_theta * self.s
        softmax_output = F.softmax(logits, dim=1)
        return cosine_theta, softmax_output

# -------------------------------------------------------------------
# 3. Data Processing
# -------------------------------------------------------------------
def collate_fn(batch):
    max_len = 64600  
    sample_rate = 16000
    batch_x = []
    
    # We will collect items 2 and 3 generically to handle swap issues
    list_2 = []
    list_3 = []

    for waveform, item2, item3 in batch:
        if isinstance(waveform, torch.Tensor):
            waveform_np = waveform.squeeze().cpu().numpy()
        else:
            waveform_np = waveform

        trimmed_np = trim_audio_silence(waveform_np, sample_rate=sample_rate)
        if len(trimmed_np) == 0: trimmed_np = waveform_np
        while len(trimmed_np) < max_len:
            trimmed_np = np.concatenate((trimmed_np, trimmed_np))
        trimmed_np = trimmed_np[:max_len] 

        trimmed_tensor = torch.tensor(trimmed_np).float().unsqueeze(0) 
        batch_x.append(trimmed_tensor)
        list_2.append(item2)
        list_3.append(item3)

    batch_x = torch.stack(batch_x)
    return batch_x, list_2, list_3

def get_custom_loader(protocol_files, config, wav_dir):
    loaders = {}
    for split, files in protocol_files.items():
        data = []
        all_label_list = []
        for file in files:
            with open(file, 'r') as f:
                lines = f.readlines()
            
            # Using .split() without arguments to handle tabs or multiple spaces automatically
            file_names = [line.strip().split()[1] for line in lines]
            labels = [line.strip().split()[3] for line in lines]

            full_paths = []
            for fname in file_names:
                ext = '.wav' if '.wav' in fname else '.flac'
                full_paths.append(fname if ext in fname else fname + ext)
            data.extend(full_paths)
            all_label_list.extend(labels)

        dataset = Dataset_Custom(data, all_label_list, base_dir=wav_dir)
        loaders[split] = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=(split == "trn"),
            drop_last=(split == "trn"),
            pin_memory=True,
            collate_fn=collate_fn,
            num_workers=4
        )
    return loaders.get("trn"), loaders.get("dev")

# -------------------------------------------------------------------
# 4. Training Loop
# -------------------------------------------------------------------
def train_epoch(trn_loader, model, criterion, optim, device, config):
    running_loss = 0
    num_total = 0.0
    model.train()

    d1 = {f"A{str(i).zfill(2)}": i - 1 for i in range(1, 7)}
    d2 = {f"AA{str(i).zfill(2)}": i - 5 for i in range(11, 14)}
    label_map = {**d1, **d2}

    
    for batch_x, item2, item3 in trn_loader:
        
        batch_y_raw = item2 
        
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)

        batch_y_indices = [label_map[y] for y in batch_y_raw]
        batch_y = torch.tensor(batch_y_indices, dtype=torch.int64).to(device)

        logits, features = model(batch_x)
        batch_loss = criterion(features, batch_y)
        
        running_loss += batch_loss.item() * batch_size

        optim.zero_grad()
        batch_loss.backward()
        optim.step()

    running_loss /= num_total
    return running_loss

def produce_evaluation_file(data_loader, model, criterion, device, save_path, trial_path):
    model.eval()
    all_scores = []
    all_labels = []
    all_preds = []

    d1 = {f"A{str(i).zfill(2)}": i - 1 for i in range(1, 7)}
    d2 = {f"AA{str(i).zfill(2)}": i - 5 for i in range(11, 14)}
    label_map = {**d1, **d2}

    with torch.no_grad():
        for batch_x, item2, item3 in data_loader:
            batch_y_raw = item2
            
            batch_x = batch_x.to(device)
            logits, features = model(batch_x)
            _, batch_out = criterion.infer(features)

            all_scores.append(batch_out.cpu())
            
            batch_labels = [label_map[y] for y in batch_y_raw]
            all_labels.extend(batch_labels)
            
            outputs_args = torch.argmax(batch_out, dim=1)
            all_preds.extend(outputs_args.cpu().numpy())

    output_tensor = torch.cat(all_scores, dim=0).numpy()
    acc = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
    EER, Att_EER = compute_EER_custom(output_tensor, all_labels)

    return EER, Att_EER, acc

# -------------------------------------------------------------------
# 5. Main
# -------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    # Load config with safety check for boolean strings
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    
    set_seed(args.seed, config)
    output_dir = Path(args.output_dir)
    database_path = Path('STOPA+ASVspoof2019/all_files/')
    protocol_files = {
        "trn": [Path('STOPA+ASVspoof2019/protocols/stopa_asvspoof2019_train.txt')],
        "dev": [Path('STOPA+ASVspoof2019/protocols/stopa_asvspoof2019_dev.txt')]
    }
    
    model_tag = "ep{}_bs{}".format(config["num_epochs"], config["batch_size"])
    if args.comment: model_tag += "_{}".format(args.comment)
    
    model_save_dir = output_dir / model_tag
    weight_save_path = model_save_dir / "weights"
    metric_path = model_save_dir / "metrics"
    
    os.makedirs(weight_save_path, exist_ok=True)
    os.makedirs(metric_path, exist_ok=True)
    
    writer = SummaryWriter(model_save_dir)
    copy(args.config, model_save_dir / "config.conf")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ResNetWithFrontend(device, config.get("model_config", {}))
    model = model.to(device)
    
    criterion = MAMLoss(in_features=256, out_features=9).to(device)

    trn_loader, dev_loader = get_custom_loader(protocol_files, config, database_path)
    print("Data loaders ready.")

    optim_config = config["optim_config"]
    params = list(model.parameters()) + list(criterion.parameters())
    
    optimizer = torch.optim.Adam(
        params,
        lr=optim_config.get("base_lr", 0.0001),
        weight_decay=optim_config.get("weight_decay", 0.0001),
        betas=tuple(optim_config.get("betas", [0.9, 0.999])),
        amsgrad=optim_config.get("amsgrad", False)
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config["num_epochs"], 
        eta_min=optim_config.get("lr_min", 0.000005)
    )
    
    best_dev_eer = 100.0
    log_file = model_save_dir / "metric_log.txt"

    for epoch in range(config["num_epochs"]):
        print(f"Start training epoch {epoch}")
        
        train_loss = train_epoch(trn_loader, model, criterion, optimizer, device, config)
        scheduler.step()
        
        torch.save(model.state_dict(), weight_save_path / f"epoch_{epoch}.pth")
        
        print("Evaluating...")
        dev_eer, att_eer, acc = produce_evaluation_file(
            dev_loader, 
            model, 
            criterion, 
            device, 
            metric_path / f"dev_score_epoch_{epoch}.txt", 
            str(protocol_files["dev"][0])
        )
        
        # Calculate the mean of the attack EERs for a single summary number
        mean_att_eer = np.mean(att_eer)

        # Create the log message using the MEAN
        log_msg = (f"Epoch: {epoch}, Train Loss: {train_loss:.5f}, "
                   f"Dev EER: {dev_eer:.3f}, Dev Att EER (Mean): {mean_att_eer:.3f}, Dev Acc: {acc:.3f}")
        
        print(log_msg)
        
        # Optionally print the full list separately so you can see individual attack performance
        print(f"Detailed Attack EERs: {[f'{x:.2f}' for x in att_eer]}")

        with open(log_file, "a") as f: 
            f.write(log_msg + "\n")
            f.write(f"Detailed Attack EERs: {att_eer}\n")

        writer.add_scalar("EER/dev", dev_eer, epoch)

        if dev_eer < best_dev_eer:
            best_dev_eer = dev_eer
            torch.save(model.state_dict(), weight_save_path / "best_model.pth")
            print("New best model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet34 Audio Anti-Spoofing")
    parser.add_argument("--config", dest="config", type=str, required=True, default = "config/Resnet34.conf")
    parser.add_argument("--output_dir", dest="output_dir", type=str, default="exp_result_ResNet34")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--comment", type=str, default=None)
    args = parser.parse_args()
    main(args)
