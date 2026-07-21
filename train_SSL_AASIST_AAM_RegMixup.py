"""
Main script for training SSL-AASIST with Additive Angular Margin (AAM) loss
and RegMixup on the combined STOPA + ASVspoof2019 dataset.

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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Custom imports
from data_utils import Dataset_Custom, trim_audio_silence
from evaluation import compute_EER_custom
from utils import create_optimizer, seed_worker, set_seed

# Import the updated model
try:
    from SSL_AASIST_AAM_RegMixup import Model
except ImportError:
    print("Error: Could not import 'Model' from SSL_AASIST_AAM_RegMixup.py")
    sys.exit(1)

warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------------------------------------------------
# 1. MAM Loss (Angular Margin) with Dynamic Margin Support
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

    def set_margin(self, new_m):
        """Allows dynamic updating of the margin during training"""
        self.m = new_m

    def forward(self, input, labels):
        # Normalize
        input_norm = F.normalize(input, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity
        cosine = F.linear(input_norm, weight_norm)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Angular margin logic
        theta = torch.acos(cosine)
        theta_m = theta + self.m
        cosine_m = torch.cos(theta_m)

        # One-hot
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # Apply margin only to target class
        output = self.s * (one_hot * cosine_m + (1.0 - one_hot) * cosine)
        
        # Standard CE on the angular logits
        loss = F.cross_entropy(output, labels)
        return loss
    
    def infer(self, input):
        # For evaluation (no margin added)
        embedding_norm = F.normalize(input, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        cosine_theta = torch.matmul(embedding_norm, weight_norm.t())
        cosine_theta = cosine_theta.clamp(-1, 1)
        
        logits = cosine_theta * self.s
        return logits

# -------------------------------------------------------------------
# 2. Data Processing
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

def get_custom_loader(protocol_files: List[Path], config: dict, wav_dir: Path):
    loaders = {}
    for split, files in protocol_files.items():
        data = []
        all_label_list = []
        for file in files:
            with open(file, 'r') as f:
                lines = f.readlines()
            
            file_names = [line.strip().split()[1] for line in lines]
            labels = [line.strip().split()[3] for line in lines]

            full_paths = []
            for fname in file_names:
                if '.wav' in fname:
                    full_paths.append(fname)
                else:
                    full_paths.append(fname + '.flac')
            
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
# 3. Training Loop (With Mixup Logic & Curriculum)
# -------------------------------------------------------------------
def train_epoch(trn_loader, model, criterion, optim, device, scheduler, config, epoch):
    running_loss = 0
    num_total = 0.0
    
    # --- Curriculum Logic ---
    # 1. Mixup Delay: Only enable Mixup after epoch 10
    mixup_start_epoch = 10
    
    if epoch >= mixup_start_epoch:
        model.apply_mixup = True
    else:
        model.apply_mixup = False
        
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

        # Forward Pass
        # Returns: (embeddings, lambda, shuffled_labels)
        embedding, mix_lambda, target_b = model(batch_x.squeeze(1), batch_y)
        
        # --- Loss Calculation ---
        loss_a = criterion(embedding, batch_y)
        
        # Handle Mixup Loss
        if model.apply_mixup and mix_lambda is not None and target_b is not None:
            loss_b = criterion(embedding, target_b)
            batch_loss = mix_lambda * loss_a + (1 - mix_lambda) * loss_b
        else:
            batch_loss = loss_a

        running_loss += batch_loss.item() * batch_size

        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()

    running_loss /= num_total
    return running_loss

def produce_evaluation_file(data_loader, model, criterion, device, save_path, trial_path):
    model.apply_mixup = False # Ensure mixup is OFF for eval
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
            
            embedding, _, _ = model(batch_x.squeeze(1), y=None)
            logits = criterion.infer(embedding)
            
            batch_out = F.softmax(logits, dim=1) 

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
# 4. Helper: Margin Scheduler
# -------------------------------------------------------------------
def adjust_margin(criterion, epoch, total_warmup_epochs=10, final_m=0.5):
    """
    Linearly increases margin from 0.0 to final_m over warmup epochs.
    """
    if epoch < total_warmup_epochs:
        new_m = (epoch / total_warmup_epochs) * final_m
    else:
        new_m = final_m
        
    criterion.set_margin(new_m)
    return new_m

# -------------------------------------------------------------------
# 5. Main
# -------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    
    config["exp_dir"] = args.output_dir
    set_seed(args.seed, config)
    
    output_dir = Path(args.output_dir)
    database_path = Path('STOPA+ASVspoof2019/all_files/')
    wav_path = database_path 
    
    protocol_files = {
        "trn": [Path('STOPA+ASVspoof2019/protocols/stopa_asvspoof2019_train.txt')],
        "dev": [Path('STOPA+ASVspoof2019/protocols/stopa_asvspoof2019_dev.txt')]
    }

    model_tag = "ep{}_bs{}".format(config["num_epochs"], config["batch_size"])
    if args.comment: model_tag += "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    metric_path = model_tag / "metrics"
    
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(metric_path, exist_ok=True)
    
    writer = SummaryWriter(model_tag)
    copy(args.config, model_tag / "config.conf")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize Model
    model = Model([], device)
    model = model.to(device)
    print('Model initialized successfully.')

    # Initialize Loss
    # Start with m=0.5, but we will overwrite this in the loop with scheduler
    criterion = MAMLoss(in_features=160, out_features=9, m=0.5).to(device)

    # Load Data
    trn_loader, dev_loader = get_custom_loader(protocol_files, config, wav_path)
    print('Data loaders defined.')

    # Optimizer
    optim_config = config["optim_config"]
    optim_config["steps_per_epoch"] = len(trn_loader)
    optim_config["epochs"] = config["num_epochs"]
    
    params = list(model.parameters()) + list(criterion.parameters())
    optimizer, scheduler = create_optimizer(params, optim_config)
    
    log_file = model_tag / "metric_log.txt"
    with open(log_file, "a") as f:
        f.write("=" * 50 + "\n")

    best_dev_eer = 100.0

    # --- Training Loop ---
    for epoch in range(config["num_epochs"]):
        print(f"Start training epoch {epoch:03d}")
        
        # 1. Update Margin (Warmup)
        # Linearly increase margin from 0 to 0.5 over 10 epochs
        current_m = adjust_margin(criterion, epoch, total_warmup_epochs=10, final_m=0.5)
        print(f"  -> Current Margin m: {current_m:.4f}")
        
        # 2. Freeze SSL Frontend for stability (First 5 epochs)
        # This prevents the massive SSL model from destroying features while the backend is random
        if epoch < 5:
            print("  -> SSL Frontend Frozen")
            model.ssl_model.eval() # Set to eval mode
            for param in model.ssl_model.parameters():
                param.requires_grad = False
        else:
            if epoch == 5: print("  -> Unfreezing SSL Frontend...")
            model.ssl_model.train()
            for param in model.ssl_model.parameters():
                param.requires_grad = True

        # 3. Train
        train_loss = train_epoch(trn_loader, model, criterion, optimizer, device, scheduler, config, epoch)
        
        print(f'\nEpoch {epoch} finished. Loss: {train_loss:.5f}')
        torch.save(model.state_dict(), model_save_path / f"epoch_{epoch}.pth")

        # 4. Evaluate
        print("Evaluating on Dev set...")
        dev_eer, att_eer, acc = produce_evaluation_file(
            dev_loader, 
            model, 
            criterion, 
            device, 
            metric_path / f"dev_score_epoch_{epoch}.txt", 
            str(protocol_files['dev'][0])
        )

        # Logging
        if isinstance(att_eer, list) or isinstance(att_eer, np.ndarray):
            mean_att_eer = np.mean(att_eer)
            att_eer_str = str(att_eer)
        else:
            mean_att_eer = att_eer
            att_eer_str = f"{att_eer:.3f}"

        log_msg = (f"Epoch: {epoch}, Loss: {train_loss:.4f}, Margin: {current_m:.2f}, "
                   f"Dev EER: {dev_eer:.3f}, Dev Att EER: {mean_att_eer:.3f}, Acc: {acc:.3f}")
        
        print(log_msg)
        with open(log_file, "a") as f:
            f.write(log_msg + "\n")
            f.write(f"Detailed Attack EERs: {att_eer_str}\n")

        if dev_eer < best_dev_eer:
            best_dev_eer = dev_eer
            torch.save(model.state_dict(), model_save_path / "best_model.pth")
            print("New best model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSL AASIST AAM with RegMixup Training")
    parser.add_argument("--config", dest="config", type=str, default="config/STOPA_ASVspoof2019.conf")
    parser.add_argument("--output_dir", dest="output_dir", type=str, default="exp_result")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--comment", type=str, default=None)
    
    args = parser.parse_args()
    main(args)