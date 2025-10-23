"""
Main script for training SSL-AASIST with Additive Angular Margin (AAM) loss
on the combined STOPA + ASVspoof2019 dataset.

"""

import argparse
import json
import os
import warnings
from pathlib import Path
from shutil import copy
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from data_utils import Dataset_Custom, trim_audio_silence
from evaluation import compute_EER_custom
from utils import create_optimizer, set_seed
from SSL_AASIST import Model

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------------------------------------------------------- #
#                               LOSS FUNCTION                                   #
# ----------------------------------------------------------------------------- #

class MAMLoss(nn.Module):
    """
    Margin-based Additive Angular Margin (AAM) loss, often used in speaker
    verification systems for discriminative feature learning.
    """

    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.50):
        super(MAMLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s  # Scaling factor
        self.m = m  # Margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute margin-based angular loss."""
        input_norm = F.normalize(input, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        cosine = F.linear(input_norm, weight_norm).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cosine)
        cosine_m = torch.cos(theta + self.m)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        output = self.s * (one_hot * cosine_m + (1.0 - one_hot) * cosine)
        return F.cross_entropy(output, labels)

    def infer(self, input: torch.Tensor):
        """Forward pass for inference (no label margin applied)."""
        embedding_norm = F.normalize(input, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        cosine_theta = torch.matmul(embedding_norm, weight_norm.t()).clamp(-1, 1)
        logits = cosine_theta * self.s
        softmax_output = F.softmax(logits, dim=1)
        return cosine_theta, softmax_output


# ----------------------------------------------------------------------------- #
#                               DATA COLLATE                                    #
# ----------------------------------------------------------------------------- #

def collate_fn(batch):
    """
    Collate function for variable-length audio.
    Trims silence, pads/truncates to fixed length, and stacks into a batch tensor.
    """
    max_len = 64600
    sample_rate = 16000

    batch_x, utt_ids = [], []

    for waveform, utt_id, _ in batch:
        waveform_np = waveform.squeeze().cpu().numpy() if isinstance(waveform, torch.Tensor) else waveform
        trimmed_np = trim_audio_silence(waveform_np, sample_rate=sample_rate) or waveform_np
        trimmed_tensor = torch.tensor(trimmed_np).float().unsqueeze(0)  # [1, T]

        # Pad or truncate to fixed length
        padded_tensor = (
            F.pad(trimmed_tensor, (0, max_len - trimmed_tensor.shape[1]))
            if trimmed_tensor.shape[1] < max_len
            else trimmed_tensor[:, :max_len]
        )

        batch_x.append(padded_tensor)
        utt_ids.append(utt_id)

    return torch.stack(batch_x), utt_ids, _


# ----------------------------------------------------------------------------- #
#                           MODEL AND DATA LOADERS                              #
# ----------------------------------------------------------------------------- #

def get_model(device: torch.device):
    """Initialize SSL-AASIST model and move to device."""
    model = Model([], device)
    nb_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {nb_params:,} parameters.")
    return model.to(device)


def get_custom_loader(protocol_files: dict, config: dict, wav_dir: Path):
    """
    Create PyTorch DataLoader objects for train and dev sets using custom dataset.
    """
    loaders = {}
    for split, files in protocol_files.items():
        data, labels = [], []

        for file in files:
            with open(file, "r") as f:
                for line in f:
                    parts = line.strip().split(" ")
                    audio_path, label = parts[1], parts[3]
                    if not audio_path.endswith((".wav", ".flac")):
                        audio_path += ".flac"
                    data.append(audio_path)
                    labels.append(label)

        dataset = Dataset_Custom(data, labels, base_dir=wav_dir)
        loaders[split] = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    return loaders["trn"], loaders["dev"]


# ----------------------------------------------------------------------------- #
#                               TRAINING LOOP                                   #
# ----------------------------------------------------------------------------- #

def train_epoch(trn_loader, model, optim, device, scheduler, config):
    """Train the model for one epoch."""
    model.train()
    running_loss, total_samples = 0.0, 0
    criterion = MAMLoss(in_features=160, out_features=9).to(device)

    # Mapping from label strings to class indices
    label_map = {f"A{str(i).zfill(2)}": i - 1 for i in range(1, 7)}
    label_map.update({f"AA{str(i).zfill(2)}": i - 5 for i in range(11, 14)})

    for batch_x, batch_y, _ in trn_loader:
        batch_x = batch_x.to(device)
        batch_y = torch.tensor([label_map[y] for y in batch_y], dtype=torch.int64).to(device)
        total_samples += batch_x.size(0)

        # Forward pass
        batch_out, _ = model(batch_x.squeeze(1))
        loss = criterion(batch_out, batch_y)

        # Backward pass and optimization
        optim.zero_grad()
        loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()

        running_loss += loss.item() * batch_x.size(0)

    return running_loss / total_samples


# ----------------------------------------------------------------------------- #
#                             EVALUATION FUNCTION                               #
# ----------------------------------------------------------------------------- #

def produce_evaluation_file(data_loader, model, device, save_path, trial_path):
    """
    Evaluate the model on a dataset and compute accuracy and EER metrics.
    """
    model.eval()
    criterion = MAMLoss(in_features=160, out_features=9).to(device)

    with open(trial_path, "r") as f:
        trial_lines = f.readlines()

    key_list, pred_key, all_outputs = [], [], []

    label_map = {f"A{str(i).zfill(2)}": i - 1 for i in range(1, 7)}
    label_map.update({f"AA{str(i).zfill(2)}": i - 5 for i in range(11, 14)})

    with torch.no_grad():
        for batch_x, key, _ in data_loader:
            batch_x = batch_x.to(device)
            last_hidden_out, _ = model(batch_x.squeeze(1))
            _, batch_out = criterion.infer(last_hidden_out)
            outputs_args = torch.argmax(batch_out, dim=1)

            all_outputs.append(batch_out)
            key_list.extend([label_map[k] for k in key])
            pred_key.extend(outputs_args.cpu().tolist())

    output = torch.cat(all_outputs, dim=0)
    acc = np.mean(np.array(pred_key) == np.array(key_list)) * 100
    EER, Att_EER = compute_EER_custom(output.cpu().numpy(), key_list)
    return EER, Att_EER, acc


# ----------------------------------------------------------------------------- #
#                                   MAIN                                        #
# ----------------------------------------------------------------------------- #

def main(args: argparse.Namespace):
    """Main training routine."""
    with open(args.config, "r") as f_json:
        config = json.load(f_json)

    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]

    # Defaults for optional configs
    config.setdefault("eval_all_best", "True")
    config.setdefault("freq_aug", "False")

    # Ensure reproducibility
    set_seed(args.seed, config)

    # Paths
    output_dir = Path(args.output_dir)
    database_path = Path("STOPA+ASVspoof2019/all_files/")
    dev_trial_path = "STOPA+ASVspoof2019/protocols/stopa_asvspoof2019_dev.txt"

    # Experiment setup
    model_tag = output_dir / f"ep{config['num_epochs']}_bs{config['batch_size']}_{args.comment or ''}"
    model_save_path = model_tag / "weights"
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")
    writer = SummaryWriter(model_tag)
    log = model_tag / "metric_log.txt"
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Initialize model and data loaders
    model = get_model(device)
    protocol_files = {
        "trn": [Path("STOPA+ASVspoof2019/protocols/stopa_asvspoof2019_train.txt")],
        "dev": [Path("STOPA+ASVspoof2019/protocols/stopa_asvspoof2019_dev.txt")],
    }
    trn_loader, dev_loader = get_custom_loader(protocol_files, config, database_path)

    # Optimizer and scheduler setup
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    with open(log, "a") as f:
        f.write("=" * 50 + "\n")

    # -------------------- Training Loop -------------------- #
    for epoch in range(config["num_epochs"]):
        print(f"\n--- Epoch {epoch + 1}/{config['num_epochs']} ---")
        running_loss = train_epoch(trn_loader, model, optimizer, device, scheduler, config)

        torch.save(model.state_dict(), model_save_path / f"epoch_{epoch}.pth")
        print(f"Model saved at epoch {epoch}")

        dev_eer, Att_EER, ACC = produce_evaluation_file(
            dev_loader,
            model,
            device,
            metric_path / f"dev_score_epoch_{epoch}.txt",
            dev_trial_path,
        )

        with open(log, "a") as f:
            f.write(f"Epoch: {epoch}, Loss: {running_loss:.5f}, Dev EER: {dev_eer:.3f}, "
                    f"Attack EER: {Att_EER:.3f}, Accuracy: {ACC:.3f}\n")

        print(f"Epoch {epoch}: Loss={running_loss:.5f}, EER={dev_eer:.3f}, ACC={ACC:.2f}%")

    print("Training complete.")


# ----------------------------------------------------------------------------- #
#                                   ENTRY POINT                                 #
# ----------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSL-AASIST Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to config file.")
    parser.add_argument("--output_dir", type=str, default="exp_result", help="Output directory.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--eval", action="store_true", help="Run evaluation only.")
    parser.add_argument("--comment", type=str, default=None, help="Optional experiment tag.")
    parser.add_argument("--eval_model_weights", type=str, default=None, help="Path to model weights for evaluation.")
    main(parser.parse_args())
