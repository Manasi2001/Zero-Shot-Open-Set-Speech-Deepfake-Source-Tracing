# Zero-Shot Open-Set Speech Deepfake Source Tracing

This repository provides the official implementation for the paper: **“Advancing Zero-Shot Open-Set Speech Deepfake Source Tracing”** [[arXiv:2509.24674](https://arxiv.org/abs/2509.24674)].

The repository provides code to reproduce the **SSL-AASIST + AAM results (Table 1)** and **backend scoring experiments (Table 2)** from the paper. For reproducing **cosine similarity**–based results in Table 2, please refer to the companion repository:  
➡️ [STOPA Repository](https://github.com/Manasi2001/STOPA)

---

## Overview

This work introduces a **zero-shot and few-shot verification framework** for speech deepfake source tracing. It adapts the **SSL-AASIST** architecture for open-set attribution, combining self-supervised front ends with additive angular margin (AAM) loss and backend scoring methods such as cosine similarity, Siamese networks, and MLP classifiers.

**Key features:**
- Zero-shot and few-shot backend scoring for open-set spoof attribution  
- SSL-AASIST embedding extractor with AAM-softmax objective  
- Evaluation of MLP and Siamese backends (cross-entropy & contrastive)  
- Reproducible STOPA + ASVspoof2019 training/evaluation protocols  

---

## Directory Structure

```
.
├── compute_eer_mlp.py                      # Compute EER for few-shot MLP backend
├── compute_eer_siamese.py                  # Compute EER for Siamese backends (few-shot, zero-shot)
├── config/
│   └── AASIST_STOPA_ASVspoof2019.conf      # Config for SSL-AASIST + AAM training
│
├── data_utils.py                           # Dataset utilities for STOPA and ASVspoof2019
├── evaluate_mlp.py                         # Evaluate few-shot MLP backend
├── evaluate_siamese_network.py             # Evaluate Siamese network backends
├── evaluation.py                           # Unified evaluation and EER computation
│
├── extract_STOPA_ASVspoof2019_embeddings.py# Extract SSL-AASIST embeddings
│
├── models/                                 # Trained model checkpoints
│   ├── SSL_AASIST_AAM.pth*                 # Selected best checkpoint (renamed for convenience)
│   ├── few_shot_siamese_network_CE.pt
│   ├── few_shot_siamese_network_CL.pt
│   └── zero_shot_siamese_network_CL.pt
│
├── trial_embeddings.npy*                   # Trial embeddings extracted by SSL_AASIST_AAM (NumPy .npy file)
├── SSL_AASIST.py                           # Model definition / architecture used for training & inference
│
├── STOPA+ASVspoof2019/                     # Dataset directory
│   ├── protocols/
│   │   ├── stopa_asvspoof2019_train.txt
│   │   └── stopa_asvspoof2019_dev.txt
│   └── all_files/                          # Place STOPA and ASVspoof2019 training files here only
│
├── evaluation_files_with_scores/*          # Score files from evaluation scripts
│   ├── evaluation_mlp.csv
│   ├── evaluation_few_shot_siamese_network_CE.csv
│   ├── evaluation_few_shot_siamese_network_CL.csv
│   └── evaluation_zero_shot_siamese_network_CL.csv
│
├── EERs_mlp/                               # EER results for few-shot MLP backend
├── EERs_few_shot_siamese_network_CE/       # EER results for Siamese (CE)
├── EERs_few_shot_siamese_network_CL/       # EER results for Siamese (CL)
├── EERs_zero_shot_siamese_network_CL/      # EER results for zero-shot Siamese
├── EERs_cosine_similarity/                 # (See STOPA repo for cosine similarity setup)
│
├── train_SSL_AASIST_AAM.py                 # Train SSL-AASIST + AAM embedding extractor
├── train_few_shot_mlp.py                   # Train few-shot MLP backend
├── train_few_shot_siamese_network_CE.py    # Train Siamese backend (cross-entropy loss)
├── train_few_shot_siamese_network_CL.py    # Train Siamese backend (contrastive loss)
├── train_zero_shot_siamese_network_CL.py   # Train zero-shot Siamese backend
│
├── utils.py                                # Utility and helper functions
└── fingerprint_all_emb.csv*                # Consolidated fingerprint embeddings
```

**Large files/folders that could not be uploaded on GitHub due to size limit can be accessed [here]().*
