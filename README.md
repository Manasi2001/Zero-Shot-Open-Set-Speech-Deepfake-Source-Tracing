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
- Reproducible STOPA [1] + ASVspoof2019 [2] training/evaluation protocols  

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

**Large files/folders that could not be uploaded on GitHub due to size limit can be accessed [here](https://studentuef-my.sharepoint.com/:f:/g/personal/manachhi_uef_fi/EswpEijCiwtPgfpeSrsoOd0BM5vC8WkJCqXMyzVLmKsoug?e=VLWp7I).*


---

## Dataset Preparation

This repository assumes access to both **STOPA** and **ASVspoof2019-LA** datasets.  

1. **Download the datasets:**
   - STOPA: [https://zenodo.org/records/15606628](https://zenodo.org/records/15606628)  
   - ASVspoof 2019-LA: [https://www.asvspoof.org](https://www.asvspoof.org)

2. **Organize as follows (only include *training files*):**

```
./STOPA+ASVspoof2019/all_files/
├── [STOPA training files]
├── [ASVspoof2019-LA training files]
└── ...
```

3. **Use provided protocols for training and development:**

```
./STOPA+ASVspoof2019/protocols/stopa_asvspoof2019_train.txt
./STOPA+ASVspoof2019/protocols/stopa_asvspoof2019_dev.txt
```

4. **Download additional protocols for evaluation:**  
The evaluation scripts require `protocols_trials_extended/`, which can be downloaded from the STOPA repository:  
➡️ [https://github.com/Manasi2001/STOPA](https://github.com/Manasi2001/STOPA)

---

## Step 1: Train SSL-AASIST + AAM Embedding Extractor

```
python train_SSL_AASIST_AAM.py --config config/AASIST_STOPA_ASVspoof2019.conf
```

- Trains the SSL-AASIST embedding extractor with **additive angular margin loss** on STOPA + ASVspoof2019 data.  
- Runs for **100 epochs**, saving intermediate checkpoints in `exp_results/` (e.g., `epoch_1.pth`, `epoch_100.pth`, etc.).  
- For this repository, one of the best-performing models (based on **minimum training loss**) was selected and renamed to `SSL_AASIST_AAM.pth` for convenience.  
- You may experiment with alternative checkpoints for performance comparison.

## Step 2: Extract Embeddings

```
python extract_STOPA_ASVspoof2019_embeddings.py
```

- Uses the trained SSL-AASIST model to extract **attack embeddings**.
- Saves the embeddings to `STOPA_ASVspoof2019_embeddings.csv`.

## Step 3: Train Backend Models

### (a) Few-Shot MLP

```
python train_few_shot_mlp.py
```

### (b) Few-Shot Siamese Networks

- Cross-Entropy loss:
  
  ```
  python train_few_shot_siamese_network_CE.py
  ```
  
- Contrastive loss:

  ```
  python train_few_shot_siamese_network_CL.py
  ```
  
### (c) Zero-Shot Siamese Network

```
python train_zero_shot_siamese_network_CL.py
```

Each model is saved under `models/` and evaluated separately in the next step.

### Step 4: Evaluate Backend Models

```
python evaluate_mlp.py
python evaluate_siamese_network.py
```

- Generates score files in `evaluation_files_with_scores/` for each backend configuration.

- Files include:

  - `evaluation_mlp.csv`
  
  - `evaluation_few_shot_siamese_network_CE.csv`
  
  - `evaluation_few_shot_siamese_network_CL.csv`
  
  - `evaluation_zero_shot_siamese_network_CL.csv`

### Step 5: Compute Equal Error Rate (EER)

```
python compute_eer_mlp.py
python compute_eer_siamese.py
```

- Computes Equal Error Rates (EER) for each backend strategy.

- Results are stored in their respective folders:

  ```
  EERs_mlp/
  EERs_few_shot_siamese_network_CE/
  EERs_few_shot_siamese_network_CL/
  EERs_zero_shot_siamese_network_CL/
  ```

---

## Notes on Cosine Similarity Experiments

The **cosine-similarity baseline** from Table 2 is *not implemented in this repository*. To reproduce those results, please refer to the **STOPA** repository:  

➡️ [https://github.com/Manasi2001/STOPA](https://github.com/Manasi2001/STOPA)

---

## Citation

If you use this repository or results from the paper, please cite:

```
@article{chhibber2025advancing,
  title={Advancing Zero-Shot Open-Set Speech Deepfake Source Tracing},
  author={Chhibber, Manasi and Mishra, Jagabandhu and Kinnunen, Tomi H},
  journal={arXiv preprint arXiv:2509.24674},
  year={2025}
}
```

---

## License

```
MIT License

Copyright (c) 2025 Manasi Chhibber

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Acknowledgements

- This work was partially supported by the **Academy of Finland** (Decision No. 349605, project “SPEECHFAKES”).  
- Computational resources were provided by **CSC – IT Center for Science, Finland**.  
- The SSL-AASIST architecture builds upon the implementation from [TakHemlata/SSL_Anti-spoofing]([https://github.com/clovaai/aasist](https://github.com/TakHemlata/SSL_Anti-spoofing)).

---

### References

[1] STOPA: A Database of Systematic VariaTion Of DeePfake Audio for Open-Set Source Tracing and Attribution
```bibtex
@misc{firc2025stopadatabasesystematicvariation,
author={Anton Firc and Manasi Chhibber and Jagabandhu Mishra and Vishwanath Pratap Singh and Tomi Kinnunen and Kamil Malinka},
year={2025},
eprint={2505.19644},
archivePrefix={arXiv},
primaryClass={cs.SD},
url={https://arxiv.org/abs/2505.19644}, 
}
```

[2] ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech
```bibtex
@article{wang2020asvspoof,
  title={ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech},
  author={Wang, Xin and Yamagishi, Junichi and Todisco, Massimiliano and Delgado, H{\'e}ctor and Nautsch, Andreas and Evans, Nicholas and Sahidullah, Md and Vestman, Ville and Kinnunen, Tomi and Lee, Kong Aik and others},
  journal={Computer Speech \& Language},
  volume={64},
  pages={101114},
  year={2020},
  publisher={Elsevier}
}
```

---
