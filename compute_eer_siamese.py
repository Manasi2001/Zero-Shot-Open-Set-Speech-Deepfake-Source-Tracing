"""
USE FOR COMPUTING EERs FOR SIAMESE NETWORK BACKEND

"""

import warnings
import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
from evaluation import compute_eer

warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("evaluation_files_with_scores/evaluation_few_shot_siamese_network_CE.csv")  # put evaluation file of your choice

def EER(df_temp):
    true_scores = df_temp[df_temp["IsTargetATK"] == True]["CosScore"].to_numpy()
    false_scores = df_temp[df_temp["IsTargetATK"] == False]["CosScore"].to_numpy()
    eer_att, _ = compute_eer(true_scores, false_scores)

    true_scores = df_temp[df_temp["IsTargetAcousticModel"] == True]["CosScore"].to_numpy()
    false_scores = df_temp[df_temp["IsTargetAcousticModel"] == False]["CosScore"].to_numpy()
    eer_am, _ = compute_eer(true_scores, false_scores)

    true_scores = df_temp[df_temp["IsTargetVocoderModel"] == True]["CosScore"].to_numpy()
    false_scores = df_temp[df_temp["IsTargetVocoderModel"] == False]["CosScore"].to_numpy()
    eer_vm, _ = compute_eer(true_scores, false_scores)

    true_scores = df_temp[df_temp["IsTargetAmArchit"] == True]["CosScore"].to_numpy()
    false_scores = df_temp[df_temp["IsTargetAmArchit"] == False]["CosScore"].to_numpy()
    eer_am_arch, _ = compute_eer(true_scores, false_scores)

    true_scores = df_temp[df_temp["IsTargetVmArchit"] == True]["CosScore"].to_numpy()
    false_scores = df_temp[df_temp["IsTargetVmArchit"] == False]["CosScore"].to_numpy()
    eer_vm_arch, _ = compute_eer(true_scores, false_scores)

    # Count occurrences for utt
    utt = {
        "true_att": len(df_temp[df_temp["IsTargetATK"] == True]),
        "false_att": len(df_temp[df_temp["IsTargetATK"] == False]),
        "true_AM": len(df_temp[df_temp["IsTargetAcousticModel"] == True]),
        "false_AM": len(df_temp[df_temp["IsTargetAcousticModel"] == False]),
        "true_VM": len(df_temp[df_temp["IsTargetVocoderModel"] == True]),
        "false_VM": len(df_temp[df_temp["IsTargetVocoderModel"] == False]),
        "true_AM_arch": len(df_temp[df_temp["IsTargetAmArchit"] == True]),
        "false_AM_arch": len(df_temp[df_temp["IsTargetAmArchit"] == False]),
        "true_VM_arch": len(df_temp[df_temp["IsTargetVmArchit"] == True]),
        "false_VM_arch": len(df_temp[df_temp["IsTargetVmArchit"] == False]),
    }

    return eer_att, eer_am, eer_vm, eer_am_arch, eer_vm_arch, utt


def EER2(df_temp_att, df_temp_am, df_temp_vm, df_temp_am_arch, df_temp_vm_arch):
    true_scores = df_temp_att[df_temp_att["IsTargetATK"] == True]["CosScore"].to_numpy()
    false_scores = df_temp_att[df_temp_att["IsTargetATK"] == False]["CosScore"].to_numpy()
    eer_att, _ = compute_eer(true_scores, false_scores)

    true_scores = df_temp_am[df_temp_am["IsTargetAcousticModel"] == True]["CosScore"].to_numpy()
    false_scores = df_temp_am[df_temp_am["IsTargetAcousticModel"] == False]["CosScore"].to_numpy()
    eer_am, _ = compute_eer(true_scores, false_scores)

    true_scores = df_temp_vm[df_temp_vm["IsTargetVocoderModel"] == True]["CosScore"].to_numpy()
    false_scores = df_temp_vm[df_temp_vm["IsTargetVocoderModel"] == False]["CosScore"].to_numpy()
    eer_vm, _ = compute_eer(true_scores, false_scores)

    true_scores = df_temp_am_arch[df_temp_am_arch["IsTargetAmArchit"] == True]["CosScore"].to_numpy()
    false_scores = df_temp_am_arch[df_temp_am_arch["IsTargetAmArchit"] == False]["CosScore"].to_numpy()
    eer_am_arch, _ = compute_eer(true_scores, false_scores)

    true_scores = df_temp_vm_arch[df_temp_vm_arch["IsTargetVmArchit"] == True]["CosScore"].to_numpy()
    false_scores = df_temp_vm_arch[df_temp_vm_arch["IsTargetAmArchit"] == False]["CosScore"].to_numpy()
    eer_vm_arch, _ = compute_eer(true_scores, false_scores)

    # Count occurrences for utt
    utt = {
        "true_att": len(df_temp_att[df_temp_att["IsTargetATK"] == True]),
        "false_att": len(df_temp_att[df_temp_att["IsTargetATK"] == False]),
        "true_AM": len(df_temp_am[df_temp_am["IsTargetAcousticModel"] == True]),
        "false_AM": len(df_temp_am[df_temp_am["IsTargetAcousticModel"] == False]),
        "true_VM": len(df_temp_vm[df_temp_vm["IsTargetVocoderModel"] == True]),
        "false_VM": len(df_temp_vm[df_temp_vm["IsTargetVocoderModel"] == False]),
        "true_AM_arch": len(df_temp_am_arch[df_temp_am_arch["IsTargetAmArchit"] == True]),
        "false_AM_arch": len(df_temp_am_arch[df_temp_am_arch["IsTargetAmArchit"] == False]),
        "true_VM_arch": len(df_temp_am_arch[df_temp_am_arch["IsTargetVmArchit"] == True]),
        "false_VM_arch": len(df_temp_vm_arch[df_temp_vm_arch["IsTargetVmArchit"] == False]),

    }

    return eer_att, eer_am, eer_vm, eer_am_arch, eer_vm_arch, utt


data = df.copy()

# Known attack types
kn_attk = ["AA01", "AA03", "AA05", "AA07", "AA10"]

kn_un = ["kn", "un"]
co_nc = ["-co-", "-nc-"]

# Initialize dictionaries for storing results
eer_kn_att, eer_kn_am, eer_kn_vm, eer_kn_am_arch, eer_kn_vm_arch = {}, {}, {}, {}, {}
eer_ukn_att, eer_ukn_am, eer_ukn_vm, eer_ukn_am_arch, eer_ukn_vm_arch = {}, {}, {}, {}, {}
utt, unk_utt = {}, {}

# Iterate over co/nc
for j, co_type in enumerate(co_nc):
    tempco = data[data["AbstractModel"].str.contains(co_type, na=False)].copy()

    if j == 0:
        dur = ["-1", "-10", "-100", "-400"]
    else:
        dur = ["-1", "-10", "-100", "-400", "-1000", "-all"]

    for k, duration in enumerate(dur):
        temp = tempco[tempco["AbstractModel"].str.endswith(duration, na=False)].copy()
        
        for i, kn_type in enumerate(kn_un):
            if kn_type == "kn":
                temp_kn = temp[temp["ATK"].isin(kn_attk)].copy()

                att_eer, am_eer, vm_eer, am_arch_eer, vm_arch_eer, utt1 = EER(temp_kn.copy())
                eer_kn_att[(j, k)] = att_eer
                eer_kn_am[(j, k)] = am_eer
                eer_kn_vm[(j, k)] = vm_eer
                eer_kn_am_arch[(j, k)] = am_arch_eer
                eer_kn_vm_arch[(j, k)] = vm_arch_eer
                utt[(j, k)] = utt1
                del temp_kn
            else:  # Unknown/unseen scenario
                temp_kn_att = temp[~temp["ATK"].isin(kn_attk) | (temp["IsTargetATK"] == True)]
                temp_kn_am = temp[~temp["ATK"].isin(kn_attk) | (temp["IsTargetAcousticModel"] == True)]
                temp_kn_vm = temp[~temp["ATK"].isin(kn_attk) | (temp["IsTargetVocoderModel"] == True)]
                temp_kn_am_arch = temp[~temp["ATK"].isin(kn_attk) | (temp["IsTargetAmArchit"] == True)]
                temp_kn_vm_arch = temp[~temp["ATK"].isin(kn_attk) | (temp["IsTargetVmArchit"] == True)]

                att_eer, am_eer, vm_eer, am_arch_eer, vm_arch_eer, utt2 = EER2(temp_kn_att.copy(), temp_kn_am.copy(), temp_kn_vm.copy(), temp_kn_am_arch.copy(), temp_kn_vm_arch.copy())
                eer_ukn_att[(j, k)] = att_eer
                eer_ukn_am[(j, k)] = am_eer
                eer_ukn_vm[(j, k)] = vm_eer
                eer_ukn_am_arch[(j, k)] = am_arch_eer
                eer_ukn_vm_arch[(j, k)] = vm_arch_eer
                unk_utt[(j, k)] = utt2
                del temp_kn_att
                del temp_kn_am
                del temp_kn_vm
                del temp_kn_am_arch
                del temp_kn_vm_arch


# Save EER values
def save_eer(eer_dict, filename):
    df = pd.DataFrame.from_dict(eer_dict, orient="index", columns=["EER"])
    df["EER"] = (df["EER"] * 100).round(2)  # Multiply by 100 and round to 2 decimals
    df.to_csv(Path(exp_dir)/ filename)

exp_dir = 'EERs_few_shot_siamese_network_CE'

save_eer(eer_kn_att, "eer_kn_att.csv")
save_eer(eer_kn_am, "eer_kn_am.csv")
save_eer(eer_kn_vm, "eer_kn_vm.csv")
save_eer(eer_kn_am_arch, "eer_kn_am_arch.csv")
save_eer(eer_kn_vm_arch, "eer_kn_vm_arch.csv")

save_eer(eer_ukn_att, "eer_ukn_att.csv")
save_eer(eer_ukn_am, "eer_ukn_am.csv")
save_eer(eer_ukn_vm, "eer_ukn_vm.csv")
save_eer(eer_ukn_am_arch, "eer_ukn_am_arch.csv")
save_eer(eer_ukn_vm_arch, "eer_ukn_vm_arch.csv")

pd.DataFrame.from_dict(utt, orient="index").to_csv(Path(exp_dir) / "utt_values.csv")
pd.DataFrame.from_dict(unk_utt, orient="index").to_csv(Path(exp_dir) / "unk_utt_values.csv")


# ----------------------------------------------------------------------------------------
# EXPLANATION OF GENERATED RESULTS
# ----------------------------------------------------------------------------------------
# This script computes the Equal Error Rate (EER) values for different detection tasks
# (attack detection, acoustic model detection, vocoder detection, etc.) under both
# **known** and **unknown** attack scenarios, for various training conditions.
#
# After execution, a directory will be created,
# containing multiple CSV files. Each file stores EER results (in percentage) for a
# specific detection type and scenario:
#
#   ┌───────────────────────────────┬─────────────────────────────────────────────┐
#   │  File Name                    │  Meaning                                   │
#   ├───────────────────────────────┼─────────────────────────────────────────────┤
#   │ eer_kn_att.csv                │ EER for **Known Attack Detection**          │
#   │ eer_kn_am.csv                 │ EER for **Known Acoustic Model Detection**  │
#   │ eer_kn_vm.csv                 │ EER for **Known Vocoder Detection**         │
#   │ eer_kn_am_arch.csv            │ EER for **Known Acoustic Model Architecture Detection** │
#   │ eer_kn_vm_arch.csv            │ EER for **Known Vocoder Architecture Detection**       │
#   │ eer_ukn_att.csv               │ EER for **Unknown Attack Detection**        │
#   │ eer_ukn_am.csv                │ EER for **Unknown Acoustic Model Detection**│
#   │ eer_ukn_vm.csv                │ EER for **Unknown Vocoder Detection**       │
#   │ eer_ukn_am_arch.csv           │ EER for **Unknown Acoustic Model Architecture Detection** │
#   │ eer_ukn_vm_arch.csv           │ EER for **Unknown Vocoder Architecture Detection**     │
#   │ utt_values.csv                │ Counts of utterances used in Known scenarios          │
#   │ unk_utt_values.csv            │ Counts of utterances used in Unknown scenarios        │
#   └───────────────────────────────┴─────────────────────────────────────────────┘
#
# Each of these CSV files has the following format:
#
#       ,EER
#       "(0, 0)",24.92
#       "(0, 1)",24.97
#       "(1, 2)",26.26
#       ...
#
# Here:
#   - The tuple index (e.g., "(0, 1)") corresponds to:
#         (j, k)
#       where:
#         j = 0 → "co" (content overlap condition)
#         j = 1 → "nc" (no content overlap condition)
#         k corresponds to the training set duration used:
#             For "co": ["-1", "-10", "-100", "-400"]
#             For "nc": ["-1", "-10", "-100", "-400", "-1000", "-all"]
#
# The EER value represents the **Equal Error Rate (%)**, a standard performance metric
# where the false acceptance rate equals the false rejection rate — lower values indicate
# better performance.
#
# When combined, these CSVs can be arranged to form a table like this:
#
#   EER%      | Known (or ID)                                    | Unknown (or OOD)
#              ├──────────────────────────────────────────────────────┼─────────────────────────────────────────────────────
#              | Attack | Acoustic Model | Vocoder | AM Arch | VM Arch | Attack | Acoustic Model | Vocoder | AM Arch | VM Arch
#   co   1-utt |  xx.xx | xx.xx          | xx.xx   | xx.xx   | xx.xx   | xx.xx  | xx.xx          | xx.xx   | xx.xx   | xx.xx
#   co  10-utt |  ...   | ...            | ...     | ...     | ...     | ...    | ...            | ...     | ...     | ...
#   nc  all    |  ...   | ...            | ...     | ...     | ...     | ...    | ...            | ...     | ...     | ...
#
# In summary:
# - "kn" = known attack types (seen during training)
# - "ukn" = unknown/unseen attack types
# - "att" = attack detection
# - "am" = acoustic model detection
# - "vm" = vocoder model detection
# - "am_arch" = acoustic model architecture detection
# - "vm_arch" = vocoder architecture detection
# - "co" = content-overlap condition
# - "nc" = no-content-overlap condition
# - Training durations: -1, -10, -100, -400, -1000, -all correspond to
#   the number of utterances used during training.
#
# ----------------------------------------------------------------------------------------
