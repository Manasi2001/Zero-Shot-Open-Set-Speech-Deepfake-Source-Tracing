"""
USE FOR COMPUTING EERs FOR MLP BACKEND

"""

import warnings
import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
from evaluation import compute_eer

warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("evaluation_files_with_scores/evaluation_mlp.csv")

def EER(df_temp):
    true_scores = df_temp[df_temp["IsTargetATK"] == True]["MLPScore"].to_numpy()
    false_scores = df_temp[df_temp["IsTargetATK"] == False]["MLPScore"].to_numpy()
    eer_att, _ = compute_eer(true_scores, false_scores)

    true_scores = df_temp[df_temp["IsTargetAcousticModel"] == True]["MLPScore"].to_numpy()
    false_scores = df_temp[df_temp["IsTargetAcousticModel"] == False]["MLPScore"].to_numpy()
    eer_am, _ = compute_eer(true_scores, false_scores)

    true_scores = df_temp[df_temp["IsTargetVocoderModel"] == True]["MLPScore"].to_numpy()
    false_scores = df_temp[df_temp["IsTargetVocoderModel"] == False]["MLPScore"].to_numpy()
    eer_vm, _ = compute_eer(true_scores, false_scores)

    true_scores = df_temp[df_temp["IsTargetAmArchit"] == True]["MLPScore"].to_numpy()
    false_scores = df_temp[df_temp["IsTargetAmArchit"] == False]["MLPScore"].to_numpy()
    eer_am_arch, _ = compute_eer(true_scores, false_scores)

    true_scores = df_temp[df_temp["IsTargetVmArchit"] == True]["MLPScore"].to_numpy()
    false_scores = df_temp[df_temp["IsTargetVmArchit"] == False]["MLPScore"].to_numpy()
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
    true_scores = df_temp_att[df_temp_att["IsTargetATK"] == True]["MLPScore"].to_numpy()
    false_scores = df_temp_att[df_temp_att["IsTargetATK"] == False]["MLPScore"].to_numpy()
    eer_att, _ = compute_eer(true_scores, false_scores)

    true_scores = df_temp_am[df_temp_am["IsTargetAcousticModel"] == True]["MLPScore"].to_numpy()
    false_scores = df_temp_am[df_temp_am["IsTargetAcousticModel"] == False]["MLPScore"].to_numpy()
    eer_am, _ = compute_eer(true_scores, false_scores)

    true_scores = df_temp_vm[df_temp_vm["IsTargetVocoderModel"] == True]["MLPScore"].to_numpy()
    false_scores = df_temp_vm[df_temp_vm["IsTargetVocoderModel"] == False]["MLPScore"].to_numpy()
    eer_vm, _ = compute_eer(true_scores, false_scores)

    true_scores = df_temp_am_arch[df_temp_am_arch["IsTargetAmArchit"] == True]["MLPScore"].to_numpy()
    false_scores = df_temp_am_arch[df_temp_am_arch["IsTargetAmArchit"] == False]["MLPScore"].to_numpy()
    eer_am_arch, _ = compute_eer(true_scores, false_scores)

    true_scores = df_temp_vm_arch[df_temp_vm_arch["IsTargetVmArchit"] == True]["MLPScore"].to_numpy()
    false_scores = df_temp_vm_arch[df_temp_vm_arch["IsTargetAmArchit"] == False]["MLPScore"].to_numpy()
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
co_nc = ["-nc-"] 

# Initialize dictionaries for storing results
eer_kn_att, eer_kn_am, eer_kn_vm, eer_kn_am_arch, eer_kn_vm_arch = {}, {}, {}, {}, {}
eer_ukn_att, eer_ukn_am, eer_ukn_vm, eer_ukn_am_arch, eer_ukn_vm_arch = {}, {}, {}, {}, {}
utt, unk_utt = {}, {}

# Iterate over co/nc
for j, co_type in enumerate(co_nc):
    tempco = data[data["AbstractModel"].str.contains(co_type, na=False)].copy()
    
    dur = ["-all"]

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

exp_dir = 'Attribute_based_AntiSpoof/Anton_Dataset_Codes/zero_shot_open_set_GitHub/EERs_mlp'

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