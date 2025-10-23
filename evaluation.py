import sys
import os
import numpy as np
np.float = float


# ===========================
#   SCORE PARSING UTILITIES
# ===========================
def parse_score_file(score_file):
    """Parse the score file and return target and non-target scores for each class."""
    scores_per_class = {f"AA{i:02d}": {"target": [], "nontarget": []} for i in range(1, 14)}
    all_target_scores = []
    all_nontarget_scores = []

    with open(score_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            score = float(parts[-1])
            attack = parts[1]

            if attack.startswith("AA") and attack in scores_per_class:
                target_class = attack
                scores_per_class[target_class]["target"].append(score)
                all_target_scores.append(score)

                # Add score as non-target for other classes
                for cls in scores_per_class:
                    if cls != target_class:
                        scores_per_class[cls]["nontarget"].append(score)
                        all_nontarget_scores.append(score)

    return scores_per_class, np.array(all_target_scores), np.array(all_nontarget_scores)


# ===========================
#   EER COMPUTATION
# ===========================
def compute_det_curve(target_scores, nontarget_scores):
    """Compute detection error trade-off (DET) curve."""
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """Returns equal error rate (EER) and the corresponding threshold."""
    if len(target_scores) == 0 or len(nontarget_scores) == 0:
        print("Error: Empty target or non-target scores.")
        return np.nan, np.nan

    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


def compute_eer_from_file(score_file):
    """Computes EER for each attack class from the score file using one-vs-rest."""
    scores_per_class, all_target_scores, all_nontarget_scores = parse_score_file(score_file)

    eer_results = {}
    for attack_class, scores in scores_per_class.items():
        target_scores = np.array(scores["target"])
        nontarget_scores = np.array(scores["nontarget"])
        eer, threshold = compute_eer(target_scores, nontarget_scores)
        eer_results[attack_class] = {"eer": eer, "threshold": threshold}

    overall_eer, overall_threshold = compute_eer(all_target_scores, all_nontarget_scores)
    eer_results['overall'] = {"eer": overall_eer, "threshold": overall_threshold}

    return eer_results, overall_eer


def save_eer_results(eer_results, output_file, epoch):
    """Append the EER results of each epoch to a text file."""
    with open(output_file, "a") as f:
        f.write(f"\n===== Epoch {epoch} =====\n")
        for attack_class, result in eer_results.items():
            eer = result["eer"]
            threshold = result["threshold"]
            if np.isnan(eer):
                f.write(f"{attack_class}: Failed to compute EER due to missing or invalid data.\n")
            else:
                f.write(f"{attack_class} - EER: {eer * 100:.2f}% at threshold: {threshold}\n")
        f.write("=" * 30 + "\n")


# ===========================
#   ASV + CM EVALUATION
# ===========================
def calculate_tDCF_EER(cm_scores_file, asv_score_file, output_file, printout=True):
    """Compute t-DCF and EER metrics for combined ASV and CM systems."""
    Pspoof = 0.05
    cost_model = {
        'Pspoof': Pspoof,
        'Ptar': (1 - Pspoof) * 0.99,
        'Pnon': (1 - Pspoof) * 0.01,
        'Cmiss': 1,
        'Cfa': 10,
        'Cmiss_asv': 1,
        'Cfa_asv': 10,
        'Cmiss_cm': 1,
        'Cfa_cm': 10,
    }

    asv_data = np.genfromtxt(asv_score_file, dtype=str)
    asv_keys = asv_data[:, 1]
    asv_scores = asv_data[:, 2].astype(np.float)

    cm_data = np.genfromtxt(cm_scores_file, dtype=str)
    cm_sources = cm_data[:, 1]
    cm_keys = cm_data[:, 2]
    cm_scores = cm_data[:, 3].astype(np.float)

    tar_asv = asv_scores[asv_keys == 'target']
    non_asv = asv_scores[asv_keys == 'nontarget']
    spoof_asv = asv_scores[asv_keys == 'spoof']

    bona_cm = cm_scores[cm_keys == 'bonafide']
    spoof_cm = cm_scores[cm_keys == 'spoof']

    eer_asv, asv_threshold = compute_eer(tar_asv, non_asv)
    eer_cm = compute_eer(bona_cm, spoof_cm)[0]

    attack_types = [f'A{_id:02d}' for _id in range(7, 20)]
    if printout:
        spoof_cm_breakdown = {atk: cm_scores[cm_sources == atk] for atk in attack_types}
        eer_cm_breakdown = {atk: compute_eer(bona_cm, spoof_cm_breakdown[atk])[0] for atk in attack_types}

    Pfa_asv, Pmiss_asv, Pmiss_spoof_asv = obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)

    tDCF_curve, CM_thresholds = compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, print_cost=False)
    min_tDCF = np.min(tDCF_curve)

    if printout:
        with open(output_file, "w") as f_res:
            f_res.write('\nCM SYSTEM\n')
            f_res.write(f'\tEER\t\t= {eer_cm * 100:8.9f} % (Equal error rate for countermeasure)\n')
            f_res.write('\nTANDEM\n')
            f_res.write(f'\tmin-tDCF\t\t= {min_tDCF:8.9f}\n')
            f_res.write('\nBREAKDOWN CM SYSTEM\n')
            for atk in attack_types:
                _eer = eer_cm_breakdown[atk] * 100
                f_res.write(f'\tEER {atk}\t\t= {_eer:8.9f} % (Equal error rate for {atk}\n')
        os.system(f"cat {output_file}")

    return eer_cm * 100, min_tDCF


def obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold):
    """Compute ASV system error rates."""
    Pfa_asv = sum(non_asv >= asv_threshold) / non_asv.size
    Pmiss_asv = sum(tar_asv < asv_threshold) / tar_asv.size

    if spoof_asv.size == 0:
        Pmiss_spoof_asv = None
    else:
        Pmiss_spoof_asv = np.sum(spoof_asv < asv_threshold) / spoof_asv.size

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv


def compute_tDCF(bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv,
                 Pmiss_spoof_asv, cost_model, print_cost):
    """Compute Tandem Detection Cost Function (t-DCF) for ASV+CM systems."""
    if cost_model['Cfa_asv'] < 0 or cost_model['Cmiss_asv'] < 0 or \
            cost_model['Cfa_cm'] < 0 or cost_model['Cmiss_cm'] < 0:
        print('WARNING: Cost values should be positive!')

    if cost_model['Ptar'] < 0 or cost_model['Pnon'] < 0 or cost_model['Pspoof'] < 0 or \
            np.abs(cost_model['Ptar'] + cost_model['Pnon'] + cost_model['Pspoof'] - 1) > 1e-10:
        sys.exit('ERROR: Your prior probabilities should be positive and sum to one.')

    if Pmiss_spoof_asv is None:
        sys.exit('ERROR: Missing ASV spoof miss rate.')

    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit('ERROR: Scores contain NaN or inf.')

    if np.unique(combined_scores).size < 3:
        sys.exit('ERROR: Provide soft CM scores, not binary decisions.')

    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(bonafide_score_cm, spoof_score_cm)

    C1 = cost_model['Ptar'] * (cost_model['Cmiss_cm'] - cost_model['Cmiss_asv'] * Pmiss_asv) - \
         cost_model['Pnon'] * cost_model['Cfa_asv'] * Pfa_asv
    C2 = cost_model['Cfa_cm'] * cost_model['Pspoof'] * (1 - Pmiss_spoof_asv)

    if C1 < 0 or C2 < 0:
        sys.exit('ERROR: tDCF weights are negative — check ASV error rates.')

    tDCF = C1 * Pmiss_cm + C2 * Pfa_cm
    tDCF_norm = tDCF / np.minimum(C1, C2)
    return tDCF_norm, CM_thresholds


# ===========================
#   CUSTOM EER COMPUTATION
# ===========================
def compute_EER_custom(output, label):
    """
    Compute the Equal Error Rate (EER) for the entire output and each attribute/class.

    Parameters:
        output (np.ndarray): 2D array (samples x classes)
        label (np.ndarray): 1D array of class labels

    Returns:
        float: Overall EER (%)
        list: Class-wise EERs (%)
    """
    target = np.zeros(output.shape)
    _, inverse_indices = np.unique(label, return_inverse=True)
    target[np.arange(len(inverse_indices)), inverse_indices] = 1

    true_score = np.array([])
    false_score = np.array([])
    AS_eer = []

    for i in range(output.shape[1]):
        ts = output[target[:, i] == 1, i]
        fs = output[target[:, i] != 1, i]

        true_score = np.concatenate((true_score, ts))
        false_score = np.concatenate((false_score, fs))

        eer_indi = compute_eer(ts, fs)[0]
        AS_eer.append(eer_indi * 100)

    eer = compute_eer(true_score, false_score)[0]
    return eer * 100, AS_eer
