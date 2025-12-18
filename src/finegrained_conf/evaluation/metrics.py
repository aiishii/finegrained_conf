from typing import Iterable, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold
import math
import warnings
from scipy.optimize import fminbound


def compute_roc_pr_auc(
    y_true: Iterable[int],
    y_score: Iterable[float],
    debug=False
) -> Tuple[float, float]:
    """
    Compute ROC-AUC and PR-AUC for a binary detection task.

    y_true: iterable of 0/1 labels, where 1 means "should be detected" (e.g., incorrect answer/triple).
    y_score: iterable of scores; higher means "more likely to be positive (incorrect)".
    Returns (roc_auc, pr_auc).
    """
    y_true_arr = np.asarray(list(y_true))
    y_score_arr = np.asarray(list(y_score))

    # Drop NaNs if necessary
    mask = ~np.isnan(y_score_arr)
    y_true_arr = y_true_arr[mask]
    y_score_arr = y_score_arr[mask]

    if len(np.unique(y_true_arr)) < 2:
        # Not enough positive/negative examples to compute AUC
        return float("nan"), float("nan")

    fpr, tpr, _ = roc_curve(y_true_arr, y_score_arr)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true_arr, y_score_arr)
    pr_auc = auc(recall, precision)

    if debug:
        return roc_auc, pr_auc, len(y_true_arr)
    else:
        return roc_auc, pr_auc


def compute_spurious_roc_pr_auc_for_questions(
    answer_correctness: Sequence[int],
    evidence_correctness: Sequence[int],
    triple_confidences_per_question: Sequence[Sequence[float]],
    debug=False
) -> Tuple[float, float]:
    """
    Compute ROC-AUC and PR-AUC for spurious correctness detection

    Only questions with correct answers are considered. For those questions,
    a spurious label is 1 when the evidence is not fully correct and 0 when
    both answer and evidence are correct. The evidence confidence for a
    question is defined as the minimum triple confidence for that question.
    Low evidence confidence should correspond to a higher spurious score, so
    ``1 - min_confidence`` is used as the score passed to the ROC/PR helpers.
    """

    labels: list[int] = []
    scores: list[float] = []

    for ans_corr, evid_corr, triple_confs in zip(
        answer_correctness, evidence_correctness, triple_confidences_per_question
    ):
        ans_corr_val = to_float_or_nan(ans_corr)
        evid_corr_val = to_float_or_nan(evid_corr)

        if ans_corr_val != 1 or math.isnan(evid_corr_val):
            continue

        try:
            conf_iterable = list(triple_confs)
        except TypeError:
            continue

        conf_values = [to_float_or_nan(c) for c in conf_iterable]
        conf_values = [c for c in conf_values if not math.isnan(c)]

        if not conf_values:
            continue

        min_confidence = float(np.min(conf_values))
        labels.append(1 if evid_corr_val != 1 else 0)
        scores.append(1 - min_confidence)

    if not labels:
        if debug: return float("nan"), float("nan"), 0
        else: return float("nan"), float("nan")
    
    return compute_roc_pr_auc(labels, scores, debug=debug)

def to_float_or_nan(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan
    
def compute_ece(confidences, correctness, num_bins: int = 10) -> float:
    conf = np.array([to_float_or_nan(x) for x in confidences], dtype=float)
    corr = np.array([to_float_or_nan(x) for x in correctness], dtype=float)
    
    mask = ~np.isnan(conf) & ~np.isnan(corr) & (conf >= 0) & (conf <= 1)
    conf, corr = conf[mask], corr[mask]

    if conf.size == 0:
        return np.nan
    
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_idx = np.digitize(conf, bin_edges[1:-1], right=True)  # 0〜num_bins-1
    
    bin_counts = np.bincount(bin_idx, minlength=num_bins)
    
    non_empty = bin_counts > 0
    
    bin_conf = np.zeros(num_bins)
    bin_acc = np.zeros(num_bins)
    
    if np.any(non_empty):
        bin_conf[non_empty] = (
            np.bincount(bin_idx, weights=conf, minlength=num_bins)[non_empty] 
            / bin_counts[non_empty]
        )
        bin_acc[non_empty] = (
            np.bincount(bin_idx, weights=corr, minlength=num_bins)[non_empty] 
            / bin_counts[non_empty]
        )
    
    bin_frac = bin_counts / conf.size
    
    ece = np.sum(bin_frac * np.abs(bin_acc - bin_conf))
    
    return float(ece)

def temperature_scaling_standard(logits, temperature):
    import numpy as np
    
    scaled_logits = logits / temperature
    
    if isinstance(scaled_logits, (int, float)):
        return 1 / (1 + np.exp(-scaled_logits))
    
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    return exp_logits / np.sum(exp_logits)

def temperature_scaling(confidences, temperature: float):
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    
    conf_arr = np.array([to_float_or_nan(x) for x in confidences], dtype=float)
    # conf_arr = np.asarray(confidences, dtype=float)
    nan_mask = np.isnan(conf_arr)
    
    valid_mask = ~nan_mask & (conf_arr >= 0) & (conf_arr <= 1)
    
    if not np.any(valid_mask):
        return conf_arr
    
    eps = np.finfo(conf_arr.dtype).eps * 10 
    conf_clipped = np.clip(conf_arr[valid_mask], eps, 1 - eps)
    logits = np.log(conf_clipped / (1 - conf_clipped))
    
    scaled_logits = logits / temperature
    max_logit = 88.0 if conf_arr.dtype == np.float32 else 700.0
    scaled_logits = np.clip(scaled_logits, -max_logit, max_logit) 
    scaled = 1.0 / (1.0 + np.exp(-scaled_logits))
    
    out = np.empty_like(conf_arr)
    out[:] = np.nan
    out[valid_mask] = scaled
    
    invalid_mask = ~nan_mask & ~valid_mask
    if np.any(invalid_mask):
        n_invalid = np.sum(invalid_mask)
        print(f"警告: {n_invalid}個の値が[0,1]の範囲外です。NaNとして扱います。")
    
    return out

def _nll(T, p, y):
    p = np.clip(p, 1e-6, 1-1e-6)
    logit = np.log(p/(1-p))
    z = logit / T
    pT = 1/(1+np.exp(-z))
    return -np.mean(y*np.log(pT)+(1-y)*np.log1p(-pT))

def find_optimal_temperature(confidences, correctness, num_bins=10):
    valid_data = []
    for conf, corr in zip(confidences, correctness):
        try:
            conf_float = float(conf) if conf is not None and (isinstance(conf, float) and not math.isnan(conf)) else None
            corr_float = float(corr) if corr is not None else None
            
            if conf_float is not None and corr_float is not None:
                valid_data.append((conf_float, corr_float))
        except (ValueError, TypeError):
            continue
    
    if not valid_data:
        print("警告: 有効なデータがありません。デフォルトの温度1.0を返します。")
        return 1.0
    
    valid_confidences, valid_correctness = zip(*valid_data)
    
    temperatures = np.linspace(0.1, 5.0, 50)
    min_nll = float('inf')
    best_temp = 1.0
    
    best_temp = fminbound(_nll, 0.05, 10.0, args=(np.array(valid_confidences),
                                              np.array(valid_correctness)),
                      xtol=1e-3)
    
    return best_temp

def compute_temperature_scaled_ece_with_cv(confidences,
                                           correctness,
                                           num_bins: int = 10,
                                           n_splits: int = 5):

    import numpy as np
    from sklearn.model_selection import KFold, LeaveOneOut

    # conf = np.asarray(confidences, dtype=float)
    # corr = np.asarray(correctness,  dtype=float)
    conf = np.array([to_float_or_nan(x) for x in confidences], dtype=float)
    corr = np.array([to_float_or_nan(x) for x in correctness], dtype=float)
    mask = ~np.isnan(conf) & ~np.isnan(corr) & (conf >= 0) & (conf <= 1)
    conf, corr = conf[mask], corr[mask]
    n = len(conf)

    if n < 2:
        return np.nan, np.nan, np.nan, [], [], []

    if n <= 10:
        cv = LeaveOneOut()
    else:
        k = min(n_splits, n) if n < n_splits else n_splits
        cv = KFold(k, shuffle=True, random_state=42)

    ece_vals, temps, fold_details = [], [], []

    for i, (tr, te) in enumerate(cv.split(conf)):
        tr_conf, tr_corr = conf[tr], corr[tr]
        te_conf, te_corr = conf[te], corr[te]

        T_opt = find_optimal_temperature(tr_conf, tr_corr, num_bins)
        temps.append(T_opt)

        te_scaled = temperature_scaling(te_conf, T_opt)

        ece = compute_ece(te_scaled, te_corr, num_bins)
        ece_vals.append(ece)
        fold_details.append(f"Fold {i+1}: ECE = {ece:.4f}, Temp = {T_opt:.4f}")

    mean_ece = float(np.mean(ece_vals))
    mean_T   = float(np.mean(temps))
    std_T    = float(np.std(temps))

    return mean_ece, mean_T, std_T, ece_vals, temps, fold_details

def compute_brier_score(confidences, correctness):
    valid_data = []
    for conf, corr in zip(confidences, correctness):
        try:
            # conf_float = float(conf) if conf is not None else None
            conf_float = float(conf) if conf is not None and (isinstance(conf, float) and not math.isnan(conf)) else None
            corr_float = float(corr) if corr is not None else None
            
            if conf_float is not None and corr_float is not None:
                valid_data.append((conf_float, corr_float))
        except (ValueError, TypeError):
            continue
    
    if not valid_data:
        print("警告: 有効なデータがありません。デフォルトのBrier Score 0.25を返します。")
        return 0.25
    
    valid_confidences, valid_correctness = zip(*valid_data)
    
    brier_score = np.mean([(conf - corr)**2 for conf, corr in zip(valid_confidences, valid_correctness)])
    
    return brier_score

import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold
import warnings

def safe_cv(confidences, correctness, metric_func,
           n_splits=5, classification=True, random_state=42):
    # conf = np.asarray(confidences, float)
    # corr = np.asarray(correctness , float)
    conf = np.array([to_float_or_nan(x) for x in confidences], dtype=float)
    corr = np.array([to_float_or_nan(x) for x in correctness], dtype=float)
    mask = ~np.isnan(conf) & ~np.isnan(corr) & (conf>=0) & (conf<=1)
    conf, corr = conf[mask], corr[mask]
    n = len(conf)

    if n < 2:
        warnings.warn("サンプル数が2未満: 交差検証不能")
        return np.nan, np.nan, []

    if n <= 10:
        warnings.warn(f"n={n} なので Leave-One-Out を使用")
        cv = LeaveOneOut()
    else:
        k = min(n_splits, n) if n < n_splits else n_splits
        if classification and len(np.unique(corr)) > 1:
            cv = StratifiedKFold(k, shuffle=True, random_state=random_state)
        else:
            cv = KFold(k, shuffle=True, random_state=random_state)

    vals = []
    for tr, te in cv.split(conf, corr if classification else None):
        try:
            vals.append(metric_func(conf[te], corr[te]))
        except Exception as e:
            warnings.warn(f"metric error: {e}")

    return np.nanmean(vals), np.nanstd(vals), vals

def compute_temperature_scaled_bs_with_cv(confidences,
                                          correctness,
                                          n_splits: int = 5):
    import numpy as np
    from sklearn.model_selection import KFold, LeaveOneOut
    conf = np.array([to_float_or_nan(x) for x in confidences], dtype=float)
    corr = np.array([to_float_or_nan(x) for x in correctness], dtype=float)
    # conf = np.asarray(confidences, dtype=float)
    # corr = np.asarray(correctness,  dtype=float)
    mask = ~np.isnan(conf) & ~np.isnan(corr) & (conf >= 0) & (conf <= 1)
    conf, corr = conf[mask], corr[mask]
    n = len(conf)

    if n < 2:
        return np.nan, np.nan, np.nan, [], [], []

    cv = LeaveOneOut() if n <= 10 else KFold(min(n_splits, n), shuffle=True, random_state=42)

    bs_vals, temps, fold_details = [], [], []

    for i, (tr, te) in enumerate(cv.split(conf)):
        T_opt = find_optimal_temperature(conf[tr], corr[tr])
        temps.append(T_opt)

        te_scaled = temperature_scaling(conf[te], T_opt)
        bs = compute_brier_score(te_scaled, corr[te])
        bs_vals.append(bs)

        fold_details.append(f"Fold {i+1}: BS = {bs:.4f}, Temp = {T_opt:.4f}")

    mean_bs = float(np.mean(bs_vals))
    mean_T  = float(np.mean(temps))
    std_T   = float(np.std(temps))

    return mean_bs, mean_T, std_T, bs_vals, temps, fold_details

def evaluate_calibration(confidences, correctness, model_name="Model"):
    """
    モデルのキャリブレーション評価を行い、結果を表示する
    
    Args:
        confidences: 信頼度のリスト
        correctness: 正誤のリスト
        model_name: モデル名（表示用）
    """
    raw_ece = compute_ece(confidences, correctness)
    
    ece_t, mean_temp_ece, std_temp_ece, ece_values, ece_temps, ece_fold_details = compute_temperature_scaled_ece_with_cv(confidences, correctness)
    
    # Brier Score
    raw_bs = compute_brier_score(confidences, correctness)
    
    bs_t, mean_temp_bs, std_temp_bs, bs_values, bs_temps, bs_fold_details = compute_temperature_scaled_bs_with_cv(confidences, correctness)
    
    print(f"{model_name} の評価結果:")
    print(f"Raw ECE: {raw_ece:.4f}")
    print(f"Temperature-scaled ECE: {ece_t:.4f} (温度平均: {mean_temp_ece:.4f} ± {std_temp_ece:.4f})")
    print("ECE 5分割の詳細:")
    for detail in ece_fold_details:
        print(f"  {detail}")
    
    print(f"Raw Brier Score: {raw_bs:.4f}")
    print(f"Temperature-scaled Brier Score: {bs_t:.4f} (温度平均: {mean_temp_bs:.4f} ± {std_temp_bs:.4f})")
    print("BS 5分割の詳細:")
    for detail in bs_fold_details:
        print(f"  {detail}")
    
    return {
        "raw_ece": raw_ece,
        "ece_t": ece_t,
        "ece_temperatures": ece_temps,
        "mean_temp_ece": mean_temp_ece,
        "std_temp_ece": std_temp_ece,
        "ece_fold_values": ece_values,
        "raw_bs": raw_bs,
        "bs_t": bs_t,
        "bs_temperatures": bs_temps,
        "mean_temp_bs": mean_temp_bs,
        "std_temp_bs": std_temp_bs,
        "bs_fold_values": bs_values
    }

from sklearn.metrics import roc_auc_score

def compute_selective_classification_auc(confidences, correctness):
    try:
        valid_data = []
        for conf, corr in zip(confidences, correctness):
            try:
                conf_float = float(conf) if conf is not None else None
                corr_float = float(corr) if corr is not None else None
                
                if conf_float is not None and corr_float is not None:
                    valid_data.append((conf_float, corr_float))
            except (ValueError, TypeError):
                continue
        
        if not valid_data:
            print("警告: 有効なデータがありません。AUCは0.5を返します。")
            return 0.5
        
        valid_data.sort(key=lambda x: x[0], reverse=True)
        
        confidences_sorted, correctness_sorted = zip(*valid_data)
        
        total_samples = len(correctness_sorted)
        accuracies = []
        coverages = []
        
        correct_count = 0
        for i, correct in enumerate(correctness_sorted):
            coverage = (i + 1) / total_samples
            correct_count += correct
            accuracy = correct_count / (i + 1)
            
            accuracies.append(accuracy)
            coverages.append(coverage)
        
        auc = 0.0
        for i in range(1, len(coverages)):
            width = coverages[i] - coverages[i-1]
            height = (accuracies[i] + accuracies[i-1]) / 2
            auc += width * height
        
        return auc
    
    except Exception as e:
        print(f"Selective Classification AUC計算エラー: {e}")
        return 0.5


def compute_auroc_for_correctness(confidences, correctness):
    from sklearn.metrics import roc_auc_score
    import numpy as np
    
    valid_data = []
    for conf, corr in zip(confidences, correctness):
        try:
            if conf is not None and not (isinstance(conf, float) and np.isnan(conf)):
                conf_float = float(conf)
                corr_float = float(corr) if corr is not None else None
                
                if corr_float is not None and 0 <= conf_float <= 1:
                    valid_data.append((conf_float, int(corr_float > 0.5)))
        except (ValueError, TypeError):
            continue
    
    if not valid_data or len(set([d[1] for d in valid_data])) <= 1:
        return 0.5
    
    confs, labels = zip(*valid_data)
    return roc_auc_score(labels, confs)


def compute_aurc(confidences, correctness):
    """
    AURC (Area Under Risk-Coverage curve)
    """
    import numpy as np
    
    valid_data = []
    for conf, corr in zip(confidences, correctness):
        try:
            if conf is not None and not (isinstance(conf, float) and np.isnan(conf)):
                conf_float = float(conf)
                corr_float = float(corr) if corr is not None else None
                
                if corr_float is not None and 0 <= conf_float <= 1:
                    valid_data.append((conf_float, corr_float))
        except (ValueError, TypeError):
            continue
    
    if not valid_data:
        return 0.5
    
    valid_data.sort(key=lambda x: x[0], reverse=True)
    confidences_sorted, correctness_sorted = zip(*valid_data)
    
    n_samples = len(correctness_sorted)
    risks = []
    coverages = []
    
    cumsum_correct = 0
    for i in range(n_samples):
        cumsum_correct += correctness_sorted[i]
        coverage = (i + 1) / n_samples
        accuracy = cumsum_correct / (i + 1)
        risk = 1 - accuracy
        
        risks.append(risk)
        coverages.append(coverage)
    
    aurc = 0.0
    for i in range(1, len(coverages)):
        width = coverages[i] - coverages[i-1]
        height = (risks[i] + risks[i-1]) / 2
        aurc += width * height
    
    return aurc

def selective_auc(confidences, correctness):
    """
    Selective Classification AUC
    """
    conf = np.array([to_float_or_nan(x) for x in confidences], dtype=float)
    corr = np.array([to_float_or_nan(x) for x in correctness], dtype=float)
    # conf = np.asarray(confidences, dtype=float)
    # corr = np.asarray(correctness, dtype=float)
    
    m = ~np.isnan(conf) & ~np.isnan(corr)
    conf, corr = conf[m], corr[m]
    
    if len(conf) == 0:
        return np.nan
    
    order = np.argsort(-conf)
    corr_sorted = corr[order]
    
    cum_correct = np.cumsum(corr_sorted)
    coverage = np.arange(1, len(corr_sorted)+1) / len(corr_sorted)
    accuracy = cum_correct / np.arange(1, len(corr_sorted)+1)
    
    auc = np.trapz(accuracy, coverage)
    return auc

def plot_calibration(confidences, correctness, num_bins=10, title="Calibration Plot"):
    plt.figure(figsize=(8, 6))
    
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges)
    
    bin_accuracies = []
    bin_confidences = []
    bin_sizes = []
    
    for bin_idx in range(1, num_bins + 1):
        bin_samples = np.where(bin_indices == bin_idx)[0]
        if len(bin_samples) == 0:
            bin_accuracies.append(0)
            bin_confidences.append((bin_edges[bin_idx-1] + bin_edges[bin_idx]) / 2)
            bin_sizes.append(0)
            continue
        
        bin_size = len(bin_samples)
        
        bin_confidence = np.mean([confidences[i] for i in bin_samples])
        
        bin_accuracy = np.mean([correctness[i] for i in bin_samples])
        
        bin_accuracies.append(bin_accuracy)
        bin_confidences.append(bin_confidence)
        bin_sizes.append(bin_size)
    
    plt.bar(range(num_bins), bin_accuracies, width=0.8, alpha=0.5, 
           color='b', label='Accuracy')
    
    plt.plot([0, num_bins-1], [0, 1], 'k--', label='Perfect Calibration')
    
    plt.xticks(range(num_bins), [f'{i/num_bins:.1f}' for i in range(num_bins)])
    plt.xlabel('Confidence Bin')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    
    ece = compute_ece(confidences, correctness, num_bins)
    plt.text(0.05, 0.95, f'ECE: {ece:.4f}', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.show()


def align_conf_corr(conf_nested, corr_nested, strict=False):
    conf_list = list(conf_nested)
    corr_list = list(corr_nested)

    keep_idx = []
    for i, row in enumerate(corr_list):
        try:
            is_empty = (len(row) == 0)
        except TypeError:
            is_empty = True
        if not is_empty:
            keep_idx.append(i)

    corr_noempty = [corr_list[i] for i in keep_idx]

    if len(conf_list) == len(corr_noempty) and len(conf_list) > 0:
        conf_iter, corr_iter = conf_list, corr_noempty
    else:
        conf_iter, corr_iter = conf_list, corr_list

    conf_flat, corr_flat = [], []
    for conf_row, corr_row in zip(conf_iter, corr_iter):
        if strict and len(conf_row) != len(corr_row):
            continue

        for c, r in zip(conf_row, corr_row):
            try:
                c_f, r_f = float(c), float(r)
                if not (math.isnan(c_f) or math.isnan(r_f)):
                    conf_flat.append(c_f)
                    corr_flat.append(r_f)
            except (TypeError, ValueError):
                pass
    return np.array(conf_flat, dtype=float), np.array(corr_flat, dtype=float)

def compute_metrics(confidences, correctness, model_name=None):
    raw_ece = compute_ece(confidences, correctness)
    
    ece_t, mean_temp_ece, std_temp_ece, ece_values, ece_temps, ece_fold_details = compute_temperature_scaled_ece_with_cv(confidences, correctness)
    
    # Brier Score
    raw_bs = compute_brier_score(confidences, correctness)
    
    bs_t, mean_temp_bs, std_temp_bs, bs_values, bs_temps, bs_fold_details = compute_temperature_scaled_bs_with_cv(confidences, correctness)
    
    # auc = compute_selective_classification_auc(confidences, correctness)
    auc = selective_auc(confidences, correctness)

    conf_arr = np.array([to_float_or_nan(x) for x in confidences], dtype=float)
    corr_arr = np.array([to_float_or_nan(x) for x in correctness], dtype=float)
    valid_mask = ~np.isnan(conf_arr) & ~np.isnan(corr_arr)
    # Convert to binary labels: 1 if incorrect (corr < 1.0), 0 if correct (corr == 1.0)
    incorrect_labels = (corr_arr[valid_mask] < 1.0).astype(int)
    incorrect_scores = 1 - conf_arr[valid_mask]
    roc_auc, pr_auc, roc_pr_processed_num = compute_roc_pr_auc(incorrect_labels, incorrect_scores, debug=True)

    print(f"{model_name} の評価結果:")
    print(f"Raw ECE: {raw_ece:.4f}")
    print(f"Temperature-scaled ECE: {ece_t:.4f} (温度平均: {mean_temp_ece:.4f} ± {std_temp_ece:.4f})")
    print("ECE 5分割の詳細:")
    for detail in ece_fold_details:
        print(f"  {detail}")
    
    print(f"Raw Brier Score: {raw_bs:.4f}")
    print(f"Temperature-scaled Brier Score: {bs_t:.4f} (温度平均: {mean_temp_bs:.4f} ± {std_temp_bs:.4f})")
    print("BS 5分割の詳細:")
    for detail in bs_fold_details:
        print(f"  {detail}")
    
    return {
        "ece": raw_ece,
        "ece_t": ece_t,
        "ece_temperatures": ece_temps,
        "mean_temp_ece": mean_temp_ece,
        "std_temp_ece": std_temp_ece,
        "ece_fold_values": ece_values,
        "raw_bs": raw_bs,
        "bs_t": bs_t,
        "bs_temperatures": bs_temps,
        "mean_temp_bs": mean_temp_bs,
        "std_temp_bs": std_temp_bs,
        "bs_fold_values": bs_values,
        "auc": auc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        'roc_pr_processed_num': roc_pr_processed_num,
    }