# ------------------------------------------------------------------------------
# Collection of metrics to compare whole 1-channel segmentation masks.
# Metrics receive two 1-channel integer arrays.
# ------------------------------------------------------------------------------
import pdb

import numpy as np
import torch
import mp.eval.metrics.scores as score_defs

# from mp.eval.metrics.scores import hausdorff_distance


def get_tp_tn_fn_fp_segmentation(target, pred, class_ix=1):
    r"""Get TP, TN, FN and FP pixel values for segmentation."""

    assert target.shape + pred.shape
    device, shape = target.device, target.shape
    zeros = torch.zeros(shape).to(device)
    ones = torch.ones(shape).to(device)
    target_class = torch.where(target == class_ix, ones, zeros)
    pred_class = torch.where(pred == class_ix, ones, zeros)
    tp = torch.where(target_class == 1, pred_class, zeros).sum()
    tn = torch.where(target_class == 0, 1 - pred_class, zeros).sum()
    fn = torch.where(target_class == 1, 1 - pred_class, zeros).sum()
    fp = torch.where(pred_class == 1, 1 - target_class, zeros).sum()
    tp, tn, fn, fp = int(tp), int(tn), int(fn), int(fp)
    # assert int(ones.sum()) == tp+tn+fn+fp
    return tp, tn, fn, fp


def dice_coefficient_with_mask(pred, target, mask):
    """
    使用掩膜以及TP, FP, FN来计算Dice系数。
    pred: 预测的分割结果，二值图像。
    target: 真实的分割结果，二值图像。
    mask: 定义有效区域的掩膜，二值图像。
    """
    # 确保预测和目标是二值的
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()

    # 应用掩膜
    pred = pred * mask
    target = target * mask

    # 计算TP, FP, FN
    TP = (pred * target).sum()
    FP = (pred * (1 - target)).sum()
    FN = ((1 - pred) * target).sum()

    # 计算Dice系数
    dice = (2 * TP + 1e-6) / (2 * TP + FP + FN + 1e-6)  # 添加一个小的常数避免除以零

    return dice


def get_mean_scores_with_mask(
    target,
    pred,
    mask,
    metrics=["ScoreDice", "ScoreIoU"],
    label_names=["background", "class 1"],
    label_weights=None,
    segmentation=True,
):
    r"""Returns the scores per label, as well as the (weighted) mean, such as
    to avoid considering "don't care" classes. The weights don't have to be
    normalized.
    """
    scores = {metric: dict() for metric in metrics}
    # Calculate metric values per each class
    metrics = {metric: getattr(score_defs, metric)() for metric in metrics}
    for label_nr, label_name in enumerate(label_names):
        # TODO: enable also for classification
        # tp, fp, fn = dice_coefficient_with_mask(target, pred, mask)
        for metric_key, metric_f in metrics.items():
            if metric_key == "ScoreHausdorff":
                score = metric_f.eval(target, pred)
            else:
                # score = metric_f.eval(tp, 0, fn, fp)
                score = dice_coefficient_with_mask(pred, target, mask)
            scores[metric_key + "[" + label_name + "]"] = score
            scores[metric_key][label_name] = score
    # Calculate metric means
    if label_weights is None:
        label_weights = {label_name: 1 for label_name in label_names}
    for metric_key in metrics.keys():
        # Replace the dictionary by the mean
        mean = sum(
            [label_score * label_weights[label_name] for label_name, label_score in scores[metric_key].items()]
        ) / sum(list(label_weights.values()))
        scores[metric_key] = mean
    return scores


def get_mean_scores(
    target,
    pred,
    metrics=["ScoreDice", "ScoreIoU", "ScoreHausdorff"],
    label_names=["background", "class 1"],
    label_weights=None,
    segmentation=True,
):
    r"""Returns the scores per label, as well as the (weighted) mean, such as
    to avoid considering "don't care" classes. The weights don't have to be
    normalized.
    """
    scores = {metric: dict() for metric in metrics}
    # Calculate metric values per each class
    metrics = {metric: getattr(score_defs, metric)() for metric in metrics}
    for label_nr, label_name in enumerate(label_names):
        # TODO: enable also for classification
        tp, tn, fn, fp = get_tp_tn_fn_fp_segmentation(target, pred, class_ix=label_nr)
        for metric_key, metric_f in metrics.items():
            if metric_key == "ScoreHausdorff":
                score = metric_f.eval(target, pred)
            else:
                score = metric_f.eval(tp, tn, fn, fp)
            scores[metric_key + "[" + label_name + "]"] = score
            scores[metric_key][label_name] = score
    # Calculate metric means
    if label_weights is None:
        label_weights = {label_name: 1 for label_name in label_names}
    for metric_key in metrics.keys():
        # Replace the dictionary by the mean
        mean = sum(
            [label_score * label_weights[label_name] for label_name, label_score in scores[metric_key].items()]
        ) / sum(list(label_weights.values()))
        scores[metric_key] = mean
    return scores
