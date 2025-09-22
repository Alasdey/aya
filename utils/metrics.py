from __future__ import annotations
from typing import List, Dict, Any

def compute_multiclass_metrics(y_true: List[str], y_pred: List[str], labels: List[str]):
    counts = {lab: {"tp": 0, "fp": 0, "fn": 0} for lab in labels}
    for gt, pr in zip(y_true, y_pred):
        for lab in labels:
            if pr == lab and gt == lab:
                counts[lab]["tp"] += 1
            elif pr == lab and gt != lab:
                counts[lab]["fp"] += 1
            elif pr != lab and gt == lab:
                counts[lab]["fn"] += 1

    per_label = {}
    for lab in labels:
        tp, fp, fn = counts[lab]["tp"], counts[lab]["fp"], counts[lab]["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else 0.0
        per_label[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}

    eval_labels = [lab for lab in labels if lab.lower() != "norel"] or labels
    macro_f1 = sum(per_label[lab]["f1"] for lab in eval_labels) / len(eval_labels)

    total_tp = sum(counts[lab]["tp"] for lab in eval_labels)
    total_fp = sum(counts[lab]["fp"] for lab in eval_labels)
    total_fn = sum(counts[lab]["fn"] for lab in eval_labels)

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2*micro_precision*micro_recall)/(micro_precision+micro_recall) if (micro_precision+micro_recall)>0 else 0.0

    return {"per_label": per_label, "macro_f1": macro_f1,
            "micro_precision": micro_precision, "micro_recall": micro_recall,
            "micro_f1": micro_f1, "total": len(y_true)}

def compute_binary_metrics(y_true: List[str], y_pred: List[str]):
    def binarize(y):
        return ["POS" if lab in ("CauseEffect","EffectCause") else "NEG" for lab in y]
    yt, yp = binarize(y_true), binarize(y_pred)
    tp = sum(1 for a,b in zip(yt,yp) if a=="POS" and b=="POS")
    fp = sum(1 for a,b in zip(yt,yp) if a=="NEG" and b=="POS")
    fn = sum(1 for a,b in zip(yt,yp) if a=="POS" and b=="NEG")
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
    return {"precision":prec, "recall":rec, "f1":f1, "support_pos": sum(1 for v in yt if v=="POS")}
