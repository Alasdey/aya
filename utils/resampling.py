from __future__ import annotations

import asyncio
import random
import uuid
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from langsmith.run_helpers import tracing_context


def majority_vote_with_configurable_tiebreak(
    predictions: List[List[Dict[str, Any]]], 
    pairs: List[Tuple[str, str, str]],
    tie_breaking: str = "random"
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Aggregate multiple independent predictions using majority voting with configurable tie-breaking.
    
    Args:
        predictions: List of prediction arrays from independent runs
        pairs: List of (Ti, gold, Tj) tuples for all pairs
        tie_breaking: "random" or "norel" - strategy for handling ties
    
    Returns:
        Tuple of (final_predictions, aggregation_stats)
    """
    # Validate tie_breaking parameter
    if tie_breaking not in ["random", "norel"]:
        raise ValueError(f"tie_breaking must be 'random' or 'norel', got '{tie_breaking}'")
    
    # Convert predictions to standardized format
    pair_votes: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    
    for pred_run in predictions:
        pred_map = _pred_map_from_json(pred_run, pairs)
        for (Ti, Tj), label in pred_map.items():
            pair_votes[(Ti, Tj)].append(label)
    
    # Perform majority voting with configurable tie-breaking
    final_predictions = []
    aggregation_stats = {
        "total_pairs": len(pairs),
        "unanimous_agreements": 0,
        "majority_decisions": 0,
        "tied_decisions": 0,
        "tie_breaking_strategy": tie_breaking,  # Record which strategy was used
        "per_pair_stats": {},
        "label_distribution": {"CauseEffect": 0, "EffectCause": 0, "NoRel": 0},
        "tie_analysis": {
            "causal_vs_causal": 0,  # CauseEffect vs EffectCause ties
            "causal_vs_norel": 0,   # Any causal vs NoRel ties
            "three_way_ties": 0     # All three labels tied
        }
    }
    
    # Add strategy-specific statistics
    if tie_breaking == "random":
        aggregation_stats["random_tiebreaks"] = 0
    elif tie_breaking == "norel":
        aggregation_stats["norel_tiebreaks"] = 0
    
    for (Ti, _gold, Tj) in pairs:
        votes = pair_votes.get((Ti, Tj), ["NoRel"] * len(predictions))
        vote_counts = Counter(votes)
        
        # Find the maximum vote count
        max_votes = max(vote_counts.values())
        winners = [label for label, count in vote_counts.items() if count == max_votes]
        
        # Statistics tracking
        if len(set(votes)) == 1:  # All votes the same
            aggregation_stats["unanimous_agreements"] += 1
            final_label = votes[0]  # All the same, so take any
        elif len(winners) == 1:  # Clear majority
            aggregation_stats["majority_decisions"] += 1
            final_label = winners[0]
        else:  # Tie - apply configured tie-breaking strategy
            aggregation_stats["tied_decisions"] += 1
            
            # Analyze the type of tie for statistics
            causal_labels = {"CauseEffect", "EffectCause"}
            tied_causal = [label for label in winners if label in causal_labels]
            tied_norel = "NoRel" in winners
            
            if len(tied_causal) == 2 and not tied_norel:
                aggregation_stats["tie_analysis"]["causal_vs_causal"] += 1
            elif len(tied_causal) >= 1 and tied_norel:
                aggregation_stats["tie_analysis"]["causal_vs_norel"] += 1  
            elif len(winners) == 3:
                aggregation_stats["tie_analysis"]["three_way_ties"] += 1
            
            # Apply tie-breaking strategy
            if tie_breaking == "random":
                final_label = random.choice(winners)
                aggregation_stats["random_tiebreaks"] += 1
            elif tie_breaking == "norel":
                # Conservative: always choose NoRel in case of ties
                final_label = "NoRel"
                aggregation_stats["norel_tiebreaks"] += 1
        
        final_predictions.append({"pair": f"{Ti},{Tj}", "label": final_label})
        aggregation_stats["label_distribution"][final_label] += 1
        aggregation_stats["per_pair_stats"][f"{Ti},{Tj}"] = {
            "votes": votes,
            "vote_counts": dict(vote_counts),
            "final_label": final_label,
            "was_tied": len(winners) > 1,
            "tie_breaking_applied": tie_breaking if len(winners) > 1 else None
        }
    
    return final_predictions, aggregation_stats


# Keep the original functions for backward compatibility (optional)
def majority_vote_with_random_tiebreak(
    predictions: List[List[Dict[str, Any]]], 
    pairs: List[Tuple[str, str, str]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Backward compatibility wrapper for random tie-breaking."""
    return majority_vote_with_configurable_tiebreak(predictions, pairs, tie_breaking="random")


def majority_vote_with_norel_tiebreak(
    predictions: List[List[Dict[str, Any]]], 
    pairs: List[Tuple[str, str, str]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Backward compatibility wrapper for NoRel tie-breaking."""
    return majority_vote_with_configurable_tiebreak(predictions, pairs, tie_breaking="norel")


def _pred_map_from_json(arr: List[Dict[str, Any]], requested_pairs: List[Tuple[str,str,str]]) -> Dict[Tuple[str,str], str]:
    """Convert JSON prediction array to pair -> label mapping (copied from main.py)"""
    pred_map: Dict[Tuple[str,str], str] = {}
    for obj in arr or []:
        p = (obj.get("pair") or "").strip()
        lab = (obj.get("label") or "").strip()
        lab = {"causeeffect":"CauseEffect","effectcause":"EffectCause","norel":"NoRel"}.get(lab.lower(), lab)
        if lab in {"CauseEffect", "EffectCause", "NoRel"} and "," in p:
            a,b = [t.strip() for t in p.split(",",1)]
            pred_map[(a,b)] = lab
    for (Ti, _gold, Tj) in requested_pairs:
        pred_map.setdefault((Ti,Tj), "NoRel")
    return pred_map