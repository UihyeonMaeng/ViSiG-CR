"""Public retrieval comparison API; implementation is shared with graph/evaluation."""
from ..clinical_matching import (
    MISSING_ATTRIBUTE_VALUES, UNCERTAIN_ASSERTIONS,
    attribute_value, best_target_match, comparison_role, online_relation_label,
    pair_non_target_context_score, target_context_score,
)

__all__ = [
    "attribute_value", "best_target_match", "comparison_role",
    "online_relation_label", "pair_non_target_context_score", "target_context_score",
]
