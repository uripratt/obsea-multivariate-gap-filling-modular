"""Evaluation module init"""

from .metrics import (
    calculate_metrics,
    calculate_skill_score,
    print_metrics,
    compare_models,
    save_metrics,
)
from .gap_analysis import (
    identify_gaps,
    calculate_error_by_gap_length,
    calculate_error_by_gap_position,
    print_gap_analysis,
)

__all__ = [
    'calculate_metrics',
    'calculate_skill_score',
    'print_metrics',
    'compare_models',
    'save_metrics',
    'identify_gaps',
    'calculate_error_by_gap_length',
    'calculate_error_by_gap_position',
    'print_gap_analysis',
]
