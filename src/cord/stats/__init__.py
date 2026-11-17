"""Statistical core of CORD. Pure NumPy/SciPy; shared by the pipeline and the synthetic studies."""

from cord.stats.conformal import ConformalEValues, conformal_evalues, conformal_pvalues, count_at_least
from cord.stats.estimation import estimate_poison_rate, per_class_counts, storey_null_proportion
from cord.stats.floors import (
    presence_min_exceedances,
    presence_threshold_approx,
    presence_threshold_exact,
    selection_floor,
)
from cord.stats.multiple_testing import (
    benjamini_hochberg,
    benjamini_yekutieli,
    e_benjamini_hochberg,
    false_discovery_proportion,
    true_positive_proportion,
)
from cord.stats.presence import (
    FoldPresence,
    PresenceResult,
    exceedance_count,
    exceedance_pvalue,
    fisher_statistic,
    higher_criticism_statistic,
    permutation_global_test,
    presence_test,
)

__all__ = [
    "ConformalEValues",
    "FoldPresence",
    "PresenceResult",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "conformal_evalues",
    "conformal_pvalues",
    "count_at_least",
    "e_benjamini_hochberg",
    "estimate_poison_rate",
    "exceedance_count",
    "exceedance_pvalue",
    "false_discovery_proportion",
    "fisher_statistic",
    "higher_criticism_statistic",
    "per_class_counts",
    "permutation_global_test",
    "presence_min_exceedances",
    "presence_test",
    "presence_threshold_approx",
    "presence_threshold_exact",
    "selection_floor",
    "storey_null_proportion",
    "true_positive_proportion",
]
