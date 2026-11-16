"""Shared logging helpers for evaluation metric summaries."""
import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def format_ece(ece_result: Optional[Dict]) -> str:
    if not ece_result:
        return "N/A"
    parts = []
    if "ece_recall" in ece_result:
        values = ece_result["ece_recall"]
        parts.append(
            "ECE_R@"
            + "/".join(str(k) for k in values)
            + ": "
            + "/".join(f"{float(v):.3f}" for v in values.values())
        )
    if "ece_map" in ece_result:
        values = ece_result["ece_map"]
        parts.append(
            "ECE_mAP@"
            + "/".join(str(k) for k in values)
            + ": "
            + "/".join(f"{float(v):.3f}" for v in values.values())
        )
    if "ece_ap" in ece_result:
        parts.append(f"ap={float(ece_result['ece_ap']):.3f}")
    return ", ".join(parts) if parts else str(ece_result)


def log_stats(prefix: str, stats: Dict[str, float], *, debug: bool = False) -> None:
    log_fn = logger.debug if debug else logger.info
    log_fn(
        "  %s: min=%.4g, max=%.4g, mean=%.4g, std=%.4g, median=%.4g",
        prefix,
        stats["min"],
        stats["max"],
        stats["mean"],
        stats["std"],
        stats["median"],
    )


def log_metric_summary(
    metric_name: str,
    ece_result: Optional[Dict[str, Any]],
    scores: np.ndarray,
    auc_pr: Optional[float],
    *,
    ece_only: bool = False,
) -> None:
    """Log summary for a learned uncertainty metric (variances + confidence rows)."""
    logger.info("\033[1m%s\033[0m", metric_name)
    logger.info("  %s", format_ece(ece_result))
    if ece_only:
        return
    score_stats = ece_result.get("score_stats") if isinstance(ece_result, dict) else None
    if score_stats:
        log_stats("variances (after clamp)", score_stats["variances_after_clamp"])
        log_stats("confidence (after clamp)", score_stats["confidence_after_clamp"])
        log_stats("variances (before clamp)", score_stats["variances_before_clamp"], debug=True)
        log_stats("confidence (before clamp)", score_stats["confidence_before_clamp"], debug=True)
    else:
        logger.info(
            "  variances (after clamp): min=%.4g, max=%.4g, mean=%.4g, std=%.4g",
            float(np.min(scores)),
            float(np.max(scores)),
            float(np.mean(scores)),
            float(np.std(scores)),
        )
    if auc_pr is not None:
        logger.info("  AUC-PR: %.4f", auc_pr)


