"""Classical multivariate process monitoring: PCA with T-squared and SPE.

A PCA model of normal operation splits each sample into a part the model
explains and a residual:

    T-squared   distance from the normal operating centre, measured inside
                the model plane (unusual, but consistent, behaviour)
    SPE (Q)     squared size of the residual (the usual relationships
                between sensors have broken)

Each statistic is divided by its control limit and the larger ratio is the
anomaly score, so a score above 1 means "outside normal limits".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from . import config


@dataclass
class PCAMonitor:
    sensors: list[str]
    mean: np.ndarray            # (V,)
    scale: np.ndarray           # (V,)
    components: np.ndarray      # (k, V) loadings
    eigenvalues: np.ndarray     # (k,) variance of each score
    t2_limit: float
    spe_limit: float

    name = "pca"

    @classmethod
    def fit(cls, normal: np.ndarray, sensors=None, variance: float = 0.9, limit_quantile: float = 0.99) -> "PCAMonitor":
        """Fit on fault-free samples (n, V).

        Components are kept until `variance` of the total is explained.
        Control limits are the empirical `limit_quantile` of each statistic
        on the fitting data. The textbook F and chi-squared limits assume
        independent Gaussian samples, which closed-loop process data are not.
        """
        normal = np.asarray(normal, dtype=np.float64)
        mean = normal.mean(axis=0)
        scale = normal.std(axis=0)
        scale[scale == 0] = 1.0
        pca = PCA(n_components=variance, svd_solver="full").fit((normal - mean) / scale)
        monitor = cls(list(sensors or config.SENSORS), mean, scale, pca.components_,
                      pca.explained_variance_, 1.0, 1.0)
        t2, spe = monitor.statistics(normal)
        monitor.t2_limit = float(np.quantile(t2, limit_quantile))
        monitor.spe_limit = float(np.quantile(spe, limit_quantile))
        return monitor

    @property
    def n_components(self) -> int:
        return len(self.eigenvalues)

    def _project(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = (np.atleast_2d(np.asarray(values, dtype=np.float64)) - self.mean) / self.scale
        scores = z @ self.components.T
        residual = z - scores @ self.components
        return z, scores, residual

    def statistics(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """T-squared and SPE for each sample of (n, V)."""
        _, scores, residual = self._project(values)
        return (scores ** 2 / self.eigenvalues).sum(axis=1), (residual ** 2).sum(axis=1)

    def score(self, values: np.ndarray) -> np.ndarray:
        """Anomaly score per sample: the larger of T2 / limit and SPE / limit."""
        t2, spe = self.statistics(values)
        return np.maximum(t2 / self.t2_limit, spe / self.spe_limit)

    def score_run(self, values: np.ndarray) -> np.ndarray:
        return self.score(values)

    def contributions(self, sample: np.ndarray) -> dict:
        """Per-sensor contributions to whichever statistic is further over its limit.

        SPE: each sensor's squared residual; they sum to SPE.
        T-squared: x_j * sum_i(t_i * p_ij / lambda_i); they sum to T-squared.
        Individual T-squared terms can be negative; those are sensors that
        pull the sample back towards normal.
        """
        z, scores, residual = self._project(sample)
        t2 = float((scores[0] ** 2 / self.eigenvalues).sum())
        spe = float((residual[0] ** 2).sum())
        if spe / self.spe_limit >= t2 / self.t2_limit:
            statistic, values = "SPE", residual[0] ** 2
        else:
            statistic, values = "T2", z[0] * ((scores[0] / self.eigenvalues) @ self.components)
        return {"statistic": statistic, "t2": t2, "spe": spe, "values": values}
