"""Auditable Black-Litterman inference and constrained portfolio optimization."""

import math
from typing import List, Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator
from scipy.optimize import minimize

from data_pipeline import DataQuality
from models.allocation_policy import AllocationPolicyResult
from models.request_validation import normalize_tickers


class ViewSpec(BaseModel):
    """A single annualized absolute or relative Black-Litterman view."""

    assets: List[str] = Field(..., min_length=1)
    relative_assets: Optional[List[str]] = Field(default=None)
    expected_return: float
    confidence: float = Field(default=0.5, gt=0.0, le=1.0)

    @field_validator("assets")
    @classmethod
    def validate_assets(cls, assets: List[str]) -> List[str]:
        return normalize_tickers(assets)

    @field_validator("relative_assets")
    @classmethod
    def validate_relative_assets(cls, assets: Optional[List[str]]) -> Optional[List[str]]:
        if assets is None:
            return None
        cleaned = [str(asset).strip() for asset in assets if str(asset).strip()]
        if not cleaned:
            return None
        return normalize_tickers(cleaned)

    @field_validator("expected_return")
    @classmethod
    def validate_expected_return(cls, expected_return: float) -> float:
        if not math.isfinite(float(expected_return)):
            raise ValueError("expected_return must be finite")
        return expected_return

    @model_validator(mode="after")
    def validate_disjoint_legs(self) -> "ViewSpec":
        long_leg = {asset.upper() for asset in self.assets}
        short_leg = {asset.upper() for asset in self.relative_assets or []}
        overlap = sorted(long_leg & short_leg)
        if overlap:
            raise ValueError("view long and relative legs must be disjoint: " + ", ".join(overlap))
        return self


class OptimizationResult(BaseModel):
    """Result of a Black-Litterman portfolio optimization."""

    tickers: List[str]
    prior_returns: List[float]
    prior_weights: List[float]
    posterior_returns: List[float]
    posterior_weights: List[float]
    raw_posterior_weights: List[float] = Field(default_factory=list)
    recommended_weights: List[float] = Field(default_factory=list)
    decision_policy: Literal["raw", "balanced_blend", "defensive_blend"] = Field(default="raw")
    turnover: float = Field(default=0.0)
    effective_min_weight: float = Field(default=0.0)
    risk_aversion: float
    allocation_model: Literal["black_litterman_views", "prior_only_bl"] = Field(
        default="prior_only_bl"
    )
    prior_source: Literal[
        "market_cap_equilibrium",
        "inverse_volatility_equilibrium",
    ] = Field(default="inverse_volatility_equilibrium")
    tau: float = Field(default=0.0)
    view_matrix: List[List[float]] = Field(default_factory=list)
    view_returns: List[float] = Field(default_factory=list)
    view_confidences: List[float] = Field(default_factory=list)
    omega_diagonal: List[float] = Field(default_factory=list)
    posterior_covariance: List[List[float]] = Field(default_factory=list)
    covariance_estimator: str = Field(default="ledoit_wolf_annualized_psd")
    effective_max_weight: float = Field(default=1.0)
    turnover_penalty: float = Field(default=0.0)
    concentration_penalty: float = Field(default=0.0)
    solver_method: str = Field(default="SLSQP")
    solver_success: bool = Field(default=False)
    solver_status: int = Field(default=-1)
    solver_message: str = Field(default="")
    solver_objective: float = Field(default=0.0)
    source: str = Field(default="unknown", description="Data source used for prices")
    source_detail: str = Field(default="unknown", description="Detailed price data provenance")
    data_warnings: List[str] = Field(default_factory=list, description="Non-fatal data quality warnings")
    data_quality: DataQuality = Field(default_factory=DataQuality, description="Unified data quality provenance")
    backtest_enabled: bool = Field(default=False)
    benchmark_symbol: str = Field(default="")
    benchmark_name: str = Field(default="")
    benchmark_source: str = Field(default="")
    benchmark_source_detail: str = Field(default="")
    risk_free_rate: float = Field(default=0.0)
    risk_free_rate_source: str = Field(default="")
    risk_free_rate_source_detail: str = Field(default="")
    methodology_warnings: List[str] = Field(default_factory=list)
    oos_dates: List[str] = Field(default_factory=list)
    oos_optimized_cum_returns: List[float] = Field(default_factory=list)
    oos_benchmark_cum_returns: List[float] = Field(default_factory=list)
    oos_prior_cum_returns: List[float] = Field(default_factory=list)
    oos_optimized_ann_vol: float = Field(default=0.0)
    oos_benchmark_ann_vol: float = Field(default=0.0)
    oos_prior_ann_vol: float = Field(default=0.0)
    oos_optimized_max_drawdown: float = Field(default=0.0)
    oos_benchmark_max_drawdown: float = Field(default=0.0)
    oos_prior_max_drawdown: float = Field(default=0.0)
    oos_excess_return: float = Field(default=0.0)
    oos_optimized_sharpe: float = Field(default=0.0)
    oos_benchmark_sharpe: float = Field(default=0.0)
    oos_prior_sharpe: float = Field(default=0.0)
    oos_optimized_ir: float = Field(default=0.0)
    model_score: float = Field(default=0.0)
    model_grade: str = Field(default="")
    model_score_risk_control: float = Field(default=0.0)
    model_score_profitability: float = Field(default=0.0)
    model_score_alpha: float = Field(default=0.0)
    model_score_stability: float = Field(default=0.0)
    model_score_win_rate: float = Field(default=0.0)
    policy_asof: str = Field(default="")
    oos_leakage_guard: bool = Field(default=False)
    oos_decision_log: List[dict] = Field(default_factory=list)
    allocation_policy: Optional[AllocationPolicyResult] = Field(default=None)


class BayesianOptimizer:
    """Black-Litterman posterior inference and mean-variance optimization."""

    TAU_MIN_OBSERVATIONS = 60
    OMEGA_CONFIDENCE_FLOOR = 1e-6
    OMEGA_VARIANCE_FLOOR = 1e-12
    SOLVER_METHOD = "SLSQP"
    SOLVER_FTOL = 1e-9
    SOLVER_MAXITER = 1000

    @staticmethod
    def _build_view_matrices(
        tickers: List[str],
        views: List[ViewSpec],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Translate human-readable views into P, Q, and confidence arrays."""
        n_assets = len(tickers)
        ticker_to_idx = {t.upper(): i for i, t in enumerate(tickers)}
        k = len(views)

        unknown_assets: list[str] = []
        for view in views:
            for asset in [*view.assets, *(view.relative_assets or [])]:
                key = asset.upper()
                if key not in ticker_to_idx and asset not in unknown_assets:
                    unknown_assets.append(asset)
        if unknown_assets:
            raise ValueError("view assets must exist in tickers: " + ", ".join(unknown_assets))

        P = np.zeros((k, n_assets))
        Q = np.zeros(k)
        confidences = np.zeros(k)

        for i, view in enumerate(views):
            long_scale = 1.0 / float(len(view.assets))
            for asset in view.assets:
                idx = ticker_to_idx[asset.upper()]
                P[i, idx] += long_scale
            if view.relative_assets:
                short_scale = 1.0 / float(len(view.relative_assets))
                for rel_asset in view.relative_assets:
                    idx = ticker_to_idx[rel_asset.upper()]
                    P[i, idx] -= short_scale
            Q[i] = view.expected_return
            confidences[i] = view.confidence

            if np.allclose(P[i], 0.0, atol=1e-15):
                raise ValueError(f"view {i} produces a zero exposure row")

        assert P.shape == (k, n_assets), f"P shape mismatch: {P.shape} != ({k}, {n_assets})"
        assert Q.shape == (k,), f"Q shape mismatch: {Q.shape} != ({k},)"
        assert confidences.shape == (k,), f"confidences shape mismatch: {confidences.shape}"
        return P, Q, confidences

    @staticmethod
    def _build_omega(
        P: np.ndarray,
        cov_matrix: np.ndarray,
        confidences: np.ndarray,
        tau: float,
        confidence_floor: float = OMEGA_CONFIDENCE_FLOOR,
        variance_floor: float = OMEGA_VARIANCE_FLOOR,
    ) -> np.ndarray:
        """Construct the uncertainty diagonal matrix Omega aligned with tau-Sigma scale."""
        if not math.isfinite(float(tau)) or float(tau) <= 0.0:
            raise ValueError("tau must be finite and positive")
        if not 0.0 < float(confidence_floor) <= 1.0:
            raise ValueError("confidence_floor must be in (0, 1]")
        if not math.isfinite(float(variance_floor)) or float(variance_floor) <= 0.0:
            raise ValueError("variance_floor must be finite and positive")
        var_views = np.diag(P @ cov_matrix @ P.T)
        safe_conf = np.clip(confidences, float(confidence_floor), 1.0)
        omega_diag = ((1.0 - safe_conf) / safe_conf) * float(tau) * var_views
        omega_diag = np.maximum(omega_diag, float(variance_floor))
        return np.diag(omega_diag)

    @staticmethod
    def _ensure_vector(v: np.ndarray, name: str = "vector") -> np.ndarray:
        """Ensure a 1D array is a column vector."""
        arr = np.asarray(v)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        if arr.ndim == 2 and arr.shape[1] == 1:
            return arr
        raise ValueError(f"{name} must be 1-dimensional, got shape {arr.shape}")

    @staticmethod
    def black_litterman(
        prior_returns: np.ndarray,
        cov_matrix: np.ndarray,
        tau: float,
        P: np.ndarray,
        Q: np.ndarray,
        omega: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute posterior expected returns and covariance via Black-Litterman."""
        Pi = BayesianOptimizer._ensure_vector(prior_returns, "prior_returns")
        n_assets = Pi.shape[0]
        Sigma = np.asarray(cov_matrix, dtype=float)
        if Sigma.shape != (n_assets, n_assets):
            raise ValueError(f"cov_matrix shape {Sigma.shape} incompatible with {n_assets} assets")
        if not np.isfinite(Pi).all() or not np.isfinite(Sigma).all():
            raise ValueError("Black-Litterman prior and covariance must be finite")
        if not np.allclose(Sigma, Sigma.T, atol=1e-10):
            raise ValueError("cov_matrix must be symmetric")
        if float(np.linalg.eigvalsh(Sigma).min()) < -1e-10:
            raise ValueError("cov_matrix must be positive semi-definite")
        if not math.isfinite(float(tau)) or float(tau) <= 0.0:
            raise ValueError("tau must be finite and positive")

        P_mat = np.asarray(P)
        Q_vec = BayesianOptimizer._ensure_vector(Q, "Q")
        k = P_mat.shape[0]
        if P_mat.shape != (k, n_assets):
            raise ValueError(f"P shape {P_mat.shape} incompatible with ({k}, {n_assets})")
        if Q_vec.shape != (k, 1):
            raise ValueError(f"Q shape {Q_vec.shape} incompatible with ({k}, 1)")
        if not np.isfinite(P_mat).all() or not np.isfinite(Q_vec).all():
            raise ValueError("P and Q must be finite")

        if k == 0:
            return Pi.flatten(), Sigma.copy()

        if omega is None:
            omega = np.diag(np.ones(k) * 1e-4)
        else:
            omega = np.asarray(omega, dtype=float)
            if omega.shape != (k, k):
                raise ValueError(f"omega shape {omega.shape} incompatible with ({k}, {k})")
        if not np.isfinite(omega).all() or not np.allclose(omega, omega.T, atol=1e-12):
            raise ValueError("omega must be finite and symmetric")
        if float(np.linalg.eigvalsh(omega).min()) <= 0.0:
            raise ValueError("omega must be positive definite")

        tau_Sigma = tau * Sigma
        middle = P_mat @ tau_Sigma @ P_mat.T + omega

        diff = Q_vec - P_mat @ Pi
        try:
            solved = np.linalg.solve(middle, diff)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Black-Litterman posterior mean solve failed") from exc
        Pi_post = Pi + tau_Sigma @ P_mat.T @ solved

        try:
            cov_solved = np.linalg.solve(middle, P_mat @ tau_Sigma)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Black-Litterman posterior covariance solve failed") from exc
        Sigma_post = Sigma + tau_Sigma - tau_Sigma @ P_mat.T @ cov_solved
        Sigma_post = (Sigma_post + Sigma_post.T) / 2.0

        if Pi_post.shape != (n_assets, 1):
            raise RuntimeError(f"posterior returns shape mismatch: {Pi_post.shape}")
        if Sigma_post.shape != (n_assets, n_assets):
            raise RuntimeError(f"posterior covariance shape mismatch: {Sigma_post.shape}")
        if not np.isfinite(Pi_post).all() or not np.isfinite(Sigma_post).all():
            raise RuntimeError("Black-Litterman posterior contains non-finite values")
        if float(np.linalg.eigvalsh(Sigma_post).min()) < -1e-9:
            raise RuntimeError("Black-Litterman posterior covariance is not positive semi-definite")
        return Pi_post.flatten(), Sigma_post

    @staticmethod
    def _effective_max_weight(max_weight: float, n_assets: int) -> float:
        """Return a feasible per-asset cap for a full-investment long-only portfolio."""
        if n_assets <= 0:
            raise ValueError("n_assets must be positive")
        return max(float(max_weight), 1.0 / n_assets)

    @staticmethod
    def _effective_min_weight(min_weight: float, n_assets: int) -> float:
        """Return a feasible per-asset floor for a full-investment long-only portfolio."""
        if n_assets <= 0:
            raise ValueError("n_assets must be positive")
        clean_min = max(float(min_weight), 0.0)
        return min(clean_min, 0.5 / n_assets)

    @staticmethod
    def _normalize_under_bounds(
        weights: np.ndarray,
        min_weight: float,
        max_weight: float,
    ) -> np.ndarray:
        """Project non-negative weights into feasible full-investment bounds."""
        weights = np.asarray(weights, dtype=float)
        n_assets = len(weights)
        effective_min = BayesianOptimizer._effective_min_weight(min_weight, n_assets)
        effective_max = BayesianOptimizer._effective_max_weight(max_weight, n_assets)
        if effective_min > effective_max:
            raise ValueError("min_weight exceeds feasible max_weight")

        total_weight = float(weights.sum())
        if (
            np.isfinite(weights).all()
            and abs(total_weight - 1.0) <= 1e-10
            and float(weights.min()) >= effective_min - 1e-10
            and float(weights.max()) <= effective_max + 1e-10
        ):
            return weights.copy()

        shifted_capacity = effective_max - effective_min
        remaining = 1.0 - effective_min * n_assets
        if remaining < -1e-10:
            raise ValueError("min_weight is not feasible for the asset count")
        if remaining <= 1e-12:
            return np.ones(n_assets, dtype=float) / n_assets

        weights = np.clip(weights, 0.0, None)
        total = float(weights.sum())
        if total <= 1e-12:
            scaled = np.ones(n_assets, dtype=float) * (remaining / n_assets)
        else:
            scaled = weights / total * remaining

        capped = np.zeros(n_assets, dtype=float)
        active = np.ones(n_assets, dtype=bool)
        residual = remaining

        while active.any() and residual > 1e-12:
            active_indices = np.flatnonzero(active)
            active_scaled = scaled[active]
            active_sum = float(active_scaled.sum())
            if active_sum <= 1e-12:
                allocation = np.ones(len(active_indices), dtype=float) * (
                    residual / len(active_indices)
                )
            else:
                allocation = residual * active_scaled / active_sum

            overweight = allocation > shifted_capacity + 1e-12
            if not overweight.any():
                capped[active_indices] = allocation
                break

            overweight_indices = active_indices[overweight]
            capped[overweight_indices] = shifted_capacity
            residual -= shifted_capacity * len(overweight_indices)
            active[overweight_indices] = False

        bounded = capped + effective_min
        bounded = np.clip(bounded, effective_min, effective_max)
        total_bounded = float(bounded.sum())
        if abs(total_bounded - 1.0) > 1e-10:
            residual = 1.0 - total_bounded
            if residual > 0.0:
                capacities = effective_max - bounded
                order = np.argsort(bounded)
                for idx in order:
                    if residual <= 1e-10:
                        break
                    addition = min(float(capacities[idx]), residual)
                    if addition > 0.0:
                        bounded[idx] += addition
                        residual -= addition
            else:
                capacities = bounded - effective_min
                order = np.argsort(-bounded)
                for idx in order:
                    if residual >= -1e-10:
                        break
                    reduction = min(float(capacities[idx]), -residual)
                    if reduction > 0.0:
                        bounded[idx] -= reduction
                        residual += reduction

        if not np.isfinite(bounded).all() or abs(float(bounded.sum()) - 1.0) > 1e-8:
            raise RuntimeError("could not normalize weights under investment bounds")
        if float(bounded.min()) < effective_min - 1e-8:
            raise RuntimeError("weights breach min_weight constraint")
        if float(bounded.max()) > effective_max + 1e-8:
            raise RuntimeError("weights breach max_weight constraint")
        return bounded

    @staticmethod
    def _enforce_max_weight(weights: np.ndarray, max_weight: float) -> np.ndarray:
        """Redistribute long-only weights so they sum to one without breaching max_weight."""
        weights = np.asarray(weights, dtype=float)
        n_assets = len(weights)
        effective_max = BayesianOptimizer._effective_max_weight(max_weight, n_assets)

        capped = np.zeros(n_assets, dtype=float)
        active = np.ones(n_assets, dtype=bool)
        remaining = 1.0

        while active.any() and remaining > 1e-12:
            active_weights = weights[active]
            active_sum = float(active_weights.sum())
            active_indices = np.flatnonzero(active)

            if active_sum <= 1e-12:
                capped[active_indices] = remaining / len(active_indices)
                break

            allocation = remaining * active_weights / active_sum
            overweight = allocation > effective_max + 1e-12
            if not overweight.any():
                capped[active_indices] = allocation
                break

            overweight_indices = active_indices[overweight]
            capped[overweight_indices] = effective_max
            remaining -= effective_max * len(overweight_indices)
            active[overweight_indices] = False

        total = float(capped.sum())
        residual = 1.0 - total
        if abs(residual) > 1e-10:
            capacities = effective_max - capped
            for idx in np.argsort(capped):
                if residual <= 1e-10:
                    break
                addition = min(float(capacities[idx]), residual)
                if addition > 0:
                    capped[idx] += addition
                    residual -= addition

        if not np.isfinite(capped).all() or abs(float(capped.sum()) - 1.0) > 1e-8:
            raise RuntimeError("could not normalize prior weights under max_weight constraint")
        if float(capped.max()) > effective_max + 1e-8:
            raise RuntimeError("prior weights exceed max_weight constraint")
        return capped

    @staticmethod
    def optimize_weights(
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        prior_weights: Optional[np.ndarray] = None,
        risk_aversion: float = 2.5,
        max_weight: float = 0.30,
        min_weight: float = 0.02,
        turnover_penalty: float = 0.02,
        concentration_penalty: float = 0.05,
        solver_method: str = SOLVER_METHOD,
        solver_ftol: float = SOLVER_FTOL,
        solver_maxiter: int = SOLVER_MAXITER,
    ) -> np.ndarray:
        """Solve mean-variance optimization with investment policy constraints."""
        weights, _ = BayesianOptimizer._solve_weights(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            prior_weights=prior_weights,
            risk_aversion=risk_aversion,
            max_weight=max_weight,
            min_weight=min_weight,
            turnover_penalty=turnover_penalty,
            concentration_penalty=concentration_penalty,
            solver_method=solver_method,
            solver_ftol=solver_ftol,
            solver_maxiter=solver_maxiter,
        )
        return weights

    @staticmethod
    def _solve_weights(
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        prior_weights: Optional[np.ndarray] = None,
        risk_aversion: float = 2.5,
        max_weight: float = 0.30,
        min_weight: float = 0.02,
        turnover_penalty: float = 0.02,
        concentration_penalty: float = 0.05,
        solver_method: str = SOLVER_METHOD,
        solver_ftol: float = SOLVER_FTOL,
        solver_maxiter: int = SOLVER_MAXITER,
    ) -> Tuple[np.ndarray, dict]:
        """Solve the allocation objective and return fail-closed solver diagnostics."""
        n = len(expected_returns)
        mu = np.asarray(expected_returns, dtype=float)
        Sigma = np.asarray(cov_matrix, dtype=float)
        if mu.shape != (n,) or not np.isfinite(mu).all():
            raise ValueError("expected_returns must be a finite one-dimensional vector")
        if Sigma.shape != (n, n) or not np.isfinite(Sigma).all():
            raise ValueError(f"cov_matrix shape {Sigma.shape} incompatible with {n} assets")
        if not math.isfinite(float(risk_aversion)) or float(risk_aversion) <= 0.0:
            raise ValueError("risk_aversion must be finite and positive")
        if str(solver_method).upper() != BayesianOptimizer.SOLVER_METHOD:
            raise ValueError("only the frozen SLSQP solver is supported")
        if not math.isfinite(float(solver_ftol)) or float(solver_ftol) <= 0.0:
            raise ValueError("solver_ftol must be finite and positive")
        if int(solver_maxiter) <= 0:
            raise ValueError("solver_maxiter must be positive")

        prior = None
        if prior_weights is not None:
            prior = np.asarray(prior_weights, dtype=float)
            if prior.shape != (n,) or not np.isfinite(prior).all():
                prior = None

        def objective(w: np.ndarray) -> float:
            base = -w @ mu + (risk_aversion / 2.0) * w @ Sigma @ w
            concentration = max(float(concentration_penalty), 0.0) * float(w @ w)
            turnover = 0.0
            if prior is not None:
                diff = w - prior
                turnover = max(float(turnover_penalty), 0.0) * float(diff @ diff)
            return float(base + concentration + turnover)

        effective_max = BayesianOptimizer._effective_max_weight(max_weight, n)
        effective_min = BayesianOptimizer._effective_min_weight(min_weight, n)
        x0_base = prior if prior is not None else np.ones(n) / n
        x0 = BayesianOptimizer._normalize_under_bounds(x0_base, effective_min, effective_max)
        bounds = [(effective_min, effective_max) for _ in range(n)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        result = minimize(
            objective,
            x0,
            method=BayesianOptimizer.SOLVER_METHOD,
            bounds=bounds,
            constraints=constraints,
            options={"ftol": float(solver_ftol), "maxiter": int(solver_maxiter)},
        )

        if not result.success:
            raise RuntimeError(f"optimization failed: {result.message}")

        weights = np.clip(result.x, effective_min, effective_max)
        if not np.isfinite(weights).all() or abs(float(np.sum(weights))) <= 1e-12:
            raise RuntimeError("optimization produced invalid weights")
        normalized = BayesianOptimizer._normalize_under_bounds(weights, effective_min, effective_max)
        diagnostics = {
            "method": BayesianOptimizer.SOLVER_METHOD,
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "objective": float(objective(normalized)),
        }
        return normalized, diagnostics

    @staticmethod
    def _normalize_prior_weights(
        weights: Optional[List[float]],
        n_assets: int,
        max_weight: float,
        min_weight: float,
    ) -> Optional[np.ndarray]:
        """Return normalized user weights, or None when they are not usable."""
        if weights is None or len(weights) != n_assets:
            return None

        weights_arr = np.asarray(weights, dtype=float)
        if not np.isfinite(weights_arr).all():
            return None

        weights_arr = np.clip(weights_arr, 0.0, None)
        weight_sum = float(weights_arr.sum())
        if abs(weight_sum) <= 1e-12:
            return None
        weights_arr = weights_arr / weight_sum
        return BayesianOptimizer._normalize_under_bounds(weights_arr, min_weight, max_weight)

    def optimize_with_views(
        self,
        tickers: List[str],
        prior_returns: np.ndarray,
        cov_matrix: np.ndarray,
        views: Optional[List[ViewSpec]],
        tau: Optional[float] = None,
        risk_aversion: float = 2.5,
        weights: Optional[List[float]] = None,
        max_weight: float = 0.30,
        min_weight: float = 0.02,
        turnover_penalty: float = 0.02,
        concentration_penalty: float = 0.05,
        market_caps: Optional[List[float]] = None,
        n_observations: Optional[int] = None,
        tau_min_observations: int = TAU_MIN_OBSERVATIONS,
        omega_confidence_floor: float = OMEGA_CONFIDENCE_FLOOR,
        omega_variance_floor: float = OMEGA_VARIANCE_FLOOR,
        solver_method: str = SOLVER_METHOD,
        solver_ftol: float = SOLVER_FTOL,
        solver_maxiter: int = SOLVER_MAXITER,
    ) -> OptimizationResult:
        """Run full Black-Litterman pipeline: views -> posterior -> optimized weights."""
        n_assets = len(tickers)
        prior_returns_arr = np.asarray(prior_returns, dtype=float)
        cov_arr = np.asarray(cov_matrix)
        if prior_returns_arr.shape != (n_assets,):
            raise ValueError(f"prior_returns shape {prior_returns_arr.shape} != ({n_assets},)")
        if cov_arr.shape != (n_assets, n_assets):
            raise ValueError(f"cov_matrix shape {cov_arr.shape} != ({n_assets}, {n_assets})")
        if not np.isfinite(prior_returns_arr).all() or not np.isfinite(cov_arr).all():
            raise ValueError("optimization inputs must be finite")
        if int(tau_min_observations) <= 0:
            raise ValueError("tau_min_observations must be positive")

        # Data-driven tau: 1/T_train when caller supplies the training-window length;
        # falls back to 1/60 to avoid division-by-tiny when n_observations is missing.
        if tau is None:
            eff_tau = 1.0 / float(
                max(int(n_observations or tau_min_observations), int(tau_min_observations))
            )
        else:
            eff_tau = float(tau)
        if not math.isfinite(eff_tau) or eff_tau <= 0.0:
            raise ValueError("tau must be finite and positive")

        # Equilibrium prior pi: prefer market caps when supplied; otherwise derive
        # an inverse-volatility-implied equilibrium so weak training samples do not
        # anchor the posterior on the noisy historical mean.
        use_caps = (
            market_caps is not None
            and len(market_caps) == n_assets
        )
        cap_weights: Optional[np.ndarray] = None
        if use_caps:
            cap_arr = np.maximum(np.asarray(market_caps, dtype=float), 0.0)
            cap_sum = float(cap_arr.sum())
            if cap_sum > 1e-12:
                cap_weights = cap_arr / cap_sum
            else:
                use_caps = False

        if use_caps and cap_weights is not None:
            pi_eq = risk_aversion * cov_arr @ cap_weights
            prior_source = "market_cap_equilibrium"
        else:
            inv_vol = 1.0 / np.sqrt(np.diag(cov_arr) + 1e-8)
            inv_vol_sum = float(inv_vol.sum())
            if inv_vol_sum > 1e-12:
                w_implied = inv_vol / inv_vol_sum
                pi_eq = risk_aversion * cov_arr @ w_implied
                prior_source = "inverse_volatility_equilibrium"
            else:
                raise RuntimeError("could not construct inverse-volatility equilibrium weights")

        P, Q, confidences = self._build_view_matrices(tickers, views or [])
        if P.shape[0] == 0:
            allocation_model = "prior_only_bl"
            posterior_returns, posterior_cov = self.black_litterman(
                pi_eq,
                cov_arr,
                eff_tau,
                P,
                Q,
            )
            omega = np.empty((0, 0), dtype=float)
        else:
            allocation_model = "black_litterman_views"
            omega = self._build_omega(
                P,
                cov_arr,
                confidences,
                eff_tau,
                confidence_floor=omega_confidence_floor,
                variance_floor=omega_variance_floor,
            )
            posterior_returns, posterior_cov = self.black_litterman(
                pi_eq,
                cov_arr,
                eff_tau,
                P,
                Q,
                omega,
            )

        effective_min = self._effective_min_weight(min_weight, n_assets)
        prior_weights = self._normalize_prior_weights(weights, n_assets, max_weight, effective_min)
        if prior_weights is None:
            # No user allocation: derive prior from the equilibrium pi under the same
            # penalty surface as the posterior so the OOS comparison stays symmetric.
            prior_weights = self.optimize_weights(
                pi_eq,
                cov_arr,
                None,
                risk_aversion,
                max_weight,
                effective_min,
                turnover_penalty=turnover_penalty,
                concentration_penalty=concentration_penalty,
                solver_method=solver_method,
                solver_ftol=solver_ftol,
                solver_maxiter=solver_maxiter,
            )
        raw_posterior_weights, solver_diagnostics = self._solve_weights(
            expected_returns=posterior_returns,
            cov_matrix=posterior_cov,
            prior_weights=prior_weights,
            risk_aversion=risk_aversion,
            max_weight=max_weight,
            min_weight=effective_min,
            turnover_penalty=turnover_penalty,
            concentration_penalty=concentration_penalty,
            solver_method=solver_method,
            solver_ftol=solver_ftol,
            solver_maxiter=solver_maxiter,
        )
        posterior_weights = raw_posterior_weights.copy()
        turnover = float(np.abs(posterior_weights - prior_weights).sum())

        if prior_weights.shape != (n_assets,):
            raise RuntimeError(f"prior_weights shape mismatch: {prior_weights.shape}")
        if posterior_weights.shape != (n_assets,):
            raise RuntimeError(f"posterior_weights shape mismatch: {posterior_weights.shape}")
        if raw_posterior_weights.shape != (n_assets,):
            raise RuntimeError(f"raw_posterior_weights shape mismatch: {raw_posterior_weights.shape}")
        if abs(float(np.sum(prior_weights)) - 1.0) >= 1e-6:
            raise RuntimeError("prior_weights do not sum to 1")
        if abs(float(np.sum(posterior_weights)) - 1.0) >= 1e-6:
            raise RuntimeError("posterior_weights do not sum to 1")

        return OptimizationResult(
            tickers=tickers,
            prior_returns=pi_eq.tolist(),
            prior_weights=prior_weights.tolist(),
            posterior_returns=posterior_returns.tolist(),
            posterior_weights=posterior_weights.tolist(),
            raw_posterior_weights=raw_posterior_weights.tolist(),
            recommended_weights=posterior_weights.tolist(),
            turnover=turnover,
            effective_min_weight=effective_min,
            risk_aversion=risk_aversion,
            allocation_model=allocation_model,
            prior_source=prior_source,
            tau=eff_tau,
            view_matrix=P.tolist(),
            view_returns=Q.tolist(),
            view_confidences=confidences.tolist(),
            omega_diagonal=np.diag(omega).tolist(),
            posterior_covariance=posterior_cov.tolist(),
            effective_max_weight=self._effective_max_weight(max_weight, n_assets),
            turnover_penalty=max(float(turnover_penalty), 0.0),
            concentration_penalty=max(float(concentration_penalty), 0.0),
            solver_method=str(solver_diagnostics["method"]),
            solver_success=bool(solver_diagnostics["success"]),
            solver_status=int(solver_diagnostics["status"]),
            solver_message=str(solver_diagnostics["message"]),
            solver_objective=float(solver_diagnostics["objective"]),
        )
