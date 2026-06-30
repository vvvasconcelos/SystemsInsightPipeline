"""Scenario discovery: find interpretable regions of the input space where an outcome of
concern concentrates.

Given an ensemble of uncertain-input draws ``X`` and a scalar outcome ``y`` (plus a definition
of which cases are "of interest", e.g. ``y < threshold``), locate boxes / rules of the input
space where the interesting cases concentrate, reported with **density** (precision — share of
points in the region that are interesting) and **coverage** (recall — share of all interesting
cases captured).

Two methods are provided:

- ``"prim"`` — a small, dependency-free implementation of the Patient Rule Induction Method
  (Friedman & Fisher, 1999): iteratively *peel* thin slices off the current hyper-box to drive
  density up, recording the full coverage/density peeling trajectory so a caller can pick a box
  on the trade-off. This replaces any reliance on the EMA Workbench's ``prim``.
- ``"cart"`` — a classification tree (scikit-learn ``DecisionTreeClassifier``) whose
  high-density leaves are turned into box rules.

References
----------
- Friedman, J. H. & Fisher, N. I. (1999). Bump hunting in high-dimensional data. *Statistics
  and Computing*, 9(2).
- Kwakkel, J. H. & Jaxa-Rozen, M. (2016). Improving scenario discovery for handling
  heterogeneous uncertainties. *Environmental Modelling & Software*, 79.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class Box:
    """A hyper-rectangle in input space, with its quality metrics.

    ``limits`` is a DataFrame indexed by the *restricted* dimensions only, with columns
    ``min``/``max``. ``coverage`` is recall, ``density`` is precision, ``mass`` is the share of
    all points inside the box.
    """
    limits: pd.DataFrame
    coverage: float
    density: float
    mass: float
    n_points: int

    @property
    def restricted_dimensions(self) -> List[str]:
        return list(self.limits.index)

    def contains(self, X: pd.DataFrame) -> np.ndarray:
        mask = np.ones(len(X), dtype=bool)
        for dim, row in self.limits.iterrows():
            mask &= (X[dim].values >= row["min"]) & (X[dim].values <= row["max"])
        return mask

    def __repr__(self) -> str:
        dims = ", ".join(f"{d} in [{r['min']:.3g}, {r['max']:.3g}]"
                         for d, r in self.limits.iterrows())
        return (f"Box(density={self.density:.3f}, coverage={self.coverage:.3f}, "
                f"mass={self.mass:.3f}; {dims or 'unrestricted'})")


@dataclass
class ScenarioResult:
    """Result of :func:`discover_scenarios`."""
    method: str
    boxes: List[Box]
    mask: np.ndarray
    columns: List[str]
    trajectory: Optional[pd.DataFrame] = None          # PRIM peeling trajectory
    splitting_variables: List[str] = field(default_factory=list)
    rules: List[str] = field(default_factory=list)
    tree: object = None                                # fitted CART estimator (CART only)

    @property
    def box(self) -> Box:
        """The recommended box (highest density among those returned)."""
        if not self.boxes:
            raise ValueError("No boxes were found.")
        return max(self.boxes, key=lambda b: (b.density, b.coverage))

    def select_box(self, min_coverage: float = 0.0) -> Box:
        """Highest-density box whose coverage is at least ``min_coverage``."""
        eligible = [b for b in self.boxes if b.coverage >= min_coverage]
        if not eligible:
            raise ValueError(f"No box reaches coverage >= {min_coverage}.")
        return max(eligible, key=lambda b: b.density)


def _make_mask(y, threshold, direction, predicate) -> np.ndarray:
    y = np.asarray(y)
    if predicate is not None:
        mask = np.asarray(predicate(y), dtype=bool)
    elif y.dtype == bool:
        mask = y
    else:
        if threshold is None:
            raise ValueError("For a continuous outcome, pass `threshold` (or a `predicate`).")
        if direction == "below":
            mask = y < threshold
        elif direction == "above":
            mask = y > threshold
        else:
            raise ValueError(f"direction must be 'below' or 'above', got {direction!r}.")
    mask = np.asarray(mask, dtype=bool).ravel()
    n_int = int(mask.sum())
    if n_int == 0:
        raise ValueError("No cases of interest under the given threshold/predicate.")
    if n_int == len(mask):
        raise ValueError("All cases are 'of interest'; the predicate does not discriminate.")
    return mask


def _box_from_limits(X, mask, lo, hi, columns) -> Box:
    n = len(X)
    total_int = int(mask.sum())
    inside = np.ones(n, dtype=bool)
    restricted = {}
    gmin, gmax = X.min(), X.max()
    for c in columns:
        inside &= (X[c].values >= lo[c]) & (X[c].values <= hi[c])
        if lo[c] > gmin[c] + 1e-12 or hi[c] < gmax[c] - 1e-12:
            restricted[c] = (lo[c], hi[c])
    n_box = int(inside.sum())
    n_box_int = int((mask & inside).sum())
    limits = pd.DataFrame(
        [(c, v[0], v[1]) for c, v in restricted.items()],
        columns=["dimension", "min", "max"],
    ).set_index("dimension")
    return Box(
        limits=limits,
        coverage=(n_box_int / total_int) if total_int else 0.0,
        density=(n_box_int / n_box) if n_box else 0.0,
        mass=n_box / n,
        n_points=n_box,
    )


def _prim(X, mask, peel_alpha, min_coverage, min_mass, columns) -> ScenarioResult:
    n = len(X)
    lo = {c: float(X[c].min()) for c in columns}
    hi = {c: float(X[c].max()) for c in columns}

    boxes = [_box_from_limits(X, mask, lo, hi, columns)]
    traj = [(boxes[0].coverage, boxes[0].density, boxes[0].mass)]

    while True:
        inside = np.ones(n, dtype=bool)
        for c in columns:
            inside &= (X[c].values >= lo[c]) & (X[c].values <= hi[c])
        cur = np.where(inside)[0]
        if len(cur) <= max(2, int(min_mass * n)):
            break

        best = None  # (density, coverage, dim, side, new_lo, new_hi)
        for c in columns:
            vals = X[c].values[cur]
            q_lo = np.quantile(vals, peel_alpha)
            q_hi = np.quantile(vals, 1.0 - peel_alpha)
            for side in ("low", "high"):
                new_lo, new_hi = dict(lo), dict(hi)
                if side == "low":
                    keep = vals > q_lo
                    if keep.sum() == len(vals) or keep.sum() == 0:
                        continue
                    new_lo[c] = float(vals[keep].min())
                else:
                    keep = vals < q_hi
                    if keep.sum() == len(vals) or keep.sum() == 0:
                        continue
                    new_hi[c] = float(vals[keep].max())
                cand = _box_from_limits(X, mask, new_lo, new_hi, columns)
                if cand.coverage < min_coverage or cand.n_points == 0:
                    continue
                key = (cand.density, cand.coverage)
                if best is None or key > best[0]:
                    best = (key, c, side, new_lo, new_hi, cand)

        if best is None or best[5].density <= boxes[-1].density + 1e-9:
            # no peel improves density without dropping below min_coverage -> stop
            if best is None:
                break
            # allow continued peeling only while density is non-decreasing
            if best[5].density < boxes[-1].density - 1e-12:
                break
        _, c, side, lo, hi, cand = best
        boxes.append(cand)
        traj.append((cand.coverage, cand.density, cand.mass))
        if cand.density >= 1.0 - 1e-12:
            break

    trajectory = pd.DataFrame(traj, columns=["coverage", "density", "mass"])
    trajectory.index.name = "peel_step"
    return ScenarioResult(method="prim", boxes=boxes, mask=mask,
                          columns=list(columns), trajectory=trajectory)


def _cart(X, mask, max_depth, min_density, seed, columns) -> ScenarioResult:
    try:
        from sklearn.tree import DecisionTreeClassifier, _tree
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "method='cart' requires scikit-learn. Install it with `pip install scikit-learn`."
        ) from exc

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    clf.fit(X.values, mask.astype(int))
    tree = clf.tree_
    feat_names = list(columns)

    splitting = sorted({feat_names[tree.feature[i]] for i in range(tree.node_count)
                        if tree.feature[i] != _tree.TREE_UNDEFINED})

    boxes: List[Box] = []
    rules: List[str] = []

    def recurse(node, lo, hi, conds):
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            name = feat_names[tree.feature[node]]
            thr = tree.threshold[node]
            left_hi = dict(hi); left_hi[name] = min(hi[name], thr)
            recurse(tree.children_left[node], lo, left_hi, conds + [f"{name} <= {thr:.4g}"])
            right_lo = dict(lo); right_lo[name] = max(lo[name], thr)
            recurse(tree.children_right[node], right_lo, hi, conds + [f"{name} > {thr:.4g}"])
            return
        # leaf
        box = _box_from_limits(X, mask, lo, hi, columns)
        if box.density >= min_density and box.n_points > 0 and len(box.restricted_dimensions) > 0:
            boxes.append(box)
            rules.append(" and ".join(conds) + f"  -> density={box.density:.3f}, "
                         f"coverage={box.coverage:.3f}, mass={box.mass:.3f}")

    lo0 = {c: float(X[c].min()) for c in columns}
    hi0 = {c: float(X[c].max()) for c in columns}
    recurse(0, lo0, hi0, [])
    boxes.sort(key=lambda b: (b.density, b.coverage), reverse=True)

    return ScenarioResult(method="cart", boxes=boxes, mask=mask, columns=list(columns),
                          splitting_variables=splitting, rules=rules, tree=clf)


def discover_scenarios(X, y, *, threshold=None, direction="below", predicate=None,
                       method="prim", peel_alpha=0.05, min_coverage=0.0, min_mass=0.05,
                       max_depth=4, min_density=0.5, seed=None) -> ScenarioResult:
    """Find interpretable regions of input space where the cases of interest concentrate.

    Parameters
    ----------
    X : pandas.DataFrame
        ``n x d`` ensemble of uncertain inputs (e.g. from :meth:`SDM.sample_outcomes`).
    y : array-like
        Length-``n`` continuous outcome, or a boolean mask of interesting cases.
    threshold, direction : float, {"below", "above"}
        For a continuous ``y``: interesting := ``y < threshold`` (``"below"``) or ``y > threshold``.
    predicate : callable, optional
        ``callable(y) -> bool mask`` overriding ``threshold``/``direction``.
    method : {"prim", "cart"}
        PRIM peeling (default) or a CART classification tree.
    peel_alpha : float
        PRIM peeling fraction per step.
    min_coverage : float
        PRIM stops peeling once coverage would drop below this.
    min_mass : float
        PRIM stops when the box holds this fraction of all points or fewer.
    max_depth, min_density : int, float
        CART tree depth and the minimum leaf density to report as a rule.
    seed : int, optional
        Seed for the CART tree (PRIM is deterministic).

    Returns
    -------
    ScenarioResult
        With ``.boxes`` (the regions found), ``.box`` (the recommended one), ``.trajectory``
        (PRIM coverage/density per peel step), and for CART ``.splitting_variables``,
        ``.rules`` and ``.tree``.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(np.asarray(X))
        X.columns = [f"x{i+1}" for i in range(X.shape[1])]
    if len(X) != len(np.asarray(y).ravel()):
        raise ValueError("X and y must have the same number of rows.")

    mask = _make_mask(y, threshold, direction, predicate)
    columns = list(X.columns)

    if method == "prim":
        return _prim(X, mask, peel_alpha, min_coverage, min_mass, columns)
    if method == "cart":
        return _cart(X, mask, max_depth, min_density, seed, columns)
    raise ValueError(f"method must be 'prim' or 'cart', got {method!r}.")
