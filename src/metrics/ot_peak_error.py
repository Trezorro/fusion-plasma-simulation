"""Unbalanced optimal-transport distance between two sets of detected peaks.

Why this exists
---------------
The rollout tables previously scored peak agreement with a 1-D Wasserstein distance between
the *prominence distributions* of the generated and the real peaks. That comparison throws
away two things that matter for ELMs. It ignores *where* the peaks are, so a model that emits
the right prominence histogram at the wrong times is scored perfect; and, being a distance
between normalized distributions, it ignores *how much* peak there is, so a model that emits
a hundred small noise peaks instead of the handful of large real ELMs pays almost nothing for
the missing large ones. Both failure modes are exactly the behaviour the paper is trying to
measure, so the metric was rewarding the thing it should penalise.

This module replaces it with an unbalanced optimal-transport cost over the peaks themselves.
Each peak is a unit of *mass* (its prominence, or its width) sitting at a *time*. Moving mass
from a generated peak to a real one costs the time displacement; mass that cannot be matched
cheaply is destroyed (a spurious generated peak) or created (a missed real ELM) at a fixed
price `lam` per unit mass. A model that produces many small peaks now pays for every one of
them in false mass, and pays the full `lam * mass` of every large ELM it fails to place.

Formulation
-----------
With G the transport plan between generated masses `a` (at times `gen_t`) and real masses `b`
(at times `real_t`), and M[i, j] = |gen_t[i] - real_t[j]|:

    E = sum(G * M) + lam * ( sum|a - G.sum(1)| + sum|b - G.sum(0)| )

which is unbalanced OT with a total-variation penalty on both marginals. It is solved exactly
rather than with an entropic or L-BFGS-B solver: adding one dummy row and one dummy column
that absorb unmatched mass at cost `lam` turns it into a *balanced* problem that the network
simplex (`ot.emd`) solves to optimality in milliseconds at the pool sizes here (a few hundred
peaks). The iterative unbalanced solvers in POT are both slower and only approximate: measured
on 200 synthetic peaks, `ot.unbalanced.lbfgsb_unbalanced` takes ~9 s against ~10 ms here and
returns a value a few percent above the true optimum, and its regularization strength would be
one more solver knob to describe in the paper. There is nothing to justify here: the number is
the exact minimum of the expression above.

Interpreting `lam`
------------------
`lam` is the price of one unit of unmatched mass, in the same time unit as `gen_t`/`real_t`
(milliseconds throughout this repo). Two consequences are worth stating explicitly.

  * A generated peak and a real peak are matched only when moving the mass is cheaper than
    destroying it on one side and creating it on the other, so the effective maximum transport
    distance is **2*lam**, not `lam`. For a "peaks may be displaced by up to X ms" reading,
    pass lam = X / 2.
  * The score is an absolute quantity in (mass * ms), not a normalized one. A quiet shot with
    few peaks scores lower than an ELMy one for both models, so shots must be aggregated as an
    unweighted mean over shots (as everything else in the tables is) and never pooled. The
    `relative` field divides by `lam * b.sum()`, the cost of generating nothing at all, which
    puts every shot on a common scale where 1.0 = "no better than an empty prediction".

Degenerate cases fall out of the same expression: with no generated peaks the cost is
`lam * b.sum()` (all real mass missed), with no real peaks it is `lam * a.sum()` (all generated
mass spurious), and with neither it is 0. Unlike the old Wasserstein column, which was
undefined whenever either side was empty and had to be restricted to windows where both sides
had peaks, this is defined everywhere, so no slice is dropped and a model that emits nothing
is scored rather than excused.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ot


@dataclass(frozen=True)
class OTPeakError:
    """The unbalanced OT cost and the three pieces it decomposes into.

    Attributes:
        total: loc + lam * (missed_mass + false_mass); the metric.
        loc: transport cost, sum(G * M), in mass * time units. Small when the matched peaks
            are close in time; it says nothing about the peaks that went unmatched.
        missed_mass: real mass with no generated counterpart (missed ELMs).
        false_mass: generated mass with no real counterpart (spurious peaks).
        gen_mass: total generated mass, for reference.
        real_mass: total real mass, for reference.
        relative: total / (lam * real_mass); 1.0 = as costly as predicting nothing. NaN when
            the real side is empty, since there is no reference cost to divide by.
    """

    total: float
    loc: float
    missed_mass: float
    false_mass: float
    gen_mass: float
    real_mass: float
    relative: float


def ot_peak_error(gen_t, gen_m, real_t, real_m, lam: float) -> OTPeakError:
    """Unbalanced OT cost between two peak sets; see the module docstring.

    Args:
        gen_t, real_t: peak times, in the same unit as `lam` (milliseconds here). Positions
            must be on a shared clock: the metric is about *when* peaks occur, so both sets
            have to be measured from the same origin.
        gen_m, real_m: peak masses, strictly positive. Prominence for "how big is this event",
            width for "how long does it last".
        lam: price per unit of unmatched mass. The effective maximum transport distance is
            2 * lam.

    Returns:
        An OTPeakError.
    """
    gen_t = np.asarray(gen_t, dtype=np.float64).ravel()
    real_t = np.asarray(real_t, dtype=np.float64).ravel()
    a = np.asarray(gen_m, dtype=np.float64).ravel()
    b = np.asarray(real_m, dtype=np.float64).ravel()
    if gen_t.shape != a.shape or real_t.shape != b.shape:
        raise ValueError("peak times and masses must have the same length on each side")
    if lam <= 0:
        raise ValueError(f"lam must be positive, got {lam}")
    # A zero-mass peak contributes nothing but makes the LP degenerate, and a negative mass is
    # meaningless: find_peaks cannot return either, so this is a wiring error, not a data case.
    if (a < 0).any() or (b < 0).any():
        raise ValueError("peak masses must be non-negative")

    gen_mass, real_mass = float(a.sum()), float(b.sum())

    if len(a) == 0 or len(b) == 0:
        total = lam * (real_mass if len(a) == 0 else gen_mass)
        return OTPeakError(
            total=total, loc=0.0,
            missed_mass=real_mass if len(a) == 0 else 0.0,
            false_mass=gen_mass if len(b) == 0 else 0.0,
            gen_mass=gen_mass, real_mass=real_mass,
            relative=np.nan if real_mass == 0 else total / (lam * real_mass),
        )

    n, m = len(a), len(b)
    # Augmented balanced problem: row n is a source holding `real_mass` that can create mass at
    # any real peak for `lam` each, column m is a sink that absorbs generated mass for `lam`
    # each, and the corner passes whatever neither side needs through at no cost. Both margins
    # then sum to gen_mass + real_mass, so `ot.emd` applies and its optimum is the unbalanced
    # optimum of the expression in the module docstring.
    cost = np.empty((n + 1, m + 1), dtype=np.float64)
    cost[:n, :m] = np.abs(gen_t[:, None] - real_t[None, :])
    cost[:n, m] = lam
    cost[n, :m] = lam
    cost[n, m] = 0.0
    margin_a = np.concatenate([a, [real_mass]])
    margin_b = np.concatenate([b, [gen_mass]])

    # numItermax is raised well above the POT default: the default 100k aborts with a warning
    # and returns a non-optimal plan on the denser pools, which would silently understate the
    # cost for exactly the noisy models the metric exists to catch.
    plan = ot.emd(margin_a, margin_b, cost, numItermax=10_000_000)

    loc = float((plan[:n, :m] * cost[:n, :m]).sum())
    false_mass = float(plan[:n, m].sum())
    missed_mass = float(plan[n, :m].sum())
    total = loc + lam * (missed_mass + false_mass)
    return OTPeakError(
        total=total, loc=loc, missed_mass=missed_mass, false_mass=false_mass,
        gen_mass=gen_mass, real_mass=real_mass,
        relative=np.nan if real_mass == 0 else total / (lam * real_mass),
    )
