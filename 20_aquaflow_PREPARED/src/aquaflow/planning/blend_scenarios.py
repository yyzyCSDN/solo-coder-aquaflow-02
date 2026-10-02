from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import product

from .source_blend import Source, Target, blended_quality

# Lifecycle states.
DRAFT = "draft"
IN_REVIEW = "in_review"
APPROVED = "approved"
REJECTED = "rejected"

# Allowed lifecycle transitions; approval is additionally gated on feasibility.
TRANSITIONS = {
    (DRAFT, "submit"): IN_REVIEW,
    (DRAFT, "reject"): REJECTED,
    (IN_REVIEW, "approve"): APPROVED,
    (IN_REVIEW, "reject"): REJECTED,
}
# States in which a plan may still be recomputed/edited.
MUTABLE_STATES = frozenset({DRAFT, IN_REVIEW})

_EPS = 1e-9


class ScenarioError(ValueError):
    """Raised for invalid lifecycle actions or unapprovable plans."""


@dataclass(frozen=True)
class BlendResult:
    """Evaluation of one concrete flow combination."""

    flows: dict[str, float]
    total_flow: float
    blended_quality: dict[str, float]
    cost: float
    feasible: bool
    rejection: tuple[str, ...]


@dataclass(frozen=True)
class RejectedCandidate:
    """A candidate mix that was cheaper than the selection but infeasible."""

    flows: dict[str, float]
    cost: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Recommendation:
    """Outcome of a blend search: selection plus cheaper rejected alternatives."""

    flows: dict[str, float]
    total_flow: float
    blended_quality: dict[str, float]
    cost: float
    feasible: bool
    rejection: tuple[str, ...]
    rejected: tuple[RejectedCandidate, ...] = ()
    # A strictly cheaper, feasible mix than a manually selected one (if any).
    cheaper_feasible_alternative: dict[str, float] | None = None
    unavailable: frozenset[str] = frozenset()


@dataclass(frozen=True)
class HistoryEvent:
    action: str
    state_from: str
    state_to: str
    at: datetime | None
    detail: str = ""


@dataclass
class Scenario:
    name: str
    sources: tuple[Source, ...]
    target: Target
    flows: dict[str, float]
    state: str = DRAFT
    step: float = 10.0
    result: BlendResult | None = None
    rejected: tuple[RejectedCandidate, ...] = ()
    cheaper_feasible_alternative: dict[str, float] | None = None
    unavailable: frozenset[str] = frozenset()
    history: tuple[HistoryEvent, ...] = ()
    version: int = 0


def _ordered_sources(sources) -> list[Source]:
    out = list(sources)
    seen: set[str] = set()
    for s in out:
        if s.name in seen:
            raise ValueError(f"duplicate source name: {s.name}")
        seen.add(s.name)
    return out


def _quality_keys(sources, target) -> list[str]:
    keys = set(target.limits)
    for s in sources:
        keys.update(s.quality)
    return sorted(keys)


def evaluate(
    sources: list[Source],
    flows: dict[str, float],
    target: Target,
    tolerance: float = _EPS,
    unavailable: frozenset[str] = frozenset(),
) -> BlendResult:
    """Evaluate a single flow combination against demand and quality limits.

    The blend is feasible iff every source is within its capacity (and not
    temporarily unavailable), total flow meets demand within ``tolerance``,
    and every flow-weighted quality parameter respects its limit.
    """
    srcs = _ordered_sources(sources)
    if set(flows) != {s.name for s in srcs}:
        raise ValueError("flows must name every source exactly once")
    vals = [float(flows[s.name]) for s in srcs]

    reasons: list[str] = []
    for s, f in zip(srcs, vals):
        if f < -_EPS:
            reasons.append(f"negative_flow:{s.name}={f:g}")
        if f > s.max_flow + _EPS:
            reasons.append(
                f"capacity_exceeded:{s.name}:flow={f:g},max={s.max_flow:g}"
            )
        if f > _EPS and s.name in unavailable:
            reasons.append(f"source_unavailable:{s.name}")

    total = sum(vals)
    if abs(total - target.demand) > tolerance:
        reasons.append(
            f"demand_unmet:flow={total:g},demand={target.demand:g}"
        )

    quality: dict[str, float] = {}
    for k in _quality_keys(srcs, target):
        v = blended_quality(srcs, vals, k)
        quality[k] = v
        limit = target.limits.get(k)
        if limit is not None and v > limit + _EPS:
            reasons.append(f"quality:{k}={v:g},limit={limit:g}")

    cost = sum(f * s.unit_cost for s, f in zip(srcs, vals))
    return BlendResult(
        flows=dict(flows),
        total_flow=total,
        blended_quality=quality,
        cost=cost,
        feasible=not reasons,
        rejection=tuple(reasons),
    )


def _grid(sources, step: float, unavailable: frozenset[str]) -> list[list[float]]:
    grid: list[list[float]] = []
    for s in sources:
        if s.name in unavailable:
            grid.append([0.0])
        else:
            grid.append([i * step for i in range(int(s.max_flow // step) + 1)])
    return grid


def recommend(
    sources: list[Source],
    target: Target,
    step: float = 10.0,
    unavailable: frozenset[str] = frozenset(),
    selected_flows: dict[str, float] | None = None,
) -> Recommendation:
    """Search the flow grid and return the deterministic best feasible mix.

    Candidates are ordered by ``(cost, flows_tuple)``: cheapest wins, and on
    equal cost the mix with the lexicographically smallest flow vector wins
    (source order is fixed), giving a deterministic decision. Every cheaper
    infeasible candidate is returned with its rejection reasons.
    """
    srcs = _ordered_sources(sources)
    if not srcs:
        raise ValueError("at least one source is required")
    if target.demand <= 0 or step <= 0:
        raise ValueError("target demand and step must be positive")
    unknown = set(unavailable) - {s.name for s in srcs}
    if unknown:
        raise ValueError(f"unknown unavailable source: {sorted(unknown)[0]}")

    names = [s.name for s in srcs]
    feasible: list[tuple[float, tuple[float, ...], BlendResult]] = []
    infeasible: list[tuple[float, tuple[float, ...], BlendResult]] = []
    for vec in product(*_grid(srcs, step, unavailable)):
        flows = dict(zip(names, vec))
        res = evaluate(srcs, flows, target, tolerance=step / 2,
                       unavailable=unavailable)
        entry = (res.cost, vec, res)
        (feasible if res.feasible else infeasible).append(entry)

    if selected_flows is not None:
        chosen = evaluate(
            srcs, selected_flows, target,
            tolerance=step / 2, unavailable=unavailable,
        )
        chosen_cost = chosen.cost
    elif feasible:
        chosen_cost, chosen_vec, chosen = min(
            feasible, key=lambda e: (e[0], e[1])
        )
    else:
        # Nothing feasible: surface the most useful diagnostic candidate.
        # Prefer mixes that meet demand (showing which quality limit blocks),
        # then closest flow to demand, then cheapest; flows break final ties.
        def diag_key(e):
            _, vec, res = e
            demand_gap = abs(res.total_flow - target.demand)
            q_violations = sum(1 for r in res.rejection if r.startswith("quality:"))
            return (demand_gap > step / 2, demand_gap, q_violations, res.cost, vec)

        chosen_cost, chosen_vec, chosen = min(infeasible, key=diag_key)

    # A strictly cheaper feasible mix only exists relative to a manual
    # selection; the optimized choice is itself the cheapest feasible mix.
    cheaper_feasible = None
    if feasible:
        alt_cost, alt_vec, _ = min(feasible, key=lambda e: (e[0], e[1]))
        if alt_cost < chosen_cost - _EPS:
            cheaper_feasible = dict(zip(names, alt_vec))

    rejected = tuple(
        RejectedCandidate(
            flows=dict(zip(names, vec)),
            cost=res.cost,
            reasons=res.rejection,
        )
        for cost, vec, res in sorted(
            infeasible, key=lambda e: (e[0], e[1])
        )
        if cost < chosen_cost - _EPS
    )

    return Recommendation(
        flows=dict(chosen.flows),
        total_flow=chosen.total_flow,
        blended_quality=chosen.blended_quality,
        cost=chosen.cost,
        feasible=chosen.feasible,
        rejection=chosen.rejection,
        rejected=rejected,
        cheaper_feasible_alternative=cheaper_feasible,
        unavailable=frozenset(unavailable),
    )


class BlendWorkbench:
    """In-memory registry of named multi-source blend scenarios.

    Approved plans are frozen: their source catalogue, demand/quality target
    and selected flows can never be mutated. Drafts and plans under review may
    be recomputed when a source becomes temporarily unavailable.
    """

    def __init__(self):
        self._scenarios: dict[str, Scenario] = {}

    # -- basic access ------------------------------------------------------
    def get(self, name: str) -> Scenario:
        try:
            return self._scenarios[name]
        except KeyError:
            raise KeyError(f"unknown scenario: {name}") from None

    def names(self) -> list[str]:
        return sorted(self._scenarios)

    # -- lifecycle ---------------------------------------------------------
    def create_draft(
        self,
        name: str,
        sources: list[Source],
        target: Target,
        step: float = 10.0,
        flows: dict[str, float] | None = None,
        unavailable: frozenset[str] = frozenset(),
        at: datetime | None = None,
    ) -> Scenario:
        if not name:
            raise ValueError("scenario name is required")
        if name in self._scenarios:
            raise ValueError(f"scenario already exists: {name}")
        srcs = tuple(_ordered_sources(sources))
        rec = recommend(srcs, target, step=step, unavailable=unavailable,
                        selected_flows=dict(flows) if flows is not None else None)
        sc = Scenario(
            name=name,
            sources=srcs,
            target=target,
            flows=dict(rec.flows),
            state=DRAFT,
            step=step,
            result=self._result_of(rec),
            rejected=rec.rejected,
            cheaper_feasible_alternative=rec.cheaper_feasible_alternative,
            unavailable=frozenset(unavailable),
            history=(HistoryEvent("create", DRAFT, DRAFT, at,
                                  "manual selection" if flows is not None
                                  else "optimized selection"),),
        )
        self._scenarios[name] = sc
        return sc

    def choose_flows(self, name: str, flows: dict[str, float],
                     at: datetime | None = None) -> Scenario:
        """Manually override the selected flow combination (drafts only)."""
        sc = self._require(name, {DRAFT})
        rec = recommend(sc.sources, sc.target, step=sc.step,
                        unavailable=sc.unavailable, selected_flows=dict(flows))
        self._apply_recommendation(sc, rec, "choose_flows", at)
        return sc

    def recalculate(self, name: str,
                    unavailable: frozenset[str] = frozenset(),
                    at: datetime | None = None) -> Scenario:
        """Re-optimize with sources temporarily unavailable.

        Only drafts and plans under review are recomputed. Approved and
        rejected historical plans are never modified.
        """
        sc = self._require(name, MUTABLE_STATES)
        rec = recommend(sc.sources, sc.target, step=sc.step,
                        unavailable=frozenset(unavailable))
        self._apply_recommendation(sc, rec, "recalculate", at,
                                   f"unavailable={sorted(unavailable)}")
        return sc

    def submit(self, name: str, at: datetime | None = None) -> Scenario:
        return self._transition(name, "submit", {DRAFT}, at)

    def reject(self, name: str, reason: str = "",
               at: datetime | None = None) -> Scenario:
        if not reason:
            raise ValueError("a rejection reason is required")
        return self._transition(name, "reject", {DRAFT, IN_REVIEW}, at, reason)

    def approve(self, name: str, at: datetime | None = None) -> Scenario:
        sc = self._require(name, {IN_REVIEW})
        assert sc.result is not None
        if not sc.result.feasible:
            raise ScenarioError(
                f"cannot approve {name}: " + "; ".join(sc.result.rejection)
            )
        return self._transition(name, "approve", {IN_REVIEW}, at)

    # -- internals ---------------------------------------------------------
    def _require(self, name, allowed) -> Scenario:
        sc = self.get(name)
        if sc.state not in allowed:
            raise ScenarioError(
                f"illegal action on {name} in state {sc.state}"
            )
        return sc

    def _transition(self, name, action, allowed, at, detail="") -> Scenario:
        sc = self._require(name, allowed)
        to = TRANSITIONS[(sc.state, action)]
        sc.history += (HistoryEvent(action, sc.state, to, at, detail),)
        sc.state = to
        sc.version += 1
        return sc

    @staticmethod
    def _result_of(rec: Recommendation) -> BlendResult:
        return BlendResult(
            flows=dict(rec.flows),
            total_flow=rec.total_flow,
            blended_quality=dict(rec.blended_quality),
            cost=rec.cost,
            feasible=rec.feasible,
            rejection=rec.rejection,
        )

    def _apply_recommendation(self, sc, rec, action, at, detail="") -> None:
        sc.flows = dict(rec.flows)
        sc.result = self._result_of(rec)
        sc.rejected = rec.rejected
        sc.cheaper_feasible_alternative = rec.cheaper_feasible_alternative
        sc.unavailable = rec.unavailable
        sc.history += (HistoryEvent(action, sc.state, sc.state, at, detail),)
        sc.version += 1
