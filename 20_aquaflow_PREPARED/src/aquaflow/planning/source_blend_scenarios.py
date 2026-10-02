"""Named multi-source blend scenarios with a draft/review/approve lifecycle.

Planners compare mixing schemes before changing intake settings.  Each
scenario snapshots the participating sources (max flow, unit cost and water
quality), the demand and the blended-quality limits, and stores the selected
flow combination.  The cheapest feasible combination on a deterministic flow
grid is proposed automatically; explicitly cheaper grid candidates that fail
demand or quality are retained with the reason they were rejected.

State machine::

    draft --submit--> review --approve--> approved
      ^                  |
      +----revise--------+
      |                  v
      +--------------- rejected

Approved scenarios are historical records: availability recalculation and
every mutating transition are refused, so an approved selection can never be
rewritten after the fact.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from itertools import product
from typing import Iterable

# Flow matching and quality comparisons tolerate float rounding noise.
EPS = 1e-9
DEFAULT_STEP = 10.0

DRAFT = "draft"
REVIEW = "review"
APPROVED = "approved"
REJECTED = "rejected"

TRANSITIONS = {
    (DRAFT, "submit"): REVIEW,
    (REVIEW, "approve"): APPROVED,
    (REVIEW, "reject"): REJECTED,
    (REVIEW, "revise"): DRAFT,
    (REJECTED, "revise"): DRAFT,
}

# Why a flow combination (or a cheaper candidate) is not feasible.
DEMAND_SHORT = "demand_short"
DEMAND_EXCESS = "demand_excess"
OVER_CAPACITY = "over_capacity"
NEGATIVE_FLOW = "negative_flow"
QUALITY_EXCEEDED = "quality_exceeded"
SOURCE_UNAVAILABLE = "source_unavailable"


@dataclass(frozen=True)
class WaterSource:
    name: str
    max_flow: float
    unit_cost: float
    quality: dict[str, float] = field(default_factory=dict)
    available: bool = True


@dataclass(frozen=True)
class BlendConstraints:
    demand: float
    limits: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RejectedCandidate:
    """A strictly cheaper grid candidate that could not be selected."""

    flows: dict[str, float]
    cost: float
    reasons: list[dict]


@dataclass(frozen=True)
class BlendResult:
    flows: dict[str, float]
    blended_quality: dict[str, float]
    cost: float
    feasible: bool
    reasons: list[dict]
    rejected_cheaper: list[RejectedCandidate]


@dataclass(frozen=True)
class BlendScenario:
    name: str
    sources: tuple[WaterSource, ...]
    constraints: BlendConstraints
    selection: dict[str, float]
    result: BlendResult
    state: str = DRAFT
    version: int = 1
    updated_at: datetime | None = None


def _ordered_sources(sources: Iterable[WaterSource]) -> list[WaterSource]:
    rows = list(sources)
    if not rows:
        raise ValueError("sources required")
    names = [s.name for s in rows]
    if len(set(names)) != len(names):
        raise ValueError("duplicate source name")
    for s in rows:
        if s.max_flow < 0 or s.unit_cost < 0:
            raise ValueError("source max_flow and unit_cost must be non-negative")
    # Name order fixes every downstream tie-break, independent of call order.
    return sorted(rows, key=lambda s: s.name)


def blended_quality(
    sources: Iterable[WaterSource], flows: dict[str, float]
) -> dict[str, float]:
    rows = list(sources)
    total = sum(flows.get(s.name, 0.0) for s in rows)
    if total <= EPS:
        return {}
    keys = sorted({k for s in rows for k in s.quality})
    return {
        key: sum(s.quality.get(key, 0.0) * flows.get(s.name, 0.0) for s in rows)
        / total
        for key in keys
    }


def total_cost(sources: Iterable[WaterSource], flows: dict[str, float]) -> float:
    return sum(s.unit_cost * flows.get(s.name, 0.0) for s in sources)


def infeasibility_reasons(
    sources: Iterable[WaterSource],
    flows: dict[str, float],
    constraints: BlendConstraints,
    unavailable: Iterable[str] = (),
) -> list[dict]:
    """Return the structured reasons a selection cannot be served."""
    rows = list(sources)
    reasons: list[dict] = []
    total = 0.0
    offline = {s.name for s in rows if not s.available} | set(unavailable)

    for s in rows:
        f = flows.get(s.name, 0.0)
        if f < -EPS:
            reasons.append({"code": NEGATIVE_FLOW, "source": s.name, "flow": f})
        if f > s.max_flow + EPS:
            reasons.append(
                {
                    "code": OVER_CAPACITY,
                    "source": s.name,
                    "flow": f,
                    "max_flow": s.max_flow,
                }
            )
        if s.name in offline and f > EPS:
            reasons.append({"code": SOURCE_UNAVAILABLE, "source": s.name})
        total += f

    if total + EPS < constraints.demand:
        reasons.append(
            {"code": DEMAND_SHORT, "total": total, "demand": constraints.demand}
        )
    if total - EPS > constraints.demand:
        reasons.append(
            {"code": DEMAND_EXCESS, "total": total, "demand": constraints.demand}
        )

    quality = blended_quality(rows, flows)
    for key in sorted(constraints.limits):
        value = quality.get(key, 0.0)
        limit = constraints.limits[key]
        if value > limit + EPS:
            reasons.append(
                {
                    "code": QUALITY_EXCEEDED,
                    "quality_key": key,
                    "value": value,
                    "limit": limit,
                }
            )
    return reasons


def evaluate(
    sources: Iterable[WaterSource],
    flows: dict[str, float],
    constraints: BlendConstraints,
    unavailable: Iterable[str] = (),
    include_rejected: bool = False,
    step: float = DEFAULT_STEP,
) -> BlendResult:
    """Evaluate one explicit flow combination against demand and quality."""
    rows = _ordered_sources(sources)
    if constraints.demand <= 0:
        raise ValueError("demand must be positive")
    full = {s.name: float(flows.get(s.name, 0.0)) for s in rows}
    reasons = infeasibility_reasons(rows, full, constraints, unavailable)
    rejected: list[RejectedCandidate] = []
    if include_rejected:
        rejected = _cheaper_rejections(rows, full, constraints, step)
    return BlendResult(
        flows=full,
        blended_quality=blended_quality(rows, full),
        cost=total_cost(rows, full),
        feasible=not reasons,
        reasons=reasons,
        rejected_cheaper=rejected,
    )


def _grid_values(source: WaterSource, step: float) -> list[float]:
    upper = int(source.max_flow // step)
    values = [i * step for i in range(upper + 1)]
    # A source whose capacity is not a whole multiple of steps can still max
    # out; include the exact cap as the final value.
    if values[-1] < source.max_flow - EPS:
        values.append(float(source.max_flow))
    return values


def _cheaper_rejections(
    rows: list[WaterSource],
    selection: dict[str, float],
    constraints: BlendConstraints,
    step: float,
) -> list[RejectedCandidate]:
    chosen_cost = total_cost(rows, selection)
    found: dict[tuple[float, ...], RejectedCandidate] = {}
    grids = [_grid_values(s, step) for s in rows]
    names = [s.name for s in rows]
    for combo in product(*grids):
        flows = dict(zip(names, combo))
        total = sum(combo)
        if abs(total - constraints.demand) > step / 2 + EPS:
            continue
        cost = total_cost(rows, flows)
        if cost >= chosen_cost - EPS:
            continue
        reasons = infeasibility_reasons(rows, flows, constraints)
        if not reasons:
            continue
        key = tuple(combo)
        if key not in found:
            found[key] = RejectedCandidate(
                flows=dict(flows), cost=cost, reasons=reasons
            )
    # Cheapest first; identical cost breaks on the ordered flow vector.
    return sorted(found.values(), key=lambda c: (c.cost, tuple(c.flows[n] for n in names)))


def optimize(
    sources: Iterable[WaterSource],
    constraints: BlendConstraints,
    unavailable: Iterable[str] = (),
    step: float = DEFAULT_STEP,
) -> BlendResult:
    """Find the cheapest feasible combination on the deterministic flow grid.

    Sources are first sorted by name.  Tie costs are broken by the
    lexicographically smallest flow vector in that name order, so equal-cost
    decisions are reproducible regardless of caller input order.  Every
    strictly cheaper demand-matching grid candidate that violates a limit is
    returned together with its rejection reason.
    """
    rows = _ordered_sources(sources)
    if constraints.demand <= 0:
        raise ValueError("demand must be positive")
    if step <= 0:
        raise ValueError("step must be positive")
    offline = {s.name for s in rows if not s.available} | set(unavailable)
    active = [s for s in rows if s.name not in offline]

    capacity = sum(s.max_flow for s in active)
    if capacity + EPS < constraints.demand:
        reasons = [
            {
                "code": DEMAND_SHORT,
                "total": capacity,
                "demand": constraints.demand,
            }
        ]
        return BlendResult(
            flows={s.name: 0.0 for s in rows},
            blended_quality={},
            cost=0.0,
            feasible=False,
            reasons=reasons,
            rejected_cheaper=[],
        )

    names = [s.name for s in rows]
    by_name = {s.name: s for s in rows}
    grids = []
    for name in names:
        if name in offline:
            grids.append([0.0])
        else:
            grids.append(_grid_values(by_name[name], step))

    best: BlendResult | None = None
    best_key = None
    quality_blocks: list[dict] = []
    closest_total: float | None = None
    closest_gap = float("inf")
    for combo in product(*grids):
        flows = dict(zip(names, combo))
        total = sum(combo)
        gap = abs(total - constraints.demand)
        if gap < closest_gap:
            closest_gap, closest_total = gap, total
        if gap > step / 2 + EPS:
            continue
        reasons = infeasibility_reasons(rows, flows, constraints)
        if not reasons:
            # Deterministic ordering: cost, then lexicographic flows by source name.
            key = (total_cost(rows, flows), tuple(combo))
            if best is None or key < best_key:
                best = BlendResult(
                    flows=flows,
                    blended_quality=blended_quality(rows, flows),
                    cost=key[0],
                    feasible=True,
                    reasons=[],
                    rejected_cheaper=[],
                )
                best_key = key
        else:
            for r in reasons:
                if r["code"] == QUALITY_EXCEEDED and r not in quality_blocks:
                    quality_blocks.append(r)

    if best is None:
        reasons: list[dict] = quality_blocks
        if not reasons:
            reasons = [
                {
                    "code": DEMAND_SHORT if (closest_total or 0.0) < constraints.demand else DEMAND_EXCESS,
                    "total": closest_total or 0.0,
                    "demand": constraints.demand,
                }
            ]
        return BlendResult(
            flows={s.name: 0.0 for s in rows},
            blended_quality={},
            cost=0.0,
            feasible=False,
            reasons=reasons,
            rejected_cheaper=[],
        )

    cheaper = _cheaper_rejections(rows, best.flows, constraints, step)
    return BlendResult(
        flows=dict(best.flows),
        blended_quality=dict(best.blended_quality),
        cost=best.cost,
        feasible=True,
        reasons=[],
        rejected_cheaper=cheaper,
    )


class BlendWorkbench:
    """In-memory registry of named blend scenarios with lifecycle control."""

    def __init__(self) -> None:
        self._scenarios: dict[str, BlendScenario] = {}

    # -- construction -------------------------------------------------------

    def create_draft(
        self,
        name: str,
        sources: Iterable[WaterSource],
        constraints: BlendConstraints,
        selection: dict[str, float] | None = None,
        unavailable: Iterable[str] = (),
        step: float = DEFAULT_STEP,
        at: datetime | None = None,
    ) -> BlendScenario:
        if not name:
            raise ValueError("scenario name required")
        if name in self._scenarios:
            raise ValueError(f"scenario already exists: {name}")
        rows = tuple(_ordered_sources(sources))
        if selection is None:
            result = optimize(rows, constraints, unavailable, step)
            if not result.feasible:
                raise ValueError("no feasible blend: " + _summarize(result.reasons))
            chosen = result.flows
        else:
            result = evaluate(
                rows, selection, constraints, unavailable, include_rejected=True, step=step
            )
            if not result.feasible:
                raise ValueError("selection infeasible: " + _summarize(result.reasons))
            chosen = result.flows
        scenario = BlendScenario(
            name=name,
            sources=rows,
            constraints=constraints,
            selection=chosen,
            result=result,
            updated_at=at,
        )
        self._scenarios[name] = scenario
        return scenario

    def get(self, name: str) -> BlendScenario:
        if name not in self._scenarios:
            raise KeyError(name)
        return self._scenarios[name]

    def list_scenarios(self) -> list[BlendScenario]:
        return [self._scenarios[name] for name in sorted(self._scenarios)]

    # -- lifecycle ----------------------------------------------------------

    def submit(self, name: str, at: datetime | None = None) -> BlendScenario:
        return self._transition(name, "submit", at)

    def approve(self, name: str, at: datetime | None = None) -> BlendScenario:
        scenario = self.get(name)
        self._guard_locked(scenario)
        if scenario.state != REVIEW:
            raise ValueError("invalid transition")
        # Demand or quality that cannot be satisfied blocks approval even if
        # the stored selection drifted since review.
        reasons = infeasibility_reasons(scenario.sources, scenario.selection, scenario.constraints)
        if reasons:
            raise ValueError("cannot approve infeasible blend: " + _summarize(reasons))
        return self._commit(scenario, APPROVED, at)

    def reject(self, name: str, at: datetime | None = None) -> BlendScenario:
        return self._transition(name, "reject", at)

    def revise(
        self,
        name: str,
        selection: dict[str, float] | None = None,
        at: datetime | None = None,
    ) -> BlendScenario:
        scenario = self.get(name)
        self._guard_locked(scenario)
        if scenario.state not in (REVIEW, REJECTED):
            raise ValueError("invalid transition")
        result = scenario.result
        chosen = scenario.selection
        if selection is not None:
            result = evaluate(
                scenario.sources,
                selection,
                scenario.constraints,
                include_rejected=True,
            )
            if not result.feasible:
                raise ValueError("selection infeasible: " + _summarize(result.reasons))
            chosen = result.flows
        return self._commit(
            replace(scenario, selection=chosen, result=result), DRAFT, at
        )

    def recalculate(
        self,
        name: str,
        unavailable: Iterable[str] = (),
        step: float = DEFAULT_STEP,
        at: datetime | None = None,
    ) -> BlendScenario:
        """Re-optimize when a source is temporarily unavailable.

        Approved history is immutable: recalculating an approved scenario
        raises instead of mutating it.  A fresh draft version is produced for
        the other states.
        """
        scenario = self.get(name)
        self._guard_locked(scenario)
        result = optimize(scenario.sources, scenario.constraints, unavailable, step)
        if not result.feasible:
            raise ValueError("no feasible blend: " + _summarize(result.reasons))
        return self._commit(
            replace(
                scenario,
                selection=result.flows,
                result=result,
                state=DRAFT,
            ),
            DRAFT,
            at,
        )

    # -- internals ----------------------------------------------------------

    def _transition(self, name: str, action: str, at: datetime | None) -> BlendScenario:
        scenario = self.get(name)
        self._guard_locked(scenario)
        target = TRANSITIONS.get((scenario.state, action))
        if target is None:
            raise ValueError("invalid transition")
        return self._commit(scenario, target, at)

    def _commit(
        self, scenario: BlendScenario, state: str, at: datetime | None
    ) -> BlendScenario:
        when = at
        if when is not None and scenario.updated_at is not None and when < scenario.updated_at:
            raise ValueError("time reversal")
        updated = replace(
            scenario,
            state=state,
            version=scenario.version + 1,
            updated_at=when or scenario.updated_at,
        )
        self._scenarios[scenario.name] = updated
        return updated

    @staticmethod
    def _guard_locked(scenario: BlendScenario) -> None:
        if scenario.state == APPROVED:
            raise ValueError("approved scenario is locked")


def _summarize(reasons: list[dict]) -> str:
    return ";".join(sorted({r["code"] for r in reasons}))
