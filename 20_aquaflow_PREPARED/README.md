# AquaFlow
Dependency-free municipal drinking-water treatment and distribution operations platform.

## Source blend scenarios
`aquaflow.planning.blend_scenarios` provides a workbench (`BlendWorkbench`) for
comparing named multi-source mixes. Each scenario stores its sources (max flow,
unit cost, quality parameters), demand/quality target, and selected flows, and
moves through `draft -> in_review -> approved|rejected`. Approval is refused
when demand or a quality limit cannot be met. `recommend(...)` returns per-source
flows, blended quality and cost, deterministic tie-breaking on equal cost
(`(cost, flows)` ordering), and the rejection reasons of every cheaper
infeasible candidate. Drafts and plans under review can be recomputed when a
source is temporarily unavailable; approved historical plans are immutable.
