import unittest

from aquaflow.api import AquaFlowService
from aquaflow.planning.source_blend import Source, Target
from aquaflow.planning.blend_scenarios import (
    BlendWorkbench,
    ScenarioError,
    recommend,
    DRAFT,
    IN_REVIEW,
    APPROVED,
    REJECTED,
)


def sources():
    # A: cheap-ish clean, B: cheapest but dirty (capacity 60), C: pricey clean.
    return [
        Source("A", max_flow=100.0, unit_cost=2.0, quality={"turbidity": 1.0}),
        Source("B", max_flow=60.0, unit_cost=1.0, quality={"turbidity": 5.0}),
        Source("C", max_flow=100.0, unit_cost=3.0, quality={"turbidity": 0.0}),
    ]


TARGET = Target(demand=100.0, limits={"turbidity": 2.0})


class RecommendationTests(unittest.TestCase):
    def test_returns_flows_quality_cost(self):
        rec = recommend(sources(), TARGET, step=10.0)
        self.assertTrue(rec.feasible)
        self.assertEqual(rec.flows, {"A": 80.0, "B": 20.0, "C": 0.0})
        self.assertEqual(rec.total_flow, 100.0)
        # (80*1 + 20*5) / 100 = 1.8
        self.assertAlmostEqual(rec.blended_quality["turbidity"], 1.8)
        self.assertEqual(rec.cost, 80 * 2 + 20 * 1)

    def test_cheaper_infeasible_candidates_are_rejected_with_reasons(self):
        rec = recommend(sources(), TARGET, step=10.0)
        self.assertTrue(rec.rejected)
        # Every rejected candidate is strictly cheaper than the selection.
        for cand in rec.rejected:
            self.assertLess(cand.cost, rec.cost)
            self.assertTrue(cand.reasons)
        # The all-cheapest-but-too-dirty mix B60/A40 (cost 140) is among them.
        dirty = next(c for c in rec.rejected if c.flows["B"] == 60.0
                     and c.flows["A"] == 40.0)
        self.assertTrue(any(r.startswith("quality:turbidity") for r in dirty.reasons))
        # Sorted deterministically by cost then flows.
        costs = [c.cost for c in rec.rejected]
        self.assertEqual(costs, sorted(costs))

    def test_deterministic_tie_break_on_equal_cost(self):
        # Two identically priced sources: multiple mixes share the same cost.
        equal = [
            Source("X", max_flow=100.0, unit_cost=1.0, quality={}),
            Source("Y", max_flow=100.0, unit_cost=1.0, quality={}),
        ]
        target = Target(demand=10.0, limits={})
        r1 = recommend(equal, target, step=10.0)
        r2 = recommend([Source("X", 100.0, 1.0, {}),
                        Source("Y", 100.0, 1.0, {})], target, step=10.0)
        # Lexicographically smallest flow vector wins, repeatably for the
        # same fixed source order.
        self.assertEqual(r1.flows, {"X": 0.0, "Y": 10.0})
        self.assertEqual(r1.flows, r2.flows)
        self.assertEqual(r1.cost, r2.cost)
        # And the rule is positional: reordering sources reorders the winner.
        r3 = recommend(list(reversed(equal)), target, step=10.0)
        self.assertEqual(r3.flows, {"X": 10.0, "Y": 0.0})

    def test_demand_cannot_be_met(self):
        small = [Source("A", 30.0, 1.0, {})]
        rec = recommend(small, Target(100.0, {}), step=10.0)
        self.assertFalse(rec.feasible)
        self.assertTrue(any(r.startswith("demand_unmet") for r in rec.rejection))

    def test_quality_constraint_unsatisfiable(self):
        dirty = [Source("A", 100.0, 1.0, {"turbidity": 9.0})]
        rec = recommend(dirty, TARGET, step=10.0)
        self.assertFalse(rec.feasible)
        self.assertTrue(any(r.startswith("quality:turbidity")
                            for r in rec.rejection))

    def test_unavailable_source_forces_recompute(self):
        rec = recommend(sources(), TARGET, step=10.0,
                        unavailable=frozenset({"B"}))
        self.assertTrue(rec.feasible)
        self.assertEqual(rec.flows["B"], 0.0)
        # Only A (turb 1 <= 2) is used; C is more expensive.
        self.assertEqual(rec.flows, {"A": 100.0, "B": 0.0, "C": 0.0})
        self.assertEqual(rec.cost, 200.0)

    def test_unavailable_source_makes_plan_infeasible(self):
        # Both sources dirty; removing either leaves no feasible blend.
        dirty = [Source("A", 100.0, 1.0, {"turbidity": 9.0}),
                 Source("B", 100.0, 2.0, {"turbidity": 8.0})]
        rec2 = recommend(dirty, TARGET, step=10.0,
                         unavailable=frozenset({"B"}))
        self.assertFalse(rec2.feasible)
        self.assertTrue(any(r.startswith("quality:turbidity")
                            for r in rec2.rejection))

    def test_manual_selection_flags_cheaper_feasible_alternative(self):
        rec = recommend(sources(), TARGET, step=10.0,
                        selected_flows={"A": 0.0, "B": 0.0, "C": 100.0})
        self.assertTrue(rec.feasible)
        self.assertEqual(rec.cost, 300.0)
        self.assertEqual(rec.cheaper_feasible_alternative,
                         {"A": 80.0, "B": 20.0, "C": 0.0})

    def test_manual_infeasible_selection_lists_cheaper_rejections(self):
        rec = recommend(sources(), TARGET, step=10.0,
                        selected_flows={"A": 40.0, "B": 60.0, "C": 0.0})
        self.assertFalse(rec.feasible)
        self.assertTrue(any(r.startswith("quality:turbidity")
                            for r in rec.rejection))


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.wb = BlendWorkbench()

    def test_named_draft_saves_optimized_selection(self):
        sc = self.wb.create_draft("summer", sources(), TARGET)
        self.assertEqual(sc.state, DRAFT)
        self.assertEqual(sc.flows, {"A": 80.0, "B": 20.0, "C": 0.0})
        self.assertEqual(self.wb.names(), ["summer"])
        self.assertTrue(sc.result.feasible)
        self.assertEqual(sc.history[0].action, "create")

    def test_full_approval_lifecycle(self):
        sc = self.wb.create_draft("summer", sources(), TARGET)
        sc = self.wb.submit("summer")
        self.assertEqual(sc.state, IN_REVIEW)
        sc = self.wb.approve("summer")
        self.assertEqual(sc.state, APPROVED)
        self.assertEqual(sc.version, 2)
        actions = [e.action for e in sc.history]
        self.assertEqual(actions, ["create", "submit", "approve"])

    def test_cannot_approve_when_demand_or_quality_unmet(self):
        # Capacity shortfall: nothing feasible -> stays draft, cannot even be
        # made approvable after submission.
        self.wb.create_draft("short", [Source("A", 30.0, 1.0, {})],
                             Target(100.0, {}))
        self.wb.submit("short")
        with self.assertRaises(ScenarioError) as cm:
            self.wb.approve("short")
        self.assertIn("demand_unmet", str(cm.exception))
        self.assertEqual(self.wb.get("short").state, IN_REVIEW)

        # Quality breach on a hand-picked mix blocks approval likewise.
        self.wb.create_draft("dirty", sources(), TARGET)
        self.wb.choose_flows("dirty",
                             {"A": 40.0, "B": 60.0, "C": 0.0})
        self.wb.submit("dirty")
        with self.assertRaises(ScenarioError) as cm:
            self.wb.approve("dirty")
        self.assertIn("quality:turbidity", str(cm.exception))
        self.assertEqual(self.wb.get("dirty").state, IN_REVIEW)

    def test_reject_requires_reason_and_closes_plan(self):
        self.wb.create_draft("p", sources(), TARGET)
        with self.assertRaises(ValueError):
            self.wb.reject("p", reason="")
        sc = self.wb.reject("p", reason="budget cut")
        self.assertEqual(sc.state, REJECTED)
        self.assertEqual(sc.history[-1].detail, "budget cut")
        with self.assertRaises(ScenarioError):
            self.wb.submit("p")

    def test_recalculate_with_temporary_unavailability(self):
        self.wb.create_draft("summer", sources(), TARGET)
        self.wb.submit("summer")
        sc = self.wb.recalculate("summer", unavailable=frozenset({"B"}))
        self.assertEqual(sc.state, IN_REVIEW)  # state preserved
        self.assertEqual(sc.flows, {"A": 100.0, "B": 0.0, "C": 0.0})
        self.assertEqual(sc.result.cost, 200.0)
        self.assertEqual(sc.history[-1].action, "recalculate")

    def test_approved_history_is_immutable(self):
        self.wb.create_draft("summer", sources(), TARGET)
        self.wb.submit("summer")
        self.wb.approve("summer")
        approved = self.wb.get("summer")
        snapshot = (dict(approved.flows), approved.state, approved.version,
                    approved.result.cost)

        with self.assertRaises(ScenarioError):
            self.wb.recalculate("summer", unavailable=frozenset({"B"}))
        with self.assertRaises(ScenarioError):
            self.wb.choose_flows("summer",
                                 {"A": 0.0, "B": 0.0, "C": 100.0})
        with self.assertRaises(ScenarioError):
            self.wb.submit("summer")
        with self.assertRaises(ScenarioError):
            self.wb.reject("summer", reason="late")
        with self.assertRaises(ScenarioError):
            self.wb.approve("summer")

        sc = self.wb.get("summer")
        self.assertEqual((dict(sc.flows), sc.state, sc.version,
                          sc.result.cost), snapshot)
        # Even though B is unavailable operationally, the approval stands.
        self.assertEqual(sc.flows, {"A": 80.0, "B": 20.0, "C": 0.0})

    def test_duplicate_and_unknown_names(self):
        self.wb.create_draft("summer", sources(), TARGET)
        with self.assertRaises(ValueError):
            self.wb.create_draft("summer", sources(), TARGET)
        with self.assertRaises(KeyError):
            self.wb.get("missing")

    def test_result_reports_mixed_quality_and_cost(self):
        sc = self.wb.create_draft("summer", sources(), TARGET)
        self.assertAlmostEqual(sc.result.blended_quality["turbidity"], 1.8)
        self.assertEqual(sc.result.cost, 180.0)
        self.assertEqual(sc.result.total_flow, 100.0)


class ServiceWiringTests(unittest.TestCase):
    def test_service_exposes_workbench_and_recommendation(self):
        svc = AquaFlowService()
        sc = svc.blend_workbench.create_draft("s", sources(), TARGET)
        self.assertEqual(sc.flows["B"], 20.0)
        rec = svc.blend_recommend(sources(), TARGET)
        self.assertTrue(rec.feasible)
        self.assertEqual(svc.health()["runtime_dependencies"], 0)


if __name__ == "__main__":
    unittest.main()
