import unittest
from datetime import datetime, timezone

from aquaflow.api import AquaFlowService
from aquaflow.planning.source_blend_scenarios import (
    APPROVED,
    DRAFT,
    REJECTED,
    REVIEW,
    BlendConstraints,
    BlendWorkbench,
    WaterSource,
    evaluate,
    optimize,
)


def sources():
    return [
        WaterSource("aquifer", 100, 1.0, {"turb": 10.0}),
        WaterSource("borehole", 100, 5.0, {"turb": 0.0}),
    ]


class OptimizeTests(unittest.TestCase):
    def test_cheapest_feasible_blend(self):
        r = optimize(sources(), BlendConstraints(100, {"turb": 3.0}), step=10)
        self.assertTrue(r.feasible)
        self.assertEqual(r.flows, {"aquifer": 30.0, "borehole": 70.0})
        self.assertEqual(r.cost, 380.0)
        self.assertAlmostEqual(r.blended_quality["turb"], 3.0)

    def test_rejected_cheaper_candidates_carry_reasons(self):
        r = optimize(sources(), BlendConstraints(100, {"turb": 3.0}), step=10)
        self.assertTrue(r.rejected_cheaper)
        cheapest = r.rejected_cheaper[0]
        # The cheapest candidate overall (100% aquifer) cannot be selected.
        self.assertEqual(cheapest.flows, {"aquifer": 100.0, "borehole": 0.0})
        self.assertEqual(cheapest.cost, 100.0)
        self.assertEqual([x["code"] for x in cheapest.reasons], ["quality_exceeded"])
        # Candidates are enumerated cheapest first.
        costs = [c.cost for c in r.rejected_cheaper]
        self.assertEqual(costs, sorted(costs))
        for c in r.rejected_cheaper:
            self.assertLess(c.cost, r.cost)

    def test_equal_cost_has_deterministic_order(self):
        rows = [
            WaterSource("zeta", 100, 1.0, {}),
            WaterSource("alpha", 100, 1.0, {}),
        ]
        first = optimize(rows, BlendConstraints(100), step=10)
        second = optimize(list(reversed(rows)), BlendConstraints(100), step=10)
        # Input order must not change the winner; sources are name-sorted and
        # the lexicographically smallest flow vector wins ties.
        self.assertEqual(first.flows, second.flows)
        self.assertEqual(first.flows, {"alpha": 0.0, "zeta": 100.0})

    def test_unavailable_source_is_excluded(self):
        r = optimize(sources(), BlendConstraints(100, {"turb": 3.0}),
                     unavailable=["aquifer"], step=10)
        self.assertTrue(r.feasible)
        self.assertEqual(r.flows, {"aquifer": 0.0, "borehole": 100.0})
        self.assertEqual(r.cost, 500.0)

    def test_unavailable_flag_on_source_matches_argument(self):
        rows = [
            WaterSource("aquifer", 100, 1.0, {"turb": 10.0}, available=False),
            WaterSource("borehole", 100, 5.0, {"turb": 0.0}),
        ]
        r = optimize(rows, BlendConstraints(100, {"turb": 3.0}), step=10)
        self.assertEqual(r.flows, {"aquifer": 0.0, "borehole": 100.0})

    def test_demand_cannot_be_met(self):
        r = optimize([WaterSource("only", 40, 1.0, {})],
                     BlendConstraints(100), step=10)
        self.assertFalse(r.feasible)
        self.assertIn("demand_short", [x["code"] for x in r.reasons])

    def test_quality_constraint_cannot_be_met(self):
        rows = [
            WaterSource("a", 100, 1.0, {"turb": 10.0}),
            WaterSource("b", 100, 1.0, {"turb": 10.0}),
        ]
        r = optimize(rows, BlendConstraints(100, {"turb": 5.0}), step=10)
        self.assertFalse(r.feasible)
        self.assertIn("quality_exceeded", [x["code"] for x in r.reasons])

    def test_evaluate_explicit_selection(self):
        r = evaluate(sources(), {"aquifer": 50.0, "borehole": 50.0},
                     BlendConstraints(100, {"turb": 3.0}),
                     include_rejected=True, step=10)
        self.assertFalse(r.feasible)
        self.assertEqual([x["code"] for x in r.reasons], ["quality_exceeded"])
        self.assertAlmostEqual(r.blended_quality["turb"], 5.0)

    def test_over_capacity_and_negative_flow_reasons(self):
        r = evaluate(sources(), {"aquifer": 150.0, "borehole": -5.0},
                     BlendConstraints(100))
        codes = {x["code"] for x in r.reasons}
        self.assertEqual(
            codes, {"over_capacity", "negative_flow", "demand_excess"}
        )


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.wb = BlendWorkbench()
        self.at = datetime(2026, 10, 2, 9, 0, tzinfo=timezone.utc)

    def _draft(self, name="summer", at=None):
        return self.wb.create_draft(
            name, sources(), BlendConstraints(100, {"turb": 3.0}),
            at=at or self.at,
        )

    def test_create_draft_proposes_and_persists_named_scenario(self):
        s = self._draft()
        self.assertEqual(s.state, DRAFT)
        self.assertEqual(s.version, 1)
        self.assertEqual(s.selection, {"aquifer": 30.0, "borehole": 70.0})
        self.assertEqual(self.wb.get("summer").name, "summer")
        self.assertEqual([x.name for x in self.wb.list_scenarios()], ["summer"])

    def test_duplicate_name_rejected(self):
        self._draft()
        with self.assertRaises(ValueError):
            self._draft()

    def test_full_lifecycle_to_approval(self):
        self._draft()
        review = self.wb.submit("summer")
        self.assertEqual(review.state, REVIEW)
        self.assertEqual(review.version, 2)
        approved = self.wb.approve("summer")
        self.assertEqual(approved.state, APPROVED)
        self.assertEqual(approved.version, 3)
        self.assertEqual(approved.selection, {"aquifer": 30.0, "borehole": 70.0})

    def test_cannot_approve_from_draft(self):
        self._draft()
        with self.assertRaises(ValueError):
            self.wb.approve("summer")

    def test_reject_then_revise_and_approve(self):
        self._draft()
        self.wb.submit("summer")
        rejected = self.wb.reject("summer")
        self.assertEqual(rejected.state, REJECTED)
        revised = self.wb.revise("summer")
        self.assertEqual(revised.state, DRAFT)
        self.wb.submit("summer")
        self.assertEqual(self.wb.approve("summer").state, APPROVED)

    def test_revise_from_review_back_to_draft(self):
        self._draft()
        self.wb.submit("summer")
        self.assertEqual(self.wb.revise("summer").state, DRAFT)

    def test_revise_with_new_selection(self):
        self._draft()
        self.wb.submit("summer")
        self.wb.reject("summer")
        revised = self.wb.revise(
            "summer", {"aquifer": 20.0, "borehole": 80.0}
        )
        self.assertEqual(revised.selection, {"aquifer": 20.0, "borehole": 80.0})
        self.assertEqual(revised.result.cost, 420.0)

    def test_revise_rejects_infeasible_selection(self):
        self._draft()
        self.wb.submit("summer")
        self.wb.reject("summer")
        with self.assertRaises(ValueError):
            self.wb.revise("summer", {"aquifer": 100.0, "borehole": 0.0})
        self.assertEqual(self.wb.get("summer").state, REJECTED)

    def test_create_draft_fails_when_constraints_unsatisfiable(self):
        with self.assertRaises(ValueError):
            self.wb.create_draft(
                "bad",
                [WaterSource("only", 40, 1.0, {})],
                BlendConstraints(100),
            )

    def test_manual_selection_must_satisfy_demand_and_quality(self):
        with self.assertRaises(ValueError):
            self.wb.create_draft(
                "manual",
                sources(),
                BlendConstraints(100, {"turb": 3.0}),
                selection={"aquifer": 100.0, "borehole": 0.0},
            )

    def test_recalculate_for_unavailable_source(self):
        s = self._draft()
        self.wb.submit("summer")
        recalculated = self.wb.recalculate("summer", unavailable=["aquifer"])
        self.assertEqual(recalculated.state, DRAFT)
        self.assertEqual(recalculated.selection,
                         {"aquifer": 0.0, "borehole": 100.0})
        self.assertEqual(recalculated.result.cost, 500.0)
        self.assertEqual(recalculated.version, s.version + 2)

    def test_recalculate_when_no_fallback_keeps_scenario_unchanged(self):
        s = self._draft()
        with self.assertRaises(ValueError):
            self.wb.recalculate("summer", unavailable=["aquifer", "borehole"])
        self.assertEqual(self.wb.get("summer").selection, s.selection)
        self.assertEqual(self.wb.get("summer").version, s.version)

    def test_approved_history_is_immutable(self):
        self._draft()
        self.wb.submit("summer")
        approved = self.wb.approve("summer")
        original_selection = dict(approved.selection)
        original_version = approved.version
        for action in (
            lambda: self.wb.submit("summer"),
            lambda: self.wb.approve("summer"),
            lambda: self.wb.reject("summer"),
            lambda: self.wb.revise("summer"),
            lambda: self.wb.recalculate("summer", unavailable=["aquifer"]),
        ):
            with self.assertRaises(ValueError):
                action()
        stored = self.wb.get("summer")
        self.assertEqual(stored.state, APPROVED)
        self.assertEqual(stored.selection, original_selection)
        self.assertEqual(stored.version, original_version)

    def test_time_reversal_refused(self):
        self._draft(at=datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc))
        with self.assertRaises(ValueError):
            self.wb.submit(
                "summer", at=datetime(2026, 10, 2, 9, 0, tzinfo=timezone.utc)
            )

    def test_manual_feasible_selection_records_rejected_cheaper(self):
        s = self.wb.create_draft(
            "manual",
            sources(),
            BlendConstraints(100, {"turb": 3.0}),
            selection={"aquifer": 30.0, "borehole": 70.0},
        )
        self.assertEqual(s.result.cost, 380.0)
        self.assertTrue(s.result.rejected_cheaper)
        self.assertEqual(
            s.result.rejected_cheaper[0].flows,
            {"aquifer": 100.0, "borehole": 0.0},
        )

    def test_result_exposes_flows_quality_cost(self):
        s = self._draft()
        r = s.result
        self.assertEqual(set(r.flows), {"aquifer", "borehole"})
        self.assertIn("turb", r.blended_quality)
        self.assertIsInstance(r.cost, float)

    def test_rejected_reason_for_unavailable_source_flow(self):
        r = evaluate(
            sources(),
            {"aquifer": 50.0, "borehole": 50.0},
            BlendConstraints(100),
            unavailable=["aquifer"],
        )
        codes = {x["code"] for x in r.reasons}
        self.assertIn("source_unavailable", codes)

    def test_invalid_step_and_demand(self):
        with self.assertRaises(ValueError):
            optimize(sources(), BlendConstraints(100), step=0)
        with self.assertRaises(ValueError):
            optimize(sources(), BlendConstraints(0))

    def test_duplicate_source_names_rejected(self):
        with self.assertRaises(ValueError):
            optimize(
                [WaterSource("a", 100, 1.0, {}), WaterSource("a", 100, 1.0, {})],
                BlendConstraints(100),
            )


class ServiceWiringTests(unittest.TestCase):
    def test_service_exposes_blend_workbench(self):
        svc = AquaFlowService()
        s = svc.create_blend_scenario(
            "s", sources(), BlendConstraints(100, {"turb": 3.0})
        )
        self.assertEqual(s.state, DRAFT)
        svc.submit_blend_scenario("s")
        self.assertEqual(svc.approve_blend_scenario("s").state, APPROVED)
        self.assertEqual(svc.get_blend_scenario("s").selection,
                         {"aquifer": 30.0, "borehole": 70.0})

    def test_health_remains_clean(self):
        self.assertEqual(AquaFlowService().health()["runtime_dependencies"], 0)


if __name__ == "__main__":
    unittest.main()
