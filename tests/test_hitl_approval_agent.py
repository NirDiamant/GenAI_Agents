import json
import sys
import types
import unittest
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "all_agents_tutorials" / "human_in_the_loop_approval_agent.ipynb"
TUTORIAL_LANGGRAPH_VERSION = "0.2.76"


def has_compatible_langgraph() -> bool:
    """Run integration tests only with the version pinned by the notebook."""
    try:
        installed_version = version("langgraph")
    except PackageNotFoundError:
        return False
    return installed_version == TUTORIAL_LANGGRAPH_VERSION


def load_notebook_namespace(include_langgraph: bool = False) -> dict:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    module_name = "notebook_under_test_full" if include_langgraph else "notebook_under_test_core"
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    namespace = module.__dict__
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        if (
            not include_langgraph
            and "requires-langgraph" in cell.get("metadata", {}).get("tags", [])
        ):
            continue
        source = "".join(cell.get("source", []))
        if source.lstrip().startswith(("!", "%")) or "IPython.display" in source:
            continue
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    return namespace


class HumanApprovalAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_notebook_namespace()

    def test_low_risk_action_executes_without_an_approval_request(self):
        state = self.ns["run_until_pause"](
            self.ns["new_request"]("lookup_order", {"order_id": "A-100"})
        )

        self.assertEqual(state["status"], "completed")
        self.assertIsNone(state["approval_request"])
        self.assertEqual(state["audit_log"][-1]["event"], "tool_executed")

    def test_high_risk_action_pauses_before_any_side_effect(self):
        state = self.ns["run_until_pause"](
            self.ns["new_request"](
                "issue_refund", {"order_id": "A-100", "amount": 240}
            )
        )

        self.assertEqual(state["status"], "awaiting_approval")
        self.assertEqual(state["approval_request"]["risk"], "high")
        self.assertNotIn("tool_result", state)
        self.assertFalse(any(e["event"] == "tool_executed" for e in state["audit_log"]))

    def test_small_mutating_refund_also_pauses_before_side_effect(self):
        state = self.ns["run_until_pause"](
            self.ns["new_request"](
                "issue_refund", {"order_id": "A-100", "amount": 80}
            )
        )

        self.assertEqual(state["status"], "awaiting_approval")
        self.assertNotIn("tool_result", state)

    def test_planned_refund_cannot_call_tool_dispatcher_directly(self):
        planned = self.ns["new_request"](
            "issue_refund", {"order_id": "A-100", "amount": 80}
        )

        with self.assertRaises(PermissionError):
            self.ns["execute_tool"](planned)

    def test_approve_resumes_and_executes_the_original_action(self):
        paused = self.ns["run_until_pause"](
            self.ns["new_request"](
                "issue_refund", {"order_id": "A-100", "amount": 240}
            )
        )
        completed = self.ns["resume_with_decision"](
            paused, self.ns["ApprovalDecision"]("approve", reviewer="ops@example.com")
        )

        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["tool_result"]["refunded"], 240)
        self.assertEqual(completed["audit_log"][-1]["reviewer"], "ops@example.com")

    def test_reject_finishes_without_executing_the_action(self):
        paused = self.ns["run_until_pause"](
            self.ns["new_request"](
                "issue_refund", {"order_id": "A-100", "amount": 240}
            )
        )
        rejected = self.ns["resume_with_decision"](
            paused,
            self.ns["ApprovalDecision"](
                "reject", reviewer="ops@example.com", reason="Policy exception required"
            ),
        )

        self.assertEqual(rejected["status"], "rejected")
        self.assertNotIn("tool_result", rejected)
        self.assertFalse(any(e["event"] == "tool_executed" for e in rejected["audit_log"]))

    def test_modify_revalidates_and_executes_reviewer_arguments(self):
        paused = self.ns["run_until_pause"](
            self.ns["new_request"](
                "issue_refund", {"order_id": "A-100", "amount": 240}
            )
        )
        modified = self.ns["resume_with_decision"](
            paused,
            self.ns["ApprovalDecision"](
                "modify",
                reviewer="ops@example.com",
                modified_args={"order_id": "A-100", "amount": 80},
            ),
        )

        self.assertEqual(modified["status"], "completed")
        self.assertEqual(modified["tool_result"]["refunded"], 80)
        self.assertEqual(modified["action"]["args"]["amount"], 80)
        modification = next(
            event for event in modified["audit_log"]
            if event["event"] == "action_modified"
        )
        self.assertEqual(modification["before"]["amount"], 240)
        self.assertEqual(modification["after"]["amount"], 80)

    def test_invalid_modified_arguments_are_blocked(self):
        paused = self.ns["run_until_pause"](
            self.ns["new_request"](
                "issue_refund", {"order_id": "A-100", "amount": 240}
            )
        )

        with self.assertRaisesRegex(ValueError, "amount"):
            self.ns["resume_with_decision"](
                paused,
                self.ns["ApprovalDecision"](
                    "modify",
                    reviewer="ops@example.com",
                    modified_args={"order_id": "A-100", "amount": -5},
                ),
            )

    def test_unexpected_tool_arguments_are_rejected(self):
        actions = [
            ("lookup_order", {"order_id": "A-100", "debug": True}),
            (
                "issue_refund",
                {"order_id": "A-100", "amount": 80, "currency": "USD"},
            ),
        ]
        for tool, args in actions:
            with self.subTest(tool=tool):
                with self.assertRaisesRegex(ValueError, "unexpected arguments"):
                    self.ns["new_request"](tool, args)

    def test_unexpected_modified_arguments_are_blocked(self):
        paused = self.ns["run_until_pause"](
            self.ns["new_request"](
                "issue_refund", {"order_id": "A-100", "amount": 240}
            )
        )

        with self.assertRaisesRegex(ValueError, "unexpected arguments"):
            self.ns["resume_with_decision"](
                paused,
                self.ns["ApprovalDecision"](
                    "modify",
                    reviewer="ops@example.com",
                    modified_args={
                        "order_id": "A-100",
                        "amount": 80,
                        "currency": "USD",
                    },
                ),
            )

    def test_boolean_and_non_finite_refund_amounts_are_rejected(self):
        for amount in (True, float("nan"), float("inf")):
            with self.subTest(amount=amount):
                with self.assertRaisesRegex(ValueError, "finite positive number"):
                    self.ns["new_request"](
                        "issue_refund", {"order_id": "A-100", "amount": amount}
                    )


@unittest.skipUnless(
    has_compatible_langgraph(),
    "requires the notebook-pinned langgraph==0.2.76; other versions skip explicitly",
)
class LangGraphApprovalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_notebook_namespace(include_langgraph=True)

    def pause(self, thread_id: str, amount: int = 240):
        config = {"configurable": {"thread_id": thread_id}}
        request = self.ns["new_request"](
            "issue_refund", {"order_id": "A-100", "amount": amount}
        )
        updates = list(
            self.ns["approval_graph"].stream(
                request, config=config, stream_mode="updates"
            )
        )
        interrupt = next(update["__interrupt__"][0] for update in updates if "__interrupt__" in update)
        return config, interrupt

    def test_graph_interrupts_before_refund_execution(self):
        config, interrupt = self.pause("graph-pause")
        snapshot = self.ns["approval_graph"].get_state(config).values

        self.assertEqual(interrupt.value["action"]["tool"], "issue_refund")
        self.assertEqual(snapshot["status"], "awaiting_approval")
        self.assertEqual(snapshot["approval_request"], interrupt.value)
        self.assertEqual(snapshot["audit_log"][-1]["event"], "approval_requested")
        self.assertNotIn("tool_result", snapshot)

    def test_graph_resumes_approve_reject_and_modify_decisions(self):
        decisions = [
            ("approve", {}, "completed", 240),
            ("reject", {"reason": "outside policy"}, "rejected", None),
            (
                "modify",
                {"modified_args": {"order_id": "A-100", "amount": 80}},
                "completed",
                80,
            ),
        ]
        for index, (decision, extra, status, refunded) in enumerate(decisions):
            with self.subTest(decision=decision):
                config, _ = self.pause(f"graph-resume-{index}")
                result = self.ns["approval_graph"].invoke(
                    self.ns["Command"](
                        resume={
                            "decision": decision,
                            "reviewer": "ops@example.com",
                            **extra,
                        }
                    ),
                    config=config,
                )
                self.assertEqual(result["status"], status)
                if refunded is None:
                    self.assertNotIn("tool_result", result)
                else:
                    self.assertEqual(result["tool_result"]["refunded"], refunded)


if __name__ == "__main__":
    unittest.main()
