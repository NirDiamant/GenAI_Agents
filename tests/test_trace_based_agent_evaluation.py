import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "all_agents_tutorials" / "trace_based_agent_evaluation.ipynb"


def load_notebook_namespace() -> dict:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {"__name__": "notebook_under_test"}
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if source.lstrip().startswith(("!", "%")):
            continue
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    return namespace


class TraceEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_notebook_namespace()

    def test_perfect_trace_receives_full_credit(self):
        case = self.ns["EvalCase"](
            case_id="weather-paris",
            prompt="What is the weather in Paris?",
            expected_calls=[{"name": "get_weather", "args": {"city": "Paris"}}],
            expected_evidence={"temperature": "18 C"},
            max_latency_ms=500,
        )
        trace = self.ns["AgentTrace"](
            answer="temperature: 18 C",
            tool_calls=[{"name": "get_weather", "args": {"city": "Paris"}}],
            evidence={"temperature": "18 C"},
            claims={"temperature": "18 C"},
            latency_ms=120,
            error=None,
        )

        score = self.ns["score_trace"](case, trace)

        self.assertEqual(score.total, 1.0)
        self.assertEqual(score.failed_checks, [])

    def test_wrong_arguments_and_ungrounded_answer_lose_independent_credit(self):
        case = self.ns["EvalCase"](
            case_id="weather-paris",
            prompt="What is the weather in Paris?",
            expected_calls=[{"name": "get_weather", "args": {"city": "Paris"}}],
            expected_evidence={"temperature": "18 C"},
            max_latency_ms=500,
        )
        trace = self.ns["AgentTrace"](
            answer="Paris is sunny.",
            tool_calls=[{"name": "get_weather", "args": {"city": "London"}}],
            evidence={},
            claims={},
            latency_ms=120,
            error=None,
        )

        score = self.ns["score_trace"](case, trace)

        self.assertEqual(score.total, 0.5)
        self.assertEqual(score.failed_checks, ["arguments", "evidence"])

    def test_errors_fail_every_execution_check(self):
        case = self.ns["EVAL_CASES"][0]
        trace = self.ns["AgentTrace"](
            answer="",
            tool_calls=[],
            evidence={},
            claims={},
            latency_ms=50,
            error="timeout",
        )

        score = self.ns["score_trace"](case, trace)

        self.assertEqual(score.total, 0.0)
        self.assertIn("error", score.failed_checks)
        self.assertEqual(score.checks, {"tool": False, "arguments": False, "evidence": False, "latency": False})

    def test_empty_error_string_still_means_execution_failed(self):
        case = self.ns["EVAL_CASES"][0]
        trace = self.ns["AgentTrace"](
            answer="temperature: 18 C",
            tool_calls=[{"name": "get_weather", "args": {"city": "Paris"}}],
            evidence={"temperature": "18 C"},
            claims={"temperature": "18 C"},
            latency_ms=120,
            error="",
        )

        score = self.ns["score_trace"](case, trace)

        self.assertEqual(score.total, 0.0)
        self.assertIn("error", score.failed_checks)

    def test_agent_exception_becomes_failed_trace_and_suite_continues(self):
        def raising_agent(case):
            if case.case_id == "weather-tokyo":
                raise TimeoutError("provider timed out")
            return self.ns["improved_agent"](case)

        report = self.ns["evaluate_suite"](raising_agent, self.ns["EVAL_CASES"])

        self.assertEqual(report["case_count"], 4)
        failed = next(run for run in report["runs"] if run["score"].case_id == "weather-tokyo")
        self.assertEqual(failed["score"].total, 0.0)
        self.assertEqual(failed["trace"].error, "TimeoutError: provider timed out")
        self.assertGreaterEqual(failed["trace"].latency_ms, 1)

    def test_empty_suite_is_rejected_with_clear_error(self):
        with self.assertRaisesRegex(ValueError, "at least one evaluation case"):
            self.ns["evaluate_suite"](self.ns["improved_agent"], [])

    def test_contradictory_claim_does_not_receive_evidence_credit(self):
        case = self.ns["EVAL_CASES"][2]
        trace = self.ns["AgentTrace"](
            answer="Order A-100 has not shipped.",
            tool_calls=[{"name": "lookup_order", "args": {"order_id": "A-100"}}],
            evidence={"status": "shipped"},
            claims={"status": "shipped"},
            latency_ms=90,
            error=None,
        )

        score = self.ns["score_trace"](case, trace)

        self.assertFalse(score.checks["evidence"])
        self.assertIn("evidence", score.failed_checks)

    def test_extra_tool_call_fails_the_complete_sequence_contract(self):
        case = self.ns["EVAL_CASES"][2]
        trace = self.ns["AgentTrace"](
            answer="Order A-100 shipped.",
            tool_calls=[
                {"name": "lookup_order", "args": {"order_id": "A-100"}},
                {"name": "delete_order", "args": {"order_id": "A-100"}},
            ],
            evidence={"status": "shipped"},
            claims={"status": "shipped"},
            latency_ms=90,
            error=None,
        )

        score = self.ns["score_trace"](case, trace)

        self.assertFalse(score.checks["tool"])
        self.assertFalse(score.checks["arguments"])

    def test_latency_budget_is_inclusive_and_rejects_over_budget(self):
        case = self.ns["EVAL_CASES"][0]
        base = {
            "answer": "temperature: 18 C",
            "tool_calls": [{"name": "get_weather", "args": {"city": "Paris"}}],
            "evidence": {"temperature": "18 C"},
            "claims": {"temperature": "18 C"},
            "error": None,
        }

        at_budget = self.ns["score_trace"](
            case, self.ns["AgentTrace"](**base, latency_ms=500)
        )
        over_budget = self.ns["score_trace"](
            case, self.ns["AgentTrace"](**base, latency_ms=501)
        )

        self.assertTrue(at_budget.checks["latency"])
        self.assertFalse(over_budget.checks["latency"])

    def test_quality_gate_rejects_p95_latency_regression(self):
        report = self.ns["evaluate_suite"](
            self.ns["improved_agent"], self.ns["EVAL_CASES"]
        )
        gate = self.ns["check_quality_gate"](
            report,
            {"pass_rate": 1.0, "tool_accuracy": 1.0, "p95_latency_ms": 100},
        )

        self.assertFalse(gate["passed"])
        self.assertEqual(gate["failures"], ["p95_latency_ms 180 > 100"])

    def test_suite_gate_reports_metric_regressions(self):
        report = self.ns["evaluate_suite"](
            self.ns["baseline_agent"], self.ns["EVAL_CASES"]
        )

        gate = self.ns["check_quality_gate"](
            report,
            {"pass_rate": 1.0, "tool_accuracy": 1.0, "p95_latency_ms": 500},
        )

        self.assertFalse(gate["passed"])
        self.assertEqual(
            gate["failures"],
            ["pass_rate 0.50 < 1.00", "tool_accuracy 0.75 < 1.00"],
        )

    def test_improved_agent_passes_the_declared_gate(self):
        report = self.ns["evaluate_suite"](
            self.ns["improved_agent"], self.ns["EVAL_CASES"]
        )
        gate = self.ns["check_quality_gate"](
            report,
            {"pass_rate": 1.0, "tool_accuracy": 1.0, "p95_latency_ms": 500},
        )

        self.assertTrue(gate["passed"])
        self.assertEqual(report["pass_rate"], 1.0)
        self.assertEqual(report["tool_accuracy"], 1.0)
        self.assertLessEqual(report["p95_latency_ms"], 500)


if __name__ == "__main__":
    unittest.main()
