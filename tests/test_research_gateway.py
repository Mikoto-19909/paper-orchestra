"""Offline contract tests. All HTTP traffic is replaced before opening a connection."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from utils.research_gateway import BudgetExceeded, Gateway, ResearchRequestError, _image, _settle, USD


MODEL = "gpt-6.2-sol"


def answer(**changes):
    data = {"id": "rsp-fixture", "model": MODEL, "status": "completed",
            "service_tier": "default", "output": [],
            "usage": {"input_tokens": 100, "output_tokens": 20,
                      "input_tokens_details": {"cached_tokens": 30, "cache_write_tokens": 40},
                      "output_tokens_details": {"reasoning_tokens": 12}}}
    return {**data, **changes}


def reply(data, status=200):
    return status, "test-request", json.dumps(data).encode()


class GatewayTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.gateway = Gateway(self.root, "run", "validation")
        self.net = patch("utils.research_gateway._post").start()
        self.addCleanup(patch.stopall)
        self.count = reply({"object": "response.input_tokens", "input_tokens": 100})
        self.net.side_effect = lambda endpoint, _: self.count if endpoint else reply(answer())

    def call(self, **kwargs):
        return self.gateway.request("writing", "one", MODEL, "tiny input", 200, **kwargs)

    def test_exact_cost_no_double_charge_and_cache_replay_after_local_parser_error(self):
        received = self.call()
        with self.assertRaises(json.JSONDecodeError):
            json.loads(str(received["output"])[0])
        restarted = Gateway(self.root, "run", "validation")
        self.assertEqual(received, restarted.request("writing", "one", MODEL, "tiny input", 200))
        self.assertEqual(self.net.call_count, 2)
        report = restarted.report()
        self.assertEqual(report["known_estimated_usd"], (30*2000 + 30*100 + 40*2500 + 20*10000)/USD)
        self.assertEqual(report["outstanding_reserved_usd"], 0)
        row = report["calls"][-1]
        self.assertEqual(json.loads((self.root / row["raw_response_path"]).read_bytes()), received)
        self.assertEqual(row["metrics"]["reasoning_tokens"], 12)
        self.assertIsNone(row["metrics"]["ttl_30m_tokens"])

    def test_budget_refusal_before_any_http_and_after_exact_count(self):
        with self.assertRaises(BudgetExceeded):
            self.gateway.request("smoke", "1", MODEL, "x", 110000)
        self.net.assert_not_called()
        self.net.return_value = reply({"object": "response.input_tokens", "input_tokens": 800000})
        self.net.side_effect = None
        with self.assertRaises(BudgetExceeded):
            self.call()
        self.assertEqual(self.net.call_count, 1)
        self.assertEqual(self.gateway.report()["calls"][0]["estimated_usd"], 0)

    def test_concurrent_different_run_ids_cannot_spend_same_validation_budget(self):
        begun, release = threading.Event(), threading.Event()

        def wait_for_peer(endpoint, _):
            if endpoint:
                return self.count
            begun.set()
            self.assertTrue(release.wait(timeout=8))
            return reply(answer())
        self.net.side_effect = wait_for_peer
        with ThreadPoolExecutor(2) as pool:
            work = pool.submit(self.gateway.request, "smoke", "1", MODEL, "x", 65000)
            self.assertTrue(begun.wait(timeout=8))
            try:
                second = Gateway(self.root, "run2", "validation")
                with self.assertRaises(BudgetExceeded):
                    second.request("smoke", "2", MODEL, "y", 65000)
            finally:
                release.set()
            self.assertEqual(work.result()["id"], "rsp-fixture")
        self.assertEqual(self.net.call_count, 2)

    def test_independent_process_sees_parent_inflight_reservation(self):
        child = """
import sys
from unittest.mock import patch
from utils.research_gateway import BudgetExceeded, Gateway
with patch('utils.research_gateway._post', side_effect=AssertionError('unexpected network')) as mock:
    try:
        Gateway(sys.argv[1], 'child', 'validation').request('smoke', '2', 'gpt-6.2-sol', 'y', 65000)
    except BudgetExceeded:
        mock.assert_not_called()
    else:
        raise AssertionError('second process bypassed shared budget')
"""

        def dispatch(endpoint, _):
            if endpoint:
                return self.count
            subprocess.run([sys.executable, "-c", child, str(self.root)],
                           cwd=Path(__file__).resolve().parents[1], check=True, timeout=15)
            return reply(answer())

        self.net.side_effect = dispatch
        self.gateway.request("smoke", "1", MODEL, "x", 65000)
        self.assertEqual([r["run_id"] for r in self.gateway.report()["calls"]], ["run", "run"])

    def test_unknown_request_blocks_replay_after_restart_and_holds_across_phases(self):
        self.net.side_effect = [self.count, TimeoutError("simulated")]
        with self.assertRaises(ResearchRequestError):
            self.call(retries=2)
        self.gateway = Gateway(self.root, "run", "validation")
        with self.assertRaises(ResearchRequestError):
            self.call(retries=2)
        self.assertEqual(self.net.call_count, 2)
        self.assertGreater(self.gateway.report()["outstanding_reserved_usd"], 0)
        self.assertEqual(self.gateway.report()["calls"][-1]["status"], "unknown")

    def test_ten_dollars_shared_by_both_phases(self):
        # Seed a known past exposure; ask for another 12.8 cents.
        self.call()
        with self.gateway._db() as con:
            con.execute("UPDATE calls SET exposure=? WHERE endpoint=''", (9_900_000_000,))
        self.net.reset_mock()
        with self.assertRaises(BudgetExceeded):
            Gateway(self.root, "formal", "formal").request("write", "2", MODEL, "z", 12800)
        self.net.assert_not_called()

    def test_raw_ill_formed_response_saved_and_never_resent(self):
        self.net.side_effect = [self.count, (200, "req-bad", b"{broken")]
        for _ in range(2):
            with self.assertRaises(ResearchRequestError):
                self.call()
        row = self.gateway.report()["calls"][-1]
        self.assertEqual((self.root / row["raw_response_path"]).read_bytes(), b"{broken")
        self.assertIsNone(row["estimated_usd"])
        self.assertGreater(row["outstanding_usd"], 0)

    def test_incomplete_output_is_still_charged(self):
        self.net.side_effect = [self.count, reply(answer(status="incomplete",
                                   incomplete_details={"reason": "max_output_tokens"}))]
        with self.assertRaises(ResearchRequestError):
            self.call()
        row = self.gateway.report()["calls"][-1]
        self.assertGreater(row["estimated_usd"], 0)
        self.assertEqual(row["reason"], {"reason": "max_output_tokens"})

    def test_images_and_web_tool_budget_match_outgoing_request(self):
        path = self.root / "page.jpg"
        path.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
        data = answer(tool_usage={"web_search": {"num_requests": 1}},
                      output=[{"type":"web_search_call"},{"type":"web_search_call"}])
        self.net.side_effect = [self.count, reply(data)]
        self.call(images=[path], search_calls=1)
        count, post = (entry.args[1] for entry in self.net.call_args_list)
        self.assertEqual(count["input"], post["input"])
        self.assertEqual(count["tools"], post["tools"])
        self.assertTrue(count["input"][0]["content"][1]["image_url"].startswith("data:image/png;base64,"))
        self.assertEqual(post["max_tool_calls"], 1)
        self.assertEqual(post["tool_choice"], "required")
        self.assertEqual(post["service_tier"], "default")
        self.assertEqual(self.gateway.report()["calls"][-1]["metrics"]["billable_searches"], 1)
        path.write_bytes(b"%PDF-invalid")
        with self.assertRaises(ValueError):
            _image(path)

    def test_missing_counts_are_null_and_not_invented(self):
        data = answer()
        del data["usage"]["input_tokens_details"]["cache_write_tokens"]
        metrics, charge, held = _settle(MODEL, data, 0, 900000)
        self.assertIsNone(metrics["cache_write_tokens"])
        self.assertIsNone(charge)
        self.assertEqual(held, 900000)

    def test_tool_actions_without_billing_count_keep_reservation(self):
        metrics, charge, held = _settle(MODEL, answer(output=[{"type":"web_search_call"}]), 1, USD)
        self.assertEqual(metrics["search_actions_observed"], 1)
        self.assertIsNone(metrics["billable_searches"])
        self.assertIsNone(charge)
        self.assertEqual(held, USD)

    def test_price_mixed_cache_lifetimes(self):
        data = answer()
        data["usage"]["input_tokens_details"]["cache_write_tokens_details"] = {
            "ttl_30m_tokens":10, "ttl_12h_tokens":30}
        self.assertEqual(_settle(MODEL, data, 0, 0)[1], 30*2000+30*100+10*2500+30*5000+20*10000)

    def test_retry_is_bounded_across_restarts_and_individually_logged(self):
        rate = reply({"error": {"code":"rate_limit_exceeded"}}, 429)
        self.net.side_effect = [self.count, rate, rate]
        with patch("utils.research_gateway.time.sleep"):
            for _ in range(2):
                with self.assertRaises(ResearchRequestError):
                    self.call(retries=1)
        self.assertEqual(self.net.call_count, 3)
        self.assertEqual([r["attempt"] for r in self.gateway.report()["calls"]], [1,1,2])

    def test_input_model_and_code_fingerprint_invalidate_cache(self):
        self.call(fingerprint="template-A")
        self.call(fingerprint="template-B")
        self.assertEqual(self.net.call_count, 4)


if __name__ == "__main__":
    unittest.main()

