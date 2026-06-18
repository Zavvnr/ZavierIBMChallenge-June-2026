"""Unit tests for agent.langflow_client + explainer.answer() orchestration. Offline."""
import os
import unittest
from unittest import mock

import requests

from agent import langflow_client as lf
from agent import explainer


class FakeResp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        return self._p


def _clear(*names):
    return {n: os.environ.pop(n, None) for n in names}

def _restore(saved):
    for n, v in saved.items():
        if v is None:
            os.environ.pop(n, None)
        else:
            os.environ[n] = v


class IsConfiguredTests(unittest.TestCase):
    def setUp(self):
        self._s = _clear("LANGFLOW_BASE_URL", "LANGFLOW_FLOW_ID", "LANGFLOW_API_KEY")

    def tearDown(self):
        _restore(self._s)

    def test_not_configured_by_default(self):
        self.assertFalse(lf.is_configured())

    def test_configured_when_base_and_flow_set(self):
        os.environ["LANGFLOW_BASE_URL"] = "http://localhost:7860"
        os.environ["LANGFLOW_FLOW_ID"] = "abc"
        self.assertTrue(lf.is_configured())


class RunFlowTests(unittest.TestCase):
    def setUp(self):
        self._s = _clear("LANGFLOW_BASE_URL", "LANGFLOW_FLOW_ID", "LANGFLOW_API_KEY")
        os.environ["LANGFLOW_BASE_URL"] = "http://localhost:7860/"
        os.environ["LANGFLOW_FLOW_ID"] = "abc"

    def tearDown(self):
        _restore(self._s)

    def test_posts_to_run_endpoint_and_parses_message(self):
        payload = {"outputs": [{"outputs": [{"results": {"message": {"text": "Offside, Law 11."}}}]}]}
        with mock.patch("requests.post", return_value=FakeResp(payload)) as post:
            out = lf.run_flow("why offside?")
        self.assertEqual(out, "Offside, Law 11.")
        called_url = post.call_args.args[0] if post.call_args.args else post.call_args.kwargs.get("url", "")
        self.assertIn("/api/v1/run/abc", called_url)

    def test_unconfigured_raises(self):
        os.environ.pop("LANGFLOW_FLOW_ID")
        with self.assertRaises(Exception):
            lf.run_flow("q")

    def test_unparseable_response_returns_empty(self):
        with mock.patch("requests.post", return_value=FakeResp({"unexpected": True})):
            self.assertEqual(lf.run_flow("q"), "")


class AnswerOrchestrationTests(unittest.TestCase):
    def test_uses_langflow_when_configured(self):
        with mock.patch.object(lf, "is_configured", return_value=True), \
             mock.patch.object(lf, "run_flow", return_value="Langflow answer"), \
             mock.patch.object(explainer, "explain") as ex:
            self.assertEqual(explainer.answer("q"), "Langflow answer")
            ex.assert_not_called()

    def test_falls_back_when_langflow_errors(self):
        with mock.patch.object(lf, "is_configured", return_value=True), \
             mock.patch.object(lf, "run_flow", side_effect=RuntimeError("down")), \
             mock.patch.object(explainer, "explain", return_value="Python answer"):
            self.assertEqual(explainer.answer("q"), "Python answer")

    def test_falls_back_on_empty_langflow_result(self):
        with mock.patch.object(lf, "is_configured", return_value=True), \
             mock.patch.object(lf, "run_flow", return_value=""), \
             mock.patch.object(explainer, "explain", return_value="Python answer"):
            self.assertEqual(explainer.answer("q"), "Python answer")

    def test_uses_python_when_not_configured(self):
        with mock.patch.object(lf, "is_configured", return_value=False), \
             mock.patch.object(explainer, "explain", return_value="Python answer") as ex:
            self.assertEqual(explainer.answer("q"), "Python answer")
            ex.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
