import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from work.pipeline2.core import write_json
from work.pipeline2.models import Chat, check_json_strings, load_chat


class ModelTests(unittest.TestCase):
    def test_explicit_registry_beats_stale_generic_environment_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "secrets.json"
            write_json(path, {"entries": [
                {"provider": "deepseek", "apiKey": "fixture-key",
                 "baseUrl": "https://api.deepseek.com", "models": ["deepseek-chat"]}]})
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "stale-fixture"}, clear=True):
                client = load_chat("text", path)
            self.assertEqual(client.api_key, "fixture-key")
            self.assertNotIn("fixture-key", repr(client))
            self.assertNotIn("fixture-key", json.dumps(client.identity))

    def test_invalid_math_json_gets_one_encoding_repair(self):
        outputs = [r'{"latex":"\lim_{x\to 0}x"}', json.dumps({"latex": r"\lim_{x\to 0}x"})]
        class Opener:
            calls = 0

            def open(self, request, timeout):
                output = outputs[self.calls]
                self.calls += 1
                data = {"choices": [{"finish_reason": "stop", "message": {"content": output}}]}
                return io.BytesIO(json.dumps(data).encode())
        opener = Opener()
        with patch("urllib.request.build_opener", return_value=opener):
            result = Chat("https://example.test", "fixture", "not-a-secret").json("JSON", {})
        self.assertEqual(result["latex"], r"\lim_{x\to 0}x")
        self.assertEqual(opener.calls, 2)

    def test_valid_json_cannot_silently_corrupt_math_escapes(self):
        for raw in [r'{"latex":"\frac{x}{y}"}', r'{"latex":"\nabla f"}', r'{"symbol":"\theta"}']:
            with self.assertRaises(ValueError):
                check_json_strings(json.loads(raw))

    def test_network_timeout_reports_original_error_not_unbound_local(self):
        """Regression: the OSError/TimeoutError handler used {error} without binding it."""
        class Opener:
            def open(self, request, timeout):
                raise TimeoutError("The read operation timed out")
        with patch("urllib.request.build_opener", return_value=Opener()), \
                patch.dict(os.environ, {"ECHONOTES_MODEL_RETRIES": "1"}, clear=True):
            with self.assertRaises(RuntimeError) as caught:
                Chat("https://example.test", "fixture", "not-a-secret").json("JSON", {})
        message = str(caught.exception)
        self.assertIn("The read operation timed out", message)
        self.assertIn("ECHONOTES_MODEL_TIMEOUT", message)

    def test_http_error_keeps_truncated_provider_message(self):
        """Regression: the HTTPError handler dropped the server error body."""
        class Opener:
            def open(self, request, timeout):
                body = json.dumps({"error": {"message": "Not found the model kimi-k3 or Permission denied"}})
                raise urllib.error.HTTPError("https://example.test", 404, "Not Found", {},
                                             io.BytesIO(body.encode()))
        with patch("urllib.request.build_opener", return_value=Opener()), \
                patch.dict(os.environ, {"ECHONOTES_MODEL_RETRIES": "2"}, clear=True):
            with self.assertRaises(RuntimeError) as caught:
                Chat("https://example.test", "fixture", "not-a-secret").json("JSON", {})
        message = str(caught.exception)
        self.assertIn("HTTP 404", message)
        self.assertIn("Not found the model kimi-k3", message)

    def test_length_finish_reason_explains_thinking_budget(self):
        """Regression: all non-stop finish reasons used to share one vague error."""
        class Opener:
            def open(self, request, timeout):
                data = {"choices": [{"finish_reason": "length", "message": {"content": ""}}],
                        "usage": {"prompt_tokens": 765, "completion_tokens": 8192,
                                  "completion_tokens_details": {"reasoning_tokens": 7300}}}
                return io.BytesIO(json.dumps(data).encode())
        with patch("urllib.request.build_opener", return_value=Opener()), \
                patch.dict(os.environ, {"ECHONOTES_MODEL_RETRIES": "1"}, clear=True):
            with self.assertRaises(ValueError) as caught:
                Chat("https://example.test", "fixture", "not-a-secret", role="text").json("JSON", {})
        message = str(caught.exception)
        self.assertIn("finish_reason=length", message)
        self.assertIn("thinking", message)
        self.assertIn("ECHONOTES_TEXT_EXTRA_BODY", message)

    def test_extra_body_is_sent_and_changes_cache_identity(self):
        captured = {}

        class Opener:
            def open(self, request, timeout):
                captured["body"] = json.loads(request.data.decode())
                data = {"choices": [{"finish_reason": "stop", "message": {"content": '{"ok":1}'}}]}
                return io.BytesIO(json.dumps(data).encode())
        with patch("urllib.request.build_opener", return_value=Opener()):
            plain = Chat("https://example.test", "fixture", "not-a-secret").json("JSON", {})
            tuned = Chat("https://example.test", "fixture", "not-a-secret",
                         extra_body={"thinking": {"type": "disabled"}}).json("JSON", {})
        self.assertEqual(plain["ok"], 1)
        self.assertEqual(captured["body"]["thinking"], {"type": "disabled"})
        self.assertNotIn("thinking", json.dumps(Chat("https://example.test", "fixture", "k").identity))
        self.assertIn("thinking", json.dumps(Chat("https://example.test", "fixture", "k",
                                                 extra_body={"thinking": {"type": "disabled"}}).identity))

    def test_load_chat_parses_extra_body_from_environment(self):
        with patch.dict(os.environ, {
                "ECHONOTES_TEXT_API_KEY": "fixture",
                "ECHONOTES_TEXT_EXTRA_BODY": '{"thinking":{"type":"disabled"}}'}, clear=True):
            client = load_chat("text")
        self.assertEqual(client.extra_body, {"thinking": {"type": "disabled"}})
        self.assertEqual(client.role, "text")
        self.assertNotIn("fixture", repr(client))

    def test_extra_body_cannot_override_reserved_request_keys(self):
        """Regression: extra_body was merged raw, so {"model": ...} desynced the
        actual request from the recorded cache identity (cache poisoning), and
        {"stream": true} broke response parsing outright."""
        with self.assertRaises(ValueError) as caught:
            Chat("https://example.test", "fixture", "k", role="text",
                 extra_body={"model": "other-model", "stream": True})
        message = str(caught.exception)
        self.assertIn("ECHONOTES_TEXT_EXTRA_BODY", message)
        self.assertIn("model", message)
        with patch.dict(os.environ, {
                "ECHONOTES_TEXT_API_KEY": "fixture",
                "ECHONOTES_TEXT_EXTRA_BODY": '{"temperature": 0.9}'}, clear=True):
            with self.assertRaises(ValueError) as caught_env:
                load_chat("text")
        self.assertIn("temperature", str(caught_env.exception))

    def test_extra_body_invalid_json_names_the_environment_variable(self):
        """Regression: a malformed EXTRA_BODY used to surface as a bare
        JSONDecodeError with no hint which of the four role variables broke."""
        with patch.dict(os.environ, {
                "ECHONOTES_VISION_API_KEY": "fixture",
                "ECHONOTES_VISION_EXTRA_BODY": "not json"}, clear=True):
            with self.assertRaises(ValueError) as caught:
                load_chat("vision")
        self.assertIn("ECHONOTES_VISION_EXTRA_BODY", str(caught.exception))

    def test_persistent_throttle_clock_survives_process_restart(self):
        """Regression: the MIN_INTERVAL clock lived in process memory, so the
        real-world "restart after a 429" flow fired the next request instantly,
        straight into the rate-limit window. The clock now persists to disk."""
        with tempfile.TemporaryDirectory() as directory:
            clock = Path(directory) / ".request-clock.json"
            sleeps = []
            with patch("time.sleep", side_effect=lambda seconds: sleeps.append(seconds)):
                Chat("https://example.test", "fixture", "k", clock_path=clock)._pace(21)
                self.assertEqual(sleeps, [])  # First ever request: nothing to wait for.
                # A "new process" (fresh instance) sharing the clock file must wait.
                Chat("https://example.test", "fixture", "k", clock_path=clock)._pace(21)
            self.assertEqual(len(sleeps), 1)
            self.assertGreater(sleeps[0], 20)
            self.assertLessEqual(sleeps[0], 21)


if __name__ == "__main__":
    unittest.main()
