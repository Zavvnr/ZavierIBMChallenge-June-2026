"""Unit tests for the profiles package. Offline: fake Wikipedia fetcher + fake Granite."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from profiles import profile_builder, wiki_client

WIKI_JSON = {
    "title": "Lionel Messi",
    "extract": "Lionel Messi is an Argentine professional footballer who plays as a forward.",
    "description": "Argentine footballer",
    "thumbnail": {"source": "https://img/messi.jpg"},
    "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Lionel_Messi"}},
}


def fake_fetcher(url):
    return WIKI_JSON


def none_fetcher(url):
    return None


def boom_fetcher(url):
    raise RuntimeError("network down")


class _Completions:
    def __init__(self, text, fail):
        self._text, self._fail = text, fail

    def create(self, **kwargs):
        if self._fail:
            raise RuntimeError("no model loaded")
        message = type("M", (), {"content": self._text})()
        return type("R", (), {"choices": [type("C", (), {"message": message})()]})()


class FakeGranite:
    """Minimal stand-in for the Granite OpenAI-compatible client."""

    def __init__(self, text="Perfil generado.", fail=False):
        self.chat = type("Chat", (), {"completions": _Completions(text, fail)})()


class WikiClientTests(unittest.TestCase):
    def test_fetch_summary_parses(self):
        s = wiki_client.fetch_summary("Lionel Messi", "en", fetcher=fake_fetcher, use_cache=False)
        self.assertEqual(s["photo_url"], "https://img/messi.jpg")
        self.assertIn("forward", s["extract"])
        self.assertEqual(s["url"], "https://en.wikipedia.org/wiki/Lionel_Messi")

    def test_fetch_summary_is_graceful(self):
        self.assertIsNone(wiki_client.fetch_summary("X", "en", fetcher=none_fetcher, use_cache=False))
        self.assertIsNone(wiki_client.fetch_summary("X", "en", fetcher=boom_fetcher, use_cache=False))
        self.assertIsNone(wiki_client.fetch_summary("", "en", fetcher=fake_fetcher, use_cache=False))


class ProfileBuilderTests(unittest.TestCase):
    def test_grounded_profile(self):
        prof = profile_builder.build_profile(
            "Lionel Messi", "es", position="Center Forward",
            granite_client=FakeGranite("Messi es un futbolista argentino."),
            fetcher=fake_fetcher, use_cache=False)
        self.assertTrue(prof["grounded"])
        self.assertEqual(prof["language"], "es-ES")
        self.assertEqual(prof["photo_url"], "https://img/messi.jpg")
        self.assertEqual(prof["profile"], "Messi es un futbolista argentino.")

    def test_falls_back_to_summary_when_granite_down(self):
        prof = profile_builder.build_profile(
            "Lionel Messi", "en", granite_client=FakeGranite(fail=True),
            fetcher=fake_fetcher, use_cache=False)
        self.assertTrue(prof["grounded"])
        self.assertIn("forward", prof["profile"])     # raw Wikipedia extract used as fallback

    def test_minimal_note_when_nothing_grounded(self):
        prof = profile_builder.build_profile(
            "Unknown Player", "en", position="Right Back",
            granite_client=FakeGranite(fail=True), fetcher=none_fetcher, use_cache=False)
        self.assertFalse(prof["grounded"])
        self.assertIn("Right Back", prof["profile"])


class CountingGranite:
    """Like FakeGranite, but counts how many times the model is actually called."""

    def __init__(self, text="Perfil generado."):
        self.calls = 0
        parent = self

        class _C:
            def create(self, **kwargs):
                parent.calls += 1
                message = type("M", (), {"content": text})()
                return type("R", (), {"choices": [type("C", (), {"message": message})()]})()

        self.chat = type("Chat", (), {"completions": _C()})()


class ProfileCacheTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_second_call_served_from_cache_no_granite(self):
        granite = CountingGranite("Messi es un crack.")
        with mock.patch("profiles.wiki_client.fetch_summary", return_value=None):
            first = profile_builder.build_profile("Lionel Messi", "es",
                                                  granite_client=granite, cache_dir=self.cache)
            second = profile_builder.build_profile("Lionel Messi", "es",
                                                   granite_client=granite, cache_dir=self.cache)
        self.assertEqual(second, first)
        self.assertEqual(granite.calls, 1)        # second read came from disk, not Granite

    def test_prewarm_then_click_is_a_cache_hit(self):
        granite = CountingGranite()
        with mock.patch("profiles.wiki_client.fetch_summary", return_value=None):
            warmed = profile_builder.prewarm_profiles(
                ["Lionel Messi", "Ángel Di María", "Lionel Messi"], "es",
                granite_client=granite, cache_dir=self.cache)
            self.assertEqual(warmed, 2)            # duplicate skipped
            self.assertEqual(granite.calls, 2)
            profile_builder.build_profile("Lionel Messi", "es",
                                          granite_client=granite, cache_dir=self.cache)
        self.assertEqual(granite.calls, 2)         # the click added no Granite call

    def test_minimal_note_is_not_cached(self):
        with mock.patch("profiles.wiki_client.fetch_summary", return_value=None):
            note = profile_builder.build_profile("Obscure Player", "en",
                                                 granite_client=FakeGranite(fail=True),
                                                 cache_dir=self.cache)
            self.assertIn("No profile information", note["profile"])
            good = CountingGranite("Now it works.")
            real = profile_builder.build_profile("Obscure Player", "en",
                                                 granite_client=good, cache_dir=self.cache)
        self.assertEqual(real["profile"], "Now it works.")
        self.assertEqual(good.calls, 1)            # the empty note never poisoned the cache


if __name__ == "__main__":
    unittest.main(verbosity=2)
