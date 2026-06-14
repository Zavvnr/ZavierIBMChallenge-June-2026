"""Unit tests for google_text_to_speech (speak + mutilingual_speaker). All offline."""
import base64
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from google_text_to_speech import speak
from google_text_to_speech import mutilingual_speaker as ms


def _capturing_transport(captured):
    def transport(url, payload, api_key):
        captured.append({"url": url, "payload": payload, "api_key": api_key})
        return {"audioContent": base64.b64encode(b"AUDIO").decode("ascii")}
    return transport


class NoOpAndFactoryTests(unittest.TestCase):
    def test_noop_returns_text_no_audio(self):
        r = speak.NoOpSpeaker().synthesize("Goal!", language="en-US")
        self.assertFalse(r.has_audio())
        self.assertEqual(r.text, "Goal!")
        self.assertIn("NoOp", r.skipped_reason)

    def test_build_speaker_toggle(self):
        self.assertIsInstance(speak.build_speaker(False), speak.NoOpSpeaker)
        self.assertIsInstance(speak.build_speaker(True, "google"), speak.GoogleCloudSpeaker)


class GoogleCloudSpeakerKeyTests(unittest.TestCase):
    def test_uses_tts_key(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("GOOGLE_TTS_API_KEY", "GOOGLE_API_KEY")}
        env["GOOGLE_TTS_API_KEY"] = "tts-key"
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(speak.GoogleCloudSpeaker().api_key, "tts-key")

    def test_google_api_key_not_used(self):
        # de-Gemini: the Gemini GOOGLE_API_KEY must NOT be picked up anymore.
        env = {k: v for k, v in os.environ.items()
               if k not in ("GOOGLE_TTS_API_KEY", "GOOGLE_API_KEY")}
        env["GOOGLE_API_KEY"] = "gemini-key"
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertIsNone(speak.GoogleCloudSpeaker().api_key)

    def test_no_key_degrades_to_text(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("GOOGLE_TTS_API_KEY", "GOOGLE_API_KEY")}
        with mock.patch.dict(os.environ, env, clear=True):
            r = speak.GoogleCloudSpeaker().synthesize("hi")
        self.assertFalse(r.has_audio())
        self.assertIn("GOOGLE_TTS_API_KEY", r.skipped_reason)


class GoogleCloudSpeakerSynthesisTests(unittest.TestCase):
    def test_synthesize_writes_audio(self):
        captured = []
        sp = speak.GoogleCloudSpeaker(transport=_capturing_transport(captured), api_key="x")
        with tempfile.TemporaryDirectory() as tmp:
            r = sp.synthesize("Goal!", language="en-US", output_dir=Path(tmp))
            self.assertTrue(r.has_audio())
            self.assertEqual(r.audio_bytes, b"AUDIO")
            self.assertTrue(r.audio_path.exists())
        self.assertEqual(captured[0]["payload"]["input"]["text"], "Goal!")

    def test_empty_text_skipped(self):
        sp = speak.GoogleCloudSpeaker(transport=_capturing_transport([]), api_key="x")
        r = sp.synthesize("   ")
        self.assertFalse(r.has_audio())
        self.assertIn("empty", r.skipped_reason)

    def test_speaking_rate_clamped(self):
        captured = []
        sp = speak.GoogleCloudSpeaker(transport=_capturing_transport(captured), api_key="x")
        with tempfile.TemporaryDirectory() as tmp:
            sp.synthesize("hi", language="en-US", output_dir=Path(tmp), speaking_rate=5.0)
            sp.synthesize("ho", language="en-US", output_dir=Path(tmp), speaking_rate=0.1)
        self.assertEqual(captured[0]["payload"]["audioConfig"]["speakingRate"], 4.0)
        self.assertEqual(captured[1]["payload"]["audioConfig"]["speakingRate"], 0.25)

    def test_voice_options_named_then_locale(self):
        opts = speak.GoogleCloudSpeaker(voice_name="MyVoice")._voice_options("en-US")
        self.assertEqual(opts, [{"languageCode": "en-US", "name": "MyVoice"},
                                {"languageCode": "en-US"}])

    def test_voice_options_unknown_locale(self):
        opts = speak.GoogleCloudSpeaker()._voice_options("xx-XX")
        self.assertEqual(opts, [{"languageCode": "xx-XX"}])


def _turn(speaker, text):
    return types.SimpleNamespace(speaker=speaker, text=text)


class FakeSingleSpeaker:
    def __init__(self, audio=b"AUDIO", skipped="", raise_exc=False):
        self.audio, self.skipped, self.raise_exc = audio, skipped, raise_exc
        self.calls = []

    def synthesize(self, text, language="en-US", speaking_rate=None):
        self.calls.append(text)
        if self.raise_exc:
            raise RuntimeError("boom")
        return types.SimpleNamespace(audio_bytes=self.audio, audio_path=None,
                                     skipped_reason=self.skipped)


class MultiSpeakerTests(unittest.TestCase):
    def test_mock_returns_placeholder_audio(self):
        da = ms.MultiSpeakerSpeaker(mock=True).synthesize_dialogue(
            [_turn("lead", "Goal!"), _turn("analyst", "Wonderful.")])
        self.assertTrue(da.has_audio())
        self.assertEqual(len(da.segments), 2)
        self.assertEqual([s.speaker for s in da.segments], ["lead", "analyst"])

    def test_path_b_uses_injected_speaker(self):
        fake = FakeSingleSpeaker(audio=b"AUDIO")
        da = ms.MultiSpeakerSpeaker(single_speaker=fake).synthesize_dialogue(
            [_turn("lead", "Goal!"), _turn("analyst", "Wonderful.")])
        self.assertTrue(da.has_audio())
        self.assertEqual(fake.calls, ["Goal!", "Wonderful."])  # sequential order preserved

    def test_skipped_reason_propagates(self):
        fake = FakeSingleSpeaker(audio=None, skipped="no key")
        da = ms.MultiSpeakerSpeaker(single_speaker=fake).synthesize_dialogue([_turn("lead", "Hi")])
        self.assertFalse(da.has_audio())
        self.assertEqual(da.segments[0].skipped_reason, "no key")

    def test_exception_degrades_to_skip(self):
        fake = FakeSingleSpeaker(raise_exc=True)
        with mock.patch("google_text_to_speech.mutilingual_speaker.time.sleep"):
            da = ms.MultiSpeakerSpeaker(single_speaker=fake).synthesize_dialogue([_turn("lead", "Hi")])
        self.assertFalse(da.has_audio())
        self.assertIn("RuntimeError", da.segments[0].skipped_reason)

    def test_build_multispeaker_speaker(self):
        spk = ms.build_multispeaker_speaker(language="es-ES")
        self.assertIsInstance(spk, ms.MultiSpeakerSpeaker)
        self.assertEqual(spk.language, "es-ES")


if __name__ == "__main__":
    unittest.main(verbosity=2)
