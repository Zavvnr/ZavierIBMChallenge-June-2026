"""Integration: the two-voice MultiSpeakerSpeaker rendering through the real GoogleCloudSpeaker.

Only the HTTP POST to Google Cloud TTS is faked (via GoogleCloudSpeaker._post); the
multispeaker -> single-speaker wiring runs for real.
"""
import base64
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from text_to_speech import speak
from text_to_speech import mutilingual_speaker as ms


def _turn(speaker, text):
    return types.SimpleNamespace(speaker=speaker, text=text)


class MultiSpeakerThroughGoogleCloud(unittest.TestCase):
    def test_dialogue_renders_two_distinct_segments(self):
        fake_resp = {"audioContent": base64.b64encode(b"AUDIO").decode("ascii")}
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(speak, "DEFAULT_OUT_DIR", Path(tmp)), \
             mock.patch.object(speak.GoogleCloudSpeaker, "_post", return_value=fake_resp), \
             mock.patch.dict(os.environ, {"GOOGLE_TTS_API_KEY": "k"}, clear=False):
            da = ms.MultiSpeakerSpeaker(language="en-US").synthesize_dialogue(
                [_turn("lead", "Goal!"), _turn("analyst", "What a finish.")])
        self.assertTrue(da.has_audio())
        self.assertEqual(len(da.segments), 2)
        self.assertTrue(all(s.has_audio() for s in da.segments))
        self.assertEqual([s.speaker for s in da.segments], ["lead", "analyst"])

    def test_dialogue_degrades_when_post_fails(self):
        with mock.patch.dict(os.environ, {"GOOGLE_TTS_API_KEY": "k"}, clear=False), \
             mock.patch.object(speak.GoogleCloudSpeaker, "_post", side_effect=RuntimeError("502")), \
             mock.patch("text_to_speech.mutilingual_speaker.time.sleep"):
            da = ms.MultiSpeakerSpeaker(language="en-US").synthesize_dialogue([_turn("lead", "Hi")])
        self.assertFalse(da.has_audio())
        self.assertTrue(da.segments[0].skipped_reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
