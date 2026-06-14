"""Unit tests for data_replayer.replayer (accelerated playback). All offline."""
import unittest

from data_replayer import replayer


def _ev(index, ts, period=1, minute=0, second=0, **extra):
    e = {"index": index, "period": period, "timestamp": ts, "minute": minute, "second": second}
    e.update(extra)
    return e


class ParseTests(unittest.TestCase):
    def test_parse_timestamp_seconds(self):
        self.assertEqual(replayer.parse_timestamp("00:01:30.500"), 90.5)
        self.assertEqual(replayer.parse_timestamp("01:00:00.000"), 3600.0)

    def test_order_key(self):
        self.assertEqual(replayer._order_key(_ev(5, "00:00:00.000", period=2, minute=3, second=4)),
                         (5, 2, 3, 4))


class ReplayTests(unittest.TestCase):
    def test_orders_by_index(self):
        events = [_ev(3, "00:00:03.000"), _ev(1, "00:00:01.000"), _ev(2, "00:00:02.000")]
        out = list(replayer.replay(events, speed=0))
        self.assertEqual([e["index"] for e in out], [1, 2, 3])

    def test_pacing_waits_scale_with_speed(self):
        waits = []
        events = [_ev(1, "00:00:00.000"), _ev(2, "00:00:10.000")]
        list(replayer.replay(events, speed=10.0, sleep=waits.append))
        self.assertEqual(waits, [1.0])  # (10s gap) / speed 10 = 1.0s

    def test_no_wait_when_speed_zero(self):
        waits = []
        events = [_ev(1, "00:00:00.000"), _ev(2, "00:00:10.000")]
        list(replayer.replay(events, speed=0, sleep=waits.append))
        self.assertEqual(waits, [])

    def test_no_sleep_across_period_change(self):
        waits = []
        events = [_ev(1, "00:45:00.000", period=1), _ev(2, "00:00:05.000", period=2)]
        list(replayer.replay(events, speed=10.0, sleep=waits.append))
        self.assertEqual(waits, [])  # half-time gap is not slept through

    def test_on_event_called_per_event(self):
        seen = []
        events = [_ev(1, "00:00:00.000"), _ev(2, "00:00:01.000")]
        list(replayer.replay(events, speed=0, on_event=seen.append))
        self.assertEqual(len(seen), 2)


class SummarizeTests(unittest.TestCase):
    def test_summarize_event(self):
        ev = _ev(1, "00:00:00.000", period=1, minute=23, second=11,
                 type={"name": "Shot"}, team={"name": "Argentina"},
                 player={"name": "Lionel Messi"})
        s = replayer.summarize_event(ev)
        for token in ("P1", "23:11", "Shot", "Argentina", "Lionel Messi"):
            self.assertIn(token, s)


if __name__ == "__main__":
    unittest.main(verbosity=2)
