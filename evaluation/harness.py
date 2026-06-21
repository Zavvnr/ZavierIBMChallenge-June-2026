"""Run MATE's commentary over a match and score faithfulness, language coverage, and latency.

Offline by default: with ``mock=True`` the CommentaryAgent uses deterministic templated
lines (no Granite), so this runs in CI and gives a faithfulness/coverage baseline. Point
it at a real Granite endpoint (``mock=False``, LM Studio loaded) to measure real model
output and per-line latency.

CLI:
    python -m evaluation.harness --mock --all-languages        # offline baseline
    python -m evaluation.harness --match-id 3869685 --language es   # real Granite
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from agent import prompts
from agent.commentary_agent import CommentaryAgent
from data_replayer.replayer import replay
from evaluation import metrics

REPO = Path(__file__).resolve().parent.parent


@dataclass
class Report:
    """The scored result of one evaluation run."""

    language: str
    n_events: int
    n_lines: int
    faithfulness: dict
    latency: dict
    coverage: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Plain-dict view for JSON output."""
        return asdict(self)

    def summary(self) -> str:
        """A compact human-readable summary."""
        faith, lat = self.faithfulness, self.latency
        lines = [
            f"events={self.n_events}  lines={self.n_lines}  language={self.language}",
            f"faithfulness = {faith['rate'] * 100:.1f}%  ({faith['clean']}/{faith['total']} clean)",
            f"latency      = mean {lat['mean_ms']}ms / median {lat['median_ms']}ms / p95 {lat['p95_ms']}ms",
        ]
        if self.coverage:
            ok = sum(1 for emitted in self.coverage.values() if emitted)
            lines.append(f"coverage     = {ok}/{len(self.coverage)} languages emit commentary")
        return "\n".join(lines)


def _run_lines(events: Sequence[dict], language: str, mock: bool,
               client=None) -> Tuple[List[tuple], List[float]]:
    """Replay events through the agent; return ([(line, event), ...], [latency_ms, ...])."""
    agent = CommentaryAgent(language=language, mock=mock, client=client)
    records: List[tuple] = []
    latencies: List[float] = []
    for event in replay(events, speed=0.0):
        started = time.perf_counter()
        line = agent.handle(event)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if line:
            records.append((line, event))
            latencies.append(elapsed_ms)
    return records, latencies


def evaluate(events: Sequence[dict], language: str = "en", mock: bool = True,
             client=None, languages: Optional[Sequence[str]] = None) -> Report:
    """Score one match's commentary; pass ``languages`` to add a coverage sweep."""
    events = list(events)
    vocab = metrics.match_vocab(events)
    records, latencies = _run_lines(events, language, mock, client)

    coverage: dict = {}
    for lang in (languages or []):
        lang_records, _ = _run_lines(events, lang, mock, client)
        coverage[lang] = len(lang_records) > 0

    return Report(
        language=prompts.normalize_language(language),
        n_events=len(events),
        n_lines=len(records),
        faithfulness=metrics.faithfulness(records, vocab),
        latency=metrics.latency_stats(latencies),
        coverage=coverage,
    )


def main(argv: Optional[list] = None) -> int:
    """Evaluate MATE commentary from the command line."""
    parser = argparse.ArgumentParser(
        description="Evaluate MATE commentary (faithfulness / language coverage / latency)."
    )
    parser.add_argument("--match-id", type=int, default=None, help="StatsBomb match (default: demo).")
    parser.add_argument("--language", default=os.getenv("DEFAULT_LANGUAGE", "en"),
                        choices=prompts.SUPPORTED_LANGUAGE_CODES)
    parser.add_argument("--limit", type=int, default=300, help="Cap events for a quick run (0 = all).")
    parser.add_argument("--all-languages", action="store_true",
                        help="Add a coverage sweep over all supported languages.")
    parser.add_argument("--mock", action="store_true", help="Offline deterministic lines (no Granite).")
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON.")
    args = parser.parse_args(argv)

    if not args.mock:  # real runs need GRANITE_* from the environment
        try:
            from dotenv import load_dotenv
            load_dotenv(REPO / ".env")
        except ImportError:
            pass

    from data_extraction.loader import fetch_events
    events = fetch_events(args.match_id)
    if args.limit:
        events = events[: args.limit]

    languages = prompts.SUPPORTED_LANGUAGE_CODES if args.all_languages else None
    report = evaluate(events, language=args.language, mock=args.mock, languages=languages)
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2) if args.json else report.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
