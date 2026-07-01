"""`python -m diagnostics` — MATE preflight health check.

Run it before a demo: it prints a clear ok / warn / fail line per dependency and exits
non-zero only on a hard failure (so it can gate a launch script). It loads ``.env`` the
same way the app does, so it sees your real GRANITE_BASE_URL / keys (names only).
"""
from __future__ import annotations

from pathlib import Path

from diagnostics import FAIL, OK, WARN, overall_status, run_all

_LABEL = {OK: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]"}


def main() -> int:
    """Print the health report; return 1 if anything is a hard FAIL, else 0."""
    try:  # load .env so the checks see the real config (values are never printed)
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except ImportError:
        pass

    results = run_all()
    print("MATE doctor — preflight check\n")
    for r in results:
        suffix = "  (required)" if r.required and r.status == FAIL else ""
        print(f"  {_LABEL.get(r.status, '[????]')}  {r.name:<18}  {r.detail}{suffix}")

    status = overall_status(results)
    print(f"\nOverall: {status.upper()}")
    if status == FAIL:
        print("Tip: tick 'Fast demo' in the UI to run offline while you fix the FAILs above.")
    return 1 if status == FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
