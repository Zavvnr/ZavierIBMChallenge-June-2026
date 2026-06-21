"""Starting line-ups and formations, grounded in StatsBomb data.

Builds a structured line-up for each team — formation, starting XI (with positions and
shirt numbers), substitutes, and manager — straight from the StatsBomb feeds we already
fetch (``lineups/{id}``, the ``Starting XI`` events, and the match metadata). Nothing is
invented: every field comes from the data. ``formation_svg`` renders the XI as a simple
pitch diagram, and ``LABELS`` localises the section headers into the 12 supported languages.

CLI:
    python -m data_extraction.lineups --match-id 3869685 --language es
    python -m data_extraction.lineups --match-id 3869685 --svg-out lineups/   # write SVGs
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Localised section labels for the line-up view. A small, reviewable i18n table — these
# are fixed UI strings (not match data), keyed by the 12 supported language codes.
LABELS = {
    "en": {"formation": "Formation", "manager": "Manager", "captain": "Captain",
           "subs": "Substitutes", "xi": "Starting XI"},
    "de": {"formation": "Formation", "manager": "Trainer", "captain": "Kapitän",
           "subs": "Auswechselspieler", "xi": "Startelf"},
    "es": {"formation": "Formación", "manager": "Entrenador", "captain": "Capitán",
           "subs": "Suplentes", "xi": "Once inicial"},
    "fr": {"formation": "Formation", "manager": "Entraîneur", "captain": "Capitaine",
           "subs": "Remplaçants", "xi": "Onze de départ"},
    "ja": {"formation": "フォーメーション", "manager": "監督", "captain": "キャプテン",
           "subs": "控え", "xi": "先発"},
    "pt": {"formation": "Formação", "manager": "Técnico", "captain": "Capitão",
           "subs": "Reservas", "xi": "Onze inicial"},
    "ar": {"formation": "التشكيلة", "manager": "المدرب", "captain": "القائد",
           "subs": "البدلاء", "xi": "التشكيلة الأساسية"},
    "cs": {"formation": "Rozestavení", "manager": "Trenér", "captain": "Kapitán",
           "subs": "Náhradníci", "xi": "Základní sestava"},
    "it": {"formation": "Formazione", "manager": "Allenatore", "captain": "Capitano",
           "subs": "Riserve", "xi": "Undici titolare"},
    "ko": {"formation": "포메이션", "manager": "감독", "captain": "주장",
           "subs": "교체 선수", "xi": "선발"},
    "nl": {"formation": "Opstelling", "manager": "Trainer", "captain": "Aanvoerder",
           "subs": "Wisselspelers", "xi": "Basiselftal"},
    "zh": {"formation": "阵型", "manager": "主教练", "captain": "队长",
           "subs": "替补", "xi": "首发"},
}


def labels_for(language: Optional[str]) -> dict:
    """Return the label set for a bare or BCP-47 code, falling back to English."""
    base = (language or "en").split("-")[0].lower()
    return LABELS.get(base, LABELS["en"])


@dataclass
class PlayerSlot:
    """One player in a line-up — only facts present in the StatsBomb data."""

    name: str
    number: Optional[int] = None
    position: str = ""
    player_id: Optional[int] = None
    position_id: Optional[int] = None
    is_captain: bool = False


@dataclass
class TeamLineup:
    """A team's grounded line-up for one match."""

    team: str
    formation: str = ""
    manager: str = ""
    starting_xi: List[PlayerSlot] = field(default_factory=list)
    substitutes: List[PlayerSlot] = field(default_factory=list)

    def as_dict(self) -> dict:
        """Serialise for the web API / JSON."""
        slot = lambda p: {"name": p.name, "number": p.number, "position": p.position,
                          "is_captain": p.is_captain}
        return {
            "team": self.team,
            "formation": self.formation,
            "manager": self.manager,
            "starting_xi": [slot(p) for p in self.starting_xi],
            "substitutes": [slot(p) for p in self.substitutes],
        }


def _manager_for(team_name: str, meta: Optional[dict]) -> str:
    """Manager name for a team from match metadata, or '' if unavailable."""
    if not meta:
        return ""
    for side in ("home_team", "away_team"):
        block = meta.get(side) or {}
        if block.get(f"{side}_name") == team_name:
            managers = block.get("managers") or []
            if managers:
                return managers[0].get("name") or managers[0].get("nickname") or ""
    return ""


def _nicknames(team_block: dict) -> dict:
    """Map player_id -> preferred display name (nickname if present) and captain flags."""
    names, captains = {}, {}
    for player in team_block.get("lineup", []):
        pid = player.get("player_id")
        names[pid] = player.get("player_nickname") or player.get("player_name") or ""
        captains[pid] = bool(player.get("captain"))
    return {"names": names, "captains": captains}


def _starting_xi_event(events: list, team_name: str) -> Optional[dict]:
    """Find a team's 'Starting XI' event in the event stream."""
    for ev in events:
        if (ev.get("type") or {}).get("name") == "Starting XI" \
                and (ev.get("team") or {}).get("name") == team_name:
            return ev
    return None


def parse_team_lineup(team_name: str, lineups: list, events: list,
                      meta: Optional[dict] = None) -> TeamLineup:
    """Assemble one team's line-up from the StatsBomb feeds (pure, no I/O)."""
    team_block = next((t for t in lineups if t.get("team_name") == team_name), {})
    info = _nicknames(team_block)
    names, captains = info["names"], info["captains"]

    xi_event = _starting_xi_event(events, team_name)
    formation = ""
    starting_xi: List[PlayerSlot] = []
    xi_ids = set()
    if xi_event:
        tactics = xi_event.get("tactics") or {}
        formation = str(tactics.get("formation") or "")
        for entry in tactics.get("lineup", []):
            player = entry.get("player") or {}
            pid = player.get("id")
            xi_ids.add(pid)
            position = entry.get("position") or {}
            starting_xi.append(PlayerSlot(
                name=names.get(pid) or player.get("name", ""),
                number=entry.get("jersey_number"),
                position=position.get("name", ""),
                player_id=pid,
                position_id=position.get("id"),
                is_captain=captains.get(pid, False),
            ))

    # Substitutes = squad players who didn't start.
    substitutes: List[PlayerSlot] = []
    for player in team_block.get("lineup", []):
        pid = player.get("player_id")
        if pid in xi_ids:
            continue
        positions = player.get("positions") or []
        substitutes.append(PlayerSlot(
            name=names.get(pid) or player.get("player_name", ""),
            number=player.get("jersey_number"),
            position=(positions[0].get("position") if positions else ""),
            player_id=pid,
            is_captain=captains.get(pid, False),
        ))

    return TeamLineup(
        team=team_name,
        formation=formation,
        manager=_manager_for(team_name, meta),
        starting_xi=sorted(starting_xi, key=lambda p: (p.position_id or 99)),
        substitutes=substitutes,
    )


def parse_lineups(lineups: list, events: list, meta: Optional[dict] = None) -> List[TeamLineup]:
    """Both teams' line-ups, in the order StatsBomb lists them."""
    return [parse_team_lineup(t.get("team_name", ""), lineups, events, meta) for t in lineups]


def fetch_lineups(match_id: Optional[int] = None) -> List[TeamLineup]:
    """Fetch + parse both line-ups for a match from StatsBomb (best-effort, fail-safe)."""
    from data_extraction.loader import (
        DEFAULT_DEMO_MATCH_ID, download_lineups, fetch_events, find_match_meta,
    )
    mid = int(match_id) if match_id else DEFAULT_DEMO_MATCH_ID
    try:
        lineups = download_lineups(mid)
        events = fetch_events(mid)
        meta = find_match_meta(mid)
        return parse_lineups(lineups, events, meta)
    except Exception:  # network / shape issues degrade to an empty list, never crash the app
        return []


# --------------------------------------------------------------------------- #
# Formation diagram (self-contained SVG, no dependencies)                      #
# --------------------------------------------------------------------------- #
def formation_lines(formation) -> List[int]:
    """Split a formation (e.g. 442 or '4-2-3-1') into outfield line sizes.

    Returns the lines *after* the goalkeeper, e.g. 442 -> [4, 4, 2]. Falls back to
    [4, 4, 2] if the formation is missing or doesn't add up to ten outfield players.
    """
    digits = [int(c) for c in str(formation) if c.isdigit()]
    return digits if digits and sum(digits) == 10 else [4, 4, 2]


def formation_svg(team: TeamLineup, language: str = "en",
                  width: int = 340, height: int = 480) -> str:
    """Render the starting XI as a vertical pitch diagram (GK at the bottom)."""
    labels = labels_for(language)
    lines = formation_lines(team.formation)
    rows = [1] + lines  # goalkeeper + outfield lines
    players = list(team.starting_xi)

    margin_y, margin_x = 46, 30
    row_gap = (height - 2 * margin_y) / max(1, len(rows) - 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'font-family="sans-serif">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#2e7d32"/>',
        f'<rect x="8" y="8" width="{width - 16}" height="{height - 16}" fill="none" '
        f'stroke="#ffffff" stroke-opacity="0.6"/>',
        f'<line x1="8" y1="{height // 2}" x2="{width - 8}" y2="{height // 2}" '
        f'stroke="#ffffff" stroke-opacity="0.4"/>',
        f'<circle cx="{width // 2}" cy="{height // 2}" r="40" fill="none" '
        f'stroke="#ffffff" stroke-opacity="0.4"/>',
        f'<text x="{width // 2}" y="22" fill="#ffffff" font-size="15" font-weight="bold" '
        f'text-anchor="middle">{_esc(team.team)} · {labels["formation"]} '
        f'{_esc(team.formation or "?")}</text>',
    ]

    idx = 0
    for r, count in enumerate(rows):
        y = height - margin_y - r * row_gap
        for c in range(count):
            x = margin_x + (width - 2 * margin_x) * (c + 1) / (count + 1)
            player = players[idx] if idx < len(players) else None
            idx += 1
            number = player.number if player and player.number is not None else ""
            name = player.name.split()[-1] if player and player.name else ""
            if player and player.is_captain:
                name += " (C)"
            parts.append(
                f'<circle cx="{x:.0f}" cy="{y:.0f}" r="15" fill="#ffffff" '
                f'stroke="#1b5e20" stroke-width="2"/>'
            )
            parts.append(
                f'<text x="{x:.0f}" y="{y + 4:.0f}" fill="#1b5e20" font-size="12" '
                f'font-weight="bold" text-anchor="middle">{number}</text>'
            )
            parts.append(
                f'<text x="{x:.0f}" y="{y + 30:.0f}" fill="#ffffff" font-size="10" '
                f'text-anchor="middle">{_esc(name)}</text>'
            )
    parts.append("</svg>")
    return "".join(parts)


def _esc(text: str) -> str:
    """Minimal XML escaping for text nodes."""
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def main(argv: Optional[list] = None) -> int:
    """CLI: print a match's line-ups, or write per-team formation SVGs."""
    parser = argparse.ArgumentParser(description="Show StatsBomb line-ups + formation diagram.")
    parser.add_argument("--match-id", type=int, default=None)
    parser.add_argument("--language", default="en")
    parser.add_argument("--svg-out", default=None, help="Directory to write <team>.svg files.")
    args = parser.parse_args(argv)

    teams = fetch_lineups(args.match_id)
    if not teams:
        print("No line-up data (check the match id / network).")
        return 1

    labels = labels_for(args.language)
    for team in teams:
        print(f"\n# {team.team} — {labels['formation']} {team.formation}  "
              f"({labels['manager']}: {team.manager or '?'})")
        for player in team.starting_xi:
            cap = f" ({labels['captain']})" if player.is_captain else ""
            print(f"  {str(player.number or '').rjust(2)}  {player.name}{cap} — {player.position}")
        print(f"  {labels['subs']}: " + ", ".join(p.name for p in team.substitutes))
        if args.svg_out:
            out_dir = Path(args.svg_out)
            out_dir.mkdir(parents=True, exist_ok=True)
            svg_path = out_dir / f"{team.team.replace(' ', '_')}.svg"
            svg_path.write_text(formation_svg(team, args.language), encoding="utf-8")
            print(f"  -> wrote {svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
