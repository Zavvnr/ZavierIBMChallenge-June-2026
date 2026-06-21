"""Formation recognition: classify a team's shape from its player positions.

Two paths mirror the rest of vision_model:
  - Offline stub: ``StubFormationPredictor`` replays scripted formation strings —
    no torch, no GPU, so tests and the demo CLI run without extra installs.
  - Real path: ``FormationClassifier`` wraps a torch ``nn.Sequential`` MLP and
    classifies normalised player positions into one of FORMATIONS.

Preprocessing (``preprocess``):
  1. Normalise x ∈ [0, 120] → [0, 1], y ∈ [0, 80] → [0, 1].
  2. Optionally flip x so the team always 'attacks' left → right in feature space.
  3. Sort by normalised x, deepest/defensive player first.
  4. Pad or truncate to exactly N_PLAYERS = 10 outfield players.
  5. Flatten to a 20-float feature vector.

Quick start (offline):
    from vision_model.formation import StubFormationPredictor
    p = StubFormationPredictor(["4-4-2"])
    print(p.predict([(20, 15), (20, 38), (20, 62), (20, 85),
                     (50, 15), (50, 38), (50, 62), (50, 85),
                     (80, 35), (80, 65)]))  # -> "4-4-2"

Real path (needs torch):
    from vision_model.formation import FormationClassifier
    clf = FormationClassifier()
    print(clf.predict(player_positions))
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from vision_model.schema import PITCH_LENGTH, PITCH_WIDTH

N_PLAYERS = 10        # outfield players per team (goalkeeper excluded)
_SENTINEL = -1.0      # padding value for absent/missing players

# Canonical formation labels in a fixed order (index = class label).
FORMATIONS: List[str] = [
    "4-4-2",
    "4-3-3",
    "4-2-3-1",
    "3-5-2",
    "3-4-3",
    "5-3-2",
    "4-5-1",
    "4-1-4-1",
]


def preprocess(
    positions: Sequence[Tuple[float, float]],
    attack_left_to_right: bool = True,
) -> List[float]:
    """Normalise, orient, sort, and flatten player positions into a 20-float vector.

    ``attack_left_to_right=False`` flips x so teams attacking right → left are
    represented identically to teams attacking left → right, keeping the network's
    input space canonical regardless of which half the team defends.
    """
    normalised: List[Tuple[float, float]] = []
    for x, y in positions:
        nx = float(x) / PITCH_LENGTH
        ny = float(y) / PITCH_WIDTH
        if not attack_left_to_right:
            nx = 1.0 - nx
        normalised.append((nx, ny))

    normalised.sort(key=lambda p: p[0])        # deepest/defensive player first
    normalised = normalised[:N_PLAYERS]
    while len(normalised) < N_PLAYERS:
        normalised.append((_SENTINEL, _SENTINEL))

    return [coord for px, py in normalised for coord in (px, py)]


def _make_net():
    """Build the formation MLP as an ``nn.Sequential`` (requires torch).

    Architecture:
        Input(20) → Linear(128) → BN → ReLU → Dropout(0.3)
                  → Linear(64)  → BN → ReLU → Dropout(0.2)
                  → Linear(32)  → ReLU → Linear(N_FORMATIONS)
    """
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise SystemExit(
            "vision_model: torch not installed. "
            "Install it (`pip install torch`) or use StubFormationPredictor."
        ) from exc

    n = len(FORMATIONS)
    return nn.Sequential(
        nn.Linear(N_PLAYERS * 2, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, n),
    )


class FormationClassifier:
    """Wraps the formation MLP with predict / predict_proba methods (requires torch).

    Use ``StubFormationPredictor`` for offline tests and the demo CLI.
    """

    def __init__(self, net=None):
        try:
            import torch
        except ImportError as exc:
            raise SystemExit(
                "vision_model: torch not installed. Use StubFormationPredictor."
            ) from exc
        self._torch = torch
        self._net = net if net is not None else _make_net()
        self._net.eval()

    def predict(
        self,
        positions: Sequence[Tuple[float, float]],
        attack_left_to_right: bool = True,
    ) -> str:
        """Return the most likely formation string for the given player positions."""
        features = preprocess(positions, attack_left_to_right)
        t = self._torch.tensor([features], dtype=self._torch.float32)
        self._net.eval()
        with self._torch.no_grad():
            idx = int(self._torch.argmax(self._net(t), dim=1).item())
        return FORMATIONS[idx]

    def predict_proba(
        self,
        positions: Sequence[Tuple[float, float]],
        attack_left_to_right: bool = True,
    ) -> List[Tuple[str, float]]:
        """Return (formation, probability) pairs sorted by probability descending."""
        import torch.nn.functional as F
        features = preprocess(positions, attack_left_to_right)
        t = self._torch.tensor([features], dtype=self._torch.float32)
        self._net.eval()
        with self._torch.no_grad():
            probs = F.softmax(self._net(t), dim=1)[0].tolist()
        return sorted(zip(FORMATIONS, probs), key=lambda x: x[1], reverse=True)

    def save(self, path: str) -> None:
        """Persist model weights to ``path``."""
        self._torch.save(self._net.state_dict(), path)

    @classmethod
    def load(cls, path: str) -> "FormationClassifier":
        """Load a previously saved model from ``path`` (requires torch)."""
        import torch
        net = _make_net()
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return cls(net)


class StubFormationPredictor:
    """Deterministic offline predictor — no torch, no GPU.

    Replays a scripted list of formation strings in round-robin order. The
    ``positions`` argument is accepted (and ignored) so callers can swap the
    stub for a real classifier without changing call sites.
    """

    def __init__(self, formations: Optional[List[str]] = None):
        self._formations = formations or ["4-4-2"]
        self._idx = 0

    def predict(
        self,
        positions: Sequence[Tuple[float, float]],
        attack_left_to_right: bool = True,
    ) -> str:
        """Return the next scripted formation (ignores position data)."""
        result = self._formations[self._idx % len(self._formations)]
        self._idx += 1
        return result


def build_predictor(kind: str = "stub", **kwargs):
    """Return a predictor by name.

    kind='stub' (offline, no torch): optional ``formations`` keyword list.
    kind='real' (needs torch):       optional ``path`` to load saved weights.
    """
    if kind == "stub":
        return StubFormationPredictor(kwargs.get("formations"))
    if kind == "real":
        path = kwargs.get("path")
        return FormationClassifier.load(path) if path else FormationClassifier()
    raise ValueError(f"unknown predictor kind: {kind!r}")
