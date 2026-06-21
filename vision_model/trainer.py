"""Synthetic training and evaluation pipeline for the formation-recognition MLP.

Real labelled broadcast video is unavailable, so training data is synthesised:
each formation is modelled as a noisy perturbation of a canonical player-position
grid, giving a lightweight MLP enough signal to distinguish structural shapes.

Data generation (``generate_dataset``):
  1. Start from a canonical grid for each formation (normalised [0,1]×[0,1]).
  2. Add per-player Gaussian noise (default σ=0.05) to simulate real variance.
  3. Scale to pitch coordinates and run through ``preprocess`` for a 20-float vector.
  4. Label = formation index in FORMATIONS.

Commands:
    python -m vision_model.trainer --train [--epochs N] [--save path.pt]
    python -m vision_model.trainer --eval  [--load path.pt]
    python -m vision_model.trainer --demo
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from vision_model.formation import FORMATIONS, N_PLAYERS, preprocess
from vision_model.schema import PITCH_LENGTH, PITCH_WIDTH

import argparse
import random

# Canonical outfield player positions in normalised [0,1]×[0,1] space,
# sorted defensive → offensive (ascending x).  Exactly N_PLAYERS each.
_CANONICAL: Dict[str, List[Tuple[float, float]]] = {
    "4-4-2": [
        (0.20, 0.15), (0.20, 0.38), (0.20, 0.62), (0.20, 0.85),
        (0.50, 0.15), (0.50, 0.38), (0.50, 0.62), (0.50, 0.85),
        (0.80, 0.35), (0.80, 0.65),
    ],
    "4-3-3": [
        (0.20, 0.15), (0.20, 0.38), (0.20, 0.62), (0.20, 0.85),
        (0.50, 0.25), (0.50, 0.50), (0.50, 0.75),
        (0.80, 0.20), (0.80, 0.50), (0.80, 0.80),
    ],
    "4-2-3-1": [
        (0.20, 0.15), (0.20, 0.38), (0.20, 0.62), (0.20, 0.85),
        (0.40, 0.35), (0.40, 0.65),
        (0.60, 0.20), (0.60, 0.50), (0.60, 0.80),
        (0.80, 0.50),
    ],
    "3-5-2": [
        (0.20, 0.25), (0.20, 0.50), (0.20, 0.75),
        (0.50, 0.10), (0.50, 0.30), (0.50, 0.50), (0.50, 0.70), (0.50, 0.90),
        (0.80, 0.35), (0.80, 0.65),
    ],
    "3-4-3": [
        (0.20, 0.25), (0.20, 0.50), (0.20, 0.75),
        (0.50, 0.20), (0.50, 0.42), (0.50, 0.58), (0.50, 0.80),
        (0.80, 0.20), (0.80, 0.50), (0.80, 0.80),
    ],
    "5-3-2": [
        (0.15, 0.10), (0.15, 0.30), (0.15, 0.50), (0.15, 0.70), (0.15, 0.90),
        (0.50, 0.25), (0.50, 0.50), (0.50, 0.75),
        (0.80, 0.35), (0.80, 0.65),
    ],
    "4-5-1": [
        (0.20, 0.15), (0.20, 0.38), (0.20, 0.62), (0.20, 0.85),
        (0.50, 0.10), (0.50, 0.30), (0.50, 0.50), (0.50, 0.70), (0.50, 0.90),
        (0.80, 0.50),
    ],
    "4-1-4-1": [
        (0.20, 0.15), (0.20, 0.38), (0.20, 0.62), (0.20, 0.85),
        (0.35, 0.50),
        (0.55, 0.15), (0.55, 0.38), (0.55, 0.62), (0.55, 0.85),
        (0.80, 0.50),
    ],
}

# Guard: every canonical entry must have exactly N_PLAYERS positions.
for _f, _pos in _CANONICAL.items():
    if len(_pos) != N_PLAYERS:
        raise AssertionError(
            f"_CANONICAL[{_f!r}] has {len(_pos)} players; expected {N_PLAYERS}"
        )


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _perturb(
    positions: List[Tuple[float, float]],
    noise_std: float,
    rng: random.Random,
) -> List[Tuple[float, float]]:
    """Add per-player Gaussian noise and clamp to [0, 1]×[0, 1]."""
    return [
        (
            max(0.0, min(1.0, x + rng.gauss(0, noise_std))),
            max(0.0, min(1.0, y + rng.gauss(0, noise_std))),
        )
        for x, y in positions
    ]


def _augment(
    positions: List[Tuple[float, float]],
    rng: random.Random,
) -> List[Tuple[float, float]]:
    """Realistic position-level augmentation in normalised space (formation-preserving).

    Mirrors left<->right (a real symmetry preprocess does NOT canonicalise), shifts the
    block up/deep and compresses or stretches its depth (high line vs low block), and
    occasionally drops a player to mimic a missed detection. This is cheaper and more
    on-distribution than image augmentation, because at inference the model sees noisy
    detector positions, not pixels.
    """
    pos = list(positions)
    if rng.random() < 0.5:                          # left-right mirror (y axis)
        pos = [(x, 1.0 - y) for x, y in pos]
    mean_x = sum(x for x, _ in pos) / len(pos)
    shift = rng.uniform(-0.12, 0.12)                # team pushed up / dropped deep
    scale = rng.uniform(0.85, 1.15)                 # compact vs stretched in depth
    pos = [(mean_x + (x - mean_x) * scale + shift, y) for x, y in pos]
    if len(pos) > 6 and rng.random() < 0.15:        # simulate a missed detection
        pos.pop(rng.randrange(len(pos)))
    return [(max(0.0, min(1.0, x)), max(0.0, min(1.0, y))) for x, y in pos]


def generate_dataset(
    n_per_formation: int = 500,
    noise_std: float = 0.05,
    seed: int = 42,
    augment: bool = False,
) -> Tuple[List[List[float]], List[int]]:
    """Generate a synthetic (X, y) dataset for formation classification.

    Each sample is a 20-float feature vector from ``preprocess``; labels are
    indices into ``FORMATIONS``.  The same seed gives the same dataset for
    reproducible evaluation.
    """
    rng = random.Random(seed)
    X: List[List[float]] = []
    y: List[int] = []

    for label_idx, formation in enumerate(FORMATIONS):
        canonical = _CANONICAL[formation]
        for _ in range(n_per_formation):
            base = _augment(canonical, rng) if augment else canonical
            noisy_norm = _perturb(base, noise_std, rng)
            # Scale normalised positions to pitch coordinates for preprocess().
            pitch = [(nx * PITCH_LENGTH, ny * PITCH_WIDTH) for nx, ny in noisy_norm]
            X.append(preprocess(pitch))
            y.append(label_idx)

    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    epochs: int = 60,
    lr: float = 1e-3,
    batch_size: int = 64,
    n_per_formation: int = 500,
    noise_std: float = 0.05,
    seed: int = 42,
    augment: bool = False,
    verbose: bool = True,
):
    """Train FormationNet on synthetic data and return the trained ``nn.Sequential``.

    Uses an 80/20 train/validation split with the same seed for reproducibility.
    Prints loss and validation accuracy every 10 epochs when ``verbose=True``.
    Requires torch.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError as exc:
        raise SystemExit("vision_model.trainer: torch not installed.") from exc

    from vision_model.formation import _make_net

    X_all, y_all = generate_dataset(n_per_formation, noise_std, seed, augment=augment)

    # Shuffle BEFORE splitting: generate_dataset emits samples grouped by formation, so a
    # raw first-80% / last-20% slice would leave whole classes out of training and make
    # validation accuracy meaningless (~chance). Seeded so the split stays reproducible.
    order = list(range(len(X_all)))
    random.Random(seed).shuffle(order)
    X_all = [X_all[i] for i in order]
    y_all = [y_all[i] for i in order]

    split = int(len(X_all) * 0.8)
    X_train, y_train = X_all[:split], y_all[:split]
    X_val, y_val = X_all[split:], y_all[split:]
    n_train = len(X_train)

    net = _make_net()
    optimiser = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size].tolist()
            xb = torch.tensor([X_train[i] for i in idx], dtype=torch.float32)
            yb = torch.tensor([y_train[i] for i in idx], dtype=torch.long)
            optimiser.zero_grad()
            loss = criterion(net(xb), yb)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * len(idx)

        if verbose and epoch % 10 == 0:
            net.eval()
            with torch.no_grad():
                xv = torch.tensor(X_val, dtype=torch.float32)
                yv = torch.tensor(y_val, dtype=torch.long)
                val_acc = (torch.argmax(net(xv), dim=1) == yv).float().mean().item()
            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"loss={epoch_loss / n_train:.4f}  val_acc={val_acc:.4f}"
            )

    return net


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(net, n_per_formation: int = 200, noise_std: float = 0.05, seed: int = 99) -> Dict:
    """Evaluate a trained net on a fresh synthetic test set (different seed).

    Returns ``{'overall': float, 'per_formation': {name: float}}``.
    Requires torch.
    """
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("vision_model.trainer: torch not installed.") from exc

    from collections import defaultdict

    X_test, y_test = generate_dataset(n_per_formation, noise_std, seed)

    net.eval()
    with torch.no_grad():
        xt = torch.tensor(X_test, dtype=torch.float32)
        yt = torch.tensor(y_test, dtype=torch.long)
        preds = torch.argmax(net(xt), dim=1).tolist()

    correct: Dict[int, int] = defaultdict(int)
    total: Dict[int, int] = defaultdict(int)
    for pred, true in zip(preds, y_test):
        total[true] += 1
        if pred == true:
            correct[true] += 1

    per_formation = {
        FORMATIONS[i]: correct[i] / total[i] if total[i] else 0.0
        for i in range(len(FORMATIONS))
    }
    overall = sum(correct.values()) / max(1, sum(total.values()))
    return {"overall": overall, "per_formation": per_formation}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """Train, evaluate, or demo the formation model from the command line."""
    parser = argparse.ArgumentParser(description="Formation model trainer/evaluator.")
    parser.add_argument("--train", action="store_true", help="Train and save the model.")
    parser.add_argument("--eval", action="store_true", help="Evaluate a saved model.")
    parser.add_argument("--demo", action="store_true", help="Quick offline stub demo.")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs.")
    parser.add_argument("--augment", action="store_true",
                        help="Add position-level augmentation (mirror, push/drop, width, dropouts).")
    parser.add_argument("--save", default="formation.pt", help="Path to save weights.")
    parser.add_argument("--load", default="formation.pt", help="Path to load weights from.")
    args = parser.parse_args(argv)

    if args.train:
        print(f"Training for {args.epochs} epochs{' + augmentation' if args.augment else ''} …")
        net = train_model(epochs=args.epochs, augment=args.augment)
        from vision_model.formation import FormationClassifier
        FormationClassifier(net).save(args.save)
        print(f"Saved to {args.save}")

    if args.eval:
        from vision_model.formation import FormationClassifier
        clf = FormationClassifier.load(args.load)
        results = evaluate_model(clf._net)
        print(f"\nOverall accuracy: {results['overall']:.4f}")
        for formation, acc in results["per_formation"].items():
            print(f"  {formation:10s}: {acc:.4f}")

    if args.demo:
        from vision_model.formation import StubFormationPredictor
        positions_442 = [
            (24, 12), (24, 30), (24, 50), (24, 68),
            (60, 12), (60, 30), (60, 50), (60, 68),
            (96, 28), (96, 52),
        ]
        stub = StubFormationPredictor(["4-4-2"])
        print(f"Demo (offline stub): predicted formation = {stub.predict(positions_442)}")

    if not any([args.train, args.eval, args.demo]):
        parser.print_help()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
