import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maskguard.temporal import box_iou
from maskguard.uncertainty import estimate_uncertainty


def test_iou_identical_boxes():
    assert box_iou((0, 0, 100, 100), (0, 0, 100, 100)) == 1.0


def test_iou_disjoint_boxes():
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_uncertainty_probabilities():
    uncertainty, uncertain, confidence = estimate_uncertainty(__import__("numpy").array([0.96, 0.01, 0.01, 0.01, 0.01]), 0.95)
    assert 0.0 <= uncertainty <= 1.0
    assert confidence > 0.9
    assert uncertain is False
