from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / 'transformer_model' / 'training'
sys.path.insert(0, str(TRAINING))

from dataset import load_data, resolve_data_dir  # noqa: E402
from model import TransformerTrajectoryPredictor  # noqa: E402


def test_preserved_flat_data_is_discoverable():
    data_dir = resolve_data_dir()
    assert data_dir == ROOT / 'transformer_model' / 'data'
    x, y, scene_ids = load_data(device='cpu')
    assert tuple(x.shape) == (233948, 4, 2)
    assert tuple(y.shape) == (233948, 12, 2)
    assert tuple(scene_ids.shape) == (233948,)


def test_transformer_output_shape_and_finiteness():
    model = TransformerTrajectoryPredictor()
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 4, 2))
    assert tuple(output.shape) == (1, 12, 2)
    assert torch.isfinite(output).all()
