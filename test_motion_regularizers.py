from types import SimpleNamespace

import pytest
import torch

from models.vit import ViTPredictor
from models.visual_world_model import VWorldModel


class MeanEncoder(torch.nn.Module):
    name = "dummy"
    emb_dim = 2

    def agg(self, tokens):
        return tokens.mean(dim=1)


def bare_model():
    model = object.__new__(VWorldModel)
    torch.nn.Module.__init__(model)
    model.encoder = MeanEncoder()
    return model


def test_calibrated_speed_is_invariant_to_units_and_global_scale():
    model = bare_model()
    state = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [6.0, 0.0]]]
    )
    features = torch.tensor(
        [[[0.0, 0.0], [2.0, 0.0], [6.0, 0.0], [12.0, 0.0]]]
    )
    assert model.calibrated_speed_loss(features, state).item() == pytest.approx(0.0)
    assert model.calibrated_speed_loss(features * 7.0, state).item() == pytest.approx(0.0)
    assert model.calibrated_speed_loss(features, state * 0.1).item() == pytest.approx(0.0)


def test_calibrated_speed_penalizes_wrong_relative_pace():
    model = bare_model()
    state = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [6.0, 0.0]]]
    )
    faithful = torch.tensor(
        [[[0.0, 0.0], [2.0, 0.0], [6.0, 0.0], [12.0, 0.0]]]
    )
    distorted = torch.tensor(
        [[[0.0, 0.0], [2.0, 0.0], [3.0, 0.0], [12.0, 0.0]]]
    )
    assert model.calibrated_speed_loss(distorted, state) > model.calibrated_speed_loss(
        faithful, state
    )


def test_predictor_returns_one_normalized_activation_per_block():
    predictor = ViTPredictor(
        num_patches=2,
        num_frames=3,
        dim=8,
        depth=3,
        heads=2,
        mlp_dim=16,
        dim_head=4,
    )
    x = torch.randn(4, 6, 8)
    final, intermediates = predictor(x, return_intermediates=True)
    assert final.shape == x.shape
    assert len(intermediates) == 3
    assert all(item.shape == x.shape for item in intermediates)
    assert torch.allclose(final, intermediates[-1])


def test_factorized_projection_modules_are_checkpointed_with_predictor():
    predictor = ViTPredictor(
        num_patches=1,
        num_frames=3,
        dim=8,
        depth=1,
        heads=1,
        mlp_dim=8,
        dim_head=8,
        motion_input_dim=2,
        direction_projection_dim=3,
        speed_projection_dim=1,
    )
    keys = predictor.state_dict()
    assert "direction_projection.weight" in keys
    assert "speed_projection.weight" in keys

