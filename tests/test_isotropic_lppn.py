import pytest
import torch

from models.isotropic_lppn import IsotropicNeighborhoodLPPN
from models.isonca import GradNormIsoGrowingNCA


@pytest.mark.parametrize("radial_distance", [False, True])
def test_isotropic_lppn_shape_and_neighbor_permutation_invariance(radial_distance):
    torch.manual_seed(0)
    decoder = IsotropicNeighborhoodLPPN(
        in_features=6,
        out_features=5,
        scale_factor=3,
        message_features=12,
        hidden_features=16,
        radial_distance=radial_distance,
    )
    state = torch.randn(2, 6, 7, 9)

    output = decoder(state)
    permuted = decoder(state, neighbor_permutation=[2, 0, 3, 1])

    assert output.shape == (2, 21, 27, 5)
    assert torch.allclose(output, permuted, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("radial_distance", [False, True])
@pytest.mark.parametrize(
    ("state_transform", "image_transform"),
    [
        (lambda x: torch.rot90(x, 1, (-2, -1)), lambda x: torch.rot90(x, 1, (-3, -2))),
        (lambda x: torch.rot90(x, 2, (-2, -1)), lambda x: torch.rot90(x, 2, (-3, -2))),
        (lambda x: torch.rot90(x, 3, (-2, -1)), lambda x: torch.rot90(x, 3, (-3, -2))),
        (lambda x: torch.flip(x, (-1,)), lambda x: torch.flip(x, (-2,))),
        (lambda x: torch.flip(x, (-2,)), lambda x: torch.flip(x, (-3,))),
    ],
)
def test_isotropic_lppn_is_d4_equivariant(radial_distance, state_transform, image_transform):
    torch.manual_seed(1)
    decoder = IsotropicNeighborhoodLPPN(
        in_features=4,
        out_features=3,
        scale_factor=4,
        message_features=8,
        hidden_features=12,
        radial_distance=radial_distance,
    )
    state = torch.randn(1, 4, 8, 8)

    transformed_input = decoder(state_transform(state))
    transformed_output = image_transform(decoder(state))

    assert torch.allclose(transformed_input, transformed_output, atol=2e-6, rtol=2e-6)


def test_isotropic_lppn_backpropagates_to_state_and_parameters():
    torch.manual_seed(2)
    decoder = IsotropicNeighborhoodLPPN(
        in_features=4,
        out_features=3,
        scale_factor=2,
        message_features=8,
        hidden_features=12,
        radial_distance=True,
    )
    state = torch.randn(1, 4, 5, 5, requires_grad=True)

    decoder(state).square().mean().backward()

    assert state.grad is not None
    assert torch.isfinite(state.grad).all()
    assert state.grad.abs().sum() > 0
    assert all(parameter.grad is not None for parameter in decoder.parameters())


def test_gradnorm_nca_accepts_explicit_update_mask():
    torch.manual_seed(3)
    model = GradNormIsoGrowingNCA(channels=8, fc_dim=16, update_prob=0.5)
    with torch.no_grad():
        torch.nn.init.normal_(model.w2.weight, std=0.01)
    state = model.seed(1, 8, 8)
    update_mask = torch.randint(0, 2, (1, 1, 8, 8), dtype=state.dtype)

    torch.manual_seed(4)
    first, _ = model(state, update_mask=update_mask)
    torch.manual_seed(5)
    second, _ = model(state, update_mask=update_mask)

    assert torch.equal(first, second)


def test_gradnorm_nca_is_pathwise_d4_equivariant_with_transformed_masks():
    torch.manual_seed(6)
    model = GradNormIsoGrowingNCA(channels=8, fc_dim=16, update_prob=0.5)
    with torch.no_grad():
        torch.nn.init.normal_(model.w2.weight, std=0.01)
    state = model.seed(1, 12, 12)
    masks = torch.randint(0, 2, (4, 1, 1, 12, 12), dtype=state.dtype)

    actual = torch.rot90(state, 1, (-2, -1))
    expected = state
    for mask in masks:
        expected, _ = model(expected, update_mask=mask)
        actual, _ = model(
            actual,
            update_mask=torch.rot90(mask, 1, (-2, -1)),
        )

    assert torch.allclose(
        actual,
        torch.rot90(expected, 1, (-2, -1)),
        atol=1e-5,
        rtol=1e-5,
    )


def test_gradnorm_nca_rejects_wrong_update_mask_shape():
    model = GradNormIsoGrowingNCA(channels=8, fc_dim=16, update_prob=0.5)
    state = model.seed(1, 8, 8)

    with pytest.raises(ValueError, match="update_mask must have shape"):
        model(state, update_mask=torch.ones(1, 8, 8))
