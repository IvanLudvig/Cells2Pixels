import torch

from losses.invariant_image_loss import InvariantImageLoss
from models.isonca import BlogpostIsoGrowingNCA, GradNormIsoGrowingNCA, IsoGrowingNCA, IsoNCA


def rollout(model, x, steps=8):
    z = None
    for _ in range(steps):
        x, z = model(x)
    return x, z


def test_isonca_seed_and_forward_shapes():
    model = IsoNCA(channels=4, fc_dim=8, update_prob=1.0, seed_mode="zeros")
    x = model.seed(2, 16, 20)
    y, z = model(x)

    assert x.shape == (2, 4, 16, 20)
    assert y.shape == x.shape
    assert z.shape == (2, 8, 16, 20)
    assert model.perception_kernels == 2


def test_isonca_zero_seed_stays_spatially_uniform():
    model = IsoNCA(channels=4, fc_dim=8, update_prob=1.0, seed_mode="zeros")
    x = model.seed(1, 16, 16)
    y, _ = rollout(model, x, steps=4)

    assert torch.allclose(y, y[:, :, :1, :1].expand_as(y))


def test_isonca_perception_is_equivariant_for_90_degree_rotations():
    torch.manual_seed(0)
    model = IsoNCA(channels=4, fc_dim=8, update_prob=1.0, seed_mode="random")
    x = model.seed(1, 16, 16)

    z = model.perception(x)
    z_rot = model.perception(torch.rot90(x, 1, dims=(-2, -1)))
    expected = torch.rot90(z, 1, dims=(-2, -1))

    assert torch.allclose(z_rot, expected, atol=1e-5, rtol=1e-5)


def test_isonca_rollout_is_equivariant_for_90_degree_rotations():
    torch.manual_seed(0)
    model = IsoNCA(channels=4, fc_dim=8, update_prob=1.0, seed_mode="random")
    torch.nn.init.normal_(model.w2.weight, std=0.01)
    x = model.seed(1, 16, 16)

    y, _ = rollout(model, x, steps=8)
    y_rot, _ = rollout(model, torch.rot90(x, 1, dims=(-2, -1)), steps=8)
    expected = torch.rot90(y, 1, dims=(-2, -1))

    assert torch.allclose(y_rot, expected, atol=1e-5, rtol=1e-5)


def test_iso_growing_nca_seed_and_forward_shapes():
    model = IsoGrowingNCA(channels=8, fc_dim=16, update_prob=1.0)
    x = model.seed(2, 16, 16)
    y, z = model(x)

    assert x.shape == (2, 8, 16, 16)
    assert y.shape == x.shape
    assert z.shape == (2, 16, 16, 16)
    assert model.perception_kernels == 2
    assert x[:, 3:, 8, 8].eq(1.0).all()
    assert x[:, :, :8, :8].sum().eq(0.0)


def test_iso_growing_nca_keeps_center_seed_alive():
    model = IsoGrowingNCA(channels=8, fc_dim=16, update_prob=1.0)
    x = model.seed(1, 16, 16)
    y, _ = model(x)

    assert model.get_living_mask(y)[:, :, 8, 8].all()


def test_blogpost_iso_growing_nca_default_hidden_width():
    model = BlogpostIsoGrowingNCA(channels=16, fc_dim=None, update_prob=1.0)

    assert model.fc_dim == 192
    assert model.w1.out_channels == 192
    assert model.perception_kernels == 2


def test_blogpost_iso_growing_nca_structured_seed_matches_blogpost_positions():
    model = BlogpostIsoGrowingNCA(channels=16, fc_dim=None, update_prob=1.0, seed_points=2, seed_radius=4)
    x = model.seed(1, 72, 72)

    assert x[0, 3:, 40, 36].eq(1.0).all()
    assert x[0, 3:, 32, 36].eq(1.0).all()
    assert x[0, :3, 40, 36].tolist() == [1.0, 0.0, 0.0]
    assert x[0, :3, 32, 36].tolist() == [0.0, 1.0, 1.0]


def test_blogpost_iso_growing_nca_uses_pre_update_alive_mask_only():
    model = BlogpostIsoGrowingNCA(channels=4, fc_dim=8, update_prob=1.0)
    x = torch.zeros(1, 4, 8, 8)
    x[:, 3:, 4, 4] = 1.0

    with torch.no_grad():
        model.w1.weight.zero_()
        model.w1.bias.fill_(1.0)
        model.w2.weight.zero_()
        model.w2.weight[3, :, 0, 0] = -1.0

    y, _ = model(x)

    assert y[:, :, 4, 4].abs().sum() > 0.0
    assert not model.get_living_mask(y)[:, :, 4, 4].all()


def test_gradnorm_iso_growing_nca_default_hidden_width():
    model = GradNormIsoGrowingNCA(channels=16, fc_dim=None, update_prob=1.0)

    assert model.fc_dim == 128
    assert model.w1.in_channels == 48
    assert model.w1.out_channels == 128
    assert model.perception_kernels == 3


def test_gradnorm_iso_growing_nca_seed_and_forward_shapes():
    model = GradNormIsoGrowingNCA(channels=16, fc_dim=32, update_prob=1.0)
    x = model.seed(2, 16, 16)
    y, z = model(x)

    assert x.shape == (2, 16, 16, 16)
    assert y.shape == x.shape
    assert z.shape == (2, 48, 16, 16)
    assert x[:, 3:, 8, 8].eq(1.0).all()


def test_gradnorm_iso_growing_nca_perception_is_equivariant_for_90_degree_rotations():
    torch.manual_seed(0)
    model = GradNormIsoGrowingNCA(channels=8, fc_dim=16, update_prob=1.0)
    x = torch.rand(1, 8, 16, 16)

    z = model.perception(x)
    z_rot = model.perception(torch.rot90(x, 1, dims=(-2, -1)))
    expected = torch.rot90(z, 1, dims=(-2, -1))

    assert torch.allclose(z_rot, expected, atol=1e-5, rtol=1e-5)


def test_invariant_image_loss_accepts_rotation_and_mirror():
    target = torch.zeros(5, 32, 32)
    target[:3, 8:16, 10:22] = 0.75
    target[3, 8:16, 10:22] = 1.0
    target[4, 8:16, 10:16] = -0.5
    target[4, 8:16, 16:22] = 0.5

    loss_fn = InvariantImageLoss.__new__(InvariantImageLoss)
    torch.nn.Module.__init__(loss_fn)
    loss_fn.device = "cpu"
    loss_fn.channel_n = target.shape[0]
    loss_fn.include_nca_alpha = False
    loss_fn.mirror = True
    loss_fn.sharpen = False
    loss_fn.l2_weight = 1.0
    loss_fn.register_buffer("target_image", target[None])
    loss_fn._build_polar_target()

    same = loss_fn({"generated_images": target[None]}, return_summary=False)[0]
    rotated = loss_fn({"generated_images": torch.rot90(target, 1, dims=(-2, -1))[None]}, return_summary=False)[0]
    mirrored = loss_fn({"generated_images": torch.flip(target, dims=(-1,))[None]}, return_summary=False)[0]

    assert rotated <= same + 1e-4
    assert mirrored <= same + 1e-4


def test_invariant_image_loss_aligns_nca_alpha_with_rendered_channels():
    target = torch.zeros(5, 32, 32)
    target[:3, 8:16, 10:22] = 0.75
    target[3, 8:16, 10:22] = 1.0
    target[4, 8:16, 10:16] = -0.5
    target[4, 8:16, 16:22] = 0.5

    loss_fn = InvariantImageLoss.__new__(InvariantImageLoss)
    torch.nn.Module.__init__(loss_fn)
    loss_fn.device = "cpu"
    loss_fn.channel_n = target.shape[0]
    loss_fn.include_nca_alpha = True
    loss_fn.mirror = True
    loss_fn.sharpen = False
    loss_fn.l2_weight = 1.0
    loss_fn.register_buffer("target_image", target[None])
    loss_fn._build_polar_target()

    alpha = target[3:4][None]
    same = loss_fn({"generated_images": target[None], "alpha": alpha}, return_summary=False)[0]
    rotated = loss_fn({
        "generated_images": torch.rot90(target, 1, dims=(-2, -1))[None],
        "alpha": torch.rot90(alpha, 1, dims=(-2, -1)),
    }, return_summary=False)[0]
    wrong_alpha = loss_fn({
        "generated_images": target[None],
        "alpha": torch.zeros_like(alpha),
    }, return_summary=False)[0]

    assert rotated <= same + 1e-4
    assert wrong_alpha > same + 1e-4
