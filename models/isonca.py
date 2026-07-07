import torch

from models.nca2d import depthwise_conv


class IsoNCA(torch.nn.Module):
    """
    Laplacian-only isotropic 2D NCA.

    The public contract matches models.nca2d.NCA: forward returns
    (updated_state, perception), and seed returns [batch, channels, h, w].
    """

    def __init__(
            self,
            channels,
            fc_dim,
            padding='circular',
            update_prob=0.5,
            seed_mode='zeros',
            noise_level=0.1,
            device=None,
            precision=torch.float32,
    ):
        super().__init__()
        if seed_mode not in {"zeros", "random"}:
            raise ValueError("seed_mode must be either 'zeros' or 'random'")

        self.channels = channels
        self.fc_dim = fc_dim
        self.padding = padding
        self.perception_kernels = 2
        self.update_prob = update_prob
        self.seed_mode = seed_mode
        self.device = device
        self.precision = precision

        self.w1 = torch.nn.Conv2d(channels * self.perception_kernels, fc_dim, 1, bias=True, device=device)
        self.w2 = torch.nn.Conv2d(fc_dim, channels, 1, bias=False, device=device)

        torch.nn.init.xavier_normal_(self.w1.weight, gain=0.2)
        torch.nn.init.zeros_(self.w2.weight)

        laplacian = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, -12.0, 2.0], [1.0, 2.0, 1.0]],
            device=device,
            dtype=precision,
        )
        self.register_buffer("filters", laplacian[None])
        self.register_buffer("noise_level", torch.tensor([noise_level], device=device, dtype=precision))

    def perception(self, s):
        state_lap = depthwise_conv(s, self.filters, self.padding)
        return torch.cat([s, state_lap], dim=1)

    def adaptation(self, s):
        z = self.perception(s)
        delta_s = self.w2(torch.relu(self.w1(z)))
        return delta_s, z

    def step_euler(self, s, dt=1.0):
        delta_s, z = self.adaptation(s)
        M = 1.0
        if self.update_prob < 1.0:
            b, _, h, w = s.shape
            M = (torch.rand(b, 1, h, w, device=s.device, dtype=self.precision) + self.update_prob).floor()

        return s + delta_s * M * dt, z

    def step_rk4(self, s, dt=1.0):
        M = 1.0
        if self.update_prob < 1.0:
            b, _, h, w = s.shape
            M = (torch.rand(b, 1, h, w, device=s.device, dtype=self.precision) + self.update_prob).floor()

        k1, z1 = self.adaptation(s)
        k2, z2 = self.adaptation(s + k1 * 0.5 * M)
        k3, z3 = self.adaptation(s + k2 * 0.5 * M)
        k4, z4 = self.adaptation(s + k3 * M)

        return s + (k1 + 2 * k2 + 2 * k3 + k4) * dt * M / 6.0, (z1 + 2 * z2 + 2 * z3 + z4) / 6.0

    def forward(self, s, dx=1.0, dy=1.0, dt=1.0, integrator='euler'):
        """
        Computes one NCA step. dx and dy are accepted for API compatibility
        with models.nca2d.NCA, but isotropic perception does not use them.
        """
        del dx, dy
        if integrator == 'euler':
            return self.step_euler(s, dt)
        if integrator == 'rk4':
            return self.step_rk4(s, dt)
        raise ValueError("Invalid integrator. Must be either 'euler' or 'rk4'")

    def seed(self, n, h=128, w=128):
        if self.seed_mode == "random":
            return (torch.rand(n, self.channels, h, w, device=self.device, dtype=self.precision) - 0.5) * self.noise_level
        return torch.zeros(n, self.channels, h, w, device=self.device, dtype=self.precision)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device = self.w1.weight.device
        return self


class IsoGrowingNCA(IsoNCA):
    """
    Growing variant of IsoNCA for RGBA morphology targets.

    The first three channels are decoded as RGB by the SIREN, and channel 4
    is the alpha/living channel used for masking the automaton state.
    """

    def __init__(self, channels, fc_dim, **kwargs):
        kwargs.setdefault("seed_mode", "zeros")
        super().__init__(channels, fc_dim, **kwargs)

        torch.nn.init.xavier_normal_(self.w1.weight, gain=0.1)
        torch.nn.init.xavier_normal_(self.w2.weight, gain=0.1)

    def forward(self, s, dx=1.0, dy=1.0, dt=1.0, integrator='euler'):
        pre_life_mask = self.get_living_mask(s)
        new_s, z = super().forward(s, dx, dy, dt, integrator)
        post_life_mask = self.get_living_mask(new_s)

        new_s = new_s * torch.logical_and(pre_life_mask, post_life_mask).to(self.precision)
        return new_s, z

    @staticmethod
    def get_living_mask(s):
        alpha = s[:, 3:4]
        return torch.nn.functional.max_pool2d(alpha, 3, stride=1, padding=1) > 0.1

    def seed(self, n, h=128, w=128):
        s = torch.zeros(n, self.channels, h, w, device=self.device, dtype=self.precision)
        s[:, 3:, h // 2, w // 2] = 1.0
        return s
