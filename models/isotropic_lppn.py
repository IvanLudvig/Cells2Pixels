import torch

from models.siren import SineLayer


def _sine_mlp(
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        first_omega_0,
        hidden_omega_0,
):
    layers = [
        SineLayer(
            in_features,
            hidden_features,
            is_first=True,
            omega_0=first_omega_0,
        )
    ]
    for _ in range(hidden_layers):
        layers.append(
            SineLayer(
                hidden_features,
                hidden_features,
                omega_0=hidden_omega_0,
            )
        )

    final = torch.nn.Linear(hidden_features, out_features)
    with torch.no_grad():
        bound = (6.0 / hidden_features) ** 0.5 / hidden_omega_0
        final.weight.uniform_(-bound, bound)
    layers.append(final)
    return torch.nn.Sequential(*layers)


class IsotropicNeighborhoodLPPN(torch.nn.Module):
    """
    Permutation-invariant local decoder for a regular 2D cell grid.

    Every output query gathers the four enclosing cells. A shared ``phi``
    network encodes each neighbor, the messages are reduced without retaining
    neighbor order, and ``rho`` maps the interpolated state plus the aggregate
    to output channels. The optional geometric input is radial distance only;
    raw x/y offsets are never exposed to the learned networks.

    Input is NCHW and output is NHWC to match ``Renderer2D.render``.
    """

    def __init__(
            self,
            in_features,
            out_features,
            scale_factor=8,
            message_features=48,
            hidden_features=96,
            phi_hidden_layers=1,
            rho_hidden_layers=1,
            radial_distance=False,
            aggregation="mean",
            first_omega_0=10.0,
            hidden_omega_0=10.0,
    ):
        super().__init__()
        if scale_factor < 1:
            raise ValueError("scale_factor must be at least 1")
        if aggregation not in {"sum", "mean"}:
            raise ValueError("aggregation must be 'sum' or 'mean'")

        self.in_features = in_features
        self.out_features = out_features
        self.output_dim = out_features
        self.scale_factor = scale_factor
        self.message_features = message_features
        self.radial_distance = radial_distance
        self.aggregation = aggregation
        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0
        self._geometry_cache = {}

        phi_in_features = 2 * in_features + int(radial_distance)
        self.phi = _sine_mlp(
            phi_in_features,
            message_features,
            phi_hidden_layers,
            message_features,
            first_omega_0,
            hidden_omega_0,
        )
        self.rho = _sine_mlp(
            in_features + message_features,
            hidden_features,
            rho_hidden_layers,
            out_features,
            first_omega_0,
            hidden_omega_0,
        )

    def _query_geometry(self, height, width, scale_factor, device, dtype):
        key = (height, width, scale_factor, device.type, device.index, dtype)
        cached = self._geometry_cache.get(key)
        if cached is not None:
            return cached

        py = (torch.arange(height * scale_factor, device=device, dtype=dtype) + 0.5) / scale_factor - 0.5
        px = (torch.arange(width * scale_factor, device=device, dtype=dtype) + 0.5) / scale_factor - 0.5
        py, px = torch.meshgrid(py, px, indexing="ij")

        y0 = torch.floor(py)
        x0 = torch.floor(px)
        y1 = y0 + 1.0
        x1 = x0 + 1.0

        neighbor_y = torch.stack([y0, y0, y1, y1], dim=-1)
        neighbor_x = torch.stack([x0, x1, x0, x1], dim=-1)
        dy = neighbor_y - py[..., None]
        dx = neighbor_x - px[..., None]
        weights = (1.0 - dy.abs()) * (1.0 - dx.abs())
        distance = torch.sqrt(dx.square() + dy.square() + torch.finfo(dtype).eps)[..., None]

        indices = (
            neighbor_y.to(torch.long).remainder(height) * width
            + neighbor_x.to(torch.long).remainder(width)
        )
        geometry = (indices, weights, distance)
        self._geometry_cache[key] = geometry
        return geometry

    def forward(self, cell_states, scale_factor=None, neighbor_permutation=None):
        if cell_states.ndim != 4:
            raise ValueError("cell_states must have shape [batch, channels, height, width]")
        batch, channels, height, width = cell_states.shape
        if channels != self.in_features:
            raise ValueError(f"expected {self.in_features} input channels, got {channels}")

        scale_factor = self.scale_factor if scale_factor is None else int(scale_factor)
        if scale_factor < 1:
            raise ValueError("scale_factor must be at least 1")

        indices, weights, distance = self._query_geometry(
            height,
            width,
            scale_factor,
            cell_states.device,
            cell_states.dtype,
        )
        out_height, out_width, neighbor_n = indices.shape
        flat_states = cell_states.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        neighbors = flat_states[:, indices.reshape(-1)].reshape(
            batch,
            out_height,
            out_width,
            neighbor_n,
            channels,
        )

        weights = weights[None, ..., None]
        interpolated = (neighbors * weights).sum(dim=-2)

        if neighbor_permutation is not None:
            permutation = torch.as_tensor(
                neighbor_permutation,
                device=cell_states.device,
                dtype=torch.long,
            )
            if permutation.shape != (neighbor_n,) or torch.unique(permutation).numel() != neighbor_n:
                raise ValueError(f"neighbor_permutation must permute {neighbor_n} entries")
            neighbors = neighbors.index_select(-2, permutation)
            distance = distance.index_select(-2, permutation)

        query = interpolated.unsqueeze(-2).expand_as(neighbors)
        message_inputs = [neighbors, neighbors - query]
        if self.radial_distance:
            message_inputs.append(
                distance[None].expand(batch, -1, -1, -1, -1).to(cell_states.dtype)
            )
        messages = self.phi(torch.cat(message_inputs, dim=-1))
        if self.aggregation == "sum":
            aggregate = messages.sum(dim=-2)
        else:
            aggregate = messages.mean(dim=-2)

        return self.rho(torch.cat([interpolated, aggregate], dim=-1))
