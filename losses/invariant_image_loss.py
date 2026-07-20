import torch
import torchvision.transforms.functional as TF

from losses.image_loss import imread, visualize_differences


def sharpen_filter(img: torch.Tensor):
    blurred = TF.gaussian_blur(img, [5, 5], [1, 1])
    return img + (img - blurred) * 2.0


class InvariantImageLoss(torch.nn.Module):
    """
    Rotation-invariant image loss from the IsoNCA blogpost, with optional mirror
    invariance and auxiliary target channels.
    """

    def __init__(
            self,
            target_path,
            image_size=(512, 512),
            padding=(32, 32),
            premultiply_alpha=True,
            aux_type="binary",
            mirror=True,
            sharpen=True,
            l2_weight=1.0,
            device='cuda:0',
    ):
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if isinstance(padding, int):
            padding = (padding, padding)
        if aux_type not in {"noaux", "binary"}:
            raise ValueError("InvariantImageLoss currently supports aux_type 'noaux' or 'binary'")

        self.target_path = target_path
        self.image_size = image_size
        self.padding = padding
        self.premultiply_alpha = premultiply_alpha
        self.aux_type = aux_type
        self.mirror = mirror
        self.sharpen = sharpen
        self.l2_weight = l2_weight
        self.device = device
        self.grid_size = (image_size[0] + 2 * padding[0], image_size[1] + 2 * padding[1])

        target_np = imread(target_path, max_size=image_size, mode='RGBA', center_crop=False)
        target = torch.tensor(target_np).permute(2, 0, 1)
        target = torch.nn.functional.pad(target, [padding[1], padding[1], padding[0], padding[0]])
        if premultiply_alpha:
            target[:3] *= target[3:4]
        target = self._add_aux_target(target, aux_type)

        self.register_buffer("target_image", target[None].to(device))
        self.channel_n = target.shape[0]
        self._build_polar_target()

    def _add_aux_target(self, target: torch.Tensor, aux_type: str):
        if aux_type == "noaux":
            return target

        _, h, w = target.shape
        if h != w:
            raise ValueError("InvariantImageLoss requires a square padded target")
        alpha = target[3]
        y_mask = torch.linspace(-1, 1, w, dtype=target.dtype, device=target.device)[:, None].sign()
        aux_target = y_mask * alpha * 0.5
        aux_target = aux_target[None] * alpha[None]
        return torch.cat([target, aux_target], dim=0)

    def _build_polar_target(self):
        target = self.target_image
        w = target.shape[-1]
        r = torch.linspace(0.5 / w, 1, w // 2, device=target.device, dtype=target.dtype)[:, None]
        angle = torch.arange(0, w * torch.pi + 1, device=target.device, dtype=target.dtype) / (w / 2)
        polar_xy = torch.stack([r * angle.cos(), r * angle.sin()], -1)[None]

        if self.sharpen:
            target = sharpen_filter(target)
        polar_target = torch.nn.functional.grid_sample(target, polar_xy, align_corners=False)

        self.register_buffer("polar_xy", polar_xy)
        self.register_buffer("polar_target", polar_target)
        self.register_buffer("fft_target", torch.fft.rfft(polar_target).conj())
        self.register_buffer("polar_target_sqnorm", polar_target.square().sum(-1, keepdim=True))

    def calc_losses(self, batch: torch.Tensor, extra_outputs=False):
        batch = batch[:, :self.channel_n].to(self.device)
        if self.sharpen:
            batch = sharpen_filter(batch)
        polar_batch = torch.nn.functional.grid_sample(
            batch,
            self.polar_xy.repeat(len(batch), 1, 1, 1),
            align_corners=False,
        )
        batch_fft = torch.fft.rfft(polar_batch)
        n = polar_batch.shape[-1]
        corr = torch.fft.irfft(batch_fft * self.fft_target, n=n)
        if self.mirror:
            corr = torch.cat([corr, torch.fft.irfft(batch_fft * self.fft_target.conj(), n=n)], -1)
        xx = polar_batch.square().sum(-1, keepdim=True)
        sqdiff = xx + self.polar_target_sqnorm - 2.0 * corr
        losses = sqdiff.mean([1, 2])
        if extra_outputs:
            return losses, batch, polar_batch
        return losses

    def forward(self, input_dict, return_summary=True):
        generated = input_dict['generated_images']
        losses = self.calc_losses(generated)
        invariant_l2 = losses.min(-1)[0].mean()
        loss = invariant_l2 * self.l2_weight
        loss_log = {"Invariant L2": invariant_l2}

        summary = None
        if return_summary:
            summary = {
                "images": visualize_differences(self.target_image[:, :4], generated[:, :4].to(self.device))
            }
        return loss, loss_log, summary
