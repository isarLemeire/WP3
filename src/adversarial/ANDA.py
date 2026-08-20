import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchattacks.attack import Attack


def is_sqr(n: int) -> bool:
    """Checks if n is a perfect square."""
    return int(math.sqrt(n)) ** 2 == n


def get_thetas(grid_size: int, min_aug: float, max_aug: float) -> torch.Tensor:
    """Generates a 2D uniform grid of translation offsets (tx, ty)."""
    axis = torch.linspace(min_aug, max_aug, grid_size)
    grid_x, grid_y = torch.meshgrid(axis, axis, indexing="ij")
    thetas = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    return thetas  # Shape: [grid_size^2, 2]


def translation(thetas: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Applies 2D spatial translation using affine grid sampling."""
    N, C, H, W = x.shape
    device = x.device

    # Construct 2x3 affine transformation matrix [1, 0, tx; 0, 1, ty]
    theta_matrix = torch.zeros(N, 2, 3, device=device, dtype=x.dtype)
    theta_matrix[:, 0, 0] = 1.0
    theta_matrix[:, 1, 1] = 1.0
    theta_matrix[:, 0, 2] = thetas[:, 0]
    theta_matrix[:, 1, 2] = thetas[:, 1]

    grid = F.affine_grid(theta_matrix, x.size(), align_corners=False) # type: ignore
    return F.grid_sample(
        x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )


class ANDA_collector:
    def __init__(self, data_shape: tuple):
        self.data_shape = data_shape  # (B, C, H, W)
        self.noise_mean = None
        self.noise_std = None

    def collect_model(self, grads: torch.Tensor) -> None:
        """Computes empirical mean and std of gradients across ensemble copies."""
        B, C, H, W = self.data_shape
        total_samples = grads.shape[0]
        n_ens = total_samples // B

        # Reshape to separate batch size and augmentation ensemble
        grads_reshaped = grads.view(B, n_ens, C, H, W)

        self.noise_mean = grads_reshaped.mean(dim=1)
        if n_ens > 1:
            self.noise_std = grads_reshaped.std(dim=1) + 1e-8
        else:
            self.noise_std = torch.zeros_like(self.noise_mean)

    def sample(self, n_sample: int = 1, scale: float = 1.0) -> torch.Tensor:
        """Samples noise vectors from the estimated gradient Gaussian distribution."""
        if self.noise_mean is None or self.noise_std is None:
            raise RuntimeError("Must call collect_model() before sampling.")

        shape = (n_sample, *self.noise_mean.shape)
        eps = torch.randn(shape, dtype=self.noise_mean.dtype)
        sampled = self.noise_mean.unsqueeze(0) + scale * self.noise_std.unsqueeze(0) * eps
        return sampled



class ANDA(Attack):
    def __init__(
        self,
        model,
        eps=8 / 255,
        alpha=2 / 255,
        steps=10,
        n_ens=4,
        aug_max=0.3,
        sample=False,
    ):
        super().__init__("ANDA", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.n_ens = n_ens
        self.aug_max = aug_max
        self.sample = sample

        assert is_sqr(
            self.n_ens
        ), "n_ens must be a perfect square (e.g., 9, 16, 25)."
        self.grid_size = int(math.sqrt(self.n_ens))
        self.thetas = get_thetas(self.grid_size, -self.aug_max, self.aug_max)

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        B, C, H, W = images.shape

        min_x = torch.clamp(images - self.eps, min=0.0)
        max_x = torch.clamp(images + self.eps, max=1.0)

        xt = images.clone().detach()
        thetas_device = self.thetas.to(self.device)

        # Initialize ANDA gradient collector for input tensor dimensions
        anda = ANDA_collector(data_shape=(B, C, H, W))

        with torch.enable_grad():
            for i in range(self.steps):
                # Expand batch for transformation ensemble: [B, C, H, W] -> [B * M, C, H, W]
                xt_batch = (
                    xt.repeat_interleave(self.n_ens, dim=0)
                    .detach()
                    .requires_grad_(True)
                )

                # Tile transformation matrices for batch processing
                thetas_batch = thetas_device.repeat(B, 1) if hasattr(thetas_device, "repeat") else thetas_device

                aug_xt_batch = translation(thetas_batch, xt_batch)
                ys = labels.repeat_interleave(self.n_ens)

                outputs = self.get_logits(aug_xt_batch)
                if outputs.ndim == 1:
                    outputs = outputs.unsqueeze(0)

                loss = F.cross_entropy(outputs, ys, reduction="sum")
                loss.backward()

                new_grad = xt_batch.grad
                anda.collect_model(new_grad) # type: ignore
                sample_noise = anda.noise_mean

                if self.sample and i == self.steps - 1:
                    sample_noises = anda.sample(n_sample=1, scale=1)
                    sample_xt = (
                        xt + self.alpha * sample_noises.squeeze(0).sign()
                    )
                    sample_xt = torch.clamp(sample_xt, min=0.0, max=1.0)
                    xt = torch.max(torch.min(sample_xt, max_x), min_x).detach()
                else:
                    xt = xt + self.alpha * sample_noise.sign() # type: ignore
                    xt = torch.clamp(xt, min=0.0, max=1.0)
                    xt = torch.max(torch.min(xt, max_x), min_x).detach()

        return xt