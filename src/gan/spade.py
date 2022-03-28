import torch.nn as nn
from torch.nn.utils import spectral_norm


# Reference:
#   https://github.com/NVlabs/SPADE


class SPADE(nn.Module):

    # noinspection PyTypeChecker
    def __init__(self, norm_channels):
        super().__init__()

        self.norm_nc = norm_channels
        self.norm = nn.InstanceNorm2d(norm_channels, affine=False)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 2 * norm_channels, kernel_size=3, padding="same")
        )

    def forward(self, x):
        images, cmaps = x

        # Part 1. generate parameter-free normalized activations
        normalized = self.norm(images)

        # Part 2. produce scaling and bias conditioned on semantic map
        cmap_embed = self.cnn(cmaps)
        gamma = cmap_embed[:, :self.norm_nc, ...]
        beta = cmap_embed[:, self.norm_nc:, ...]

        # apply scale and bias
        out = normalized * (1 + gamma) + beta
        return out


class SPADEResnetBlock(nn.Module):

    # noinspection PyTypeChecker
    def __init__(self, in_channels, out_channels):
        super().__init__()

        nc_in, nc_out = in_channels, out_channels
        nc_mid = min(nc_in, nc_out)
        alpha = 0.2

        self.blocks = nn.ModuleList([
            nn.Sequential(
                SPADE(nc_in),
                nn.LeakyReLU(alpha),
                spectral_norm(nn.Conv2d(nc_in, nc_mid, kernel_size=3, padding="same"))
            ),
            nn.Sequential(
                SPADE(nc_mid),
                nn.LeakyReLU(alpha),
                spectral_norm(nn.Conv2d(nc_mid, nc_out, kernel_size=3, padding="same"))
            )
        ])

        if nc_in != nc_out:
            self.shortcut = nn.Sequential(
                SPADE(nc_in),
                nn.LeakyReLU(alpha),
                spectral_norm(nn.Conv2d(nc_in, nc_out, kernel_size=1, bias=False))
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        images, cmaps = x
        images_shortcut = self.shortcut((images, cmaps))
        for block in self.blocks:
            images = block((images, cmaps))
        return images + images_shortcut
