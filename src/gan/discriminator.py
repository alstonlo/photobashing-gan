import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class MultiScaleDiscriminator(nn.Module):

    def __init__(self, base_channels):
        super().__init__()

        self.n_subunits = 2
        self.fine_disc = Discriminator(base_channels)
        self.coarse_disc = Discriminator(base_channels)

    def downsample(self, image):
        out_size = image.shape[-1] // 2
        return F.interpolate(image, out_size, mode="nearest")

    def forward(self, images, cmaps):
        return [
            self.fine_disc(images, cmaps),
            self.coarse_disc(self.downsample(images), self.downsample(cmaps))
        ]


class Discriminator(nn.Module):

    # noinspection PyTypeChecker
    def __init__(self, base_channels):
        super().__init__()

        nc = base_channels
        alpha = 0.2

        blocks = [
            nn.Sequential(
                nn.Conv2d(6, nc, 4, stride=2, padding=1),
                nn.LeakyReLU(alpha)
            ),
            nn.Sequential(
                spectral_norm(nn.Conv2d(nc, nc * 2, 4, stride=2, padding=1, bias=False)),
                nn.InstanceNorm2d(nc * 2), nn.LeakyReLU(alpha)
            ),
            nn.Sequential(
                spectral_norm(nn.Conv2d(nc * 2, nc * 4, 4, stride=2, padding=1, bias=False)),
                nn.InstanceNorm2d(nc * 4), nn.LeakyReLU(alpha)
            ),
            nn.Sequential(
                spectral_norm(nn.Conv2d(nc * 4, nc * 8, 5, padding="same", bias=False)),
                nn.InstanceNorm2d(nc * 8), nn.LeakyReLU(alpha)
            ),
            nn.Conv2d(nc * 8, 1, 5, padding="same")
        ]

        self.blocks = nn.ModuleList(blocks)

    def forward(self, images, cmaps):
        x = torch.concat([images, cmaps], dim=1)
        features = [x]
        for module in self.blocks:
            features.append(module(features[-1]))
        return features
