import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from src.gan.spade import SPADEResnetBlock


class Generator(nn.Module):

    # noinspection PyTypeChecker
    def __init__(self, latent_dim, base_channels):
        super().__init__()

        nc = base_channels

        self.lin1 = nn.Linear(latent_dim, 4 * 4 * nc * 16)

        self.blocks = nn.ModuleList([
            SPADEResnetBlock(16 * nc + 3, 16 * nc),
            SPADEResnetBlock(16 * nc + 3, 16 * nc),
            SPADEResnetBlock(16 * nc + 3, 8 * nc),
            SPADEResnetBlock(8 * nc + 3, 8 * nc),
            SPADEResnetBlock(8 * nc + 3, 4 * nc),
            SPADEResnetBlock(4 * nc + 3, 2 * nc),
            SPADEResnetBlock(2 * nc + 3, 1 * nc),
        ])

        self.cnn_out = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(nc, 3, 3, padding="same"),
            nn.Tanh(),
        )

    def forward(self, noise, cmaps):
        noise = self.lin1(noise).reshape(noise.shape[0], -1, 4, 4)
        pyramid = self._build_gaussian_pyramid(cmaps)

        image = noise
        for block, sub_cmap in zip(self.blocks, reversed(pyramid)):
            image = torch.concat([image, sub_cmap], dim=1)
            image = block((image, sub_cmap))
            if image.shape[-1] < cmaps.shape[-1]:
                image = F.interpolate(image, scale_factor=2, mode="bilinear")
        return self.cnn_out(image)

    # noinspection PyTypeChecker
    def _build_gaussian_pyramid(self, image):
        pyramid = [image]
        while image.shape[-1] > 4:
            image = TF.gaussian_blur(image, kernel_size=5, sigma=1.0)
            image = image[:, :, ::2, ::2]
            pyramid.append(image)
        return pyramid
