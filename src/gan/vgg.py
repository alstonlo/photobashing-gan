import torch.nn as nn
import torchvision


class VGG19(nn.Module):

    def __init__(self):
        super().__init__()

        vgg_features = torchvision.models.vgg19(pretrained=True).features

        self.slices = nn.ModuleList([
            nn.Sequential(*[vgg_features[x] for x in range(2)]),
            nn.Sequential(*[vgg_features[x] for x in range(2, 7)]),
            nn.Sequential(*[vgg_features[x] for x in range(7, 12)]),
            nn.Sequential(*[vgg_features[x] for x in range(12, 21)]),
            nn.Sequential(*[vgg_features[x] for x in range(21, 30)]),
        ])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, image):
        h = image
        features = []
        for s in self.slices:
            h = s(h)
            features.append(h)
        return features
