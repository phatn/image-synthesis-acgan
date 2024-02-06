import numpy as np
import torch.nn as nn

from utils import opt

img_shape = (opt.channels, opt.img_size, opt.img_size)


class GANGenerator(nn.Module):
    def __init__(self):
        super(GANGenerator, self).__init__()

        def create_generator_block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.conv_blocks = nn.Sequential(
            *create_generator_block(opt.latent_dim, 128, normalize=False),
            *create_generator_block(128, 256),
            *create_generator_block(256, 512),
            *create_generator_block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    ''' Describe how the output of the model is calculated '''

    def forward(self, z):
        img = self.conv_blocks(z)
        img = img.view(img.size(0), *img_shape)
        return img


class GANDiscriminator(nn.Module):
    def __init__(self):
        super(GANDiscriminator, self).__init__()

        self.conv_blocks = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.conv_blocks(img_flat)

        return validity
