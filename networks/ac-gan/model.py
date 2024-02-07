import torch.nn as nn
import torch

from utils import opt


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Embedding layer for label information
        self.label_embedding = nn.Embedding(opt.n_classes, opt.latent_dim)

        # Initial size before upsampling
        self.initial_size = opt.img_size // 4

        # Linear layer followed by upsampling and convolutional blocks
        self.linear_layer = nn.Sequential(
            nn.Linear(opt.latent_dim, 128 * self.initial_size ** 2)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    ''' Describe how the output of the model is calculated '''

    def forward(self, noise, labels):
        # Combine noise with label information
        embedded_labels = torch.mul(self.label_embedding(labels), noise)

        # Pass through linear layer
        intermediate_output = self.linear_layer(embedded_labels)

        # Reshape output
        reshaped_output = intermediate_output.view(intermediate_output.size(0), 128, self.initial_size,
                                                   self.initial_size)

        # Pass through convolutional blocks
        generated_image = self.conv_blocks(reshaped_output)

        return generated_image


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def create_discriminator_block(in_channels, out_channels, is_block=True):
            """Creates layers for each discriminator block"""
            block = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)]
            if is_block:
                block.append(nn.BatchNorm2d(out_channels, momentum=0.8))
            return block

        # Sequential layers for convolutional blocks
        self.conv_blocks = nn.Sequential(
            *create_discriminator_block(opt.channels, 16, is_block=False),
            *create_discriminator_block(16, 32),
            *create_discriminator_block(32, 64),
            *create_discriminator_block(64, 128),
        )

        # Height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1),
            nn.Sigmoid()
        )
        self.aux_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, opt.n_classes),
            nn.Softmax(dim=1)
        )

    ''' Describe how the output of the model is calculated '''

    def forward(self, img):
        # Pass image through convolutional blocks
        out = self.conv_blocks(img)

        # Flatten the output
        out = out.view(out.shape[0], -1)

        # Pass through output layers
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label
