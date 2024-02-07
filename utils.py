import argparse
import os

import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image

# Instantiate an ArgumentParser object
parser = argparse.ArgumentParser()

# Define command-line arguments
parser.add_argument("--n_epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for Adam optimizer")
parser.add_argument("--b1", type=float, default=0.5, help="Decay rate of first-order momentum of gradient for Adam")
parser.add_argument("--b2", type=float, default=0.999, help="Decay rate of second-order momentum of gradient for Adam")
parser.add_argument("--latent_dim", type=int, default=100, help="Dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="Number of classes in the dataset")
parser.add_argument("--img_size", type=int, default=32, help="Size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="Number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="Interval between image sampling during training")

# Parse the command-line arguments
opt = parser.parse_args()

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


os.makedirs("../../output/ac-gan-images", exist_ok=True)
os.makedirs("../../output/gan-images", exist_ok=True)


# Configure data loader
def create_dataloader():
    os.makedirs("../../data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )
    return dataloader


def create_sample_image(isACGan, generator, n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels) if isACGan else generator(z)
    prefix = "ac-gan" if isACGan else "gan"
    save_image(gen_imgs.data, "../../output/" + prefix + "-images/%d.png" % batches_done, nrow=n_row, normalize=True)
