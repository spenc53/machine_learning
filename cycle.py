
# General
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import torch.optim as optim
import itertools

# Importing data
import torchvision.datasets as dset
from torchvision import datasets, transforms

# Plotting
import matplotlib.pyplot as plt;
import torchvision.utils as vutils

# NN
from Autoencoder import Autoencoder, SmoothAutoencoder, Discriminator # this will be used as our generator
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Try to generate a larger image
# Use regular Autoencoder to generate the image in 1 shot
image_size = 64
chunk_size = 64
# generator_g = SmoothAutoencoder(3, 1000, 64, image_size, chunk_size).to(device)
# generator_f = SmoothAutoencoder(3, 1000, 64, image_size, chunk_size).to(device)
generator_g = Autoencoder(3, 1000, 64, image_size).to(device)
generator_f = Autoencoder(3, 1000, 64, image_size).to(device)

discriminator_x = Discriminator(64, 3, 1, image_size).to(device)
discriminator_y = Discriminator(64, 3, 1, image_size).to(device)

# IMPORT DATA
batch_size = 128
monet_data = dset.ImageFolder(root='../ae/monet',
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
monet_dataloader = torch.utils.data.DataLoader(monet_data, batch_size=batch_size,
                                         shuffle=True)

real_data = dset.ImageFolder(root='../ae/real',
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
real_dataloader = torch.utils.data.DataLoader(real_data, batch_size=batch_size,
                                         shuffle=True)
# get the data to train with
# need both of them
show_training_images = False
if show_training_images:
    real_batch = next(iter(monet_dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()


# Setup Adam optimizers for both generators and discriminators
lr = 0.0005
beta1 = 0.5 
optimizerG = optim.Adam(itertools.chain(generator_g.parameters(), generator_f.parameters()), lr=lr, betas=(beta1, 0.999))
# optimizerGF = optim.Adam(generator_f.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerDX = optim.Adam(discriminator_x.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerDY = optim.Adam(discriminator_y.parameters(), lr=lr, betas=(beta1, 0.999))


dataloader_iterator_monet = iter(monet_dataloader)
def get_monet_data():
    global dataloader_iterator_monet
    try:
        data, target = next(dataloader_iterator_monet)
    except StopIteration:
        dataloader_iterator_monet = iter(monet_dataloader)
        data, target = next(dataloader_iterator_monet)
    return data

dataloader_iterator_real = iter(real_dataloader)
def get_real_data():
    global dataloader_iterator_real
    try:
        data, target = next(dataloader_iterator_real)
    except StopIteration:
        dataloader_iterator_real = iter(real_dataloader)
        data, target = next(dataloader_iterator_real)
    return data

LAMBDA = 10
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

def train(real_x, real_y, epochs = 20):

    # g(x) => produces fake y
    # g(y) => real y
    # f(y) => produces fake x
    # f(x) => real x

    optimizerG.zero_grad()

    # Identity loss
    loss_identity_y = criterion_identity(generator_g(real_y), real_y)
    loss_identity_x = criterion_identity(generator_f(real_x), real_x)

    loss_identity = (loss_identity_x + loss_identity_y) / 2

    # GAN loss
    fake_y = generator_g(real_x).detach()
    disc_fake_x = discriminator_y(fake_y)
    loss_GAN_G = criterion_GAN(disc_fake_x, torch.ones_like(disc_fake_x))
    fake_x = generator_f(real_y).detach()
    disc_fake_y = discriminator_x(fake_x)
    loss_GAN_F = criterion_GAN(disc_fake_y, torch.ones_like(disc_fake_y))

    loss_GAN = (loss_GAN_G + loss_GAN_F) / 2

    # cycle loss
    recov_x = generator_f(fake_y)
    loss_cycle_x = criterion_cycle(recov_x, real_x)
    recov_y = generator_g(fake_x)
    loss_cycle_y = criterion_cycle(recov_y, real_y)

    loss_cycle = (loss_cycle_x + loss_cycle_y) / 2

    # Total Loss
    loss_G = loss_GAN +  LAMBDA * loss_cycle + LAMBDA * loss_identity

    loss_G.backward()
    optimizerG.step()

    ### discriminator A
    optimizerDX.zero_grad()

    # REAL LOSS
    discrim_x = discriminator_x(real_x)
    loss_real = criterion_GAN(discrim_x, torch.ones_like(discrim_x))
    discrim_fake_x = discriminator_x(fake_x)
    loss_fake = criterion_GAN(discrim_fake_x, torch.zeros_like(discrim_fake_x))

    loss_discriminator_x = (loss_real + loss_fake) / 2

    loss_discriminator_x.backward()
    optimizerDX.step()

    ### discriminator B
    optimizerDY.zero_grad()

    # REAL LOSS
    discrim_y = discriminator_y(real_y)
    loss_real_y = criterion_GAN(discrim_y, torch.ones_like(discrim_y))
    discrim_fake_y = discriminator_y(fake_y)
    loss_fake_y = criterion_GAN(discrim_fake_y, torch.zeros_like(discrim_fake_y))

    loss_discriminator_y = (loss_real_y + loss_fake_y) / 2

    loss_discriminator_y.backward()
    optimizerDY.step()

def show_image(autoencoder, data):
    for i, (x, y) in enumerate(data):
        z = autoencoder(x.to(device))
        z = z.to('cpu').detach()
        
        plt.imshow(z[0].permute(1,2,0))
        plt.show()
        break

i = 0
while True:
    try:
        real_x = get_monet_data().to(device)
        real_y = get_real_data().to(device)

        train(real_x, real_y)
        print(i)
        i += 1
    except KeyboardInterrupt:
        c = input("show images? ")
        if c.lower() == 'y':
            show_image(generator_g, monet_dataloader)
            show_image(generator_g, monet_dataloader)

            show_image(generator_f, real_dataloader)
            show_image(generator_f, real_dataloader)

        c = input("continue? (y/n) ")
        if c.lower() == "n":
            break

show_image(generator_g, monet_dataloader)
show_image(generator_g, real_dataloader)
