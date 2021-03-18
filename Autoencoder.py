import torch
import torch.nn as nn
import torch.functional as F
import math
from functools import reduce

class Shrink(nn.Module):
    def __init__(self, in_channels, out_channels, show_shape = False):
        super(Shrink, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            ShapePrinter(False)
        )
        pass

    def forward(self, x):
        return self.main(x)

class Encoder(nn.Module):
    def __init__(self, ndf, nc, nz, img_size, show_shape = False):
        super(Encoder, self).__init__()
        
        # 4 steps
        # 64 => 4
        # 32 => 2
        # 16 => 1
        # step up from 1 (ndf -> number of features)

        steps = []

        # add initial layer
        steps.append(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        steps.append(nn.LeakyReLU(0.2, inplace=True))
        steps.append(ShapePrinter(show_shape))

        max_step = 3
        curr_step = 0
        while curr_step < max_step:
            in_channels = int(ndf * 2**curr_step)
            curr_step += 1
            out_channels = int(ndf * 2**curr_step)
            steps.append(Shrink(in_channels, out_channels, show_shape))

        r = img_size // (2**(max_step+1))
        
        steps.append(nn.Flatten(1))
        steps.append(nn.Linear(ndf*(2**curr_step)*r*r, nz))

        self.main = nn.Sequential(*steps)

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, ndf, nc, nz, img_size, show_shape = False):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            Encoder(ndf, nc, nz, img_size, show_shape),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

class ShapePrinter(nn.Module):
    def __init__(self, verbose = False):
        super(ShapePrinter, self).__init__()
        self.verbose = verbose

    def forward(self, x):
        if self.verbose:
            print("SHAPE:", x.shape)
        return x

class DoubleSize(nn.Module):
    def __init__(self, in_channels, out_channels, kernel = 4, stride = 2, padding = 1):
        super(DoubleSize, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.main(x)


class Decoder(nn.Module):
    def __init__(self, ngf, nc, nz, img_size, is_output = True):
        super(Decoder, self).__init__()

        steps = []

        mult = img_size // 8

        steps.append(nn.Unflatten(1, (nz, 1, 1)))
        steps.append(ShapePrinter())
        steps.append(DoubleSize(nz, ngf * mult, 4, 1, 0))
        steps.append(ShapePrinter())
        
        while mult > 1:
            prev = mult
            mult = mult // 2
            steps.append(DoubleSize(ngf * prev, ngf * mult))
            steps.append(ShapePrinter())

        if is_output:
            steps.append(nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False))
            steps.append(nn.Tanh())
        else:
            steps.append(DoubleSize(ngf, nc, 4, 2, 1))
            
        steps.append(ShapePrinter())

        self.main = nn.Sequential(*steps)

    def forward(self, x):
        return self.main(x)

class Smoother(nn.Module):
    def __init__(self):
        super(Smoother, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 3, 5, padding=2),
            nn.BatchNorm2d(3),
            nn.ReLU(True),

            nn.Conv2d(3, 3, 3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(True),

            nn.Conv2d(3, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class SmoothAutoencoder(nn.Module):
    def __init__(self, nc, nz, ngf, img_size, chunk_size = 64):
        super(SmoothAutoencoder, self).__init__()
        self.autoencoder = Autoencoder(nc, nz, ngf, chunk_size, False)
        self.smoother = Smoother()

        self.chunks = img_size // chunk_size

    def forward(self, x):
        values = list(map(lambda y: list(map(lambda seg: self.autoencoder(seg), torch.chunk(y, self.chunks, 3))), torch.chunk(x, self.chunks, 2)))
        temp = list(map(lambda t: reduce(lambda x_1, x_2: torch.cat((x_1, x_2), 3), t[:]), values))
        x = reduce(lambda x_1, x_2: torch.cat((x_1, x_2), 2), temp[:])
        x = self.smoother(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, nc, nz, ngf, img_size = 64, is_output = True):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(ngf, nc, nz, img_size)
        self.decoder = Decoder(ngf, nc, nz, img_size, is_output)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
