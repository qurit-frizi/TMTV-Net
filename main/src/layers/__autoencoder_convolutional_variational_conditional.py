from typing import Union, List, Tuple

import torch.nn as nn
import torch
from .crop_or_pad import crop_or_pad_fun
from .autoencoder_convolutional_variational import AutoencoderConvolutionalVariational
from ..utils import flatten
import numpy as np


class AutoencoderConvolutionalVariationalConditional(nn.Module):
    """
    Conditional Variational convolutional auto-encoder implementation

    Most of the implementation is shared with regular variational convolutional auto-encoder.

    The main difference if the auto-encoder is conditioned on a variable ``y``. The model learns a
    latent given y. In this implementation, the encoder is not using ``y``, only the decoder is aware
    of it. This is done by concatenating the latent variable calculated by the encoder and ``y``.
    """
    def __init__(
            self,
            input_shape: Union[torch.Size, List[int], Tuple[int, ...]],
            encoder: nn.Module,
            decoder: nn.Module,
            z_size: int,
            y_size: int,
            input_type=torch.float32):
        """

        Args:
            input_shape: the shape, including the ``N`` and ``C`` components (e.g., [N, C, H, W...]) of
                the encoder
            encoder: the encoder taking ``x`` and returning an encoding
            decoder: taking an input of size [``y_size`` + ``z_size``] + [1] * ``cnn_dim`` and returning ``x``
            z_size: the size of the latent variable
            y_size: the size of the ``y`` variable
            input_type: the type of ``x`` variable
        """
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.decoder = decoder
        self.encoder = encoder
        self.z_size = z_size
        self.y_size = y_size

        # calculate the encoding size
        with torch.no_grad():
            encoding = encoder(torch.zeros(input_shape, dtype=input_type))
            # remove the N component, then multiply the rest
            self.encoder_output_size = np.asarray(encoding.shape[1:]).prod()
        self.cnn_dim = len(input_shape) - 2  # remove the N, C components

        # in the original paper (Kingma & Welling 2015, we
        # have a z_mean and z_var, but the problem is that
        # the z_var can be negative, which would cause issues
        # in the log later. Hence we assume that latent vector
        # has a z_mean and z_log_var component, and when we need
        # the regular variance or std_dev, we simply use
        # an exponential function
        self.z_logvar = nn.Linear(self.encoder_output_size, z_size)
        self.z_mu = nn.Linear(self.encoder_output_size, z_size)

    def encode(self, x):
        assert x.shape[1:] == self.input_shape[1:], f'expected shape=Nx{self.input_shape[1:]}, got=Nx{x.shape[1:]}'
        n = self.encoder(x)
        n = flatten(n)

        mu = self.z_mu(n)
        logvar = self.z_logvar(n)
        return mu, logvar

    def decode(self, mu, logvar, y, sample_parameters=None):
        if sample_parameters is None:
            sample_parameters = self.training
        z = AutoencoderConvolutionalVariational.reparameterize(sample_parameters, mu, logvar)
        assert y.shape[0] == mu.shape[0]
        assert y.shape[1] == self.y_size
        z_y = torch.cat((z, y), dim=1)

        shape = [z_y.shape[0], z_y.shape[1]] + [1] * self.cnn_dim
        nd_z_y = z_y.view(shape)
        recon = self.decoder(nd_z_y)
        return recon

    def sample_given_y(self, y):
        assert y.shape[1:] == (self.y_size,)
        random_z = torch.randn([len(y), self.z_size], dtype=torch.float32, device=y.device)

        random_encoding = torch.cat([random_z, y], dim=1)
        shape = [random_encoding.shape[0], random_encoding.shape[1]] + [1] * self.cnn_dim
        random_encoding = random_encoding.view(shape)
        random_recon = self.decoder(random_encoding)
        return random_recon

    def forward(self, x, y):
        mu, logvar = self.encode(x)
        recon = self.decode(mu, logvar, y)

        # make the recon exactly the same size!
        recon = crop_or_pad_fun(recon, x.shape[2:])
        assert recon.shape == x.shape, f'recon ({recon.shape}) and x ({x.shape}) must have the same shape.' \
                                       f'problem with the decoded!'
        return recon, mu, logvar
