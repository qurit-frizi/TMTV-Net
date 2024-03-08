import collections

import torch.nn as nn
import torch
import numpy as np


from ..train import get_device
from ..train.collate import collate_list_of_dicts
from ..train.utilities import prepare_loss_terms
from ..utils import len_batch, get_batch_n


def process_outputs_and_extract_loss(outputs, batch, is_training):
    loss_terms = prepare_loss_terms(outputs, batch, is_training)
    sum_losses = 0.0
    for name, loss_term in loss_terms.items():
        loss = loss_term.get('loss')
        if loss is not None:
            # if the loss term doesn't contain a `loss` attribute, it means
            # this is not used during optimization (e.g., embedding output)
            sum_losses += loss
    return sum_losses


class GanDataPool:
    def __init__(self, pool_size, replacement_probability=0.5, insertion_probability=0.1):
        self.pool_size = pool_size
        self.generated_images = []
        self.batches = []
        self.replacement_probability = replacement_probability
        self.insertion_probability = insertion_probability

    def get_data(self, batch, images_fake):
        return_images = []
        return_batch = []
        nb_images = len(images_fake)
        assert len_batch(batch) == nb_images, 'expected same number of images and batch samples!'

        for i in range(nb_images):
            if len(self.generated_images) < self.pool_size:
                # insert the image into the pool
                batch_sample = get_batch_n(batch, nb_images, np.asarray([i]), None, True)
                image = images_fake[i].unsqueeze(0)

                self.generated_images.append(image)
                self.batches.append(batch_sample)
                return_images.append(image)
                return_batch.append(batch_sample)
            else:
                p = np.random.uniform(0, 1)
                if p < self.insertion_probability:
                    # replace the data
                    random_id = np.random.randint(0, self.pool_size - 1)
                    self.generated_images[random_id] = images_fake[i].unsqueeze(0)
                    self.batches[random_id] = get_batch_n(batch, nb_images, np.asarray([i]), None, True)

                p = np.random.uniform(0, 1)
                if p < self.replacement_probability:
                    random_id = np.random.randint(0, self.pool_size - 1)
                    return_images.append(self.generated_images[random_id])
                    return_batch.append(self.batches[random_id])
                else:
                    return_images.append(images_fake[i].unsqueeze(0))
                    return_batch.append(get_batch_n(batch, nb_images, np.asarray([i]), None, True))

        # collate the results to be similar to inputs
        return_images = torch.cat(return_images, dim=0)
        return_batch = collate_list_of_dicts(return_batch, device=images_fake.device)
        return return_batch, return_images


class Gan(nn.Module):
    """
    Generic GAN implementation. Support conditional GANs.

    Examples:
        - generator conditioned by concatenating a one-hot attribute to the latent or conditioned
            by another image (e.g., using UNet)
        - discriminator conditioned by concatenating a one-hot image sized to the image
            or one-hot concatenated to intermediate layer
        - simple GAN (i.e., no observation)

    Notes:
        Here the module will have its own optimizer. The :class:`trw.train.Trainer` should have ``optimizers_fn``
        set to ``None``.
    """

    def __init__(
            self,
            discriminator,
            generator,
            latent_size,
            optimizer_discriminator_fn,
            optimizer_generator_fn,
            real_image_from_batch_fn,
            train_split_name='train',
            loss_from_outputs_fn=process_outputs_and_extract_loss,
            image_pool=None,
    ):
        """

        Args:
            discriminator: a discriminator taking input ``image_from_batch_fn(batch)`` and
                returning Nx2 output (without the activation function applied)
            generator: a generator taking as input [N, latent_size, [1] * dim], with dim=2 for 2D images
                and returning as output the same shape and type as ``image_from_batch_fn(batch)``
            latent_size: the latent size (random vector to seed the generator), `None` for no random
                latent (e.g., pix2pix)
            optimizer_discriminator_fn: the optimizer function to be used for the discriminator. Takes
                as input a model and return the trainable parameters
            optimizer_generator_fn: the optimizer function to be used for the generator. Takes
                as input a model and return the trainable parameters
            real_image_from_batch_fn: a function to extract the relevant image from the batch. Takes as input
                a batch and return an image
            train_split_name: only this split will be used for the training
            image_pool: if not `None`, use a history of generated images to train the discriminator
        """
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator

        self.optmizer_discriminator = optimizer_discriminator_fn(params=self.discriminator.parameters())
        self.scheduler_discriminator = None
        if isinstance(self.optmizer_discriminator, tuple):
            assert len(self.optmizer_discriminator) == 2
            self.optmizer_discriminator, self.scheduler_discriminator = self.optmizer_discriminator

        self.optmizer_generator = optimizer_generator_fn(params=self.generator.parameters())
        self.scheduler_generator = None
        if isinstance(self.optmizer_generator, tuple):
            assert len(self.optmizer_generator) == 2
            self.optmizer_generator, self.scheduler_generator = self.optmizer_generator

        self.latent_size = latent_size
        self.train_split_name = train_split_name
        self.real_image_from_batch_fn = real_image_from_batch_fn
        self.loss_from_outputs_fn = loss_from_outputs_fn
        self.image_pool = image_pool

    def _generate_latent(self, nb_samples):
        device = get_device(self)
        if self.latent_size is not None:
            with torch.no_grad():
                z = torch.randn(nb_samples, self.latent_size, device=device)
        else:
            z = None

        return z

    @staticmethod
    def _merge_generator_discriminator_outputs(
            generator_outputs,
            discriminator_real_outputs,
            discriminator_fake_outputs):

        output_prefix = [
            ('gen_', generator_outputs),
            ('real_', discriminator_real_outputs),
            ('fake_', discriminator_fake_outputs)
        ]

        all_outputs = []
        for prefix, outputs in output_prefix:
            if outputs is None:
                continue
            for output_name, output in outputs.items():
                all_outputs.append((prefix + output_name, output))

        return collections.OrderedDict(all_outputs)

    def forward(self, batch):
        if 'split_name' not in batch:
            # MUST have the split name!
            return {}

        # generate a fake image
        nb_samples = len_batch(batch)
        latent = self._generate_latent(nb_samples)

        o = self.generator(batch, latent)
        assert isinstance(o, tuple) and len(o) == 2, 'must return a tuple (torch.Tensor, dict of outputs)'
        images_fake_orig, generator_outputs = o

        if batch['split_name'] != self.train_split_name or not self.training:
            # we are in valid/test mode, return only the generated image!
            all_outputs = self._merge_generator_discriminator_outputs(
                generator_outputs,
                None,
                None)
            return all_outputs

        images_real = self.real_image_from_batch_fn(batch)
        assert type(images_real) == type(images_fake_orig) or \
            (isinstance(images_fake_orig, list) and type(images_fake_orig[0]) == type(images_real)), \
            'return must be of the same type!'

        if isinstance(images_fake_orig, collections.Sequence) and not isinstance(images_fake_orig, torch.Tensor):
            assert isinstance(images_fake_orig[0], torch.Tensor), 'if list, elements must be tensors!'

        #
        # train discriminator
        #

        # discriminator: train with all fakes
        batch_fake = batch
        if self.image_pool is not None and batch.get('dataset_name') is not None:
            # `dataset_name` is None, this means we are not in the real training loop yet
            # keep a history of generated images and pass them to the classifier to
            # remember previous iterations
            batch_fake, images_fake = self.image_pool.get_data(batch_fake, images_fake_orig)
        else:
            images_fake = images_fake_orig

        if isinstance(images_fake, list):
            fake = [i.detach() for i in images_fake]
        else:
            fake = images_fake.detach()
        discriminator_outputs_fake = self.discriminator(batch_fake, fake, is_real=False)

        # discriminator: train with all real
        self.optmizer_discriminator.zero_grad()
        batch_real = batch
        discriminator_outputs_real = self.discriminator(batch_real, images_real, is_real=True)

        # extend the real, fake label to the size of the discriminator output (e.g., PatchGan)
        discriminator_loss_fake = self.loss_from_outputs_fn(discriminator_outputs_fake, batch_fake, is_training=True)
        discriminator_loss_real = self.loss_from_outputs_fn(discriminator_outputs_real, batch_real, is_training=True)
        discriminator_loss = (discriminator_loss_fake + discriminator_loss_real) / 2

        if discriminator_loss.requires_grad:
            discriminator_loss.backward()
            self.optmizer_discriminator.step()

        #
        # train generator
        #
        self.optmizer_generator.zero_grad()
        discrimator_outputs_generator = self.discriminator(batch, images_fake_orig, is_real=True)
        discrimator_outputs_generator_and_generator_outputs = {**discrimator_outputs_generator, **generator_outputs}
        generator_loss = self.loss_from_outputs_fn(discrimator_outputs_generator_and_generator_outputs, batch, is_training=True)

        #
        # update generator and discriminator parameters all at once
        #

        if generator_loss.requires_grad:
            generator_loss.backward()
            self.optmizer_generator.step()

        if self.training and batch.get('batch_id') == 0:
            # the first batch of each epoch should change the
            # learning rate
            if self.scheduler_generator is not None:
                self.scheduler_generator.step()
            if self.scheduler_discriminator is not None:
                self.scheduler_discriminator.step()

        all_outputs = self._merge_generator_discriminator_outputs(
            generator_outputs,
            discriminator_outputs_real,
            discriminator_outputs_fake)
        return all_outputs
