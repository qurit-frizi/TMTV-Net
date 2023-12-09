import logging
import collections
import functools


import guided_back_propagation
import outputs
import utilities
from torch.nn import functional as F
from filter_gaussian import FilterGaussian
from layers.utils import upsample as upsample_fn


def to_value(v):
    """
    Convert where appropriate from tensors to numpy arrays

    Args:
        v: an object. If ``torch.Tensor``, the tensor will be converted to a numpy
            array. Else returns the original ``v``

    Returns:
        ``torch.Tensor`` as numpy arrays. Any other type will be left unchanged
    """
    if isinstance(v, torch.Tensor):
        return v.cpu().data.numpy()
    return v


def len_batch(batch):
    """

    Args:
        batch: a data split or a `collections.Sequence`

    Returns:
        the number of elements within a data split
    """
    if isinstance(batch, (collections.Sequence, torch.Tensor)):
        return len(batch)

    assert isinstance(batch, collections.Mapping), 'Must be a dict-like structure! got={}'.format(type(batch))

    for name, values in batch.items():
        if isinstance(values, (list, tuple)):
            return len(values)
        if isinstance(values, torch.Tensor) and len(values.shape) != 0:
            return values.shape[0]
        if isinstance(values, np.ndarray) and len(values.shape) != 0:
            return values.shape[0]
    return 0

from losses import total_variation_norm
import torch

logger = logging.getLogger(__name__)


def default_optimizer(params, nb_iters, learning_rate=0.5):
    """
    Create a default optimizer for :class:`trw.train.MeaningfulPerturbation`

    Args:
        params: the parameters to optimize
        nb_iters: the number of iterations
        learning_rate: the default learning rate

    Returns:
        a tuple (:class:`torch.optim.Optimizer`, :class:`torch.optim.lr_scheduler._LRScheduler`)
    """
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=nb_iters // 3, gamma=0.1)
    return optimizer, scheduler


def create_inputs(batch, modified_input_name, modified_input):
    """
    Create the model inputs depending on whether the input is a dictionary or tensor
    """
    if isinstance(batch, torch.Tensor):
        return modified_input
    elif isinstance(batch, collections.Mapping):
        new_batch = {}
        for feature_name, feature_value in batch.items():
            if feature_name == modified_input_name:
                new_batch[feature_name] = modified_input
            else:
                new_batch[feature_name] = feature_value
        return new_batch
    else:
        raise NotImplemented()


def default_information_removal_smoothing(image, blurring_sigma=5, blurring_kernel_size=23, explanation_for=None):
    """
    Default information removal (smoothing).

    Args:
        image: an image
        blurring_sigma: the sigma of the blurring kernel used to "remove" information from the image
        blurring_kernel_size: the size of the kernel to be used. This is an internal parameter to approximate the gaussian kernel. This is exposed since
            in 3D case, the memory consumption may be high and having a truthful gaussian blurring is not crucial.
        explanation_for: the class to explain

    Returns:
        a smoothed image
    """
    logger.info('default_perturbate_smoothed: default_perturbate_smoothed, blurring_sigma={}, blurring_kernel_size={}'.format(
        blurring_sigma, blurring_kernel_size
    ))
    gaussian_filter = FilterGaussian(input_channels=image.shape[1], sigma=blurring_sigma, nb_dims=len(image.shape) - 2, kernel_sizes=blurring_kernel_size, device=image.device)
    blurred_img = gaussian_filter(image)
    return blurred_img


class MeaningfulPerturbation:
    """
    Implementation of "Interpretable Explanations of Black Boxes by Meaningful Perturbation", arXiv:1704.03296

    Handle only 2D and 3D inputs. Other inputs will be discarded.

    Deviations:
    - use a global smoothed image to speed up the processing
    """
    def __init__(
            self,
            model,
            iterations=150,
            l1_coeff=0.1,
            tv_coeff=0.2,
            tv_beta=3,
            noise=0.2,
            model_output_postprocessing=functools.partial(F.softmax, dim=1),
            mask_reduction_factor=8,
            optimizer_fn=default_optimizer,
            information_removal_fn=default_information_removal_smoothing,
            export_fn=None,
    ):
        """

        Args:
            model: the model
            iterations: the number of iterations optimization
            l1_coeff: the strength of the penalization for the size of the mask
            tv_coeff: the strength of the penalization for "unnatural" artefacts
            tv_beta: exponentiation of the total variation
            noise: the amount of noise injected at each iteration.
            model_output_postprocessing: function to post-process the model output
            mask_reduction_factor: the size of the mask will be `mask_reduction_factor` times smaller than the input. This is used as regularization to
                remove "unnatural" artifacts
            optimizer_fn: how to create the optimizer
            export_fn: if not None, a function taking (iter, perturbated_input_name, perturbated_input, mask) will be called each iteration (e.g., for debug purposes)
            information_removal_fn: information removal function (e.g., the paper originally used a smoothed image to remove objects)
        """
        self.model = model
        self.iterations = iterations
        self.l1_coeff = l1_coeff
        self.tv_coeff = tv_coeff
        self.tv_beta = tv_beta
        self.noise = noise
        self.model_output_postprocessing = model_output_postprocessing
        self.mask_reduction_factor = mask_reduction_factor
        self.optimizer_fn = optimizer_fn
        self.information_removal_fn = information_removal_fn
        self.export_fn = export_fn

    def __call__(self, inputs, target_class_name, target_class=None):
        """

        Args:
            inputs: a tensor or dictionary of tensors. Must have `require_grads` for the inputs to be explained
            target_class: the index of the class to explain the decision. If `None`, the class output will be used
            target_class_name: the output node to be used. If `None`:
                * if model output is a single tensor then use this as target output

                * else it will use the first `OutputClassification` output

        Returns:
            a tuple (output_name, dictionary (input, explanation mask))
        """
        logger.info('started MeaningfulPerturbation ...')
        logger.info('parameters: iterations={}, l1_coeff={}, tv_coeff={}, tv_beta={}, noise={}, mask_reduction_factor={}'.format(
            self.iterations, self.l1_coeff, self.tv_beta, self.tv_beta, self.noise, self.mask_reduction_factor
        ))
        self.model.eval()  # make sure we are in eval mode

        inputs_with_gradient = dict(guided_back_propagation.GuidedBackprop.get_floating_inputs_with_gradients(inputs))
        if len(inputs_with_gradient) == 0:
            logger.error('MeaningfulPerturbation.__call__: failed. No inputs will collect gradient!')
            return None
        else:
            logger.info('MeaningfulPerturbation={}'.format(inputs_with_gradient.keys()))

        outputs = self.model(inputs)
        if target_class_name is None and isinstance(outputs, collections.Mapping):
            for output_name, output in outputs.items():
                if isinstance(output, outputs.OutputClassification):
                    logger.info(f'output found={output_name}')
                    target_class_name = output_name
                    break
        output = MeaningfulPerturbation._get_output(target_class_name, outputs, self.model_output_postprocessing)
        logger.info(f'original model output={output}')
        output_start = to_value(output)

        if target_class is None:
            target_class = torch.argmax(output, dim=1)
            logger.info(f'target_class by sample={target_class}, value={output[:, target_class]}')
        else:
            logger.info(f'target class={target_class}')

        # construct our gradient target
        model_device = utilities.get_device(self.model, batch=inputs)
        nb_samples = len_batch(inputs)
        assert nb_samples == 1, 'only a single sample is handled at once! (TODO fix this!)'

        masks_by_feature = {}
        for input_name, input_value in inputs_with_gradient.items():
            img = input_value.detach()  # do not keep the gradient! This will be recorded by Callback_explain_decision
            if len(img.shape) != 4 and len(img.shape) != 5:
                # must be (Sample, channel, Y, X) or (Sample, channel, Z, Y, X) input
                logging.info(f'input={input_name} was discarded as the shape={img.shape} do not match (Sample, '
                             f'channel, Y, X) or (Sample, channel, Z, Y, X)')
                continue

            logger.info('processing feature_name={}'.format(input_name))

            # must have a SINGLE channel and decreased dim for all others
            mask_shape = [nb_samples, 1] + [d // self.mask_reduction_factor for d in img.shape[2:]]
            logging.info('mask size={}'.format(mask_shape))

            # do not follow the paper (non-masked = 1) so that we can use tricks on the mask
            # optimization (e.g., weight decay)
            mask = torch.zeros(mask_shape, requires_grad=True, dtype=torch.float32, device=model_device)

            # https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351
            blurred_img = self.information_removal_fn(img, explanation_for=target_class)
            optimizer, scheduler = self.optimizer_fn([mask], self.iterations)

            assert self.iterations > 0
            c_start = 0.0
            upsampled_mask = None
            perturbated_input = None
            c = 1e1000
            for i in range(self.iterations):
                optimizer.zero_grad()

                upsampled_mask = upsample_fn(mask, img.shape[2:])
                # make the same number of channels for the mask as there is in the image
                upsampled_mask = upsampled_mask.expand([upsampled_mask.shape[0], img.shape[1]] + list(img.shape[2:]))

                # Use the mask to perturbate the input image.
                perturbated_input = img.mul(1 - upsampled_mask) + blurred_img.mul(upsampled_mask)

                noise = torch.zeros(img.shape, device=model_device)
                if self.noise > 0:
                    noise.normal_(mean=0, std=self.noise)

                perturbated_input = perturbated_input + noise
                batch = create_inputs(inputs, input_name, perturbated_input)
                outputs = self.model(batch)

                output = MeaningfulPerturbation._get_output(target_class_name, outputs, self.model_output_postprocessing)
                assert len(output.shape) == 2, 'expected a `N * nb_classes` output'

                # mask = 1, we keep original image. Mask = 0, replace with blurred image
                # we want to minimize the number of mak voxels with 0 (l1 loss), keep the
                # mask smooth (tv + mask upsampling), finally, we want to decrease the
                # probability of `target_class`
                l1 = self.l1_coeff * torch.mean(torch.abs(mask))
                tv = self.tv_coeff * total_variation_norm(mask, self.tv_beta)
                c = output[:, target_class]

                loss = l1 + tv + c

                if i == 0:
                    # must be collected BEFORE backward!
                    c_start = to_value(c)

                loss.backward()
                optimizer.step()
                scheduler.step()

                # Optional: clamping seems to give better results
                mask.data.clamp_(0, 1)

                if i % 20 == 0:
                    logger.info('iter={}, total_loss={}, l1_loss={}, tv_loss={}, c_loss={}'.format(
                        i,
                        to_value(loss),
                        to_value(l1),
                        to_value(tv),
                        to_value(c),
                    ))

                    if self.export_fn is not None:
                        self.export_fn(i, input_name, perturbated_input, upsampled_mask)

            logger.info('class loss start={}, end={}'.format(c_start, to_value(c)))
            logger.info('final output={}'.format(to_value(output)))

            masks_by_feature[input_name] = {
                'mask': to_value(upsampled_mask),
                'perturbated_input': to_value(perturbated_input),
                'smoothed_input': to_value(blurred_img),
                'loss_c_start': c_start,
                'loss_c_end': to_value(c),
                'output_start': output_start,
                'output_end': to_value(output),
            }

        return target_class_name, masks_by_feature

    @staticmethod
    def _get_output(target_class_name, outputs, postprocessing):
        if target_class_name is not None:
            output = outputs[target_class_name]
            if isinstance(output, outputs.Output):
                output = output.output
        else:
            output = outputs

        assert isinstance(output, torch.Tensor)
        output = postprocessing(output)
        return output
