from typing import Any, Callable, Optional, Tuple, Mapping

import torch
import logging



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


import outputs
import utilities
import guided_back_propagation
import numpy as np
from torch import nn


logger = logging.getLogger(__name__)


def is_feature_metadata(name: str, value: Any) -> bool:
    """
    Return True is a feature name/value should be considered as metadata
    """

    if not isinstance(value, (torch.Tensor, np.ndarray)):
        return True

    return False


class IntegratedGradients:
    """
    Implementation of `Integrated gradients`, a method of attributing the prediction of a deep network
        to its input features.

    This is implementing the paper `Axiomatic Attribution for Deep Networks`,
    Mukund Sundararajan, Ankur Taly, Qiqi Yan
    as described in https://arxiv.org/abs/1703.01365
    """
    def __init__(self,
                 model: nn.Module,
                 steps: int = 100,
                 baseline_inputs: Any = None,
                 use_output_as_target: bool = False,
                 post_process_output: Callable[[Any], torch.Tensor] = guided_back_propagation.post_process_output_id):
        """

        Args:
            model: the model
            steps: the number of intermediate steps to perform the gradient integration
            baseline_inputs: this will be used as input baseline. This should be an input such that `model(reference_inputs)`
                is close to 0 (e.g. black image for CNNs). If `None`, return inputs filled with zeros
            post_process_output: a function to post-process the output of a model so that it is suitable for gradient attribution
        """
        self.baseline_inputs = baseline_inputs
        self.model = model
        self.steps = steps
        self.use_output_as_target = use_output_as_target
        self.post_process_output = post_process_output

    def __call__(self, inputs: Any, target_class_name: str, target_class: Optional[int] = None) \
            -> Optional[Tuple[str, Mapping]]:
        """
            Generate the guided back-propagation gradient

            Args:
                inputs: a tensor or dictionary of tensors. Must have `require_grads` for the inputs to be explained
                target_class: the index of the class to explain the decision. If `None`, the class output will be used
                target_class_name: the output node to be used. If `None`:
                    * if model output is a single tensor then use this as target output

                    * else it will use the first `OutputClassification` output

            Returns:
                a tuple (output_name, dictionary (input, integrated gradient))
            """
        logger.info('started integrated gradient ...')
        self.model.eval()  # make sure we are in eval mode
        input_names_with_gradient = dict(guided_back_propagation.GuidedBackprop.get_floating_inputs_with_gradients(inputs)).keys()
        if len(input_names_with_gradient) == 0:
            logger.error('IntegratedGradients.__call__: failed. No inputs will collect gradient!')
            return None
        else:
            logger.info('input_names_with_gradient={}'.format(input_names_with_gradient))

        outputs = self.model(inputs)
        model_output = outputs.get(target_class_name)
        if model_output is None:
            for output_name, output in outputs.items():
                if isinstance(output, outputs.OutputClassification):
                    logger.info('IntegratedGradients.__call__: output found={}'.format(output_name))
                    target_class_name = output_name
                    model_output = output
                    break
        if model_output is None:
            logger.error('IntegratedGradients.__call__: failed. No suitable output could be found!')
            return None
        model_output = self.post_process_output(model_output)

        if target_class is None:
            target_class = torch.argmax(model_output, dim=1)  # type: ignore

        # construct our gradient target
        model_device = utilities.get_device(self.model, batch=inputs)
        nb_classes = model_output.shape[1]
        nb_samples = len_batch(inputs)

        if self.use_output_as_target:
            one_hot_output = model_output.clone()
        else:
            one_hot_output = torch.FloatTensor(nb_samples, nb_classes).to(device=model_device).zero_()
            one_hot_output[:, target_class] = 1.0

        # construct our reference inputs
        baseline_inputs = {}
        for feature_name, feature_value in inputs.items():
            if is_feature_metadata(feature_name, feature_value):
                # if metadata, we can't interpolate!
                continue
            baseline_inputs[feature_name] = torch.zeros_like(feature_value)

        # construct our integrated gradients
        integrated_gradients = {name: torch.zeros_like(inputs[name]) for name in input_names_with_gradient}

        # start integration
        for n in range(self.steps):
            integrated_inputs = {}
            with torch.no_grad():
                # here do no propagate the gradient (mixture of input and baseline)
                # We just want the gradient for the `inputs`
                for feature_name, feature_value in inputs.items():
                    if is_feature_metadata(feature_name, feature_value) or not torch.is_floating_point(feature_value):
                        # metadata or non floating point tensors: keep original value
                        integrated_inputs[feature_name] = feature_value
                    else:
                        baseline_value = baseline_inputs[feature_name]
                        integrated_inputs[feature_name] = baseline_value + float(n) / self.steps * (feature_value - baseline_value)
                        integrated_inputs[feature_name].requires_grad = True

            integrated_outputs = self.model(integrated_inputs)
            integrated_output = self.post_process_output(integrated_outputs[target_class_name])

            self.model.zero_grad()
            integrated_output.backward(gradient=one_hot_output, retain_graph=True)

            for name in input_names_with_gradient:
                if integrated_inputs[name].grad is not None:
                    integrated_gradients[name] += integrated_inputs[name].grad

        # average the gradients and multiply by input
        for name in list(integrated_gradients.keys()):
            integrated_gradients[name] = to_value((inputs[name] - baseline_inputs[name]) * integrated_gradients[name] / self.steps)

        logger.info('integrated gradient successful!')
        return target_class_name, integrated_gradients
