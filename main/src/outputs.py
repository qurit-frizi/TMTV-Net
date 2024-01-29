import warnings
from typing import Callable, Any, List

import torch
import functools
import collections
import torch.nn as nn
import torch.nn.functional as F
import metrics
from sequence_array import sample_uid_name as default_sample_uid_name
import losses
from flatten import flatten
from losses import LossDiceMulticlass, LossFocalMulticlass


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


def dict_torch_values_to_numpy(d):
    """
    Transform all torch.Tensor to numpy arrays of a dictionary like object
    """
    if d is None:
        return
    assert isinstance(d, collections.Mapping), 'must be a dict like object'

    for name, value in d.items():
        if isinstance(value, torch.Tensor):
            d[name] = to_value(value)


class Output:
    """
    This is a tag name to find the output reference back from `outputs`
    """
    output_ref_tag = 'output_ref'

    def __init__(self, metrics, output, criterion_fn, collect_output=False, sample_uid_name=None):
        """
        :param metrics: the metrics to be reported for each output
        :param output: a `torch.Tensor` to be recorded
        :param criterion_fn: the criterion function to be used to evaluate the output
        :param collect_output: if True, the output values will be collected (and possibly exported for debug purposes)
        :pram sample_uid_name: collect sample UID along with the output
        """
        self.output = output
        self.criterion_fn = criterion_fn
        self.collect_output = collect_output
        self.metrics = metrics

        # this can be used to collect the UIDs of the sample the output was calculated from.
        # this can be particularly useful for various tasks: track data augmentation,
        self.sample_uid_name = sample_uid_name

    def evaluate_batch(self, batch, is_training):
        """
        Evaluate a batch of data and extract important outputs
        :param batch: the batch of data
        :param is_training: if True, this was a training batch
        :return: tuple(a dictionary of values, dictionary of metrics)
        """
        assert 0, 'this needs to be implemented in derived classes!'
        
    def loss_term_cleanup(self, loss_term):
        """
        This function is called for each batch just before switching to another batch.

        It can be used to clean up large arrays stored or release CUDA memory
        """
        dict_torch_values_to_numpy(loss_term)
        metrics_results = loss_term.get('metrics_results')
        if metrics_results is not None:
            dict_torch_values_to_numpy(metrics_results)


def extract_metrics(metrics_outputs, outputs):
    """
    Extract metrics from an output

    Args:
        metrics_outputs: a list of metrics
        outputs: the result of `Output.evaluate_batch`

    Returns:
        a dictionary of key, value
    """
    history = collections.OrderedDict()
    if metrics_outputs is not None:
        for metric in metrics_outputs:
            metric_result = metric(outputs)
            if metric_result is not None:
                assert isinstance(metric_result, collections.Mapping), 'must be a dict like structure'
                history.update({metric: metric_result})
    return history


class OutputEmbedding(Output):
    """
    Represent an embedding

    This is only used to record a tensor that we consider an embedding (e.g., to be exported to tensorboard)
    """
    def __init__(self, output, clean_loss_term_each_batch=False, sample_uid_name=default_sample_uid_name, functor=None):
        """
        
        Args:
            output: the output from which the embedding will be created
            clean_loss_term_each_batch: if ``True``, the loss term output will be removed from the output in
                order to free memory just before the next batch. For example, if we want to collect statistics
                on the embedding, we do not need to keep track of the output embedding and in particular for
                large embeddings. If ``False``, the output will not be cleaned (and possibly accumulated
                for the whole epoch)
            sample_uid_name: UID name to be used for collecting the embedding of the samples
            functor: apply a function on the output to create the embedding
        """
        super().__init__(
            output=output,
            criterion_fn=None,
            collect_output=True,
            sample_uid_name=sample_uid_name,
            metrics=None)
        self.clean_loss_term_each_batch = clean_loss_term_each_batch
        self.functor = functor

    def evaluate_batch(self, batch, is_training):
        loss_term = collections.OrderedDict()

        # do not keep track of GPU torch.Tensor, specially for large embeddings.
        # ``clean_loss_term_each_batch`` may be ``False``, so the output
        # would never be cleaned up.
        self.output = to_value(self.output)
        if self.functor is not None:
            self.output = self.functor(self.output)

        loss_term['output'] = self.output
        loss_term[Output.output_ref_tag] = self  # keep a back reference
        return loss_term
    
    def loss_term_cleanup(self, loss_term):
        if self.clean_loss_term_each_batch:
            del loss_term['output']
            self.output = None


def segmentation_criteria_ce_dice(output, truth, per_voxel_weights=None, ce_weight=0.5, per_class_weights=None, power=1.0, smooth=1.0, focal_gamma=None):
    """
    loss combining cross entropy and multi-class dice

    Args:
        output: the output value, with shape [N, C, Dn...D0]
        truth: the truth, with shape [N, 1, Dn..D0]
        ce_weight: the weight of the cross entropy to use. This controls the importance of the
            cross entropy loss to the overall segmentation loss. Range in [0..1]
        per_class_weights: the weight per class. A 1D vector of size C indicating the weight of the classes. This
            will be used for the cross-entropy loss
        per_voxel_weights: the weight of each truth voxel. Must be of shape [N, Dn..D0]

    Returns:
        a torch tensor
    """
    assert per_class_weights is None or len(per_class_weights) == output.shape[1], f'per_class_weights must have a' \
                                                                                 f'weight per class. ' \
                                                                                 f'Got={len(per_class_weights)}, ' \
                                                                                 f'expected={output.shape[1]}'

    assert per_voxel_weights is None or per_voxel_weights.shape == truth.shape, 'incorrect per-voxel weight'
    assert truth.shape[1] == 1, f'expected a single channel! Got={truth.shape[1]}'

    if ce_weight > 0:
        if focal_gamma is None or focal_gamma <= 0:
            if output.shape[1] == 1:
                cross_entropy_loss = nn.BCEWithLogitsLoss(reduction='none', weight=per_class_weights)(output, truth.float())
            else:
                cross_entropy_loss = nn.CrossEntropyLoss(reduction='none', weight=per_class_weights)(output, truth.squeeze(1))
        else:
            cross_entropy_loss = LossFocalMulticlass(gamma=focal_gamma)(output, truth)

        if per_voxel_weights is not None:
            cross_entropy_loss = cross_entropy_loss * per_voxel_weights
        cross_entropy_loss = cross_entropy_loss.mean(tuple(range(1, len(cross_entropy_loss.shape))))
    else:
        cross_entropy_loss = 0

    if ce_weight < 1:
        dice_loss = losses.LossDiceMulticlass(power=power, smooth=smooth)(output, truth)
    else:
        dice_loss = 0

    loss = ce_weight * cross_entropy_loss + (1 - ce_weight) * dice_loss
    return loss


def criterion_softmax_cross_entropy(output, output_truth):
    assert len(output.shape) == len(output_truth.shape), '`output` and `output_truth` must have the same dimensionality'
    assert output_truth.shape[1] == 1, 'truth must have a single channel'
    assert output_truth.shape[2:] == output.shape[2:], 'all the input must be covered by truth'
    assert output.shape[1] >= 2, 'output must have N channels, one for each class! (else binary version should be used!)'
    #return F.cross_entropy(output, output_truth.squeeze(1), reduction='none')
    return nn.CrossEntropyLoss(reduction='none')(output, output_truth.squeeze(1))


class OutputClassification(Output):
    """
    Classification output
    """
    def __init__(
            self,
            output,
            output_truth,
            *,
            criterion_fn=lambda: criterion_softmax_cross_entropy,
            collect_output=True,
            collect_only_non_training_output=False,
            metrics: List[metrics.Metric] = metrics.default_classification_metrics(),
            loss_reduction=torch.mean,
            weights=None,
            per_voxel_weights=None,
            loss_scaling=1.0,
            output_postprocessing=functools.partial(torch.argmax, dim=1, keepdim=True),  # =1 as we export the class
            maybe_optional=False,
            classes_name='unknown',
            sample_uid_name=default_sample_uid_name):
        """

        Args:
            output: the raw output values (no activation applied, i.e., logits). Should be of shape [N, C, ...]
            output_truth: the tensor to be used as target. Should be of shape [N, C, ...] and be
                compatible with ``criterion_fn``
            criterion_fn: the criterion to minimize between the output and the output_truth. If ``None``, the returned
                loss will be 0
            collect_output: if True, the output values will be collected (and possibly exported for debug purposes)
            collect_only_non_training_output: if True, only the non-training splits will have the outputs collected
            metrics: the metrics to be reported each epoch
            loss_reduction: a function to reduce the N-d losses to a single loss value
            weights: if not None, the weight name. the loss of each sample will be weighted by this vector
            loss_scaling: scale the loss by a scalar
            output_postprocessing: the output will be post-processed by this function for the classification report.
                For example, the classifier may return logit scores for each class and we can use argmax of the
                scores to get the class.
            maybe_optional: if True, the loss term may be considered optional if the ground
                truth is not part of the batch
            sample_uid_name (str): if not None, collect the sample UID
            per_voxel_weights: a per voxel weighting that will be passed to criterion_fn
            classes_name: the name of the class. This is used only to map a class ID to a string (e.g., for the
                classification report)
        """
        super().__init__(
            output=output,
            criterion_fn=criterion_fn,
            collect_output=collect_output,
            sample_uid_name=sample_uid_name,
            metrics=metrics)
        self.output_truth = output_truth
        self.loss_reduction = loss_reduction
        self.output_postprocessing = output_postprocessing
        self.collect_only_non_training_output = collect_only_non_training_output
        self.loss_scaling = loss_scaling
        self.weights = weights
        self.per_voxel_weights = per_voxel_weights
        self.maybe_optional = maybe_optional
        self.classes_name = classes_name

        if self.per_voxel_weights is not None:
            assert self.per_voxel_weights.shape[2:] == self.output.shape[2:]
            assert len(self.per_voxel_weights) == len(self.output)

        if output_truth is not None:
            if len(output.shape) != len(output_truth.shape):
                if len(output.shape) == len(output_truth.shape) + 1:
                    warnings.warn('output and output_truth must have the same shape!'
                                'For binary classification, output_truth.shape == (X, 1).'
                                'This will be disabled in the future! Simply replace by'
                                '`output_truth` by `output_truth.unsqueeze(1)`', FutureWarning)
                    self.output_truth = output_truth.unsqueeze(1)

    def evaluate_batch(self, batch, is_training):
        truth = self.output_truth
        if truth is None and self.maybe_optional:
            return None
        assert truth is not None, 'truth is `None` use `maybe_optional` to True'
        assert len(truth) == len(self.output), f'expected len output ({len(self.output)}) == len truth ({len(truth)})!'
        assert isinstance(truth, torch.Tensor), 'feature must be a torch.Tensor!'
        assert truth.dtype == torch.long, f'the truth vector must be a `long` type feature, got={truth.dtype}'

        # make sure the class is not out of bound. This is a very common mistake!
        """
        max_index = int(torch.max(truth).cpu().numpy())
        min_index = int(torch.min(truth).cpu().numpy())
        assert max_index < self.output.shape[1], f'index out of bound. Got={max_index}, ' \
                                                 f'maximum={self.output.shape[1]}. Make sure the input data is correct.'
        assert min_index >= 0, f'incorrect index! got={min_index}'
        """

        criterion_args = {}
        if self.per_voxel_weights is not None:
            criterion_args['per_voxel_weights'] = self.per_voxel_weights

        loss_term = {}
        if self.criterion_fn is not None:
            losses = self.criterion_fn()(self.output, truth, **criterion_args)
        else:
            losses = torch.zeros(len(self.output), device=self.output.device)
        assert isinstance(losses, torch.Tensor), 'must `loss` be a `torch.Tensor`'

        if self.weights is not None:
            weights = self.weights
            assert len(weights) == len(losses), 'must have a weight per sample'
            assert len(weights.shape) == 1, 'must be a 1D vector'
        else:
            weights = torch.ones_like(losses)

        # weight the loss of each sample by the corresponding weight
        weighted_losses = weights * losses

        if len(weighted_losses.shape) >= 2:
            # average the per-sample loss
            weighted_losses = flatten(weighted_losses).mean(dim=1)

        assert truth.shape[0] == losses.shape[0], 'loos must have 1 element per sample'

        loss_term['output_raw'] = self.output
        output_postprocessed = self.output_postprocessing(self.output)
        assert torch.is_tensor(output_postprocessed)
        assert output_postprocessed.dtype == torch.long, f'output must have a `long` type. ' \
                                                         f'Got={output_postprocessed.dtype}'
        loss_term['output'] = output_postprocessed
        loss_term['output_truth'] = truth

        if self.sample_uid_name is not None and self.sample_uid_name in batch:
            uid = to_value(batch[self.sample_uid_name])
            assert len(uid) == len(truth), f'1 UID for 1 sample! Got={len(uid)} for samples={len(truth)}'
            loss_term['uid'] = uid

        # TODO label smoothing
        loss_term['losses'] = weighted_losses
        loss_term['loss'] = self.loss_scaling * self.loss_reduction(weighted_losses)
        loss_term['weights'] = weights
        loss_term[Output.output_ref_tag] = self  # keep a back reference
        loss_term['metrics_results'] = extract_metrics(self.metrics, loss_term)
        return loss_term

    def loss_term_cleanup(self, loss_term):
        if not self.collect_output:
            del loss_term['output_raw']
            del loss_term['output']
            del loss_term['output_truth']

        # delete possibly large outputs
        self.output = None
        self.output_truth = None

        


bce_logits_loss = lambda output, target: nn.functional.binary_cross_entropy_with_logits(
    output,
    target.type(output.dtype),
    reduction='none'
)


class OutputClassificationBinary(OutputClassification):
    """
    Classification output for binary classification

    Args:
        output: the output with shape [N, 1, {X}], without any activation applied (i.e., logits)
        output_truth: the truth with shape [N, 1, {X}]
    """
    def __init__(
            self,
            output,
            output_truth,
            *,
            criterion_fn=lambda: bce_logits_loss,
            collect_output=True,
            collect_only_non_training_output=False,
            metrics: List[metrics.Metric] = metrics.default_classification_metrics(),
            loss_reduction=torch.mean,
            weights=None,
            per_voxel_weights=None,
            loss_scaling=1.0,
            output_postprocessing=lambda x: (torch.sigmoid(x) >= 0.5).long(),
            maybe_optional=False,
            classes_name='unknown',
            sample_uid_name=default_sample_uid_name):

        if len(output.shape) != len(output_truth.shape):
            if len(output.shape) == len(output_truth.shape) + 1:
                warnings.warn('output and output_truth must have the same shape!'
                              'For binary classification, output_truth.shape == (X, 1).'
                              'This will be disabled in the future! Simply replace by'
                              '`output_truth` by `output_truth.unsqueeze(1)`', FutureWarning)
                output_truth = output_truth.unsqueeze(1)

        assert len(output.shape) == len(output_truth.shape), 'must have the same dimensionality!'
        assert output.shape[1] == 1, 'binary classification!'
        assert output_truth.shape[1] == 1, 'binary classification!'

        super().__init__(
            output=output,
            output_truth=output_truth,
            criterion_fn=criterion_fn,
            collect_output=collect_output,
            collect_only_non_training_output=collect_only_non_training_output,
            metrics=metrics,
            loss_reduction=loss_reduction,
            weights=weights,
            per_voxel_weights=per_voxel_weights,
            loss_scaling=loss_scaling,
            output_postprocessing=output_postprocessing,  # =1 as we export the class
            maybe_optional=maybe_optional,
            classes_name=classes_name,
            sample_uid_name=sample_uid_name
        )


class OutputSegmentation(OutputClassification):
    def __init__(
            self,
            output: torch.Tensor,
            output_truth: torch.Tensor,
            criterion_fn: Callable[[], Any] = LossDiceMulticlass,
            collect_output: bool = False,
            collect_only_non_training_output: bool = False,
            metrics: List[metrics.Metric] = metrics.default_segmentation_metrics(),
            loss_reduction: Callable[[torch.Tensor], torch.Tensor] = torch.mean,
            weights=None,
            per_voxel_weights=None,
            loss_scaling=1.0,
            output_postprocessing=functools.partial(torch.argmax, dim=1, keepdim=True),  # =1 as we export the class
            maybe_optional=False,
            sample_uid_name=default_sample_uid_name):
        """

        Args:
            output: the raw output values (`criterion_fn` will apply normalization such as sigmoid)
            output_truth: the tensor to be used as target. Targets must be compatible with ``criterion_fn``
            criterion_fn: the criterion to minimize between the output and the output_truth. If ``None``, the returned
                loss will be 0
            collect_output: if True, the output values will be collected (and possibly exported for debug purposes)
            collect_only_non_training_output: if True, only the non-training splits will have the outputs collected
            metrics: the metrics to be reported each epoch
            loss_reduction: a function to reduce the N-d losses to a single loss value
            weights: if not None, the weight name. the loss of each sample will be weighted by this vector
            loss_scaling: scale the loss by a scalar
            output_postprocessing: the output will be post-processed by this function for the segmentation result.
                For example, the classifier may return logit scores for each class and we can use argmax of the
                scores to get the class.
            maybe_optional: if True, the loss term may be considered optional if the ground
                truth is not part of the batch
            sample_uid_name (str): if not None, collect the sample UID
            per_voxel_weights: a per voxel weighting that will be passed to criterion_fn
        """
        if output_truth is not None:
            assert len(output.shape) == len(output_truth.shape), 'must have the same dimensionality!'

        super().__init__(
            output=output,
            output_truth=output_truth,
            criterion_fn=criterion_fn,
            collect_output=collect_output,
            collect_only_non_training_output=collect_only_non_training_output,
            metrics=metrics,
            loss_reduction=loss_reduction,
            weights=weights,
            loss_scaling=loss_scaling,
            output_postprocessing=output_postprocessing,
            maybe_optional=maybe_optional,
            sample_uid_name=sample_uid_name,
            per_voxel_weights=per_voxel_weights
        )


class OutputSegmentationBinary(OutputSegmentation):
    """
    Output for binary segmentation.

    Parameters:
        output: shape N * 1 * X format, must be raw logits
        output_truth: should have N * 1 * X format, with values 0 or 1
    """
    def __init__(
            self,
            output: torch.Tensor,
            output_truth: torch.Tensor,
            criterion_fn: Callable[[], Any] = LossDiceMulticlass,
            collect_output: bool = False,
            collect_only_non_training_output: bool = False,
            metrics: List[metrics.Metric] = metrics.default_segmentation_metrics(),
            loss_reduction: Callable[[torch.Tensor], torch.Tensor] = torch.mean,
            weights=None,
            per_voxel_weights=None,
            loss_scaling=1.0,
            output_postprocessing=lambda x: (torch.sigmoid(x) > 0.5).type(torch.long),  # =1 as we export the class
            maybe_optional=False,
            sample_uid_name=default_sample_uid_name):
        super().__init__(
            output=output,
            output_truth=output_truth,
            criterion_fn=criterion_fn,
            collect_output=collect_output,
            collect_only_non_training_output=collect_only_non_training_output,
            metrics=metrics,
            loss_reduction=loss_reduction,
            weights=weights,
            per_voxel_weights=per_voxel_weights,
            loss_scaling=loss_scaling,
            output_postprocessing=output_postprocessing,
            maybe_optional=maybe_optional,
            sample_uid_name=sample_uid_name
        )


def mean_all(x):
    """
    :param x: a Tensor
    :return: the mean of all values
    """
    return torch.mean(x.view((-1)))


class OutputRegression(Output):
    """
    Regression output
    """
    def __init__(
            self,
            output,
            output_truth,
            criterion_fn=lambda: nn.MSELoss(reduction='none'),
            collect_output=True,
            collect_only_non_training_output=False,
            metrics=metrics.default_regression_metrics(),
            loss_reduction=mean_all,
            weights=None,
            loss_scaling=1.0,
            output_postprocessing=lambda x: x,
            target_name=None,
            sample_uid_name=default_sample_uid_name):
        """

        :param output:
        :param target_name:
        :param criterion_fn:
        :param collect_output:
        :param collect_only_non_training_output:
        :param metrics:
        :param loss_reduction:
        :param weights: if not None, the weight. the loss of each sample will be weighted by this vector
        :param loss_scaling: scale the loss by a scalar
        :param output_postprocessing:
        """
        super().__init__(
            output=output,
            criterion_fn=criterion_fn,
            collect_output=collect_output,
            sample_uid_name=sample_uid_name,
            metrics=metrics)
        self.target_name = target_name
        self.output_truth = output_truth
        self.loss_reduction = loss_reduction
        self.output_postprocessing = output_postprocessing
        self.collect_only_non_training_output = collect_only_non_training_output
        self.weights = weights
        self.loss_scaling = loss_scaling

    def evaluate_batch(self, batch, is_training):
        loss_term = {}
        losses = self.criterion_fn()(self.output, self.output_truth)
        assert isinstance(losses, torch.Tensor), 'must `loss` be a `torch.Tensor`'
        assert len_batch(batch) == losses.shape[0], 'loos must have 1 element per sample'
        if self.collect_output:
            # we may not want to collect any outputs or training outputs to save some time
            if not self.collect_only_non_training_output or not is_training:
                # detach the output so as not to calculate gradients. Keep the truth so that we
                # can calculate statistics (e.g., accuracy, FP/FN...)
                loss_term['output_raw'] = self.output
                loss_term['output'] = self.output_postprocessing(self.output)
                loss_term['output_truth'] = self.output_truth

        # do NOT keep the original output else memory will be an issue
        del self.output
        self.output = None

        if self.weights is not None:
            weights = self.weights
            assert weights is not None, f'weight `{self.weights}` could not be found!'
            assert len(weights) == len(losses), 'must have a weight per sample'
            assert len(weights.shape) == 1, 'must be a 1D vector'
        else:
            weights = torch.ones_like(losses)
            
        if self.sample_uid_name is not None and self.sample_uid_name in batch:
            loss_term['uid'] = to_value(batch[self.sample_uid_name])

        # weight the loss of each sample by the corresponding weight
        weighted_losses = weights * losses

        loss_term['losses'] = weighted_losses
        # here we MUST be able to calculate the gradient so don't detach
        loss_term['loss'] = self.loss_scaling * self.loss_reduction(weighted_losses)
        loss_term[Output.output_ref_tag] = self  # keep a back reference
        loss_term['metrics_results'] = extract_metrics(self.metrics, loss_term)
        return loss_term


class OutputTriplets(Output):
    def __init__(
            self,
            samples,
            positive_samples,
            negative_samples,
            criterion_fn=lambda: losses.LossTriplets(),
            metrics=metrics.default_generic_metrics(),
            loss_reduction=mean_all,
            weight_name=None,
            loss_scaling=1.0,
            sample_uid_name=default_sample_uid_name
    ):
        super().__init__(metrics=metrics, output=samples, criterion_fn=criterion_fn)
        self.loss_reduction = loss_reduction
        self.sample_uid_name = sample_uid_name
        self.loss_scaling = loss_scaling
        self.weight_name = weight_name
        self.criterion_fn = criterion_fn
        self.negative_samples = negative_samples
        self.positive_samples = positive_samples

    def evaluate_batch(self, batch, is_training):
        loss_term = collections.OrderedDict()
        losses = self.criterion_fn()(self.output, self.positive_samples, self.negative_samples)
        assert isinstance(losses, torch.Tensor), 'must `loss` be a `torch.Tensor`'
        assert len_batch(batch) == losses.shape[0], 'loss must have 1 element per sample'

        loss_term['output_raw'] = self.output

        # do NOT keep the original output else memory will be an issue
        # (e.g., CUDA device)
        del self.output
        self.output = None
        del self.negative_samples
        self.negative_samples = None
        del self.positive_samples
        self.positive_samples = None

        if self.weight_name is not None:
            weights = batch.get(self.weight_name)
            assert weights is not None, f'weight `{self.weight_name}` could not be found!'
            assert len(weights) == len(losses), 'must have a weight per sample'
            assert len(weights.shape) == 1, 'must be a 1D vector'
            # expand to same shape size so that we can easily broadcast the weight
            weights = weights.reshape([weights.shape[0]] + [1] * (len(losses.shape) - 1))
        else:
            weights = torch.ones_like(losses)

        if self.sample_uid_name is not None and self.sample_uid_name in batch:
            loss_term['uid'] = to_value(batch[self.sample_uid_name])

        # weight the loss of each sample by the corresponding weight
        weighted_losses = weights * losses

        loss_term['losses'] = weighted_losses
        loss_term['loss'] = self.loss_scaling * self.loss_reduction(weighted_losses)
        loss_term[Output.output_ref_tag] = self  # keep a back reference
        loss_term['metrics_results'] = extract_metrics(self.metrics, loss_term)
        return loss_term


class OutputLoss(Output):
    """
    Represent a given loss as an output.

    This can be useful to add additional regularizer to the training (e.g., :class:`LossCenter`).
    """
    def __init__(
            self,
            losses,
            loss_reduction=torch.mean,
            metrics=metrics.default_generic_metrics(),
            sample_uid_name=default_sample_uid_name):
        super().__init__(
            metrics=metrics,
            output=losses,
            criterion_fn=None,
            collect_output=True,
            sample_uid_name=sample_uid_name)
        self.loss_reduction = loss_reduction

    def evaluate_batch(self, batch, is_training):
        loss_term = {
            'losses': self.output,
            'loss': self.loss_reduction(self.output),  # to be optimized, we MUST have a `loss` key
            Output.output_ref_tag: self,  # keep a back reference
        }

        if self.sample_uid_name is not None and self.sample_uid_name in batch:
            loss_term['uid'] = to_value(batch[self.sample_uid_name])

        loss_term['metrics_results'] = extract_metrics(self.metrics, loss_term)
        return loss_term

    def loss_term_cleanup(self, loss_term):
        super().loss_term_cleanup(loss_term)

        # delete possibly large outputs
        self.output = None

