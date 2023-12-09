from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np

from utilities import to_value
import losses
from sklearn import metrics
import collections
import torch
from analysis_plots import auroc
import torch.nn as nn

# TODO discard samples where weight is <= 0 for all the metrics


def fast_confusion_matrix(
        y: torch.Tensor,
        y_pred: torch.Tensor,
        num_classes: int,
        ignore_y_out_of_range: bool = False,
        device=torch.device('cpu')) -> torch.Tensor:
    """
    Compute confusion matrix to evaluate the accuracy of a classification.

    By definition a confusion matrix :math:`C` is such that :math:`C_{i, j}`
    is equal to the number of observations known to be in group :math:`i` and
    predicted to be in group :math:`j`.

    Thus in binary classification, the count of true negatives is
    :math:`C_{0,0}`, false negatives is :math:`C_{1,0}`, true positives is
    :math:`C_{1,1}` and false positives is :math:`C_{0,1}`.

    Similar to :func:`sklearn.metrics.confusion_matrix`

    Args:
        y_pred: prediction (tensor of integers)
        y: tensor of integers
        num_classes: the number of classes
        ignore_y_out_of_range: if `True`, indices of `y` greater than `num_classes` will be ignored
        device: device where to perform the calculation
    """
    assert y_pred.shape == y.shape
    y_pred = y_pred.flatten().to(device)
    y = y.flatten().to(device)

    if ignore_y_out_of_range:
        target_mask = (y >= 0) & (y < num_classes)
        y = y[target_mask]
        y_pred = y_pred[target_mask]

    indices = num_classes * y + y_pred
    m = torch.bincount(indices, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return m


class Metric(ABC):
    """
    A metric base class

    Calculate interesting metric
    """
    @abstractmethod
    def __call__(self, outputs: Dict) -> Optional[Dict]:
        """

        Args:
            outputs:
                the outputs of a batch
        Returns:
            a dictionary of metric names/values or None
        """
        pass

    @abstractmethod
    def aggregate_metrics(self, metric_by_batch: List[Dict]) -> Dict[str, float]:
        """
        Aggregate all the metrics into a consolidated metric.

        Args:
            metric_by_batch: a list of metrics, one for each batch

        Returns:
            a dictionary of result name and value
        """
        pass


class MetricLoss(Metric):
    """
    Extract the loss from the outputs
    """
    def __call__(self, outputs):
        loss = to_value(outputs.get('loss'))
        if loss is not None:
            return {'loss': float(loss)}
        return None

    def aggregate_metrics(self, metric_by_batch):
        loss = 0.0
        for m in metric_by_batch:
            loss += m['loss']
        return {'loss': loss / len(metric_by_batch)}


class MetricClassificationBinaryAUC(Metric):
    """
    Calculate the Area under the Receiver operating characteristic (ROC) curve.

    For this, the output needs to provide an ``output_raw`` of shape [N, 2] (i.e., binary
    classification framed as a multi-class classification) or of shape [N, 1] (binary classification)
    """
    def __call__(self, outputs):
        truth = to_value(outputs.get('output_truth'))
        found = to_value(outputs.get('output_raw'))
        if truth is None or found is None:
            # data is missing
            return None

        if len(found.shape) != len(truth.shape) or found.shape[1] > 2:
            # dimensions are of the expected shape
            return None

        if len(found.shape) > 2:
            # TODO: implement for N-dimensions! We probably can't keep everything in memory
            return None

        return {
            'output_raw': found,
            'output_truth': truth,
        }

    def aggregate_metrics(self, metric_by_batch):
        all_output_raw = [m['output_raw'] for m in metric_by_batch]
        all_output_raw = np.concatenate(all_output_raw)
        all_output_truth = [m['output_truth'] for m in metric_by_batch]
        all_output_truth = np.concatenate(all_output_truth)

        auc = auroc(all_output_truth, all_output_raw[:, -1])
        if np.isnan(auc):
            auc = 0.0
        return {'1-auc': 1.0 - auc}


class MetricClassificationError(Metric):
    """
    Calculate the ``1 - accuracy`` using the `output_truth` and `output`
    """
    def __call__(self, outputs):
        truth = to_value(outputs.get('output_truth'))
        found = to_value(outputs.get('output'))
        weights = to_value(outputs.get('weights'))
        assert found.shape == truth.shape
        if truth is not None and found is not None:
            if weights is not None:
                min_weight = weights.min()
                if min_weight <= 0:
                    # if we have invalid indices (i.e., weights <= 0),
                    # discard these samples
                    valid_samples = np.where(weights > 0)
                    truth = truth[valid_samples]
                    found = found[valid_samples]

            return collections.OrderedDict([
                ('nb_trues', np.sum(found == truth)),
                ('total', truth.size),  # for multi-dimension, use the size! (e.g., patch discriminator, segmentation)
            ])
        return None

    def aggregate_metrics(self, metric_by_batch):
        nb_trues = 0
        total = 0
        for m in metric_by_batch:
            nb_trues += m['nb_trues']
            total += m['total']
        return {'classification error': 1.0 - nb_trues / total}


class MetricSegmentationDice(Metric):
    """
    Calculate the average dice score of a segmentation map 'output_truth' and class
    segmentation logits 'output_raw'.

    Notes:
        * by default, nn.Sigmoid function will be applied on the output to force a range [0..1] of the output

        * the aggregation will aggregate all the foregrounds/backgrounds THEN calculate the dice (but NOT average
            of dices). Using this aggregation, it is possible to calculate the true dice on a partitioned input
            (e.g., 3D segmentations, we often use sub-volumes)
    """
    def __init__(
            self,
            dice_fn=losses.LossDiceMulticlass(
                normalization_fn=None,  # we use discrete values, not probabilities
                return_dice_by_class=True,
                smooth=0),
            aggregate_by: Optional[str] = None):
        """

        Args:
            dice_fn: the function to calculate the dice score of each class
            aggregate_by: if not None, the dice scores will be aggregated first by `aggregate_by`. This can be useful
                when the metrics is calculated from pieces of the input data and we want to calculate a dice per
                case
        """
        self.dice_fn = dice_fn
        self.aggregate_by = aggregate_by

    def __call__(self, outputs):
        # keep the torch variable. We want to use GPU if available since it can
        # be slow use numpy for this
        truth = outputs.get('output_truth')
        found = outputs.get('output')
        raw = outputs.get('output_raw')
        assert raw is not None, 'missing value=`output_raw`'
        assert found is not None, 'missing value=`output`'
        assert truth is not None, 'missing value=`output_truth`'

        nb_classes = raw.shape[1]

        if self.aggregate_by is not None:
            aggregate_by = outputs.get(self.aggregate_by)
            assert aggregate_by is not None, f'cannot find the aggregate_by={self.aggregate_by} in batch! if using ' \
                                             f'`trw.train.OutputSegmentation`, make sure to set `sample_uid_name`' \
                                             f'appropriately'
        else:
            aggregate_by = None

        #assert found.min() >= 0, 'Unexpected value: `output` must be in range [0..1]'

        if found is None or truth is None:
            return None

        assert found.shape[1] == 1, 'output must have a single channel!'
        if raw.shape[1] > 1:
            # one hot encode the output
            found_one_hot = losses.one_hot(found[:, 0], nb_classes)
            assert len(found_one_hot.shape) == len(truth.shape), f'expecting dim={len(truth.shape)}, ' \
                                                                 f'got={len(found_one_hot)}'
            assert found_one_hot.shape[2:] == truth.shape[2:]
        else:
            # it is already one hot encoded for a binary classification!
            found_one_hot = found

        with torch.no_grad():
            numerator, cardinality = self.dice_fn(found_one_hot, truth)

        return {
            # sum the samples: we have to do this to support variably sized
            # batch size
            'numerator': to_value(numerator),
            'cardinality': to_value(cardinality),
            'aggregate_by': aggregate_by
        }

    @staticmethod
    def _aggregate_dices(metric_by_batch):
        eps = 1e-5  # avoid div by 0

        # aggregate all the patches at once to calculate the global dice (and not average of dices)
        numerator = metric_by_batch[0]['numerator'].copy()
        cardinality = metric_by_batch[0]['cardinality'].copy()
        assert len(numerator.shape) == 2, 'must be NxC matrix'
        assert numerator.shape == cardinality.shape

        numerator = numerator.sum(axis=0)
        cardinality = cardinality.sum(axis=0)
        for m in metric_by_batch[1:]:
            numerator += m['numerator'].sum(axis=0)
            cardinality += m['cardinality'].sum(axis=0)
        # calculate the dice score by class
        dice = numerator / (cardinality + eps)
        return dice

    @staticmethod
    def _aggregate_dices_by_uid(metric_by_batch):
        eps = 1e-5  # avoid div by 0

        # group the dice's (numerator, denominator) by UID
        num_card_by_uid = {}
        for m in metric_by_batch:
            numerators = m['numerator']
            cardinalitys = m['cardinality']
            uids = m.get('aggregate_by')
            assert uids is not None
            assert len(numerators) == len(cardinalitys)
            assert len(numerators) == len(uids)
            for numerator, cardinality, uid in zip(numerators, cardinalitys, uids):
                numerator_cardinality = num_card_by_uid.get(uid)
                if numerator_cardinality is None:
                    num_card_by_uid[uid] = [numerator, cardinality]
                else:
                    numerator_cardinality[0] += numerator
                    numerator_cardinality[1] += cardinality

        # then calculate the average dice by UID
        dice_sum = 0
        for uid, (numerator, cardinality) in num_card_by_uid.items():
            # if cardinality[class] == 0, then numerator[class] == 0
            # so it is ok to just add `eps` when cardinality == 0 to avoid
            # div by 0
            dice = numerator / (cardinality + eps)
            dice_sum += dice

        return dice_sum / len(num_card_by_uid)

    def aggregate_metrics(self, metric_by_batch):
        nb_batches = len(metric_by_batch)
        if nb_batches > 0:
            if self.aggregate_by is None:
                dice = MetricSegmentationDice._aggregate_dices(metric_by_batch)
            else:
                dice = MetricSegmentationDice._aggregate_dices_by_uid(metric_by_batch)

            # to keep consistent with the other metrics
            # calculate the `1 - metric`
            one_minus_dice = 1 - dice
            r = collections.OrderedDict()
            for c in range(len(dice)):
                r[f'1-dice[class={c}]'] = one_minus_dice[c]
            r['1-dice'] = np.average(one_minus_dice)
            return r

        # empty, so assume the worst
        return {'1-dice': 1}


class MetricClassificationF1(Metric):
    def __init__(self, average=None):
        """
        Calculate the Multi-class ``1 - F1 score``.

        Args:
            average: one of ``binary``, ``micro``, ``macro`` or ``weighted`` or None. If ``None``, use
                ``binary`` if only 2 classes or ``macro`` if more than two classes
        """
        self.average = average
        self.max_classes = 0

    def __call__(self, outputs):
        output_raw = to_value(outputs.get('output_raw'))
        if output_raw is None:
            return None
        if len(output_raw.shape) != 2:
            return None

        truth = to_value(outputs.get('output_truth'))
        if truth is None:
            return None

        self.max_classes = max(self.max_classes, output_raw.shape[1])
        found = to_value(outputs['output'])
        return {
            'truth': truth,
            'found': found
        }

    def aggregate_metrics(self, metric_by_batch):
        truth = [m['truth'] for m in metric_by_batch]
        truth = np.concatenate(truth)
        found = [m['found'] for m in metric_by_batch]
        found = np.concatenate(found)

        if self.average is None:
            if self.max_classes <= 1:
                average = 'binary'
            else:
                average = 'macro'
        else:
            average = self.average

        score = 1.0 - metrics.f1_score(y_true=truth, y_pred=found, average=average)
        return {
            f'1-f1[{average}]': score
        }


class MetricClassificationBinarySensitivitySpecificity(Metric):
    """
    Calculate the sensitivity and specificity for a binary classification using the `output_truth` and `output`
    """
    def __call__(self, outputs):
        output_raw = outputs.get('output_raw')
        if output_raw is None:
            return None

        # here we MUST have binary classification problem
        # so make sure the `C` == 2 (binary classification with a single output, or multi-class with 2 outputs)
        if len(output_raw.shape) < 2 or output_raw.shape[1] > 2:
            return None

        truth = outputs.get('output_truth')
        found = outputs.get('output')
        if truth.shape != found.shape:
            # shape must be the same, else something is wrong!
            return None

        # make it a 1D tensor
        truth = truth.reshape((-1))
        found = found.reshape((-1))
        if truth is not None and found is not None:
            with torch.no_grad():
                cm = fast_confusion_matrix(y_pred=found, y=truth, num_classes=2).numpy()

            if len(cm) == 2:
                # special case: only binary classification
                tn, fp, fn, tp = cm.ravel()

                return collections.OrderedDict([
                    ('tn', tn),
                    ('fn', fn),
                    ('fp', fp),
                    ('tp', tp),
                ])
            else:
                if truth[0] == 0:
                    # 0, means perfect classification of the negative
                    return collections.OrderedDict([
                        ('tn', cm[0, 0]),
                        ('fn', 0),
                        ('fp', 0),
                        ('tp', 0),
                    ])
                else:
                    # 1, means perfect classification of the positive
                    return collections.OrderedDict([
                        ('tp', cm[0, 0]),
                        ('fn', 0),
                        ('fp', 0),
                        ('tn', 0),
                    ])

        # something is missing, don't calculate the stats
        return None

    def aggregate_metrics(self, metric_by_batch):
        tn = 0
        fp = 0
        fn = 0
        tp = 0

        for m in metric_by_batch:
            tn += m['tn']
            fn += m['fn']
            tp += m['tp']
            fp += m['fp']

        if tp + fn > 0:
            one_minus_sensitivity = 1.0 - tp / (tp + fn)
        else:
            # invalid! `None` will be discarded
            one_minus_sensitivity = None

        if fp + tn > 0:
            one_minus_specificity = 1.0 - tn / (fp + tn)
        else:
            # invalid! `None` will be discarded
            one_minus_specificity = None

        return collections.OrderedDict([
            # we return the 1.0 - metric, since in the history we always keep the smallest number
            ('1-sensitivity', one_minus_sensitivity),
            ('1-specificity', one_minus_specificity),
        ])


def default_classification_metrics():
    """"
    Default list of metrics used for classification
    """
    return [
        MetricLoss(),
        MetricClassificationError(),
        MetricClassificationBinarySensitivitySpecificity(),
        MetricClassificationBinaryAUC(),
        MetricClassificationF1(),
    ]


def default_regression_metrics():
    """"
    Default list of metrics used for regression
    """
    return [
        MetricLoss(),
    ]


def default_segmentation_metrics():
    """"
    Default list of metrics used for segmentation
    """
    return [
        MetricLoss(),
        MetricSegmentationDice(),

        # this implementation is slow. By default it should be disabled!
        #MetricClassificationBinarySensitivitySpecificity(),
    ]


def default_generic_metrics():
    """"
    Default list of metrics
    """
    return [
        MetricLoss(),
    ]
