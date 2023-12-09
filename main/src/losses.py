from functools import partial
from typing import Callable, Sequence, Optional

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing_extensions import Literal
from basic_typing import TensorNCX, TorchTensorNX, TorchTensorNCX


def one_hot(
        targets: TorchTensorNX, 
        num_classes: int, 
        dtype=torch.float32, 
        device: Optional[torch.device]=None) -> TorchTensorNCX:
    """
    Encode the targets (an tensor of integers representing a class)
    as one hot encoding.

    Support target as N-dimensional data (e.g., 3D segmentation map).

    Equivalent to torch.nn.functional.one_hot for backward compatibility with pytorch 1.0

    Args:
        num_classes: the total number of classes
        targets: a N-dimensional integral tensor (e.g., 1D for classification, 2D for 2D segmentation map...)
        dtype: the type of the output tensor
        device: the device of the one-hot encoded tensor. If `None`, use the target's device

    Returns:
        a one hot encoding of a N-dimentional integral tensor
    """
    if device is None:
        device = targets.device

    nb_samples = len(targets)
    if len(targets.shape) == 2:
        # 2D target (e.g., classification)
        encoded_shape = (nb_samples, num_classes)
    else:
        # N-d target (e.g., segmentation map)
        encoded_shape = tuple([nb_samples, num_classes] + list(targets.shape[1:]))

    with torch.no_grad():
        encoded_target = torch.zeros(encoded_shape, dtype=dtype, device=device)
        encoded_target.scatter_(1, targets.unsqueeze(1), 1)
    return encoded_target


class LossDiceMulticlass(nn.Module):
    """
    Implementation of the soft Dice Loss (multi-class) for N-d images
    
    If multi-class, compute the loss for each class then average the losses

    References:
        [1] "V-Net: Fully Convolutional Neural Networks for Volumetric Medical
        Image Segmentation" https://arxiv.org/pdf/1606.04797.pdf
    """
    def __init__(self,
                 normalization_fn: Callable[[torch.Tensor], torch.Tensor]=partial(nn.Softmax, dim=1),
                 eps: float =1e-5,
                 return_dice_by_class: bool =False,
                 smooth: float =1e-3,
                 power: float =1.0,
                 per_class_weights: Sequence[float] = None,
                 discard_background_loss: bool = True):
        """

        Args:
            normalization_fn: apply a normalization function on the `output` in the forward method. This
                should  normalize the output from logits to probability (i.e. range [0..1])
            eps: epsilon to avoid division by zero
            return_dice_by_class: if True, returns the (numerator, cardinality) by class and by sample from
                the dice can be calculated, else returns the per sample dice loss `1 - average(dice by class)`
            smooth: a smoothing factor
            per_class_weights: a weighting of the classes
            power: power of the denominator components
            discard_background_loss: if True, the loss will NOT include the background `class`

        Notes:
        * if return_dice_by_class is True, to calculate dice by class by sample:
            dice_by_class_by_sample = (numerator / cardinality).numpy()  # axis 0: sample, axis 1: class
            average_dice_by_class = dice_by_class_by_sample.mean(axis=0) # axis 0: average dice by class

        * To calculate metrics (and not loss for optimization), smooth==0 and probably a
            smaller eps (e.g., eps=1e-7)

        """
        super().__init__()

        self.eps = eps
        self.normalization = None
        self.return_dice_by_class = return_dice_by_class
        self.smooth = smooth
        self.per_class_weights = torch.tensor(per_class_weights).unsqueeze(0) if per_class_weights is not None else None
        self.power = power
        self.discard_background_loss = discard_background_loss

        if normalization_fn is not None:
            self.normalization = normalization_fn()
        
    def forward(self, output, target):
        """
        
        Args:
            output: must have N x C x d0 x ... x dn shape, where C is the total number of classes to predict
            target: must have N x 1 x d0 x ... x dn shape

        Returns:
            if return_dice_by_class is False, return 1 - dice score suitable for optimization.
            Else, return the (numerator, cardinality) by class and by sample
        """
        assert len(output.shape) > 2
        assert len(output.shape) == len(target.shape), 'output: must have N x C x d0 x ... x dn shape and ' \
                                                       'target: must have N x 1 x d0 x ... x dn shape'
        assert output.shape[0] == target.shape[0]
        assert target.shape[1] == 1, 'segmentation must have a single channel!'

        if self.normalization is not None:
            proba = self.normalization(output)
        else:
            proba = output

        # for each class (including background!), create a mask
        # so that class N is encoded as one hot at dimension 1
        if output.shape[1] > 1:
            encoded_target = one_hot(target[:, 0], proba.shape[1], dtype=proba.dtype)
        else:
            # the target is already in a one-hot encoded when class is 0-1
            assert target.max() <= 1, 'should be binary classification!'
            encoded_target = target.type(proba.dtype)
        
        intersection = proba * encoded_target
        indices_to_sum = tuple(range(2, len(proba.shape)))
        numerator = 2 * intersection.sum(indices_to_sum) + self.smooth
        if self.power is not None and self.power != 1.0:
            cardinality = proba ** self.power + encoded_target ** self.power
        else:
            cardinality = proba + encoded_target
        cardinality = cardinality.sum(indices_to_sum) + self.eps + self.smooth

        if not self.return_dice_by_class:
            # loss per samples (classes are averaged)
            average_loss_per_channel = (1 - numerator / cardinality)

            if self.per_class_weights is not None:
                # apply the per class weighting
                if self.per_class_weights.device != average_loss_per_channel.device:
                    self.per_class_weights = self.per_class_weights.to(average_loss_per_channel.device)
                average_loss_per_channel = average_loss_per_channel * self.per_class_weights.detach()

            if self.discard_background_loss:
                # the background portion most likely will
                # overwhelm the foreground so discard it
                average_loss_per_channel = average_loss_per_channel[:, 1:]

            average_loss_per_channel = average_loss_per_channel.mean(dim=1)
            return average_loss_per_channel
        else:
            return numerator, cardinality


class LossCrossEntropyCsiMulticlass(nn.Module):
    """
    Optimize a metric similar to ``Critical Success Index`` (CSI) on the cross-entropy

    A loss for heavily unbalanced data (order of magnitude more negative than positive)
    Calculate the cross-entropy and use only the loss using the TP, FP and FN. Loss from
    TN is simply discarded.
    """
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets, important_class=1):
        """
        Args:
            outputs: a N x C tensor with ``N`` the number of samples and ``C`` the number of classes
            targets: a ``N`` integral tensor
            important_class: the class to keep the cross-entropy loss even if classification is correct

        Returns:
            a ``N`` floating tensor representing the loss of each sample
        """
        ce = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
        classification = outputs.argmax(dim=1)
        w = ~ (classification == targets) | (classification == important_class)
        return ce * w.type(ce.dtype)


class LossFocalMulticlass(nn.Module):
    r"""
    This criterion is a implementation of Focal Loss, which is proposed in
    Focal Loss for Dense Object Detection, https://arxiv.org/pdf/1708.02002.pdf

        Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

    Args:
        alpha(1D Tensor, Variable) : the scalar factor for this criterion. One weight factor for each class.
        gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                               putting more focus on hard, misclassiﬁed examples
    """

    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        if alpha is None:
            self.alpha = None
        else:
            if isinstance(alpha, torch.Tensor):
                self.alpha = alpha
            else:
                assert isinstance(alpha, (list, np.ndarray))
                self.alpha = torch.from_numpy(np.asarray(alpha))
            assert len(alpha.shape) == 1
            assert alpha.shape[0] > 1

        self.gamma = gamma
        self.reduction = reduction
        assert reduction in (None, 'mean')

    def forward(self, outputs, targets):
        #assert len(outputs.shape) >= 3, 'must have NCX shape!'
        assert len(outputs.shape) == len(targets.shape), 'output: must have W x C x d0 x ... x dn shape and ' \
                                                         'target: must have W x 1 x d0 x ... x dn shape'
        assert targets.shape[1] == 1, '`C` must be size 1'
        targets = targets.squeeze(1)

        if self.alpha is not None:
            assert len(self.alpha) == outputs.shape[1], 'there must be one alpha weight by class!'
            if self.alpha.device != outputs.device:
                self.alpha = self.alpha.to(outputs.device)

        if outputs.shape[1] == 1:
            # binary cross entropy when we have a single class output
            ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets.unsqueeze(1).float(), reduction='none', weight=self.alpha)
        else:
            ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # for segmentation maps, make sure we average all values by sample
        if self.reduction == 'mean':
            axes = tuple(range(1, focal_loss.dim()))
            if len(axes) == 0:
                # we already have a 1D loss vector. if we do focal_loss.mean(dim=()),
                # it is the same as focal_loss.mean(), which we don't want!
                return focal_loss
            return focal_loss.mean(dim=axes)
        elif self.reduction is None:
            return focal_loss
        else:
            raise ValueError()


class LossTriplets(nn.Module):
    r"""
    Implement a triplet loss

    The goal of the triplet loss is to make sure that:

    - Two examples with the same label have their embeddings close together in the embedding space
    - Two examples with different labels have their embeddings far away.

    However, we don’t want to push the train embeddings of each label to collapse into very small clusters.
    The only requirement is that given two positive examples of the same class and one negative example,
    the negative should be farther away than the positive by some margin. This is very similar to the
    margin used in SVMs, and here we want the clusters of each class to be separated by the margin.

    The loss implements the following equation:

    \mathcal{L} = max(d(a, p) - d(a, n) + margin, 0)

    """
    def __init__(self, margin=1.0, distance=nn.PairwiseDistance(p=2)):
        """

        Args:
            margin: the margin to separate the positive from the negative
            distance: the distance to be used to compare (samples, positive_samples) and (samples, negative_samples)
        """
        super().__init__()
        self.distance = distance
        self.margin = margin

    def forward(self, samples, positive_samples, negative_samples):
        """
        Calculate the triplet loss

        Args:
            samples: the samples
            positive_samples: the samples that belong to the same group as `samples`
            negative_samples: the samples that belong to a different group than `samples`

        Returns:
            a 1D tensor (N) representing the loss per sample
        """
        assert samples.shape == positive_samples.shape
        assert samples.shape == negative_samples.shape

        nb_samples = len(samples)

        # make sure we have a nb_samples x C shape
        samples = samples.view((nb_samples, -1))
        positive_samples = positive_samples.view((nb_samples, -1))
        negative_samples = negative_samples.view((nb_samples, -1))

        d = self.distance(samples, positive_samples) - self.distance(samples, negative_samples) + self.margin
        d = torch.max(d, torch.zeros_like(d))
        return d


class LossBinaryF1(nn.Module):
    """
    The macro F1-score is non-differentiable. Instead use a surrogate that is differentiable
        and correlates well with the Macro F1 score by working on the class probabilities rather
        than the discrete classification.

    For example, if the ground truth is 1 and the model prediction is 0.8, we calculate it as 0.8 true
        positive and 0.2 false negative
    """
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, outputs, targets):
        assert len(outputs.shape) == len(targets.shape) + 1, 'output: must have W x C x d0 x ... x dn shape and ' \
                                                            'target: must have W x d0 x ... x dn shape'
        assert outputs.shape[1] == 2, 'only for binary classification!'

        y_true = one_hot(targets, 2).type(torch.float32)
        y_pred = F.softmax(outputs, dim=1)

        tp = (y_true * y_pred).sum(dim=0).type(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).type(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).type(torch.float32)

        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)

        f1 = 2 * (precision * recall) / (precision + recall + self.eps)
        f1 = f1.clamp(min=0.0, max=1.0)
        one_minus_f1 = 1 - f1.mean()

        # per design, we MUST return a per-sample loss. This is not valid for F1
        # so instead repeat the ``one_minus_f1`` to have correct dimension.
        return one_minus_f1.repeat(targets.shape[0])


class LossCenter(nn.Module):
    """
    Center loss, penalize the features falling further from the feature class center.

    In most of the available CNNs, the softmax loss function is used as the supervision
    signal to train the deep model. In order to enhance the discriminative power of the
    deeply learned features, this loss can be used as a new supervision signal. Specifically,
    the center loss simultaneously learns a center for deep features of each class and penalizes
    the distances between the deep features and their corresponding class centers.

    An implementation of center loss: Wen et al. A Discriminative Feature Learning Approach for Deep
    Face Recognition. ECCV 2016.

    Note:
        This loss *must* be part of a `parent` module or explicitly optimized by an optimizer. If not,
        the centers will not be modified.
    """
    def __init__(self, number_of_classes, number_of_features, alpha=1.0):
        """

        Args:
            number_of_classes: the (maximum) number of classes
            number_of_features: the (exact) number of features
            alpha: the loss will be scaled by ``alpha``
        """
        super().__init__()
        self.alpha = alpha

        # me MUST have a randomly initialized center to help with
        # convergence
        self.centers = nn.Parameter(torch.randn(number_of_classes, number_of_features))

    def forward(self, x, classes):
        """

        Args:
            x: the features, an arbitrary n-d tensor (N * C * ...). Features should ideally be in range [0..1]
            classes: a 1D integral tensor (N) representing the class of each ``x``

        Returns:
            a 1D tensor (N) representing the loss per sample
        """
        assert len(classes.shape) == 1, f'must be a 1D tensor. Got={classes.shape}'
        assert len(classes) == len(x), f'must have the same dim in input ({len(x)}) and classes ({len(classes)})!'
        flattened_x = x.view(x.shape[0], -1)
        criterion = torch.nn.MSELoss(reduction='none')
        losses = criterion(self.centers[classes], flattened_x)
        return self.alpha * losses.mean(dim=1)


class LossContrastive(torch.nn.Module):
    """
    Implementation of the contrastive loss.
    
    L(x0, x1, y) = 0.5 * (1 - y) * d(x0, x1)^2 + 0.5 * y * max(0, m - d(x0, x1))^2

    with y = 0 for samples x0 and x1 deemed dissimilar while y = 1 for similar samples. Dissimilar pairs
    contribute to the loss function only if their distance is within this radius ``m`` and minimize d(x0, x1)
    over the set of all similar pairs.

    See Dimensionality Reduction by Learning an Invariant Mapping, Raia Hadsell, Sumit Chopra, Yann LeCun, 2006.
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, x0, x1, same_target):
        """

        Args:
            x0: N-D tensor
            x1: N-D tensor
            same_target: ``0`` or ``1`` 1D tensor. ``1`` means the ``x0`` and ``x1`` belongs to the same class, while
                ``0`` means they are from a different class

        Returns:
            a 1D tensor (N) representing the loss per sample
        """
        nb_samples = len(x0)
        assert nb_samples == len(x1)
        assert nb_samples == len(same_target)
        assert len(same_target.shape) == 1
        assert same_target.shape[0] == nb_samples

        distances = F.pairwise_distance(x1, x0, p=2)
        distances_sqr = distances.pow(2)

        m_or_p = (1 + -1 * same_target).float()

        losses = 0.5 * (same_target.float() * distances_sqr +
                        m_or_p *
                        F.relu(self.margin - distances))
        return losses


def _total_variation_norm_2d(x, beta):
    assert len(x.shape) == 4, 'expeted N * C * H * W format!'
    assert x.shape[1] == 1, 'single channel only tested'
    row_grad = torch.mean(torch.abs((x[:, :, :-1, :] - x[:, :, 1:, :])).pow(beta))
    col_grad = torch.mean(torch.abs((x[:, :, :, :-1] - x[:, :, :, 1:])).pow(beta))
    return row_grad + col_grad


def _total_variation_norm_3d(x, beta):
    assert len(x.shape) == 5, 'expeted N * C * D * H * W format!'
    assert x.shape[1] == 1, 'single channel only tested'
    depth_grad = torch.mean(torch.abs((x[:, :, :-1, :, :] - x[:, :, 1:, :, :])).pow(beta))
    row_grad = torch.mean(torch.abs((x[:, :, :, :-1, :] - x[:, :, :, 1:, :])).pow(beta))
    col_grad = torch.mean(torch.abs((x[:, :, :, :, :-1] - x[:, :, :, :, 1:])).pow(beta))
    return row_grad + col_grad + depth_grad


def total_variation_norm(x, beta):
    """
    Calculate the total variation norm

    Args:
        x: a tensor with format (samples, components, dn, ..., d0)
        beta: the exponent

    Returns:
        a scalar
    """
    if len(x.shape) == 4:
        return _total_variation_norm_2d(x, beta)
    elif len(x.shape) == 5:
        return _total_variation_norm_3d(x, beta)
    else:
        raise NotImplemented()


class LossMsePacked(nn.Module):
    """
    Mean squared error loss with target packed as an integer (e.g., classification)

    The ``packed_target`` will be one hot encoded and the mean squared error is applied with the ``tensor``.
    """
    def __init__(self, reduction: Literal['mean', 'none'] = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, tensor, packed_target):
        """
        Args:
            tensor: a NxCx... tensor
            packed_target:  a Nx1x... tensor
        """
        assert len(tensor.shape) == len(packed_target.shape), '`tensor` must be encoded as NxCx... while' \
                                                              '`packed_target` must be encoded as Nx1x...'
        assert tensor.shape[0] == packed_target.shape[0]
        assert packed_target.shape[1] == 1, 'target MUST be a'
        assert tensor.shape[2:] == packed_target.shape[2:], '`tensor` and `packed_target` must have the same shape' \
                                                            '(except the N, C components)'

        nb_classes = tensor.shape[1]
        if nb_classes >= 2:
            assert packed_target.max() < nb_classes, f'error: target larger than the number ' \
                                                     f'of classes ({packed_target.max()} vs {nb_classes})'
            target = one_hot(packed_target.squeeze(1), nb_classes)
        else:
            # no need of unpacking when we have binary classifier
            assert nb_classes == 1
            target = packed_target.type(tensor.dtype)

        assert tensor.shape == target.shape
        loss = F.mse_loss(tensor, target, reduction='none')
        if self.reduction == 'none':
            pass
        elif self.reduction == 'mean':
            dims = torch.arange(1, len(loss.shape))
            loss = loss.mean(axis=tuple(dims))
        else:
            raise ValueError(f'reduction={self.reduction} not implemented!')
        return loss
