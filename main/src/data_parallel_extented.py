import torch
import torch.nn as nn
from outputs import Output


def gather_extended(outputs, target_device, dim=0):
    r"""
    Gathers tensors from different GPUs on a specified device
      (-1 means the CPU).

    This is an extended version of `` compared to pytorch to support :class:`Output`
    """

    from torch.nn.parallel._functions import Gather

    def gather_map(outputs):
        # original + modifications of pytorch `gather_map` function
        out = outputs[0]
        if isinstance(out, torch.Tensor):
            return Gather.apply(target_device, dim, *outputs)
        if out is None:
            return None

        #
        # Torch 1.3 modification
        #
        if isinstance(out, Output):
            # TODO need to be extensible! create a output.gather function
            # TODO merge metrics too!
            outputs_t = [o.output for o in outputs]
            out.output = gather_map(outputs_t)

            if hasattr(out, 'output_truth'):
                output_truth = [o.output_truth for o in outputs]
                out.output_truth = gather_map(output_truth)
            return out
        #
        # end modification
        #

        if isinstance(out, dict):
            if not all((len(out) == len(d) for d in outputs)):
                raise ValueError('All dicts must have the same number of keys')
            return type(out)(((k, gather_map([d[k] for d in outputs]))
                              for k in out))
        return type(out)(map(gather_map, zip(*outputs)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        res = gather_map(outputs)
    finally:
        gather_map = None
    return res


class DataParallelExtended(nn.DataParallel):
    """
    Customized version of :class:`torch.nn.DataParallel` to support model with
    complex outputs such as :class:`Output`
    """
    def __init__(self, *arg, **argv):
        super().__init__(*arg, **argv)

    def gather(self, outputs, output_device):
        return gather_extended(outputs, output_device, dim=self.dim)
