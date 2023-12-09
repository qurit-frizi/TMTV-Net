import importlib


class _LazyRaise:
    def __init__(self, msg):
        self.msg = msg

    def __getattr__(self, name):
        raise RuntimeError(self.msg)

    def __call__(self, *args, **kwargs):
        raise RuntimeError(self.msg)


def optional_import(module_name: str, additional_error_message=''):
    """
    Optional module import.

    Raise an error only when a module is being used

    Args:
        module_name: the name of the module to import
        additional_error_message: add a custom error message

    Returns:
        the module

    Examples:
        >>> nn = optional_import('torch.nn')
        >>> print(nn.ReLU)
    """
    try:
        m = importlib.import_module(module_name)
        if m is not None:
            return m
    except:
        return _LazyRaise(f'optional module cannot be imported `{module_name}`. '
                          f'To use this functionality, this module must be installed! {additional_error_message}')
