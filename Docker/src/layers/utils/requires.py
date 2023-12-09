import torch


def torch_requires(min_version: str, silent_fail: bool = False):
    """
    Requires a minimum pytorch version.

    If the version is not satisfy, the requested function will throw a :class:`RuntimeError` when called
    (but not when declared) if `silent_fail` is `False`. Otherwise, the function is not called and `None`
    is returned.

    Args:
        min_version: a version represented as string
        silent_fail: if True, the decorated function will not be executed and return `None`. If `False`,
            throw a :class:`RuntimeError`
    """
    def version_error(*args, **kwargs):
        raise RuntimeError(f'Requires pytorch version >= {min_version}')

    def return_none(*args, **kwargs):
        return None

    def inner(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        from packaging.version import Version
        current_version = Version(torch.__version__)

        target_version = Version(min_version)
        if current_version >= target_version:
            return wrapper
        else:
            if silent_fail:
                return return_none
            else:
                return version_error

    return inner
