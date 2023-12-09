import importlib


def find_global_name(name):
    """
    Find a function or class from its name. If not found, raise a :class:`RuntimeError`

    Examples:
        >>> find_global_name('trw.utils.find_global_name')

    Args:
        name: a name with possibly namespaces

    Returns:
        an object
    """
    name_parts = name.split('.')
    if len(name_parts) > 1:
        module_name = '.'.join(name_parts[:-1])
        module = importlib.import_module(module_name)
        fn = getattr(module, name_parts[-1])
        if fn is None:
            raise RuntimeError(f'could not find function=`{name_parts[-1]}` in module={module_name}')
        return fn
    else:
        fn = globals().get(name)
        if fn is None:
            raise RuntimeError(f'cannot find name=`{name}`')
        return fn
