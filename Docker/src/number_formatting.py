from typing import Union


def bytes2human(n: Union[int, float]) -> str:
    """
    Format large number of bytes into readable string for a human

    Examples:
        >>> bytes2human(10000)
        '9.8K'

        >>> bytes2human(100001221)
        '95.4M'

    """
    # http://code.activestate.com/recipes/578019
    # >>> bytes2human(10000)
    # '9.8K'
    # >>> bytes2human(100001221)
    # '95.4M'
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10
    for s in reversed(symbols):
        if n >= prefix[s]:
            value = float(n) / prefix[s]
            return '%.1f%s' % (value, s)
    return "%.2f" % n


def number2human(n: Union[int, float]) -> str:
    """
    Format large number into readable string for a human

    Examples:
        >>> number2human(1000)
        '1.0K'

        >>> number2human(1200000)
        '1.2M'

    """
    # http://code.activestate.com/recipes/578019
    # >>> bytes2human(10000)
    # '9.8K'
    # >>> bytes2human(100001221)
    # '95.4M'
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = (10 ** 3) ** (i + 1)
    for s in reversed(symbols):
        if n >= prefix[s]:
            value = float(n) / prefix[s]
            return '%.1f%s' % (value, s)
    return "%.2f" % n
