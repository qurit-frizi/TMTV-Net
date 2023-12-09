def safe_filename(filename: str, replace_with: str = '_') -> str:
    """
    Replace problematic characters (e.g., '/' or '#') considering Windows/Linux based OSes
    Args:
        replace_with: the character to be replaced with
        filename: a filename

    Returns:
        a string that can be used as filename
    """
    return filename.replace('/', replace_with).replace('\\', replace_with).replace('#', replace_with)
