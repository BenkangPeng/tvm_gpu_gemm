def dump(content: str, path: str, fname: str):
    """
    Dump code to file
    Args:
        content: str, the content to dump
        path: str, the path to dump
        fname: str, the name of the file
    """
    import os
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, fname), "w") as f:
        f.write(content)