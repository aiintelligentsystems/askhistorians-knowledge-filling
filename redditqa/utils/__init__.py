def prefix_dict(d: dict, prefix: str, sep: str = "_"):
    return {f"{prefix}{sep}{k}": v for k, v in d.items()}
