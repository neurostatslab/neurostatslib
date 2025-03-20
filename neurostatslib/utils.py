def flatten(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            yield from flatten({key + "/" + k: v for k, v in value.items()})
        else:
            yield key, value
