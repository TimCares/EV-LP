from typing import Dict

def add_to_registry(obj:object, registry:Dict[str, object], name:str=None) -> None:
    """Add an object to a registry.

    Args:
        obj (object): The object to add to the registry.
        registry (Dict[str, object]): The registry to add the object to.
        name (str, optional): The name under which to add the object. Defaults to None, meaning
            the object name will be used.

    Raises:
        ValueError: If the object is already registered.

    Returns:
        Callable: Unmodified input object
    """   
    model_name = name or obj.__name__ # use obj name if no name is provided
    if model_name in registry:
        raise ValueError(f"Model '{model_name}' is already registered.")
    registry[model_name] = obj
    return obj