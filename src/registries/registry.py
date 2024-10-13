"""
inspired by "timm" package
"""
from typing import Dict, Callable
from ._utils import add_to_registry

MODEL_REGISTRY:Dict[str, Callable] = {}
MODEL_CONFIG_REGISTRY:Dict[str, Callable] = {}
DATASET_REGISTRY:Dict[str, Callable] = {}
DATAMODULE_REGISTRY:Dict[str, Callable] = {}

def create_register_factory(registry:Dict[str, Callable]) -> Callable:
    """Generates a decorator that registers an object in the given registry.

    Args:
        registry (Dict[str, Callable]): The registry to register objects in.

    Returns:
        Callable: Decorator wrapper function that registers the object in the registry.
    """    
    def decorator(name:str=None) -> Callable:
        """Register an object in the given registry, optionally with a custom name.

        Args:
            name (str, optional): The name under which to register the object. Defaults to None, meaning
                the object name will be used.

        Returns:
            Callable: Actual decorator function that registers the object in the registry.
        """        
        return lambda func: add_to_registry(func, registry, name)
    return decorator

# here are the actual functions that are used to model, configs, datasets and datamodules
# they are usually used as decorators to register the respective object, e.g.:
# @register_model("my_model")
# class MyModel(nn.Module):
#     ...
register_model = create_register_factory(MODEL_REGISTRY)
register_model_config = create_register_factory(MODEL_CONFIG_REGISTRY)
register_dataset = create_register_factory(DATASET_REGISTRY)
register_datamodule = create_register_factory(DATAMODULE_REGISTRY)
