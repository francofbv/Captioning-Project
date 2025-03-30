from .cnn import CNN
from .transformer import Transformer

def get_model(model_name, num_classes, **kwargs):
    if model_name == 'CNN':
        return CNN(num_classes=num_classes, **kwargs)
    elif model_name == 'Transformer':
        return Transformer(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Model {model_name} not found")