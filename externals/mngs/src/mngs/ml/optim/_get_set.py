# import torch.nn as nn
import torch.optim as optim
from .Ranger_Deep_Learning_Optimizer.ranger.ranger2020 import Ranger


# def set_an_optim(models, optim_str, lr):
def set(models, optim_str, lr):    
    """Sets an optimizer to models"""
    if not isinstance(models, list):
        models = [models]
    learnable_params = []
    for m in models:
        learnable_params += list(m.parameters())
    # optim = mngs.ml.switch_optim(optim_str)
    optim = get(optim_str)    
    return optim(learnable_params, lr)

def get(optim_str):
    optims_dict = {
        "adam": optim.Adam,
        "ranger": Ranger,
        "rmsprop": optim.RMSprop,
        "sgd": optim.SGD
        }
    return optims_dict[optim_str]
