"""
metrics boilerplate

compute_metrics_torch is the main function, 
consisting of several component function

All the classes have similar style
Input to forward is dictionaries of 
prediction and data from batch

"""


import torch

@torch.no_grad()
def compute_metrics_torch(prediction, batch):
    
    metrics_dict = {}

    return metrics


# put other functions here