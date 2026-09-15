"""
model class boilerplate

Entire model should be put in a single file
the components as different classes in the same file

All the classes should have predefined default params

The main model class takes the batch as dictionary input
The output will be another dictionary
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        # construct the object
        
        super().__init__()        

        pass

    
    def forward(self, batch):
        
        # do stuff here 

        return {}



## model component classes go here