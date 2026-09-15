"""
loss function boilerplate

LossFunction is the main class, 
consisting of several component loss classes

All the classes have similar style
Input to forward is dictionaries of 
prediction and data from batch

"""

import torch
import torch.nn as nn

class LossFunction(nn.Module):
    def __init__(self):
        # construct the object
        
        super().__init__()

        self.weights = {}       # put component loss function weights

        self.loss1 = LossFunction1()
        self.loss2 = LossFunction2() 

        pass

    def update_weights(self, epoch):
        # update weights
        pass 

    def forward(self, output, batch):
        # call loss functions and combine

        l1 = self.loss1(output, batch)
        l2 = self.loss2(output, batch)

        loss = self.weights['l1'] * l1 + self.weights['l2'] * l2 


        loss_dict = {
            'loss' : loss,
            'l1' : l1,
            'l2' : l2
        }

        return loss_dict


class LossFunction1(nn.Module):
    def __init__(self):
        # construct the object
        super().__init__()
        pass 
        
    def forward(self, output, batch):
        # compute loss
        pass

class LossFunction2(nn.Module):
    def __init__(self):
        # construct the object
        super().__init__()
        pass 
        
    def forward(self, output, batch):
        # compute loss
        pass