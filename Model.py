import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset,RandomSampler

class NNet(nn.Module):
    def __init__(self,input_n,NL,NN):
        super(NNet,self).__init__()
        self.input_layer=nn.Linear(input_n,NN)
        self.hidden_layer=nn.ModuleList([nn.Linear(NN,NN) for i in range(NL)])
        self.output_layer=nn.Linear(NN,1)

    def forward(self, x):
        L = torch.sigmoid(self.input_layer(x))*self.input_layer(x)
        for i, li in enumerate(self.hidden_layers):
            L = self.act(li(L))
        output = self.output_layer(L)

        return output