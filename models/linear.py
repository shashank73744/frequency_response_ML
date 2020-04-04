import torch
import torch.nn as nn
import torch.nn.functional as F

class LIN(torch.nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim,):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(LIN, self).__init__()
        self.linear1 = torch.nn.Linear(in_dim, out_dim)
        # self.linear2 = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear1(x)
        return y_pred