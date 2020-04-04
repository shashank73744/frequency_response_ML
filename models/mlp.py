import math
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,in_dim, hidden_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.activation = nn.ReLU6()

        layers = []
        layers.append(nn.Linear(in_dim,hidden_dim))
        layers.append(nn.ReLU6())
        layers.append(nn.Linear(hidden_dim,out_dim))
        self.model = nn.Sequential(*layers)

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        out = self.model(x)
        return out
