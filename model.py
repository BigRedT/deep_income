import copy
import torch
import torch.nn as nn


class IncomeClassifierConstants():
    def __init__(self):
        self.in_dim = 105
        self.hidden_dim = 105
        self.num_hidden_blocks = 2
        self.drop_prob = 0.2
        self.out_dim =2


class IncomeClassifier(nn.Module):
    def __init__(self,const):
        super().__init__()
        self.const = copy.deepcopy(const)

        if self.const.num_hidden_blocks==0:
            self.layers = nn.Linear(self.const.in_dim,self.const.out_dim)

        else:
            layers = []
            
            # Add input layers
            layers.append(self.input_block())
            
            # Add hidden layers
            for i in range(self.const.num_hidden_blocks):
                layers.append(self.hidden_block())
            
            # Add output layers
            layers.append(self.output_block())
            
            self.layers = nn.Sequential(*layers)

        self.softmax_layer = nn.Softmax(1)


    def input_block(self):
        return nn.Sequential(
            nn.Linear(self.const.in_dim,self.const.hidden_dim),
            nn.BatchNorm1d(self.const.hidden_dim),
            nn.Dropout(self.const.drop_prob),
            nn.Sigmoid())

    def hidden_block(self):
        return nn.Sequential(
            nn.Linear(self.const.hidden_dim,self.const.hidden_dim),
            nn.BatchNorm1d(self.const.hidden_dim),
            nn.Dropout(self.const.drop_prob),
            nn.Sigmoid())

    def output_block(self):
        return nn.Linear(self.const.hidden_dim,self.const.out_dim)

    def forward(self,x):
        logits = self.layers(x)
        probs = self.softmax_layer(logits)
        return logits, probs


def test_classifier():
    const = IncomeClassifierConstants()
    const.hidden_dim = 50
    model = IncomeClassifier(const)
    x = torch.rand([5,105])
    logits, probs = model(x)
    import pdb; pdb.set_trace()


if __name__=='__main__':
    test_classifier()
