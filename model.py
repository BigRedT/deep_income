import torch
import torch.nn as nn


class IncomeClassifier(nn.Module):
    def __init__(self,in_dim,hidden_dim=None):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = 2 
        # self.layers = nn.Sequential(
        #     nn.Linear(self.in_dim,self.out_dim),
        # )
        self.layers = nn.Sequential(
            nn.Linear(self.in_dim,self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim,self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim,self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim,self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim,self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim,self.out_dim)
        )
        self.softmax_layer = nn.Softmax(1)

    def forward(self,x):
        logits = self.layers(x)
        probs = self.softmax_layer(logits)
        return logits, probs


def test_classifier():
    model = IncomeClassifier(105)
    x = torch.rand([5,105])
    logits, probs = model(x)
    import pdb; pdb.set_trace()


if __name__=='__main__':
    test_classifier()
