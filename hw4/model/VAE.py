import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

''' reference
https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/Variational_autoencoder.py
'''


class VAE(nn.Module):
    def __init__(self, feat_size=(4096*3), hidden_size=400, bottleneck_size=20):
        super(VAE, self).__init__()
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size

        self.fc1 = nn.Linear(feat_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, bottleneck_size)
        self.fc22 = nn.Linear(hidden_size, bottleneck_size)
        self.fc3 = nn.Linear(bottleneck_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, feat_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)


    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar