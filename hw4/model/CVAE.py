import torch
from torch import nn
from torch.autograd import Variable

''' reference
https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/Variational_autoencoder.py
'''


class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        
        # output shape : ((X - K + 2P) / S) + 1 | for all X in integer, X indice Width, Height
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False) # (b, 16, 64, 64)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4,
                               stride=2, padding=1, bias=False) # (b, 32, 32, 32)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3,
                               stride=1, padding=1, bias=False) # (b, 32, 32, 32)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=4,
                               stride=2, padding=1, bias=False) # (b, 16, 16, 16)
        self.bn4 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(16 * 16 * 16, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc21 = nn.Linear(1024, 1024)
        self.fc22 = nn.Linear(1024, 1024)

        # Decoder
        self.fc3 = nn.Linear(1024, 1024)
        self.fc_bn3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 16 * 16 * 16)
        self.fc_bn4 = nn.BatchNorm1d(16 * 16 * 16)

        self.conv5 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1,
                                        output_padding=1, bias=False) # (b, 32, 16, 16)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1,
                                        padding=1, bias=False) # (b, 32, 32, 32)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1,
                                        output_padding=1, bias=False) # (b, 16, 64, 64)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1,
                                        padding=1, bias=False) # (b, 3, 64, 64)

        self.relu = nn.ReLU()

    def encode(self, x):
        conv1 = self.conv1(x)
        conv1 = self.bn1(conv1)
        conv1 = self.relu(conv1)
        
        conv2 = self.relu(self.bn2(self.conv2(conv1)))
        conv3 = self.relu(self.bn3(self.conv3(conv2)))
        conv4 = self.relu(self.bn4(self.conv4(conv3))).view(-1, 16 * 16 * 16)

        fc1 = self.relu(self.fc_bn1(self.fc1(conv4)))
        return self.fc21(fc1), self.fc22(fc1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3))).view(-1, 16, 16, 16)

        conv5 = self.relu(self.bn5(self.conv5(fc4)))
        # print('conv5 : ', conv5.shape)
        conv6 = self.relu(self.bn6(self.conv6(conv5)))
        # print('conv6 : ', conv6.shape)
        conv7 = self.relu(self.bn7(self.conv7(conv6)))
        # print('conv7 : ',conv7.shape)
        return self.conv8(conv7).view(-1, 3, 64, 64)

    def forward(self, x):
        # print(x.shape)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

