import torch
import torch.nn as nn

class FCs(nn.Module):
    def __init__(self, in_ch, out_ch, h_ch=1000):
        super(FCs, self).__init__()
        self.main = nn.Sequential(
                nn.Linear(in_ch, h_ch),
                nn.ReLU(),
                nn.Linear(h_ch, 100),
                nn.ReLU(),
                nn.Linear(100, out_ch),
            )

    def forward(self, x):
        return self.main(x)

class NewFCs(nn.Module):
    def __init__(self, in_ch, out_ch, h_ch=1000):
        super(NewFCs, self).__init__()
        self.main = nn.Sequential(
                nn.Linear(in_ch, h_ch),
                nn.ReLU(),
                nn.Linear(h_ch, 100),
                nn.ReLU(),
                nn.Linear(100, out_ch),
            )

    def forward(self, x):
        return self.main(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 2e-4)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.0, 2e-4)
        m.bias.data.fill_(1e-4)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 2e-4)
        m.bias.data.fill_(1e-4)

class decoder_fl(nn.Module):
    def __init__(self, dim, nc=20):
        super(decoder_fl, self).__init__()
        self.dim = dim
        self.fc = NewFCs(nc, dim)
        self.apply(weights_init)
    
    def forward(self, x):
        return self.fc(x).view(-1, self.dim)

class discriminator_fl(nn.Module):
    def __init__(self, dim=1, nc=50):
        super(discriminator_fl, self).__init__()
        self.dim = dim
        self.fc = FCs(nc, dim)
        self.apply(weights_init)

    def forward(self, x):
        h1 = self.fc(x)
        h2 = torch.sigmoid(h1)
        return h2.view(-1, self.dim)

class discriminator_wgan(nn.Module):
    def __init__(self, dim=1, nc=50):
        super(discriminator_wgan, self).__init__()
        self.dim = dim
        self.fc = FCs(nc, dim)
        self.bn = nn.BatchNorm1d(1)
        self.apply(weights_init)

    def forward(self, x):
        h1 = self.bn(self.fc(x))
        h2 = torch.sigmoid(h1)
        return h2.view(-1, self.dim)
