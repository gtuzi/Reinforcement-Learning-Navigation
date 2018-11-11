import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.channels = [state_size] + [64, 128, 256, 128, 64, action_size]
        self.layers = nn.ModuleList([nn.Linear(self.channels[i], self.channels[i+1])
                       for i in range(len(self.channels) - 1)])
        self.bn = nn.ModuleList([nn.BatchNorm1d(self.channels[i+1]) for i in range(len(self.layers) - 1)])

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if (i + 1) < len(self.layers): # nonlinearity applied to hidden layers
                # Batch normalize
                x = self.bn[i](x)
                x = F.leaky_relu(x, negative_slope=0.1)

        # Return logits
        return x




class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Feature backend
        self.feat_channels = [state_size] + [64, 128, 256, 128, 64]
        self.feat_layers = nn.ModuleList([nn.Linear(self.feat_channels[i], self.feat_channels[i + 1]) for i in range(len(self.feat_channels) - 1)])
        self.feat_bn = nn.ModuleList([nn.BatchNorm1d(self.feat_channels[i+1]) for i in range(len(self.feat_layers))])

        # Advantage stream
        self.adv_channels = [self.feat_channels[-1], action_size]
        self.adv_layers = nn.ModuleList([nn.Linear(self.adv_channels[i], self.adv_channels[i + 1]) for i in range(len(self.adv_channels) - 1)])
        # self.adv_bn = nn.ModuleList([nn.BatchNorm1d(self.adv_channels[i+1]) for i in range(len(self.adv_layers))])

        # Value stream
        self.val_channels = [self.feat_channels[-1], 1]
        self.val_layers = nn.ModuleList([nn.Linear(self.val_channels[i], self.val_channels[i+1]) for i in range(len(self.val_channels) - 1)])
        # self.val_bn = nn.ModuleList([nn.BatchNorm1d(self.val_channels[i+1]) for i in range(len(self.val_layers))])

        pass


    def forward(self, x):

        # Feed the network through the features
        for i, layer in enumerate(self.feat_layers):
            x = layer(x)
            x = F.leaky_relu(x, negative_slope=0.1)
            x = self.feat_bn[i](x)

        val_x = x
        adv_x = x
        batch_size = int(x.size(0))

        # Value stream
        for i, layer in enumerate(self.val_layers):
            val_x = layer(val_x)
            # val_x = F.leaky_relu(val_x, negative_slope=0.1)
            # val_x = self.val_bn[i](val_x)


        # Advantage stream
        for i, layer in enumerate(self.adv_layers):
            adv_x = layer(adv_x)
            # Batch normalize
            # adv_x = F.leaky_relu(adv_x, negative_slope=0.1)
            # adv_x = self.adv_bn[i](adv_x)


        # The duel !!
        q = val_x + (adv_x - torch.mean(adv_x, dim=-1).view(batch_size, 1))
        # q = val_x + (adv_x - torch.max(adv_x, dim=-1)[0][..., None])

        return q





class Conv2DLeakyReLUBN(nn.Module):
    def __init__(self, nci, nco, kernel=3, stride=1, padding=0):
        super(Conv2DLeakyReLUBN, self).__init__()
        self.conv = nn.Conv2d(nci, nco, kernel, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(nco)
        self.act = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.bn(x)
        return x

class LinearLeakyReLUBN(nn.Module):
    def __init__(self, nci, nco):
        super(LinearLeakyReLUBN, self).__init__()
        self.lin = nn.Linear(nci, nco)
        self.bn = nn.BatchNorm1d(nco)
        self.act = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        x = self.lin(x)
        x = self.act(x)
        x = self.bn(x)
        return x


class LinearBN(nn.Module):
    def __init__(self, nci, nco):
        super(LinearBN, self).__init__()
        self.lin = nn.Linear(nci, nco)
        self.bn = nn.BatchNorm1d(nco)

    def forward(self, x):
        x = self.lin(x)
        x = self.bn(x)
        return x


class VizQNet(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(VizQNet, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.feats = nn.ModuleList([
            Conv2DLeakyReLUBN(state_size, 32, 8, 4, 0),    # 20
            Conv2DLeakyReLUBN(32, 64, 4, 2, 0),            # 9
            Conv2DLeakyReLUBN(64, 128, 3, 2, 0),           # 4
        ])

        self.qest = nn.ModuleList([
            LinearLeakyReLUBN(128*4*4, 512),
            nn.Linear(512, action_size)
        ])


    def forward(self, x):
        # Features
        for l in self.feats:
            x = l(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Estimator
        for l in self.qest:
            x = l(x)

        return x



class DuelingVizQNet(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(DuelingVizQNet, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.feats = nn.ModuleList([
            Conv2DLeakyReLUBN(state_size, 32, 8, 4, 0),  # 20
            Conv2DLeakyReLUBN(32, 64, 4, 2, 0),  # 9
            Conv2DLeakyReLUBN(64, 128, 3, 2, 0),  # 4
        ])


        self.adv_stream = nn.ModuleList([LinearLeakyReLUBN(128*4*4, 512),
                                         nn.Linear(512, action_size)])
        self.val_stream = nn.Linear(128*4*4, 1)



    def forward(self, x):
        # Features
        for l in self.feats:
            x = l(x)

        # Flatten
        x = x.view(x.size(0), -1)
        adv_x = x
        val_x = x
        batch_size = x.size(0)

        # Estimator
        for l in self.adv_stream:
            adv_x = l(adv_x)

        val_x = self.val_stream(val_x)

        # The duel !!
        q = val_x + (adv_x - torch.mean(adv_x, dim=-1).view(batch_size, 1))

        return q