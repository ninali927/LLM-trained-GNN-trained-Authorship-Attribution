import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv


class GCN(torch.nn.Module):
    """
    Graph Convolutional Network.
    Supports optional edge weights.
    """

    def __init__(self, in_channels, all_hidden_channels, out_channels, activation, aggr='sum'):
        super(GCN, self).__init__()

        assert activation in ['sigmoid', 'softmax', 'relu', None], 'Invalid activation function.'

        self.name = 'GCN'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = torch.nn.ModuleList()

        widths = [in_channels] + all_hidden_channels + [out_channels]

        for i in range(len(widths) - 1):
            self.layers.append(GCNConv(widths[i], widths[i + 1]))

        if activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = torch.nn.Softmax(dim=1)
        elif activation == 'relu':
            self.activation = torch.nn.ReLU()
        else:
            self.activation = None

    def forward(self, x, edge_index, weights=None):
        """
        Forward pass.

        Inputs:
        - x: node feature matrix
        - edge_index: graph connectivity
        - weights: optional edge weights
        """
        if weights is None:
            x = self.layers[0](x, edge_index)
            for i in range(1, len(self.layers)):
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.layers[i](x, edge_index)
        else:
            x = self.layers[0](x, edge_index, weights)
            for i in range(1, len(self.layers)):
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.layers[i](x, edge_index, weights)

        if self.activation is not None:
            x = self.activation(x)

        return x


class SAGE(torch.nn.Module):
    """
    GraphSAGE model.
    This version does not use edge weights directly.
    """

    def __init__(self, in_channels, all_hidden_channels, out_channels, activation, aggr='sum'):
        super(SAGE, self).__init__()

        assert activation in ['sigmoid', 'softmax', 'relu', None], 'Invalid activation function.'

        self.name = 'SAGE'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = torch.nn.ModuleList()

        widths = [in_channels] + all_hidden_channels + [out_channels]

        for i in range(len(widths) - 1):
            self.layers.append(SAGEConv(widths[i], widths[i + 1], aggr=aggr))

        if activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = torch.nn.Softmax(dim=1)
        elif activation == 'relu':
            self.activation = torch.nn.ReLU()
        else:
            self.activation = None

    def forward(self, x, edge_index, weights=None):
        """
        Forward pass.

        Inputs:
        - x: node feature matrix
        - edge_index: graph connectivity
        - weights: unused, kept for interface consistency
        """
        x = self.layers[0](x, edge_index)

        for i in range(1, len(self.layers)):
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.layers[i](x, edge_index)

        if self.activation is not None:
            x = self.activation(x)

        return x


class GIN(torch.nn.Module):
    """
    Graph Isomorphism Network.
    This version does not use edge weights directly.
    """

    def __init__(self, in_channels, all_hidden_channels, out_channels, activation, aggr='sum'):
        super(GIN, self).__init__()

        assert activation in ['sigmoid', 'softmax', 'relu', None], 'Invalid activation function.'

        self.name = 'GIN'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = torch.nn.ModuleList()

        widths = [in_channels] + all_hidden_channels + [out_channels]

        for i in range(len(widths) - 1):
            linear_layer = torch.nn.Linear(widths[i], widths[i + 1])
            self.layers.append(GINConv(linear_layer, aggr=aggr))

        if activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = torch.nn.Softmax(dim=1)
        elif activation == 'relu':
            self.activation = torch.nn.ReLU()
        else:
            self.activation = None

    def forward(self, x, edge_index, weights=None):
        """
        Forward pass.

        Inputs:
        - x: node feature matrix
        - edge_index: graph connectivity
        - weights: unused, kept for interface consistency
        """
        x = self.layers[0](x, edge_index)

        for i in range(1, len(self.layers)):
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.layers[i](x, edge_index)

        if self.activation is not None:
            x = self.activation(x)

        return x


class GAT(torch.nn.Module):
    """
    Graph Attention Network.
    This version does not use edge weights directly.
    """

    def __init__(self, in_channels, all_hidden_channels, out_channels, activation, aggr='sum'):
        super(GAT, self).__init__()

        assert activation in ['sigmoid', 'softmax', 'relu', None], 'Invalid activation function.'

        self.name = 'GAT'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = torch.nn.ModuleList()

        widths = [in_channels] + all_hidden_channels + [out_channels]

        for i in range(len(widths) - 1):
            self.layers.append(GATConv(widths[i], widths[i + 1]))

        if activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = torch.nn.Softmax(dim=1)
        elif activation == 'relu':
            self.activation = torch.nn.ReLU()
        else:
            self.activation = None

    def forward(self, x, edge_index, weights=None):
        """
        Forward pass.

        Inputs:
        - x: node feature matrix
        - edge_index: graph connectivity
        - weights: unused, kept for interface consistency
        """
        x = self.layers[0](x, edge_index)

        for i in range(1, len(self.layers)):
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.layers[i](x, edge_index)

        if self.activation is not None:
            x = self.activation(x)

        return x
    
    
    