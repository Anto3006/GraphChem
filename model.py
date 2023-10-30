import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import mask_select,index_to_mask

class GNN(nn.Module):

    def __init__(self,node_features,edge_features,hidden_channels,heads):
        super().__init__()
        torch.manual_seed(3006)
        self.conv1 = GATConv(node_features,hidden_channels,edge_dim=edge_features,heads=heads,dropout=0.4)
        self.conv2 = GATConv(heads*hidden_channels,hidden_channels,edge_dim=edge_features,heads=heads,dropout=0.4)
        self.conv3 = GATConv(heads*hidden_channels,hidden_channels,edge_dim=edge_features,heads=heads,dropout=0.4)
        self.dropout0 = nn.Dropout1d(0.25)
        self.linear1 = Linear(heads*hidden_channels,hidden_channels)
        self.dropout1 = nn.Dropout1d(0.5)
        self.linear2 = Linear(hidden_channels,hidden_channels)
        self.dropout2 = nn.Dropout1d(0.5)
        self.linear3 = Linear(hidden_channels,1)
    
    def forward(self,x,edge_index,edge_features, batch, is_mask=False,masked_node=None):
        x = self.conv1(x,edge_index,edge_attr=edge_features)
        x = x.relu()
        x = self.conv2(x,edge_index,edge_attr=edge_features)
        x = x.relu()
        x = self.conv3(x,edge_index,edge_attr=edge_features)
        x = x.relu()
        if is_mask:
            if masked_node != None:
                mask = 1*index_to_mask(torch.tensor([i for i in range(len(x)) if i != masked_node]),size=len(x))
                mask = mask.view(-1,1).to("cuda")
                x = torch.mul(mask,x)
        x = global_add_pool(x,batch)
        #x = self.dropout0(x)
        x = self.linear1(x)
        x = x.relu()
        #x = self.dropout1(x)
        x = self.linear2(x)
        x = x.relu()
        #x = self.dropout2(x)
        x = self.linear3(x)
        x = x.relu()
        return x
    
