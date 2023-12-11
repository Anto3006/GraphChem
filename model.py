import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_add_pool, SAGPooling
from torch_geometric.utils import mask_select,index_to_mask
from torch_geometric.nn import BatchNorm


class GNN(nn.Module):

    def __init__(self,conv_layer_count,conv_layer_dropout,fc_layer_count,fc_layer_droupout,node_features, \
                 edge_features,hidden_channels,heads,skip_connection=False,graph_feature_count=0):
        super().__init__()
        torch.manual_seed(3006)
        self.skip_connection = skip_connection
        self.conv_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        in_features = node_features
        out_features = hidden_channels
        self.l = Linear(in_features,out_features)
        in_features = out_features
        for count in range(conv_layer_count):
            self.batch_norm_layers.append(BatchNorm(in_features))
            self.conv_layers.append(GATConv(in_features,out_features,edge_dim=edge_features,heads=heads,dropout=conv_layer_dropout[count],concat=False))
        in_features += graph_feature_count
        for count in range(fc_layer_count):
            if count == fc_layer_count-1:
                out_features = 1
            if fc_layer_droupout[count] > 0:
                self.fc_layers.append(nn.Dropout1d(fc_layer_droupout[count]))
            self.fc_layers.append(Linear(in_features,out_features))
            in_features = out_features
    
    def forward(self,x,edge_index,edge_features, batch,is_mask=False,masked_node=None,graph_features=None):
        x = self.l(x)
        x = x.relu()
        x_skip = None
        if self.skip_connection:
            x_skip = torch.clone(x)
        for batch_layer,conv_layer in zip(self.batch_norm_layers,self.conv_layers):
            x = batch_layer(x)
            x = conv_layer(x,edge_index,edge_attr=edge_features)
            x = x.relu()
            if self.skip_connection:
                x = x + x_skip
                x_skip = torch.clone(x)
        if is_mask:
            if masked_node != None:
                mask = 1*index_to_mask(torch.tensor([i for i in range(len(x)) if i not in masked_node]),size=len(x))
                mask = mask.view(-1,1).to("cuda")
                x = torch.mul(mask,x)
        x = global_add_pool(x,batch)
        if graph_features != None:
            x = torch.cat([x,graph_features])
        for layer in self.fc_layers:
            x = layer(x)
            x = x.relu()
        return x
    
