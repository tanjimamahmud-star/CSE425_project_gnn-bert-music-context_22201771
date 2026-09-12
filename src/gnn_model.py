import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE, GATConv, global_mean_pool

class MusicGNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, model_type='graphsage'):
        """
        Task 2: GNN Encoder on Music Structure Graphs
        """
        super(MusicGNNEncoder, self).__init__()
        self.model_type = model_type
        
        if model_type == 'graphsage':
            self.gnn = GraphSAGE(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                out_channels=out_channels,
                dropout=0.2
            )
        elif model_type == 'gat':
            self.convs = nn.ModuleList()
            self.convs.append(GATConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_channels, hidden_channels))
            self.convs.append(GATConv(hidden_channels, out_channels))
        else:
            raise ValueError("Unsupported GNN model type. Choose 'graphsage' or 'gat'.")

    def forward(self, x, edge_index, batch):
        """
        Forward pass for GNN.
        Args:
            x: Node features (e.g., segment embeddings or chords)
            edge_index: Graph connectivity
            batch: Batch vector indicating which graph each node belongs to
        """
        if self.model_type == 'graphsage':
            node_embeddings = self.gnn(x, edge_index)
        else:
            h = x
            for i, conv in enumerate(self.convs):
                h = conv(h, edge_index)
                if i < len(self.convs) - 1:
                    h = F.elu(h)
                    h = F.dropout(h, p=0.2, training=self.training)
            node_embeddings = h

        # Graph Readout (Mean Pooling) as specified in Task 2
        g = global_mean_pool(node_embeddings, batch)
        
        return g, node_embeddings
