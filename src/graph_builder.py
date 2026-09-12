import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data

def build_segment_graph(segment_features, tau=0.75):
    """
    Constructs a PyTorch Geometric graph from segmented audio features.
    
    Args:
        segment_features: numpy array of shape (num_segments, feature_dim, time_frames)
        tau: threshold for cosine similarity edges
        
    Returns:
        torch_geometric.data.Data object
    """
    num_segments = segment_features.shape[0]
    
    # Node features: we flatten (or pool) the time frames to get a single vector per segment
    # e.g., (num_segments, feature_dim * time_frames) -> you might want to use mean pooling over time instead
    # Here we use mean pooling over the time axis to keep feature dim reasonable: (num_segments, feature_dim)
    pooled_features = np.mean(segment_features, axis=-1) 
    
    # 1. Temporal Adjacency Edges (segment i is connected to i+1)
    edge_list = []
    for i in range(num_segments - 1):
        # Add bidirectional temporal edges
        edge_list.append([i, i + 1])
        edge_list.append([i + 1, i])
        
    # 2. Cosine Similarity Edges
    # Compute pairwise similarity between all segments
    sim_matrix = cosine_similarity(pooled_features)
    
    for i in range(num_segments):
        for j in range(i + 1, num_segments):
            if sim_matrix[i, j] > tau:
                edge_list.append([i, j])
                edge_list.append([j, i])
                
    # Remove duplicates
    if len(edge_list) > 0:
        edge_list = np.unique(edge_list, axis=0)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
    # Convert node features to torch tensor
    x = torch.tensor(pooled_features, dtype=torch.float)
    
    # Create PyTorch Geometric Data object
    graph_data = Data(x=x, edge_index=edge_index)
    
    return graph_data

def build_chord_graph(chord_sequence):
    """
    Optional alternative task: Chord-transition graph.
    Nodes = unique chords, Edges = observed transitions weighted by count.
    """
    unique_chords = list(set(chord_sequence))
    chord_to_idx = {chord: i for i, chord in enumerate(unique_chords)}
    
    num_nodes = len(unique_chords)
    adj_matrix = np.zeros((num_nodes, num_nodes))
    
    # Count transitions
    for i in range(len(chord_sequence) - 1):
        src = chord_to_idx[chord_sequence[i]]
        dst = chord_to_idx[chord_sequence[i+1]]
        adj_matrix[src, dst] += 1
        
    # Extract edges
    src_nodes, dst_nodes = np.nonzero(adj_matrix)
    edge_weights = adj_matrix[src_nodes, dst_nodes]
    
    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float)
    
    # One-hot encode node features (or use an embedding later)
    x = torch.eye(num_nodes, dtype=torch.float)
    
    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return graph_data
