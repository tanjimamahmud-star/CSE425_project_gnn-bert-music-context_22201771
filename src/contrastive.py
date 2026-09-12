import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveDualEncoder(nn.Module):
    def __init__(self, gnn_encoder, bert_encoder, d_model=768, tau=0.07):
        """
        Task 4: Cross-Modal MusicCaps Alignment using Contrastive Learning (InfoNCE)
        """
        super(ContrastiveDualEncoder, self).__init__()
        self.gnn = gnn_encoder
        self.bert = bert_encoder
        self.tau = tau
        
        # Projection heads to bring both to the same embedding space d_model
        gnn_out_dim = self.gnn.gnn.out_channels if hasattr(self.gnn, 'gnn') else self.gnn.convs[-1].out_channels
        self.gnn_proj = nn.Linear(gnn_out_dim, d_model)
        self.text_proj = nn.Linear(self.bert.hidden_size, d_model)

    def forward(self, x_graph, edge_index, batch, input_ids, attention_mask):
        # 1. GNN embeddings for audio graphs
        g, _ = self.gnn(x_graph, edge_index, batch)
        g_emb = self.gnn_proj(g)
        
        # 2. BERT embeddings for text captions
        _, t_cls = self.bert(input_ids, attention_mask)
        t_emb = self.text_proj(t_cls)
        
        # 3. L2 Normalize representations (crucial for cosine similarity in InfoNCE)
        g_emb = F.normalize(g_emb, p=2, dim=1)
        t_emb = F.normalize(t_emb, p=2, dim=1)
        
        return g_emb, t_emb

def compute_infonce_loss(g_emb, t_emb, tau):
    """
    Computes InfoNCE contrastive loss for paired (graph, caption) representations.
    g_emb: (batch_size, d_model)
    t_emb: (batch_size, d_model)
    """
    batch_size = g_emb.size(0)
    
    # Compute similarity matrix S = (g * t^T) / tau
    # (batch_size, batch_size)
    similarity_matrix = torch.matmul(g_emb, t_emb.t()) / tau
    
    # Target is the diagonal (matching audio-caption pairs)
    labels = torch.arange(batch_size, device=g_emb.device)
    
    # Contrastive loss (cross entropy over similarities)
    # We compute loss in both directions (Graph->Text and Text->Graph)
    loss_g2t = F.cross_entropy(similarity_matrix, labels)
    loss_t2g = F.cross_entropy(similarity_matrix.t(), labels)
    
    loss = (loss_g2t + loss_t2g) / 2
    
    return loss
