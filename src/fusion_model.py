import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttention(nn.Module):
    def __init__(self, d_model):
        super(CrossAttention, self).__init__()
        self.d_model = d_model
        
        # Q comes from GNN (graph representation), K, V come from BERT (text representation)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
    def forward(self, g, H_text):
        """
        g: Graph-level readout (batch_size, d_model)
        H_text: BERT contextual embeddings (batch_size, seq_len, d_model)
        """
        # Ensure g has sequence length dimension for attention calculation
        Q = self.W_q(g.unsqueeze(1))          # (batch_size, 1, d_model)
        K = self.W_k(H_text)                  # (batch_size, seq_len, d_model)
        V = self.W_v(H_text)                  # (batch_size, seq_len, d_model)
        
        # A = softmax(Q K^T / sqrt(d))
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_model)  # (batch_size, 1, seq_len)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Context vector from cross attention
        attended_text = torch.bmm(attention_weights, V).squeeze(1)          # (batch_size, d_model)
        
        return attended_text, attention_weights

class GNNBERTFusion(nn.Module):
    def __init__(self, gnn_encoder, bert_encoder, num_classes, d_model=768, fusion_type='cross_attention'):
        """
        Task 3: GNN-BERT Fusion for Multi-Context Understanding
        """
        super(GNNBERTFusion, self).__init__()
        self.gnn = gnn_encoder
        self.bert = bert_encoder
        self.fusion_type = fusion_type
        
        # Projection layer if GNN output dim doesn't match BERT dim
        self.gnn_proj = nn.Linear(self.gnn.gnn.out_channels if hasattr(self.gnn, 'gnn') else self.gnn.convs[-1].out_channels, d_model)
        
        if fusion_type == 'cross_attention':
            self.cross_attn = CrossAttention(d_model)
            # z = CONCAT(g, A H_text) -> dim is 2 * d_model
            self.classifier = nn.Linear(2 * d_model, num_classes)
        elif fusion_type == 'early_concat':
            # Simply concat GNN readout and BERT CLS token
            self.classifier = nn.Linear(2 * d_model, num_classes)
        else:
            raise ValueError("Unsupported fusion type.")

    def forward(self, x_graph, edge_index, batch, input_ids, attention_mask):
        # 1. GNN Representation
        g, _ = self.gnn(x_graph, edge_index, batch)
        g_proj = self.gnn_proj(g)
        
        # 2. BERT Representation
        H_text, t_cls = self.bert(input_ids, attention_mask)
        
        # 3. Fusion
        if self.fusion_type == 'cross_attention':
            attended_text, attn_weights = self.cross_attn(g_proj, H_text)
            z = torch.cat([g_proj, attended_text], dim=-1)
        elif self.fusion_type == 'early_concat':
            z = torch.cat([g_proj, t_cls], dim=-1)
            
        # 4. Classification
        logits = self.classifier(z)
        # Note: BCEWithLogitsLoss combines sigmoid and BCE, so we return logits
        
        return logits, z
