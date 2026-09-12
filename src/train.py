import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from bert_encoder import BertTextEncoder
from gnn_model import MusicGNNEncoder
from fusion_model import GNNBERTFusion
from contrastive import ContrastiveDualEncoder, compute_infonce_loss

# --- Task 3: Supervised Training Loop (Fusion) ---
def train_fusion_model(train_loader, num_classes, d_model=768, epochs=10, lr=1e-4, device='cpu'):
    print("Initializing Task 3 Models (GNN-BERT Fusion)...")
    bert = BertTextEncoder().to(device)
    # in_channels depends on whether you use mel (128) or chroma (12). Let's assume 128.
    gnn = MusicGNNEncoder(in_channels=128, hidden_channels=256, out_channels=d_model, model_type='graphsage').to(device)
    fusion_model = GNNBERTFusion(gnn, bert, num_classes=num_classes, d_model=d_model, fusion_type='cross_attention').to(device)
    
    optimizer = Adam(fusion_model.parameters(), lr=lr)
    # Using BCEWithLogitsLoss for multi-label classification (tags/genres)
    criterion = nn.BCEWithLogitsLoss()
    
    fusion_model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Task 3 Epoch {epoch+1}/{epochs}")
        
        # train_loader should yield: (graphs, input_ids, attention_mask, labels)
        for batch_graphs, input_ids, attention_mask, labels in progress_bar:
            batch_graphs = batch_graphs.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device).float()
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, _ = fusion_model(
                x_graph=batch_graphs.x, 
                edge_index=batch_graphs.edge_index, 
                batch=batch_graphs.batch, 
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
            
            # Loss calculation
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(train_loader):.4f}")
        
    return fusion_model

# --- Task 4: Contrastive Training Loop (MusicCaps) ---
def train_contrastive_model(train_loader, d_model=768, epochs=10, lr=1e-4, tau=0.07, device='cpu'):
    print("Initializing Task 4 Models (Contrastive Dual-Encoder)...")
    bert = BertTextEncoder().to(device)
    gnn = MusicGNNEncoder(in_channels=128, hidden_channels=256, out_channels=d_model, model_type='graphsage').to(device)
    dual_encoder = ContrastiveDualEncoder(gnn, bert, d_model=d_model, tau=tau).to(device)
    
    optimizer = Adam(dual_encoder.parameters(), lr=lr)
    
    dual_encoder.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Task 4 Epoch {epoch+1}/{epochs}")
        
        # train_loader should yield matched pairs: (graphs, input_ids, attention_mask)
        for batch_graphs, input_ids, attention_mask in progress_bar:
            batch_graphs = batch_graphs.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass to get normalized embeddings
            g_emb, t_emb = dual_encoder(
                x_graph=batch_graphs.x, 
                edge_index=batch_graphs.edge_index, 
                batch=batch_graphs.batch, 
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
            
            # Contrastive InfoNCE Loss
            loss = compute_infonce_loss(g_emb, t_emb, tau=tau)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"InfoNCE loss": loss.item()})
            
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(train_loader):.4f}")
        
    return dual_encoder
