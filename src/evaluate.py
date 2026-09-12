import torch
import numpy as np
from sklearn.metrics import f1_score, average_precision_score

# --- Evaluate Task 3 (Multi-Label Classification) ---
def evaluate_fusion_model(model, test_loader, device='cpu'):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_graphs, input_ids, attention_mask, labels in test_loader:
            batch_graphs = batch_graphs.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            logits, _ = model(
                x_graph=batch_graphs.x, 
                edge_index=batch_graphs.edge_index, 
                batch=batch_graphs.batch, 
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.cpu().numpy())
            
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Threshold probabilities to get binary predictions
    binary_preds = (all_preds > 0.5).astype(int)
    
    # Calculate metrics
    macro_f1 = f1_score(all_labels, binary_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(all_labels, binary_preds, average='micro', zero_division=0)
    auc_pr = average_precision_score(all_labels, all_preds, average='macro')
    
    print("--- Task 3 Evaluation Results ---")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Micro-F1: {micro_f1:.4f}")
    print(f"AUC-PR:   {auc_pr:.4f}")
    
    return {"macro_f1": macro_f1, "micro_f1": micro_f1, "auc_pr": auc_pr}

# --- Evaluate Task 4 (Contrastive Retrieval) ---
def evaluate_contrastive_retrieval(model, test_loader, device='cpu'):
    model.eval()
    all_g_emb = []
    all_t_emb = []
    
    with torch.no_grad():
        for batch_graphs, input_ids, attention_mask in test_loader:
            batch_graphs = batch_graphs.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            g_emb, t_emb = model(
                x_graph=batch_graphs.x, 
                edge_index=batch_graphs.edge_index, 
                batch=batch_graphs.batch, 
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
            
            all_g_emb.append(g_emb.cpu())
            all_t_emb.append(t_emb.cpu())
            
    # Concatenate all embeddings
    g_emb = torch.cat(all_g_emb, dim=0) # (N, d_model)
    t_emb = torch.cat(all_t_emb, dim=0) # (N, d_model)
    
    N = g_emb.size(0)
    
    # Compute full similarity matrix (N x N)
    # Since they are L2 normalized, dot product is cosine similarity
    sim_matrix = torch.mm(t_emb, g_emb.t()) # Text query -> Audio retrieval
    
    # Calculate Recall@K
    def calculate_recall_at_k(sim_matrix, k):
        # Top-K predictions for each row (text query)
        _, topk_indices = sim_matrix.topk(k, dim=1)
        
        # Check if the correct index (diagonal i == j) is in the top K
        correct = 0
        for i in range(N):
            if i in topk_indices[i]:
                correct += 1
        return correct / N
        
    r1 = calculate_recall_at_k(sim_matrix, 1)
    r5 = calculate_recall_at_k(sim_matrix, 5)
    r10 = calculate_recall_at_k(sim_matrix, 10)
    
    print("--- Task 4 Retrieval Results (Text -> Audio) ---")
    print(f"R@1:  {r1:.4f}")
    print(f"R@5:  {r5:.4f}")
    print(f"R@10: {r10:.4f}")
    
    return {"R@1": r1, "R@5": r5, "R@10": r10}
