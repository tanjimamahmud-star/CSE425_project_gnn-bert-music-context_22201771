import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from transformers import BertTokenizer

import sys
# Add src to python path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from audio_features import get_segmented_features
from graph_builder import build_segment_graph
from train import train_fusion_model

class GTZANDemoDataset(Dataset):
    def __init__(self, data_dir, max_files_per_genre=5):
        """
        Loads a subset of the GTZAN dataset to demonstrate the end-to-end pipeline.
        """
        self.data_dir = data_dir
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        self.genres = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.genre_to_id = {g: i for i, g in enumerate(self.genres)}
        
        self.data = []
        
        print(f"Found {len(self.genres)} genres. Processing dataset...")
        
        for genre in self.genres:
            genre_dir = os.path.join(data_dir, genre)
            files = glob.glob(os.path.join(genre_dir, '*.wav'))
            
            # Process just a few files per genre so the demo runs quickly
            for f in files[:max_files_per_genre]:
                print(f"Processing: {os.path.basename(f)}")
                
                # 1. Audio Processing: Extract segments and log-mel features
                seg_feats = get_segmented_features(f, feature_type='log_mel', segment_length_sec=3.0)
                if seg_feats is None or seg_feats.shape[0] == 0:
                    continue
                
                # 2. Graph Building: Create segment graph with temporal & cosine edges
                graph_data = build_segment_graph(seg_feats, tau=0.7)
                
                # 3. Text Processing: Use genre as a proxy for captions/tags
                text_description = f"This music track belongs to the {genre} genre."
                tokens = self.tokenizer(
                    text_description, 
                    padding='max_length', 
                    max_length=16, 
                    truncation=True, 
                    return_tensors='pt'
                )
                
                # 4. Labeling: Create a multi-label target (required for BCEWithLogitsLoss)
                label = torch.zeros(len(self.genres))
                label[self.genre_to_id[genre]] = 1.0
                
                self.data.append((
                    graph_data, 
                    tokens['input_ids'].squeeze(0), 
                    tokens['attention_mask'].squeeze(0), 
                    label
                ))
                
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # Returns (graph, input_ids, attention_mask, label)
        return self.data[idx]

if __name__ == "__main__":
    # Point this to your actual GTZAN dataset location
    DATA_DIR = "/mnt/d/Uni/CSE425/neural-networks-project-main/Data/genres_original"
    
    print("==================================================")
    print("  GNN-BERT Fusion End-to-End Demo (Task 3)        ")
    print("==================================================")
    
    # 1. Initialize Dataset
    dataset = GTZANDemoDataset(DATA_DIR, max_files_per_genre=3) # ~30 files total for a fast demo
    
    if len(dataset) == 0:
        print("No valid audio files processed. Check the dataset path.")
    else:
        print(f"\nSuccessfully processed {len(dataset)} tracks into graphs and text tokens.")
        
        # 2. Create DataLoader
        # Note: PyTorch Geometric's DataLoader automatically batches the graph Data objects
        # while normally batching the text tensors and labels!
        train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # 3. Determine Device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\nStarting training loop on device: {device}")
        
        # 4. Train the Model
        num_classes = len(dataset.genres)
        trained_model = train_fusion_model(
            train_loader=train_loader, 
            num_classes=num_classes, 
            epochs=3, 
            device=device
        )
        
        print("\n==================================================")
        print("  Demo Finished Successfully!                     ")
        print("==================================================")
