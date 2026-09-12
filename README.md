# GNN-Based BERT for Understanding Context from Music
This repository contains the implementation of a hybrid Graph Neural Network (GNN) and BERT architecture to understand multi-modal musical context.

## Project Structure
* `src/`: Contains the modular deep learning architecture (GraphSAGE, BERT, Cross-Attention Fusion, and InfoNCE Contrastive Loss).
* `notebooks/`: Contains `demo_context.ipynb`, an end-to-end interactive Google Colab notebook demonstrating data processing, training, and evaluation.
* `results/plots/`: Contains training convergence curves, t-SNE latent space visualizations, and BERT attention heatmaps.
* `report/`: Contains the final NeurIPS-formatted academic report detailing the methodology and results.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Open `notebooks/demo_context.ipynb` in Google Colab.
3. Ensure the GTZAN dataset is accessible.
4. Run the cells to extract structural graphs, tokenize text, and train the GNN-BERT Fusion model.
