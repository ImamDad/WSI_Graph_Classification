# WSI_Graph_Classification
HiGATE: Hierarchical Graph Attention with Cross-Level Interaction for Multi-Scale Representation Learning in Computational Pathology


# Hierarchical GNN for WSI Graph Classification

A PyTorch implementation of hierarchical graph neural networks for whole slide image classification using the PanNuke dataset.

## Features

- Hierarchical graph construction (cell-level and tissue-level)
- DINOv2 feature extraction
- Multi-scale graph attention networks
- Explainable AI with feature attribution
- External validation on multiple datasets

## Installation

bash
git clone https://github.com/ImamDad/WSI_Graph_Classification.git
cd WSI_Graph_Classification
pip install -r requirements.txt


# Training
python main.py --mode train

# Evaluation
python main.py --mode evaluate --model-path saved_models/best_model.pth

# Explainability
python main.py --mode explain --model-path saved_models/best_model.pth

# External validation
python main.py --mode validate --model-path saved_models/best_model.pth
