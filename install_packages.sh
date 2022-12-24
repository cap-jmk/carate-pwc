#!/bin/bash
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu 
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+${CUDA}.html
pip install torch-geometric rdkit-pypi networkx[default] matplotlib
pip install torch-cluster 
pip install torch-spline-conv