
# Graph Neural Network-Based Phase Unwrapping (GNNPU)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 1.12+](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![PyG 2.2+](https://img.shields.io/badge/PyG-2.2+-brightgreen.svg)](https://pytorch-geometric.readthedocs.io/en/latest/)

This repository provides the official PyTorch implementation for the paper:

> **Graph Neural Network-Based Phase Unwrapping for Sparse Discontinuous Surfaces in Laser Interferometry**  
> *[List of Authors, e.g., Your Name, Co-author's Name, ..., Ping Zhong]*  
> Submitted to *Optics and Lasers in Engineering*

Our work introduces GNNPU, a novel algorithm for phase unwrapping from sparse and discontinuous data. By representing sparse points as a graph and leveraging a Graph Attention Network (GATv2), GNNPU accurately predicts wrap count differences and reconstructs a globally consistent phase map.

<p align="center">
  <img src="assets/teaser.png" width="800" alt="GNNPU Workflow">
</p>
<p align="center">
  <em><b>Fig. 1</b>: The GNNPU workflow. From a sparse wrapped phase (left), a Delaunay graph is constructed. The GNN predicts wrap count differences on edges, which are then integrated to produce the final, continuous unwrapped phase (right).</em>
</p>

---

## 🚀 Getting Started

### 1. Environment Setup

First, clone the repository and navigate into the project directory:
```bash
git clone https://github.com/ttxsg/Graph-Neural-Network-Based-Phase-Unwrapping.git
cd Graph-Neural-Network-Based-Phase-Unwrapping
We recommend creating a dedicated conda environment:
code
Bash
conda create -n gnnpu python=3.8
conda activate gnnpu
Install the required dependencies. Please follow the official instructions for PyTorch and PyG to match your CUDA version.
a. Install PyTorch:
code
Bash
# Example for CUDA 11.7
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
b. Install PyTorch Geometric (PyG):
code
Bash
# Corresponds to PyTorch 1.13.1 and CUDA 11.7
pip install pyg_lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
c. Install other dependencies:
code
Bash
pip install -r requirements.txt
2. Dataset and Pre-trained Model
Download the pre-processed dataset and our pre-trained model from the following link:
[➡️ Download Dataset & Model from Google Drive (Your Link Here) ⬅️]
After downloading, unzip the file and place the contents into the project's root directory. The expected file structure should be:
code
Text
.
├── checkpoints/
│   └── gnnpu_pretrained.pth      # Pre-trained model
├── data/
│   └── sparse_dataset_1018/
│       ├── train/
│       │   ├── sample_001.mat
│       │   └── ...
│       └── test/
│           ├── sample_001.mat
│           └── ...
└── ... (other project files)
Our dataset consists of .mat files, each containing true_phase and wrapped_phase arrays.
💻 Usage
Training
To train the GNNPU model from scratch on the provided dataset, simply run the main training script. All hyperparameters are configured within train.py.
code
Bash
python train.py
Training progress will be displayed, and the best model weights will be saved to train3_927_11_28_data_all.pth by default.
Inference
We provide a simple script, inference.py, to perform phase unwrapping on a single .mat file using a pre-trained model.
code
Bash
python inference.py \
    --model_path "checkpoints/gnnpu_pretrained.pth" \
    --input_path "data/sparse_dataset_1018/test/sample_001.mat" \
    --output_path "results/unwrapped_sample_001.png"
This will process the input file and save a visual comparison of the wrapped input, ground truth, and unwrapped result.
🏛️ Model Architecture
The core of our model is EdgeRegressionGNN_v2, which is built upon multiple layers of GATv2Conv. Key features include:
Rich Node Features: Combining local phase gradients with absolute positional encodings.
Edge-Aware Attention: Incorporating relative position and distance as edge attributes into the GATv2 attention mechanism.
Multi-layer Deep GNN: A deep stack of 6 GATv2 layers to capture long-range dependencies across the graph.
Edge Regressor: A final MLP that predicts the wrap count difference from the concatenated features of two connected nodes.
📜 Citation
If you find our work or this repository useful for your research, please consider citing our paper:
code
Bibtex
@article{zhong2024gnnpu,
  title   = {Graph Neural Network-Based Phase Unwrapping for Sparse Discontinuous Surfaces in Laser Interferometry},
  author  = {Your Name, Another Author, Ping Zhong},
  journal = {Optics and Lasers in Engineering},
  year    = {2024},
  note    = {(Under review)}
}
(Please update the BibTeX entry once the paper is officially published.)
📄 License
This project is licensed under the MIT License. See the LICENSE file for more details.
🙏 Acknowledgements
This implementation relies heavily on the fantastic PyTorch Geometric library. We also thank the authors of previous works for making their code and data available, which has greatly benefited the community.
