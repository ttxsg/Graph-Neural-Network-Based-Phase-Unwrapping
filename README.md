# Graph Neural Network-Based Phase Unwrapping (GNNPU)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 1.12+](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![PyG 2.2+](https://img.shields.io/badge/PyG-2.2+-brightgreen.svg)](https://pytorch-geometric.readthedocs.io/en/latest/)

This repository contains the **official PyTorch implementation** of the following work:

> **Graph Neural Network-Based Phase Unwrapping for Sparse and Discontinuous Surfaces in Laser Interferometry**  
> *Zheng Xinxin, Ping Zhong, et al.*  
> Submitted to *Optics and Lasers in Engineering (2025)*

---

### 🧠 Overview
GNNPU is a graph-based phase unwrapping framework designed for **sparse** and **discontinuous** phase data.  
By constructing a Delaunay graph from sparse samples and leveraging a Graph Attention Network (GATv2), the method predicts **integer wrapping count differences** between connected nodes and integrates them to reconstruct a **globally consistent unwrapped phase**.

<p align="center">
  <img src="assets/teaser.png" width="800" alt="GNNPU Workflow">
</p>
<p align="center">
  <em><b>Fig. 1</b>: Overview of GNNPU. From sparse wrapped phase data (left), a Delaunay graph is constructed. The GNN predicts integer wrapping count differences along edges, which are integrated to yield a globally continuous unwrapped phase (right).</em>
</p>

📜 Citation Section（更合适投稿阶段）
### 📜 Citation

If you find this repository useful in your research, please cite:

```bibtex
@article{zheng2025gnnpu,
  title   = {Graph Neural Network-Based Phase Unwrapping for Sparse and Discontinuous Surfaces in Laser Interferometry},
  author  = {Zheng, Xinxin and Zhong, Ping},
  journal = {Optics and Lasers in Engineering},
  year    = {2025},
  note    = {Manuscript submitted for publication}
}
