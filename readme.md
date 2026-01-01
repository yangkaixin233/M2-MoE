# M²-MoE: Joint Manifold and Message-Passing Mixture of Experts for Graph Fraud Detection

This repository contains the official implementation of **M²-MoE**, a unified framework that jointly selects geometry and message-passing architecture at the node level for graph-based fraud detection.

## Overview

M²-MoE addresses the coupled dual heterogeneity in fraud detection graphs:
- **Geometry MoE**: Selects among Euclidean, Spherical, and Hyperbolic manifolds
- **Architecture MoE**: Selects among GCN, ResGCN, and SGC message-passing operators
- **Collaborative Context Module**: Enables bidirectional feedback between the two expert systems

## Datasets

We evaluate on four public fraud detection benchmarks:

| Dataset | #Nodes | #Edges | #Features | Fraud(%) | Homophily |
|---------|--------|--------|-----------|----------|-----------|
| FDCompCN | 5,317 | 30K | 57 | 10.51 | 0.96 |
| Amazon | 11,944 | 8.8M | 25 | 6.87 | 0.95 |
| YelpChi | 45,954 | 7.7M | 32 | 14.53 | 0.77 |
| T-Finance | 39,357 | 21.2M | 10 | 4.58 | 0.97 |

### Dataset Download

**YelpChi & Amazon**:
- These datasets are publicly available and can be automatically downloaded from [DGL](https://www.dgl.ai/) or obtained from the [CARE-GNN repository](https://github.com/YingtongDou/CARE-GNN).
- Alternatively, download from the [GADBench](https://github.com/squareRoot3/GADBench) benchmark.

**T-Finance**:
- Download from the BWGNN paper's [Google Drive](https://drive.google.com/drive/folders/1PpNwvZx_YRSCDiHaBUmRIS3x1rZR7fMr?usp=sharing).
- Reference: Tang et al., "Rethinking Graph Neural Networks for Anomaly Detection", ICML 2022.

**FDCompCN**:
- A financial statement fraud dataset of Chinese companies constructed from the CSMAR database.
- Contains three relations: C-I-C (investment), C-P-C (customer), C-S-C (supplier).
- Available from the [SplitGNN repository](https://github.com/split-gnn/SplitGNN).
- Reference: Wu et al., "SplitGNN: Spectral Graph Neural Network for Fraud Detection against Heterophily", CIKM 2023.

### Data Format

Each dataset should be organized as:
```
data/
├── amazon/
│   └── Amazon.mat
├── yelp/
│   └── YelpChi.mat
├── tfinance/
│   └── tfinance.npz
└── fdcompcn/
    └── comp.dgl
```

## Requirements

```
python >= 3.8
torch >= 1.11.0
dgl >= 0.9.1
numpy >= 1.22.4
scipy >= 1.4.1
scikit-learn >= 1.1.2
```

## Usage

```bash
# Train on YelpChi
python train.py --dataset yelp --epochs 200 --patience 50

# Train on Amazon
python train.py --dataset amazon --epochs 200 --patience 50

# Train on T-Finance
python train.py --dataset tfinance --epochs 200 --patience 50

# Train on FDCompCN
python train.py --dataset fdcompcn --epochs 200 --patience 50
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{m2moe2026,
  title={M$^2$-MoE: Joint Manifold and Message-Passing Mixture of Experts for Graph Fraud Detection},
  author={Anonymous},
  booktitle={IEEE International Conference on Multimedia and Expo (ICME)},
  year={2026}
}
```

## License

This project is licensed under the MIT License.