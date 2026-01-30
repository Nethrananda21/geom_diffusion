# Geometry-Complete Equivariant Diffusion for Structure-Based Drug Design

A PyTorch implementation of E(3)-equivariant diffusion models for pocket-conditioned molecular generation.

## 🎯 Key Features

- **E(3)-Equivariant Architecture**: Rotation and translation equivariant graph neural networks
- **Protein Pocket Conditioning**: Classifier-free guidance for target-aware generation
- **Cosine Noise Schedule**: Optimized for molecular geometry
- **T4-Safe Design**: Memory-efficient with hard constraints for 16GB VRAM
- **Full Evaluation Suite**: Validity, uniqueness, novelty, QED, SA score

## 📊 SOTA Targets

| Metric     | Target | Description                |
| ---------- | ------ | -------------------------- |
| Validity   | >95%   | RDKit sanitization success |
| Uniqueness | >90%   | Tanimoto < 0.9             |
| Novelty    | >80%   | Tanimoto < 0.4 to training |
| Vina Score | < -7.0 | kcal/mol binding affinity  |
| QED        | > 0.5  | Drug-likeness              |
| SA Score   | < 4.0  | Synthetic accessibility    |

## 🗂 Project Structure

```
geom_diffusion/
├── configs/
│   ├── debug_t4.yaml      # T4 GPU config (1K samples)
│   └── train_a100.yaml    # A100 GPU config (5K samples)
├── data/
│   ├── filters.py         # Quality filtering (RMSD, Vina, etc.)
│   ├── diversity.py       # MaxMin diversity selection
│   ├── pockets.py         # Pocket extraction & processing
│   └── dataset.py         # PyTorch Geometric dataset
├── models/
│   ├── egnn.py            # E(n)-Equivariant GNN layers
│   ├── diffusion.py       # DDPM with cosine schedule
│   └── encoder.py         # Frozen pocket encoder
├── train.py               # Training script
├── evaluate.py            # Evaluation metrics
└── requirements.txt       # Dependencies
```

## 🚀 Quick Start

### Installation

```bash
# Clone and enter directory
cd geom_diffusion

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Training

**Debug on T4 (Day 1)**:

```bash
python train.py --config configs/debug_t4.yaml --wandb
```

**Production on A100 (Days 2-4)**:

```bash
python train.py --config configs/train_a100.yaml --wandb
```

### Evaluation

```bash
# From checkpoint
python evaluate.py --checkpoint checkpoints/latest.pt --n_samples 100

# From SDF file
python evaluate.py --sdf generated.sdf --training_sdf training.sdf
```

## ⚠️ Critical Constraints (T4 GPU)

These constraints are **required** for T4 GPU (16GB VRAM):

| Constraint   | Value | Reason               |
| ------------ | ----- | -------------------- |
| Pocket atoms | ≤ 250 | OOM prevention       |
| Edge cutoff  | 5.0 Å | O(N²) memory scaling |
| Ligand atoms | 15-40 | Graph size limit     |
| Batch size   | 4     | Memory headroom      |

## 📈 Training Stages

### Stage A: Debug (T4)

- Dataset: 1,000 pairs (50 pockets × 20 ligands)
- Goal: Loss < 0.1, no OOM, >80% validity
- Duration: ~1 day

### Stage B: Production (A100)

- Dataset: 5,000 pairs (100 pockets × 50 ligands)
- Goal: SOTA metrics on held-out pockets
- Duration: ~3 days
- Budget: ~$75-100

## 🧪 Data Preparation

The data pipeline expects CrossDocked2020 format:

1. **Quality Filtering**: RMSD < 1.5Å, Vina ≤ -6.0, Resolution < 2.5Å
2. **Diversity Selection**: MaxMin on ECFP4, Tanimoto ≤ 0.8
3. **Size Stratification**: 30% small (15-25), 40% medium (26-35), 30% large (36-40)
4. **Preprocessing**: COM centering, heavy atoms only, 6Å pocket radius

```python
from data.filters import QualityFilter, FilterConfig
from data.diversity import select_diverse_ligands
from data.pockets import PocketProcessor

# Filter
filter_config = FilterConfig(pocket_atoms_max=250)
quality_filter = QualityFilter(filter_config)

# Select diverse ligands
selected_indices = select_diverse_ligands(mols, n_select=20)

# Process pockets
processor = PocketProcessor()
processed = processor.process(protein_coords, protein_elements,
                               ligand_coords, ligand_elements)
```

## 🔬 Model Architecture

### Conditional EGNN

```
Input: Noised ligand (x_t, h_t) + Pocket (x_p, h_p) + Timestep t
  │
  ├── Pocket Encoder (2 EGNN layers, cached)
  │
  ├── Ligand Denoiser (4-8 EGNN layers)
  │     └── E(3)-Equivariant message passing
  │
  ├── Cross-Attention (ligand → pocket)
  │
  └── Output Heads
        ├── Coordinate noise: ε_x
        └── Type noise: ε_h
```

### Diffusion Process

- Forward: q(x_t | x_0) = N(√ᾱ_t x_0, (1-ᾱ_t)I)
- Reverse: Model predicts ε, reconstruct x\_{t-1}
- Schedule: Cosine (500 steps T4, 1000 steps A100)

## 📝 Configuration

Key config options in YAML:

```yaml
model:
  egnn:
    hidden_dim: 128 # 256 for A100
    n_layers: 4 # 6-8 for A100
    edge_cutoff: 5.0 # CRITICAL: Don't exceed on T4

diffusion:
  timesteps: 500 # 1000 for A100
  guidance:
    scale: 2.0 # Higher = stronger pocket conditioning

training:
  batch_size: 4 # 32 for A100
  gradient_accumulation_steps: 4
  stability:
    clip_norm: 1.0 # Essential for EGNN
    mixed_precision: "fp16" # "bf16" for A100
```

## 📚 References

- [E(n) Equivariant Graph Neural Networks](https://arxiv.org/abs/2102.09844) - Satorras et al., 2021
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - Ho et al., 2020
- [Equivariant Diffusion for Molecule Generation](https://arxiv.org/abs/2203.17003) - Hoogeboom et al., 2022
- [CrossDocked2020](https://github.com/gnina/models) - Francoeur et al., 2020

## 📄 License

MIT License - See LICENSE file for details.
