# Seq2Kd: Protein-DNA Binding Affinity Prediction

Seq2Kd is a deep learning model for predicting protein-DNA binding affinities ($K_d$) using cross-attention mechanisms between protein and DNA sequence representations. The model leverages high-dimensional embeddings from the ESM2 protein language model to achieve high-precision affinity estimation directly from sequences.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Basic Inference](#basic-inference)
- [Input Format](#input-format)
- [Benchmarks](#benchmarks)
- [Configuration](#configuration)
- [Model Checkpoints](#model-checkpoints)
- [Dependencies](#dependencies)
- [License](#license)

## Features

- **Affinity Prediction**: Predicts protein-DNA binding affinities from sequence
- **ESM2 integration**: Uses state-of-the-art protein language model embeddings
- **Cross-attention mechanism**: Models protein-DNA interactions through bidirectional attention

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/luojunchen/Seq2Kd.git
cd Seq2Kd
```

### 2. Create conda environment

**Option A: Using environment.yml (Recommended)**

```bash
conda env create -f environment.yml
conda activate seq2kd
```

**Option B: Manual installation**

```bash
# Create environment
conda create -n seq2kd python=3.10
conda activate seq2kd

# Install PyTorch with CUDA 12.4
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Install PyTorch Geometric
pip install torch-geometric==2.6.0
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

# Install ESM2
pip install fair-esm==2.0.0

# Install other dependencies
pip install numpy==1.26.4 pandas==2.2.3 scipy==1.14.0
pip install click==8.0.4 tqdm==4.66.5 ruamel.yaml==0.16.12
pip install matplotlib==3.9.1 einops==0.8.0 biopython==1.84
```

### 3. Download ESM2 model weights

The ESM2 model weights will be automatically downloaded on first use. Alternatively, you can pre-download them:

```bash
mkdir -p checkpoints/esm
cd checkpoints/esm
wget https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt
# Weights will be downloaded to this directory on first run
```

For more information about ESM2, see: https://github.com/facebookresearch/esm


## Project Structure

```
Seq2Kd/
├── cfg/                         # Configuration files
│   └── cfg01.yaml
├── checkpoints/                 # Model checkpoints
│   ├── esm/                     # ESM2 model weights
│   ├── seq2kd_basic.pt          # Seq2Kd model weights
│   └── seq2kd_distillation.pt   # Weights including self-distillation
├── data/ 
│   ├── DNA_mutation_dataset.txt # Benchmark data for DNA sequence variations
│   └── TF_mutation_dataset.txt  # Benchmark data for TF amino acid mutations
├── src/
│   ├── inference.py             # Main inference script
│   └── models/
│       ├── seq2kd.py            # Main Seq2Kd model
│       ├── esm_embedding.py     # ESM2 embedding wrapper
│       ├── datasets.py          # Dataset classes
│       ├── data_utils.py        # Data utilities
│       └── common.py            # Common utilities
├── _inputs/                     # Input data directory
├── _outputs/                    # Output results directory
├── run.sh                       # Batch script for standard inference pipeline
├── environment.yml              # Conda environment file
└── README.md                    # This file
```

## Usage

### Basic Inference

To run binding affinity prediction, prepare your sequences in the input format and run:

```bash
python src/inference.py \
  --model-path ./checkpoints/seq2kd_basic.pt \
  --config-path ./cfg/cfg01.yaml \
  --input-path ./_inputs/your_input.txt \
  --output-dir ./_outputs
```

Or use the provided script:

```bash
bash run.sh
```

## Input Format

Input files should be plain text files with space-separated DNA and protein sequences:

```
ATCGATCG... MKTVRQERLK...
GCTAGCTA... MPPEPTIDE...
```

Each line contains:
1. DNA sequence (A, T, C, G characters)
2. Protein sequence (amino acid characters)

## Benchmarks
Seq2Kd has been extensively tested on experimental datasets to verify its predictive performance across different biological scenarios:

- DNA Mutation Benchmark: Evaluated using other experimental data focusing on DNA sequence variations and their impact on binding strength across diverse TFs.

- TF Mutation Benchmark: Tested against datasets containing experimental affinity measurements for various protein sequence variants, demonstrating the model's sensitivity to protein-side changes.

## Configuration

Configuration files are located in `cfg/` directory. Key parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `d_model` | Hidden dimension size | 512 |
| `num_cross` | Number of cross-attention layers | 12 |
| `use_protein_len` | Maximum protein sequence length | 512 |
| `dna_len` | Maximum DNA sequence length | 41 |
| `esm_path` | Path to ESM2 checkpoints | ./checkpoints/esm |

## Model Checkpoints

Place model checkpoint files in the `checkpoints/` directory:

| Checkpoint | Description |
|------------|-------------|
| `seq2kd_basic.pt` | Base model trained on FOODIE experimental binding affinity data |
| `seq2kd_distillation.pt` | Enhanced model with additional self-distillation training |

## Dependencies

### Main Environment (seq2kd)

The main dependencies are specified in `environment.yml`. Key packages include:

| Package | Version | Description |
|---------|---------|-------------|
| Python | 3.10 | Programming language |
| PyTorch | 2.4.0 | Deep learning framework (CUDA 12.4) |
| torch-geometric | 2.6.0 | Graph neural network library |
| torch-cluster | 1.6.3 | Clustering algorithms for PyTorch |
| torch-scatter | 2.1.2 | Scatter operations for PyTorch |
| torch-sparse | 0.6.18 | Sparse tensor operations |
| fair-esm | 2.0.0 | ESM2 protein language model |
| numpy | 1.26.4 | Numerical computing |
| pandas | 2.2.3 | Data manipulation |
| scipy | 1.14.0 | Scientific computing |
| click | 8.0.4 | Command-line interface |
| tqdm | 4.66.5 | Progress bars |
| ruamel.yaml | 0.16.12 | YAML parsing |
| matplotlib | 3.9.1 | Visualization |
| einops | 0.8.0 | Tensor operations |
| biopython | 1.84 | Biological computation |

### ESM2 Model

This project uses ESM2 (Evolutionary Scale Modeling) for protein sequence embeddings. The model weights (~2.5GB for esm2_t33_650M_UR50D) are automatically downloaded from Facebook AI Research servers on first use.

- Model used: `esm2_t33_650M_UR50D` (650M parameters)
- Output dimension: 1280
- GitHub: https://github.com/facebookresearch/esm

## License

This project is licensed under the MIT License.

## Acknowledgments

- [ESM](https://github.com/facebookresearch/esm) - Protein language models from Facebook AI Research
