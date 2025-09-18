
# GloCTM: Global Cross-lingual Topic Model

A cross-lingual topic modeling framework that learns shared topic representations across multiple languages.

## Installation

Install the required dependencies:

```bash
pip install matplotlib==3.7.2 numpy==1.26.4 PyYAML==6.0.2 scikit_learn==1.2.2 scipy==1.16.1 sentence_transformers==4.1.0 torch==2.6.0+cu124 tqdm==4.67.1 transformers==4.52.4 wandb==0.21.0
```

## Quick Start

### 1. Prepare Embeddings

Generate document and word embeddings:

```bash
cd path/to/GloCTM
python create_doc_embeddings.py
python create_word_embeddings.py
```

### 2. Train Model

Start training with the provided script:

```bash
bash run.sh
```

## Reproduce

To reproduce the results from the paper:

- **Document Embeddings**: Run on Tesla T4 GPU
- **Word Embeddings & Training**: Run on Tesla P100 GPU

## Topic Coherence CNPMI

To calculate topic coherence using CNPMI metric, use the evaluation tool from:
 [Link](https://github.com/BobXWu/CNPMI)

## Configuration

Model and dataset configurations are located in:
- `configs/model/GloCTM.yaml` - Model hyperparameters
- `configs/dataset/` - Dataset-specific settings

## Output

Results are saved to the `output/` directory, including:
- Learned topic words for each language
- Model checkpoints
- Evaluation metrics