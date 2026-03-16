# MaJEPA: Self-Supervised Audio Representation Learning

This project implements a masked Joint Embedding Predictive Architecture (JEPA) for learning representations from piano audio using a self-supervised predictive objective.

The model is trained on log-mel spectrograms from the Google MAESTRO dataset and uses a ViT-based encoder with an EMA teacher–student framework. Instead of optimizing for downstream classification, the focus is on understanding how representations evolve during training.

## Key Features

- **JEPA-based training**: Predicts latent representations of masked audio patches rather than reconstructing inputs or using contrastive objectives  
- **Transformer architecture**: Vision Transformer (ViT) encoder applied to time–frequency spectrogram patches  
- **Self-supervised objective**: Masked prediction with no labels  
- **Variance regularization**: SIGReg-inspired constraint to encourage isotropic and well-conditioned embeddings  

## Analysis & Evaluation

The project emphasizes **representation diagnostics** rather than benchmark performance:

- Temporal consistency across time offsets  
- Representation isotropy (random-pair cosine similarity)  
- Effective dimensionality (participation ratio)  
- Intra- vs inter-piece embedding geometry  
- Downstream classification as an auxiliary probe  

## Results

Key observations include:

- Increasing temporal localization over training  
- Improved isotropy and expansion of effective dimensionality  
- Reduced piece-level separability despite better-conditioned representations  
- Peak downstream performance at intermediate training stages  

## Compute

- Trained to 100k steps on a single NVIDIA T4 GPU (~9 hours)

## Paper

See the full report for details:  
[PDF](./MaJEPA.pdf)

## Notes

This project focuses on **understanding representation structure** in predictive self-supervised learning, rather than maximizing downstream task performance.
