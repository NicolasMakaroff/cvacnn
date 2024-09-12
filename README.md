# CVACNN: Chan-Vese Attention Convolutional Neural Network

## Description
This repository contains the implementation of **CVACNN (Chan-Vese Attention Convolutional Neural Network)** in **JAX**. The project is part of the implementation described in the accompanying research paper, where we propose a novel integration of the Chan-Vese active contour model with attention mechanisms in a convolutional neural network for robust image segmentation.

CVACNN leverages the Chan-Vese model's capacity for unsupervised image segmentation and incorporates attention mechanisms to capture contextual information effectively. The implementation in JAX ensures fast and efficient training with GPU/TPU support.

## Features
- **Hybrid Segmentation Model**: Integration of Chan-Vese active contours with modern attention-based neural networks.
- **Attention Mechanism**: Explores the use of self-attention to enhance feature representations for image segmentation.
- **JAX Implementation**: Fully implemented in JAX for efficient computation, GPU/TPU acceleration, and functional transformations.
- **Custom Loss Function**: Implements a hybrid loss that combines a classical energy-based loss from the Chan-Vese model with standard deep learning losses.

## Paper Reference
If you use this code for your research, please cite the following paper:

**[Your Paper Title]**

> Authors: Nicolas Makaroff, Laurent D. Cohen  
> Conference/Journal: International Conference on Geometric Science of Information
> Year: 2023
> [link](https://link.springer.com/chapter/10.1007/978-3-031-38299-4_59)

```bibtex
@inproceedings{makaroff2023chan,
  title={Chan-Vese Attention U-Net: An attention mechanism for robust segmentation},
  author={Makaroff, Nicolas and Cohen, Laurent D},
  booktitle={International Conference on Geometric Science of Information},
  pages={574--582},
  year={2023},
  organization={Springer}
}

```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cvacnn.git
   cd cvacnn
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. (Optional) Install JAX with CUDA support for GPU acceleration:
   ```bash
   pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```

## Usage

### Training
To train the CVACNN model on your dataset, use the following command:
```bash
python train.py 
```


