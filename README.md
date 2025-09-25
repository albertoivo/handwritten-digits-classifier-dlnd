# MNIST Handwritten Digits Classifier

A neural network implementation for classifying handwritten digits using the MNIST dataset. This project demonstrates building and training a fully connected neural network from scratch using PyTorch.

## Results

- **Baseline Model**: 97.26% test accuracy
- **Improved Model**: 97.45% test accuracy
- **Architecture**: 3-layer fully connected network (784 → 128 → 64 → 10)
- **Parameters**: 109,386 trainable parameters

## Environment Setup

Create and activate the conda environment:

```bash
conda create -n "dl_env" python=3.10 pandas numpy pytorch torchvision torchaudio matplotlib -c pytorch -c conda-forge
conda activate dl_env
```

Alternatively, use pip with the requirements file:

```bash
pip install -r requirements.txt
```

## Usage

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd handwritten-digits-classifier-dlnd
   ```

2. **Run the notebook**:
   ```bash
   jupyter notebook MNIST_Handwritten_Digits.ipynb
   ```

3. **Follow the notebook sections**:
   - Data loading and preprocessing
   - Model architecture definition
   - Training and validation
   - Testing and evaluation
   - Model improvements

## Model Architecture

```
MNISTNet(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc1): Linear(in_features=784, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=10, bias=True)
  (dropout): Dropout(p=0.2, inplace=False)
)
```

## Key Features

- **Data Preprocessing**: Normalization with mean=0.5, std=0.5
- **Regularization**: Dropout (0.2) and L2 weight decay (1e-4)
- **Optimization**: Adam optimizer with learning rate scheduling
- **Evaluation**: Comprehensive testing with per-class accuracy metrics

## Benchmarks

| Model | Accuracy |
|-------|----------|
| Lecun et al. (1998) - Linear | 88.0% |
| Lecun et al. (1998) - CNN | 95.3% |
| **This Project** | **97.45%** |
| Ciresan et al. (2011) - CNN | 99.65% |

## Files

- `MNIST_Handwritten_Digits.ipynb` - Main notebook with complete implementation
- `requirements.txt` - Python dependencies
- `data/` - MNIST dataset (auto-downloaded)
- `mnist_model.pth` - Saved model weights

## License

This project is part of a Deep Learning Nanodegree program.